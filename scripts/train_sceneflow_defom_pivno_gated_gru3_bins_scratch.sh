#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source "${SCRIPT_DIR}/dataset_env.sh"

# Train the final bins + gated-GRU3 architecture entirely from scratch.
# No checkpoint is loaded and every parameter is trainable from step one.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo310}"
CONDA_BIN="${CONDA_BIN:-/home/u2025140689/anaconda3/bin/conda}"
MASTER_PORT="${MASTER_PORT:-29547}"

IMAGE_HEIGHT="${IMAGE_HEIGHT:-320}"
IMAGE_WIDTH="${IMAGE_WIDTH:-768}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-4}"
NUM_STEPS="${NUM_STEPS:-200000}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
TRAIN_ITERS="${TRAIN_ITERS:-16}"
CORR_RADIUS="${CORR_RADIUS:-4}"
MAX_DISP="${MAX_DISP:-768}"

NUM_INIT_BINS="${NUM_INIT_BINS:-48}"
INIT_CORR_SCALE="${INIT_CORR_SCALE:-10.0}"
INIT_MAX_OFFSET="${INIT_MAX_OFFSET:-48.0}"

NAME="${NAME:-defom_pivno_gated_gru3_bins_scratch_d${MAX_DISP}_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_b${BATCH_SIZE}_200k}"
CHECKPOINT_DIR="checkpoints/${NAME}"
EXTRA_ARGS=("$@")

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable does not exist or is not executable: ${CONDA_BIN}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_LIST[@]}"
if (( NUM_GPUS < 1 )); then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU." >&2
  exit 2
fi
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "Global batch ${BATCH_SIZE} must be divisible by ${NUM_GPUS} GPUs." >&2
  exit 2
fi
LOCAL_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))

GPU_IDS=()
for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
  GPU_IDS+=("${gpu_index}")
done

for subset in FlyingThings3D Monkaa Driving; do
  if [[ ! -d "${SCENEFLOW_ROOT}/${subset}" ]]; then
    echo "SceneFlow subset is missing: ${SCENEFLOW_ROOT}/${subset}" >&2
    exit 2
  fi
done

mkdir -p "${CHECKPOINT_DIR}"

echo "Launching DEFOM-PIVNO bins training from scratch:"
echo "  GPUs=${CUDA_VISIBLE_DEVICES}, global_batch=${BATCH_SIZE}, local_batch=${LOCAL_BATCH_SIZE}"
echo "  data=${SCENEFLOW_ROOT}"
echo "  steps=${NUM_STEPS}, all_parameter_lr=${LEARNING_RATE}"
echo "  output=${CHECKPOINT_DIR}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train_stereo.py \
  --distributed \
  --launcher pytorch \
  --gpu_ids "${GPU_IDS[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS_PER_GPU}" \
  --train_datasets sceneflow \
  --train_folds 1 \
  --image_size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
  --max_disp "${MAX_DISP}" \
  --mixed_precision \
  --n_downsample 2 \
  --n_gru_layers 3 \
  --hidden_dims 128 128 128 \
  --context_norm instance \
  --train_iters "${TRAIN_ITERS}" \
  --valid_iters 32 \
  --scale_iters 0 \
  --corr_radius "${CORR_RADIUS}" \
  --pivno_num_init_bins "${NUM_INIT_BINS}" \
  --pivno_bins_max_offset "${INIT_MAX_OFFSET}" \
  --pivno_init_corr_scale "${INIT_CORR_SCALE}" \
  --save_latest_ckpt_freq 1000 \
  --save_ckpt_freq 20000 \
  --val_freq 20000 \
  "${EXTRA_ARGS[@]}" \
  --model defom_pivno_gated_gru3_bins \
  --pivno_bins_stage joint \
  --pivno_bins_pretrained_lr "${LEARNING_RATE}" \
  --name "${NAME}" \
  --num_steps "${NUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
