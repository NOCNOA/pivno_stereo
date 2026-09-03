#!/usr/bin/env bash
set -euo pipefail

export SCENEFLOW_ROOT="${SCENEFLOW_ROOT:-/data/public_data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo}"
CONDA_BIN="${CONDA_BIN:-/home/yijiayi/anaconda3/bin/conda}"
MASTER_PORT="${MASTER_PORT:-29546}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-320}"
IMAGE_WIDTH="${IMAGE_WIDTH:-768}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
NUM_STEPS="${NUM_STEPS:-200000}"
TRAIN_ITERS="${TRAIN_ITERS:-16}"
CORR_RADIUS="${CORR_RADIUS:-4}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-4}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable does not exist or is not executable: ${CONDA_BIN}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_LIST[@]}"
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "Global batch ${BATCH_SIZE} must be divisible by ${NUM_GPUS} GPUs." >&2
  exit 2
fi
LOCAL_BATCH_SIZE=$((BATCH_SIZE / NUM_GPUS))
if (( LOCAL_BATCH_SIZE != 2 )); then
  echo "Expected per-GPU batch 2, got ${LOCAL_BATCH_SIZE}." >&2
  exit 2
fi

GPU_IDS=()
for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
  GPU_IDS+=("${gpu_index}")
done

NAME="${NAME:-defom_pivno_gwc4_enc16_concat_gru3_nogate_noleft_rgb_d768_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_b${BATCH_SIZE}_${NUM_GPUS}gpu_200k}"
CHECKPOINT_DIR="checkpoints/${NAME}"
mkdir -p "${CHECKPOINT_DIR}"

for subset in FlyingThings3D Monkaa Driving; do
  if [[ ! -d "${SCENEFLOW_ROOT}/${subset}" ]]; then
    echo "SceneFlow subset is missing: ${SCENEFLOW_ROOT}/${subset}" >&2
    exit 2
  fi
done

echo "Launching DEFOM-PIVNO GWC4/enc16 direct-concat GRU3: GPUs=${CUDA_VISIBLE_DEVICES}, world_size=${NUM_GPUS}, global_batch=${BATCH_SIZE}, local_batch=${LOCAL_BATCH_SIZE}, lr=${LEARNING_RATE}, steps=${NUM_STEPS}, crop=${IMAGE_HEIGHT}x${IMAGE_WIDTH}, iters=${TRAIN_ITERS}, data=${SCENEFLOW_ROOT}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train_stereo.py \
  --model defom_pivno_gwc4_enc16_concat_gru3 \
  --distributed \
  --launcher pytorch \
  --gpu_ids "${GPU_IDS[@]}" \
  --name "${NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS_PER_GPU}" \
  --train_datasets sceneflow \
  --train_folds 1 \
  --image_size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
  --max_disp 768 \
  --num_steps "${NUM_STEPS}" \
  --lr "${LEARNING_RATE}" \
  --mixed_precision \
  --n_downsample 2 \
  --n_gru_layers 3 \
  --hidden_dims 128 128 128 \
  --context_norm instance \
  --train_iters "${TRAIN_ITERS}" \
  --valid_iters 32 \
  --scale_iters 0 \
  --corr_radius "${CORR_RADIUS}" \
  --save_latest_ckpt_freq 1000 \
  --save_ckpt_freq 10000 \
  --val_freq 10000 \
  "$@" 2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
