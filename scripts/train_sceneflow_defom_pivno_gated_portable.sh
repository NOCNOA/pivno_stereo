#!/usr/bin/env bash
set -euo pipefail

export SCENEFLOW_ROOT="${SCENEFLOW_ROOT:-/data/public_data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo}"
MASTER_PORT="${MASTER_PORT:-29543}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-320}"
IMAGE_WIDTH="${IMAGE_WIDTH:-768}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-0.00002}"
GATE_LEARNING_RATE="${GATE_LEARNING_RATE:-0.0002}"
NUM_STEPS="${NUM_STEPS:-20000}"
TRAIN_ITERS="${TRAIN_ITERS:-18}"
CORR_RADIUS="${CORR_RADIUS:-4}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-4}"
BASE_CKPT="${BASE_CKPT:-checkpoints/defom_pivno_rgb_d768_320x768_b4_2gpu/defom_pivno_rgb_d768_320x768_b4_2gpu_200000.pth}"

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_LIST[@]}"
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "Global batch ${BATCH_SIZE} must be divisible by ${NUM_GPUS} GPUs." >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Base checkpoint does not exist: ${BASE_CKPT}" >&2
  exit 2
fi

GPU_IDS=()
for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
  GPU_IDS+=("${gpu_index}")
done

NAME="${NAME:-defom_pivno_gated_rgb_d768_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_b${BATCH_SIZE}_${NUM_GPUS}gpu_ft${NUM_STEPS}}"
CHECKPOINT_DIR="checkpoints/${NAME}"
mkdir -p "${CHECKPOINT_DIR}"

echo "Launching gated DEFOM-PIVNO: base=${BASE_CKPT}, GPUs=${CUDA_VISIBLE_DEVICES}, global_batch=${BATCH_SIZE}, base_lr=${BASE_LEARNING_RATE}, gate_lr=${GATE_LEARNING_RATE}, steps=${NUM_STEPS}, iters=${TRAIN_ITERS}"

conda run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train_stereo.py \
  --model defom_pivno_gated \
  --distributed \
  --launcher pytorch \
  --gpu_ids "${GPU_IDS[@]}" \
  --name "${NAME}" \
  --resume_ckpt "${BASE_CKPT}" \
  --no_resume_optimizer \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS_PER_GPU}" \
  --train_datasets sceneflow \
  --train_folds 1 \
  --image_size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
  --max_disp 768 \
  --num_steps "${NUM_STEPS}" \
  --lr "${BASE_LEARNING_RATE}" \
  --pivno_gate_lr "${GATE_LEARNING_RATE}" \
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
  --save_ckpt_freq 5000 \
  --val_freq 5000 \
  "$@" 2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
