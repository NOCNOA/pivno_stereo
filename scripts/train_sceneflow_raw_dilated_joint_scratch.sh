#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/dataset_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo310}"
CONDA_BIN="${CONDA_BIN:-/home/u2025140689/anaconda3/bin/conda}"
MASTER_PORT="${MASTER_PORT:-29545}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-320}"
IMAGE_WIDTH="${IMAGE_WIDTH:-768}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
NUM_STEPS="${NUM_STEPS:-200000}"
TRAIN_ITERS="${TRAIN_ITERS:-16}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-4}"

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_LIST[@]}"
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "Global batch ${BATCH_SIZE} must be divisible by ${NUM_GPUS} GPUs." >&2
  exit 2
fi
GPU_IDS=()
for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
  GPU_IDS+=("${gpu_index}")
done

NAME="${NAME:-pivno_raw_dilated_joint_scratch_mobilenet_200k}"
CHECKPOINT_DIR="checkpoints/${NAME}"
mkdir -p "${CHECKPOINT_DIR}"

echo "Launching raw-dilated joint scratch training: GPU=${CUDA_VISIBLE_DEVICES}, batch=${BATCH_SIZE}, steps=${NUM_STEPS}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train_stereo.py \
  --model defom_pivno_gated_gru3_mobilenetv2_raw_dilated_joint \
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
  --pivno_gate_lr "${LEARNING_RATE}" \
  --mixed_precision \
  --n_downsample 2 \
  --n_gru_layers 3 \
  --hidden_dims 128 128 128 \
  --context_norm instance \
  --train_iters "${TRAIN_ITERS}" \
  --valid_iters 32 \
  --scale_iters 0 \
  --corr_radius 4 \
  --save_latest_ckpt_freq 1000 \
  --save_ckpt_freq 10000 \
  --val_freq 50000 \
  --wdecay 1e-5 \
  "$@" 2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
