#!/usr/bin/env bash
set -euo pipefail

export SCENEFLOW_ROOT="${SCENEFLOW_ROOT:-/data/public_data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo}"
CONDA_BIN="${CONDA_BIN:-/home/yijiayi/anaconda3/bin/conda}"
MASTER_PORT="${MASTER_PORT:-29560}"

# Reuse the completed hidden-SR head. The new fusion initially copies hidden
# exactly and gives RGB zero contribution.
SOURCE_CKPT="${SOURCE_CKPT:-checkpoints/defom_pivno_gated_gru3_gwc4_mask_sr_head_d768_20k/checkpoint_latest.pth}"
NAME="${NAME:-defom_pivno_gated_gru3_gwc4_mask_rgb_hidden_sr_head_d768_10k}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-320}"
IMAGE_WIDTH="${IMAGE_WIDTH:-768}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-4}"
NEW_MODULE_LR="${NEW_MODULE_LR:-0.0001}"
PRETRAINED_HEAD_LR="${PRETRAINED_HEAD_LR:-0.00001}"
NUM_STEPS="${NUM_STEPS:-10000}"
TRAIN_ITERS="${TRAIN_ITERS:-32}"
VALID_ITERS="${VALID_ITERS:-32}"
RESIDUAL_MAX="${RESIDUAL_MAX:-4.0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable is unavailable: ${CONDA_BIN}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "Trained hidden-SR checkpoint does not exist: ${SOURCE_CKPT}" >&2
  exit 2
fi
for subset in FlyingThings3D Monkaa Driving; do
  if [[ ! -d "${SCENEFLOW_ROOT}/${subset}" ]]; then
    echo "SceneFlow subset is missing: ${SCENEFLOW_ROOT}/${subset}" >&2
    exit 2
  fi
done

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_LIST[@]}"
if (( GLOBAL_BATCH_SIZE <= 0 || GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be positive and divisible by ${NUM_GPUS}." >&2
  exit 2
fi
LOCAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))
GPU_IDS=()
for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
  GPU_IDS+=("${gpu_index}")
done
CHECKPOINT_DIR="checkpoints/${NAME}"
mkdir -p "${CHECKPOINT_DIR}"

echo "C32/GWC4 gated-GRU3 RGB/hidden fusion-SR launch"
echo "  source=${SOURCE_CKPT}"
echo "  name=${NAME} stage=head base=frozen"
echo "  new_module_lr=${NEW_MODULE_LR} pretrained_head_lr=${PRETRAINED_HEAD_LR}"
echo "  GPUs=${CUDA_VISIBLE_DEVICES} global_batch=${GLOBAL_BATCH_SIZE} local_batch=${LOCAL_BATCH_SIZE}"
echo "  crop=${IMAGE_HEIGHT}x${IMAGE_WIDTH} steps=${NUM_STEPS} residual_max=${RESIDUAL_MAX}"
echo "  train_iters=${TRAIN_ITERS} valid_iters=${VALID_ITERS} SR_calls_per_forward=1"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: configuration is valid; training was not started."
  exit 0
elif [[ "${DRY_RUN}" != "0" ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train_stereo.py \
  --model defom_pivno_gated_gru3_gwc4_mask_rgb_hidden_sr \
  --distributed \
  --launcher pytorch \
  --gpu_ids "${GPU_IDS[@]}" \
  --name "${NAME}" \
  --resume_ckpt "${SOURCE_CKPT}" \
  --no_resume_optimizer \
  --pivno_mask_sr_stage head \
  --pivno_mask_sr_residual_max "${RESIDUAL_MAX}" \
  --pivno_mask_sr_pretrained_lr "${PRETRAINED_HEAD_LR}" \
  --batch_size "${GLOBAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS_PER_GPU}" \
  --train_datasets sceneflow \
  --train_folds 1 \
  --image_size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
  --max_disp 768 \
  --num_steps "${NUM_STEPS}" \
  --lr "${NEW_MODULE_LR}" \
  --mixed_precision \
  --n_downsample 2 \
  --n_gru_layers 3 \
  --hidden_dims 128 128 128 \
  --context_norm instance \
  --train_iters "${TRAIN_ITERS}" \
  --valid_iters "${VALID_ITERS}" \
  --scale_iters 0 \
  --corr_radius 4 \
  --save_latest_ckpt_freq 1000 \
  --save_ckpt_freq 5000 \
  --val_freq 2500 \
  "$@" 2>&1 | tee -a "${CHECKPOINT_DIR}/train.log"
