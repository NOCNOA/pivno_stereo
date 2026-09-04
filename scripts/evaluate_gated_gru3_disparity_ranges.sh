#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/dataset_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo}"
CONDA_BIN="${CONDA_BIN:-/home/yijiayi/anaconda3/bin/conda}"
MASTER_PORT="${MASTER_PORT:-29631}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
VALID_ITERS="${VALID_ITERS:-32}"
RESTORE_CKPT="${RESTORE_CKPT:-checkpoints/defom_pivno_gated_gru3_gwc4gate_enc16_noleft_rgb_d768_320x768_b4_2gpu_200k/200000_epoch_defom_pivno_gated_gru3_gwc4gate_enc16_noleft_rgb_d768_320x768_b4_2gpu_200k.pth.gz}"
EVAL_THRESHOLDS="${EVAL_THRESHOLDS:-192 384 512 768}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable does not exist or is not executable: ${CONDA_BIN}" >&2
  exit 2
fi
if [[ ! -f "${RESTORE_CKPT}" ]]; then
  echo "Checkpoint does not exist: ${RESTORE_CKPT}" >&2
  exit 2
fi
if [[ ! -d "${SCENEFLOW_ROOT}/FlyingThings3D" ]]; then
  echo "FlyingThings3D is missing: ${SCENEFLOW_ROOT}/FlyingThings3D" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_LIST[@]}"
read -r -a THRESHOLD_LIST <<< "${EVAL_THRESHOLDS}"

echo "Evaluating completed C32/GWC4 gated-GRU3 in one pass: checkpoint=${RESTORE_CKPT}, GPUs=${CUDA_VISIBLE_DEVICES}, world_size=${NUM_GPUS}, thresholds=${EVAL_THRESHOLDS}, valid_iters=${VALID_ITERS}, batch_per_gpu=${EVAL_BATCH_SIZE}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  evaluate_stereo.py \
  --distributed \
  --model defom_pivno_gated_gru_kernel_ablation \
  --pivno_gru_kernel_size 3 \
  --restore_ckpt "${RESTORE_CKPT}" \
  --datasets things \
  --max_disp 768 \
  --eval_disp_thresholds "${THRESHOLD_LIST[@]}" \
  --mixed_precision \
  --valid_iters "${VALID_ITERS}" \
  --scale_iters 0 \
  --n_downsample 2 \
  --n_gru_layers 3 \
  --hidden_dims 128 128 128 \
  --context_norm instance \
  --corr_radius 4 \
  --eval_batch_size "${EVAL_BATCH_SIZE}"
