#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/dataset_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo}"
CONDA_BIN="${CONDA_BIN:-/home/yijiayi/anaconda3/bin/conda}"
MASTER_PORT="${MASTER_PORT:-29559}"
RESTORE_CKPT="${RESTORE_CKPT:?Set RESTORE_CKPT to a gated-GRU3-GWC4-mask-RGB-SR checkpoint}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
VALID_ITERS="${VALID_ITERS:-32}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable is unavailable: ${CONDA_BIN}" >&2
  exit 2
fi
if [[ ! -f "${RESTORE_CKPT}" ]]; then
  echo "Checkpoint does not exist: ${RESTORE_CKPT}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_LIST[@]}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  evaluate_stereo.py \
  --model defom_pivno_gated_gru3_gwc4_mask_rgb_sr \
  --restore_ckpt "${RESTORE_CKPT}" \
  --datasets things \
  --max_disp 768 \
  --eval_max_disp 768 \
  --mixed_precision \
  --valid_iters "${VALID_ITERS}" \
  --scale_iters 0 \
  --n_downsample 2 \
  --n_gru_layers 3 \
  --hidden_dims 128 128 128 \
  --context_norm instance \
  --corr_radius 4 \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --distributed \
  "$@"
