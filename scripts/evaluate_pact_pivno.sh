#!/usr/bin/env bash
set -euo pipefail

export SCENEFLOW_ROOT="${SCENEFLOW_ROOT:-/data/public_data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo}"
MASTER_PORT="${MASTER_PORT:-29529}"
RESTORE_CKPT="${RESTORE_CKPT:-checkpoints/defom_pact_pivno_d768_320x768_b4_2gpu/defom_pact_pivno_d768_320x768_b4_2gpu_200000.pth}"
MODEL_MAX_DISP="${MODEL_MAX_DISP:-768}"
EVAL_MAX_DISP="${EVAL_MAX_DISP:-768}"
VALID_ITERS="${VALID_ITERS:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"

if [[ ! -f "${RESTORE_CKPT}" ]]; then
  echo "Checkpoint does not exist: ${RESTORE_CKPT}" >&2
  exit 2
fi
if [[ ! -d "${SCENEFLOW_ROOT}/FlyingThings3D" ]]; then
  echo "FlyingThings3D does not exist under: ${SCENEFLOW_ROOT}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_LIST[@]}"

echo "Evaluating full PACT-PIVNO: checkpoint=${RESTORE_CKPT}, GPUs=${CUDA_VISIBLE_DEVICES}, world_size=${NUM_GPUS}, batch_per_gpu=${EVAL_BATCH_SIZE}, iters=${VALID_ITERS}, model_max_disp=${MODEL_MAX_DISP}, eval_max_disp=${EVAL_MAX_DISP}"

conda run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  evaluate_stereo.py \
  --model pact_pivno \
  --restore_ckpt "${RESTORE_CKPT}" \
  --datasets things \
  --max_disp "${MODEL_MAX_DISP}" \
  --eval_max_disp "${EVAL_MAX_DISP}" \
  --n_downsample 2 \
  --n_gru_layers 3 \
  --hidden_dims 128 128 128 \
  --context_norm instance \
  --corr_radius 4 \
  --valid_iters "${VALID_ITERS}" \
  --scale_iters 0 \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --mixed_precision \
  --distributed
