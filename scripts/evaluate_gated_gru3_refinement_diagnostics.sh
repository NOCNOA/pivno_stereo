#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/dataset_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo}"
CONDA_BIN="${CONDA_BIN:-/home/yijiayi/anaconda3/bin/conda}"
MASTER_PORT="${MASTER_PORT:-29641}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESTORE_CKPT="${RESTORE_CKPT:-checkpoints/defom_pivno_gated_gru3_gwc4gate_enc16_noleft_rgb_d768_320x768_b4_2gpu_200k/200000_epoch_defom_pivno_gated_gru3_gwc4gate_enc16_noleft_rgb_d768_320x768_b4_2gpu_200k.pth.gz}"
OUTPUT_JSON="${OUTPUT_JSON:-evaluation_results/gated_gru3_200k_refinement_diagnostics.json}"

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
mkdir -p "$(dirname "${OUTPUT_JSON}")"

echo "Evaluating gated-GRU3 refinement diagnostics: checkpoint=${RESTORE_CKPT}, GPUs=${CUDA_VISIBLE_DEVICES}, world_size=${NUM_GPUS}, batch_per_gpu=${BATCH_SIZE}, output=${OUTPUT_JSON}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  tools/evaluate_pivno_refinement.py \
  --model defom_pivno_gated_gru3 \
  --restore_ckpt "${RESTORE_CKPT}" \
  --sceneflow_root "${SCENEFLOW_ROOT}" \
  --iters 32 \
  --report_iters 1 2 4 8 16 32 \
  --disp_thresholds 192 384 512 768 \
  --support_thresholds 4 8 16 \
  --corr_radius 4 \
  --batch_size "${BATCH_SIZE}" \
  --mixed_precision \
  --deterministic \
  --output_json "${OUTPUT_JSON}"
