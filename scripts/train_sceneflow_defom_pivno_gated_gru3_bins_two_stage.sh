#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source "${SCRIPT_DIR}/dataset_env.sh"

# Usage:
#   bash scripts/train_sceneflow_defom_pivno_gated_gru3_bins_two_stage.sh
#   BASE_CKPT=/path/to/base.pth CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZE=4 bash "$0"
#
# Stage 1 trains only bin_initializer + initial_weight_head. Stage 2 loads the
# completed stage-1 model and jointly fine-tunes all reused and new modules.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"

CONDA_ENV="${CONDA_ENV:-defomstereo310}"
CONDA_BIN="${CONDA_BIN:-/home/u2025140689/anaconda3/bin/conda}"
MASTER_PORT="${MASTER_PORT:-29546}"

IMAGE_HEIGHT="${IMAGE_HEIGHT:-320}"
IMAGE_WIDTH="${IMAGE_WIDTH:-768}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-16}"
CORR_RADIUS="${CORR_RADIUS:-4}"
MAX_DISP="${MAX_DISP:-768}"

NUM_INIT_BINS="${NUM_INIT_BINS:-48}"
INIT_CORR_SCALE="${INIT_CORR_SCALE:-10.0}"
INIT_MAX_OFFSET="${INIT_MAX_OFFSET:-48.0}"

INIT_STEPS="${INIT_STEPS:-20000}"
JOINT_STEPS="${JOINT_STEPS:-180000}"
INIT_LR="${INIT_LR:-0.0002}"
JOINT_NEW_LR="${JOINT_NEW_LR:-0.00002}"
JOINT_PRETRAINED_LR="${JOINT_PRETRAINED_LR:-0.000002}"

BASE_CKPT="${BASE_CKPT:-checkpoints/defom_pivno_gated_gru3_gwc4gate_enc16_noleft_rgb_d768_320x768_b2_1gpu_200k_64channel/defom_pivno_gated_gru3_gwc4gate_enc16_noleft_rgb_d768_320x768_b2_1gpu_200k_64channel_ 40000.pth}"
RUN_PREFIX="${RUN_PREFIX:-defom_pivno_gated_gru3_bins_d${MAX_DISP}_${IMAGE_HEIGHT}x${IMAGE_WIDTH}_b${BATCH_SIZE}}"
INIT_NAME="${INIT_NAME:-${RUN_PREFIX}_init_${INIT_STEPS}}"
JOINT_NAME="${JOINT_NAME:-${RUN_PREFIX}_joint_${JOINT_STEPS}}"
INIT_FINAL_CKPT="checkpoints/${INIT_NAME}.pth"

EXTRA_ARGS=("$@")

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable does not exist or is not executable: ${CONDA_BIN}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Base gated-GRU3 checkpoint does not exist: ${BASE_CKPT}" >&2
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

run_stage() {
  local stage="$1"
  local name="$2"
  local steps="$3"
  local learning_rate="$4"
  local source_checkpoint="$5"
  local checkpoint_dir="checkpoints/${name}"
  local -a stage_args=()

  if [[ "${stage}" == "joint" ]]; then
    stage_args+=(
      --pivno_bins_pretrained_lr "${JOINT_PRETRAINED_LR}"
    )
  fi

  mkdir -p "${checkpoint_dir}"
  echo "Launching bins ${stage} stage:"
  echo "  GPUs=${CUDA_VISIBLE_DEVICES}, global_batch=${BATCH_SIZE}, local_batch=${LOCAL_BATCH_SIZE}"
  echo "  source=${source_checkpoint}"
  echo "  steps=${steps}, new_lr=${learning_rate}, output=${checkpoint_dir}"
  if [[ "${stage}" == "joint" ]]; then
    echo "  reused_lr=${JOINT_PRETRAINED_LR}"
  fi

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
    --save_ckpt_freq 10000 \
    --val_freq 10000 \
    "${EXTRA_ARGS[@]}" \
    --model defom_pivno_gated_gru3_bins \
    --pivno_bins_stage "${stage}" \
    --resume_ckpt "${source_checkpoint}" \
    --no_resume_optimizer \
    --name "${name}" \
    --num_steps "${steps}" \
    --lr "${learning_rate}" \
    "${stage_args[@]}" \
    2>&1 | tee -a "${checkpoint_dir}/train.log"
}

echo "Two-stage DEFOM-PIVNO bins training"
echo "  data=${SCENEFLOW_ROOT}"
echo "  base=${BASE_CKPT}"

run_stage init "${INIT_NAME}" "${INIT_STEPS}" "${INIT_LR}" "${BASE_CKPT}"

if [[ ! -f "${INIT_FINAL_CKPT}" ]]; then
  echo "Stage-1 final checkpoint was not created: ${INIT_FINAL_CKPT}" >&2
  exit 3
fi

run_stage joint "${JOINT_NAME}" "${JOINT_STEPS}" "${JOINT_NEW_LR}" "${INIT_FINAL_CKPT}"

echo "Two-stage training completed."
echo "  init checkpoint=${INIT_FINAL_CKPT}"
echo "  joint checkpoint=checkpoints/${JOINT_NAME}.pth"
