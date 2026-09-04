#!/usr/bin/env bash
# Shared dataset setup for repository launch scripts.
# A caller-provided environment variable always takes precedence.

DATASET_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ENV_FILE="${DATASET_ENV_FILE:-${DATASET_PROJECT_ROOT}/.dataset_env}"

if [[ -f "${DATASET_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${DATASET_ENV_FILE}"
fi

export SCENEFLOW_ROOT="${SCENEFLOW_ROOT:-/data/public_data}"
export KITTI_ROOT="${KITTI_ROOT:-/home/share/yijiayi/kitti}"
export ETH3D_ROOT="${ETH3D_ROOT:-/home/share/yijiayi/eth3d}"
export MIDDLEBURY_ROOT="${MIDDLEBURY_ROOT:-/home/share/yijiayi/Middlebury}"

unset DATASET_PROJECT_ROOT DATASET_ENV_FILE
