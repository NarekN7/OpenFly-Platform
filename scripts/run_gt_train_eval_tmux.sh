#!/usr/bin/env bash
# Ground-truth frame dump for the Annotation/train.json split (same logic as seen / eval.py GT mode).
# Writes under: $OPENFLY_NFS_GT_ROOT/<env_airsim_*>/astar_data/... (default train root below).
#
# Example (env 16 only — set GT_ENV_PREFIXES in train/eval.py to ("env_airsim_16/",)):
#   bash scripts/run_gt_train_eval_tmux.sh
# Or tmux:
#   tmux new-session -d -s openfly_gt_train 'bash -lc "exec bash /path/to/OpenFly-Platform/scripts/run_gt_train_eval_tmux.sh"'
#
# Env overrides:
#   OPENFLY_AIRSIM_ENV          default env_airsim_16
#   OPENFLY_NFS_GT_ROOT         default /nfs/np/mnt/xtb/vln/train
#   OPENFLY_GT_JSON_PATH        default Annotation/train.json
#   OPENFLY_GT_START_INDEX      resume after crash (skip first N trajectories after prefix filter)
#   OPENFLY_GT_TRAIN_LOG        log file (default REPO_ROOT/gt_frame_dump_train.log; use e.g. gt_frame_dump_train_env18.log per env)
#   OPENFLY_AIRSIM_WAIT_BEFORE_EVAL  seconds after starting sim (default 40)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${OPENFLY_AIRSIM_ENV:-env_airsim_16}"
WAIT_SEC="${OPENFLY_AIRSIM_WAIT_BEFORE_EVAL:-40}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OF3
cd "$REPO_ROOT"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/runtime-openfly-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR"

export OPENFLY_GT_JSON_PATH="${OPENFLY_GT_JSON_PATH:-Annotation/train.json}"
export OPENFLY_NFS_GT_ROOT="${OPENFLY_NFS_GT_ROOT:-/nfs/np/mnt/xtb/vln/train}"
mkdir -p "$OPENFLY_NFS_GT_ROOT" 2>/dev/null || true
for _e in env_airsim_16 env_airsim_18 env_airsim_23 env_airsim_26 env_airsim_gz env_airsim_sh; do
  mkdir -p "${OPENFLY_NFS_GT_ROOT}/${_e}"
done

START_SH="${REPO_ROOT}/envs/airsim/${ENV_NAME}/LinuxNoEditor/start.sh"
if [[ ! -f "$START_SH" ]]; then
  echo "Missing $START_SH" >&2
  exit 1
fi

echo "GT train dump: JSON=$OPENFLY_GT_JSON_PATH  NFS_GT_ROOT=$OPENFLY_NFS_GT_ROOT  AirSim=$ENV_NAME"
echo "Starting AirSim (${ENV_NAME})..."
bash "$START_SH" >> "${REPO_ROOT}/airsim_start.log" 2>&1 &
echo "Waiting ${WAIT_SEC}s for simulator..."
sleep "${WAIT_SEC}"

export OPENFLY_SKIP_AIRSIM_LAUNCH=1
export OPENFLY_AIRSIM_WAIT_SEC=0

GT_LOG="${OPENFLY_GT_TRAIN_LOG:-${REPO_ROOT}/gt_frame_dump_train.log}"
# Append log when resuming so you keep prior progress lines
# Use ${var:-0} in arithmetic too: with `set -u`, (( VAR > 0 )) errors if VAR is unset.
GT_START="${OPENFLY_GT_START_INDEX:-0}"
if [[ "${GT_START}" =~ ^[0-9]+$ ]] && (( GT_START > 0 )); then
  exec python train/eval.py 2>&1 | tee -a "${GT_LOG}"
else
  exec python train/eval.py 2>&1 | tee "${GT_LOG}"
fi
