#!/usr/bin/env bash
# Ground-truth frame dump for data_curated/seen_curated.json into NFS seen_curated.
# Runs the six AirSim envs sequentially (one simulator at a time); eval.py cannot
# switch worlds without restarting the correct AirSim binary.
#
# Usage:
#   bash scripts/run_seen_curated_gt_all_envs.sh
# Env overrides:
#   OPENFLY_GT_AFTER_POSE_SLEEP_SEC  (default 0.1)
#   OPENFLY_AIRSIM_WAIT_BEFORE_EVAL  (default 40)

set -euo pipefail
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SEC="${OPENFLY_AIRSIM_WAIT_BEFORE_EVAL:-40}"
POSE_SLEEP="${OPENFLY_GT_AFTER_POSE_SLEEP_SEC:-0.1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OF3
cd "$REPO_ROOT"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/runtime-openfly-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR"

SEEN_JSON="${REPO_ROOT}/data_curated/seen_curated.json"
NFS_SEEN="/nfs/np/mnt/xtb/vln/seen_curated"

export OPENFLY_GT_JSON_PATH="${SEEN_JSON}"
export OPENFLY_NFS_GT_ROOT="${NFS_SEEN}"
mkdir -p "${NFS_SEEN}" 2>/dev/null || true
for _e in env_airsim_16 env_airsim_18 env_airsim_23 env_airsim_26 env_airsim_gz env_airsim_sh; do
  mkdir -p "${NFS_SEEN}/${_e}"
done

kill_sims() {
  pkill -TERM -f 'python train/eval.py' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'python train/eval.py' 2>/dev/null || true
  pkill -TERM -f 'AirVLN-Linux-Shipping' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'AirVLN-Linux-Shipping' 2>/dev/null || true
}

ENVS=(env_airsim_16 env_airsim_18 env_airsim_23 env_airsim_26 env_airsim_gz env_airsim_sh)

for ENV_NAME in "${ENVS[@]}"; do
  SUFFIX="${ENV_NAME#env_airsim_}"
  GT_LOG="${REPO_ROOT}/gt_seen_curated_env${SUFFIX}_full.log"

  kill_sims

  export OPENFLY_AIRSIM_ENV="${ENV_NAME}"
  export OPENFLY_GT_ENV_PREFIXES="${ENV_NAME}/"
  export OPENFLY_GT_AFTER_POSE_SLEEP_SEC="${POSE_SLEEP}"
  export OPENFLY_GT_START_INDEX="${OPENFLY_GT_START_INDEX:-0}"

  echo "========== SEEN curated: ${ENV_NAME} -> log ${GT_LOG} =========="

  START_SH="${REPO_ROOT}/envs/airsim/${ENV_NAME}/LinuxNoEditor/start.sh"
  if [[ ! -f "$START_SH" ]]; then
    echo "Missing $START_SH" >&2
    exit 1
  fi

  echo "Starting AirSim (${ENV_NAME})..."
  bash "$START_SH" >> "${REPO_ROOT}/airsim_start.log" 2>&1 &
  echo "Waiting ${WAIT_SEC}s for simulator..."
  sleep "${WAIT_SEC}"

  export OPENFLY_SKIP_AIRSIM_LAUNCH=1
  export OPENFLY_AIRSIM_WAIT_SEC=0

  GT_START="${OPENFLY_GT_START_INDEX:-0}"
  if [[ "${GT_START}" =~ ^[0-9]+$ ]] && (( GT_START > 0 )); then
    python train/eval.py 2>&1 | tee -a "${GT_LOG}"
  else
    python train/eval.py 2>&1 | tee "${GT_LOG}"
  fi

  echo "Finished ${ENV_NAME}. Killing simulator before next env."
  kill_sims
  # Next env must start fresh from index 0 (each env has its own filtered slice)
  export OPENFLY_GT_START_INDEX=0
done

echo "All six seen_curated envs completed."
