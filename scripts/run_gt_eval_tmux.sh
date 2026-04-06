#!/usr/bin/env bash
# Ground-truth frame dump: start AirSim, wait for it to be ready, then run train/eval.py (GT mode).
# AirSim start.sh uses -windowed -RenderOffScreen -nosound for headless-style runs.
# Usage (from anywhere):
#   bash scripts/run_gt_eval_tmux.sh
# Or start in tmux:
#   tmux new-session -d -s openfly_gt_frames "bash /path/to/OpenFly-Platform/scripts/run_gt_eval_tmux.sh"
#
# Env overrides:
#   OPENFLY_AIRSIM_ENV          default env_airsim_16
#   OPENFLY_NFS_SEEN_ROOT       default /nfs/np/mnt/xtb/vln/seen (GT dump output root)
#   OPENFLY_NFS_GT_ROOT         if set, overrides output root for GT dumps (use for train: .../train)
#   OPENFLY_GT_JSON_PATH        default Annotation/seen.json (use Annotation/train.json for train)
#   OPENFLY_AIRSIM_WAIT_BEFORE_EVAL  seconds after starting sim (default 40)
#   OPENFLY_GT_START_INDEX      skip first N trajectories (resume after sim crash; e.g. 103)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${OPENFLY_AIRSIM_ENV:-env_airsim_16}"
WAIT_SEC="${OPENFLY_AIRSIM_WAIT_BEFORE_EVAL:-40}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate OF3
cd "$REPO_ROOT"

# UE/AirSim Linux binary often requires this when not on a full desktop session
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/runtime-openfly-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR"

export OPENFLY_NFS_SEEN_ROOT="${OPENFLY_NFS_SEEN_ROOT:-/nfs/np/mnt/xtb/vln/seen}"
mkdir -p "$OPENFLY_NFS_SEEN_ROOT" 2>/dev/null || true
# Layout: $OPENFLY_NFS_SEEN_ROOT/<env_airsim_*>/astar_data/<subpath>/<traj_id>/*.png
for _e in env_airsim_16 env_airsim_18 env_airsim_23 env_airsim_26 env_airsim_gz env_airsim_sh; do
  mkdir -p "${OPENFLY_NFS_SEEN_ROOT}/${_e}"
done

START_SH="${REPO_ROOT}/envs/airsim/${ENV_NAME}/LinuxNoEditor/start.sh"
if [[ ! -f "$START_SH" ]]; then
  echo "Missing $START_SH" >&2
  exit 1
fi

echo "Starting AirSim (${ENV_NAME})..."
bash "$START_SH" >> "${REPO_ROOT}/airsim_start.log" 2>&1 &
echo "Waiting ${WAIT_SEC}s for simulator..."
sleep "${WAIT_SEC}"

# eval.py should not spawn a second simulator process
export OPENFLY_SKIP_AIRSIM_LAUNCH=1
export OPENFLY_AIRSIM_WAIT_SEC=0

exec python train/eval.py 2>&1 | tee "${REPO_ROOT}/gt_frame_dump.log"
