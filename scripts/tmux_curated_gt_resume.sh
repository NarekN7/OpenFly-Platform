#!/usr/bin/env bash
# Start (or restart) curated GT dump inside tmux with resume index.
# Usage:
#   bash scripts/tmux_curated_gt_resume.sh env_airsim_23 2531
# Optional 3rd–5th args (defaults keep prior behavior):
#   $3  OPENFLY_GT_JSON_PATH (default: data_curated/train_curated.json)
#   $4  OPENFLY_GT_TRAIN_LOG   (default: gt_frame_dump_train_curated_env<SUFFIX>_full.log)
#   $5  session tag after env suffix (default: full) e.g. redump -> openfly_gt_curated_env26_redump
# Attach: tmux attach -t <session printed below>
#
# Kills any existing session with the same name first. Stop any other
# train/eval.py / AirSim yourself if ports are busy.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${1:?usage: $0 <e.g. env_airsim_23> <OPENFLY_GT_START_INDEX>}"
START="${2:?usage: $0 <env> <start_index>}"

SUFFIX="${ENV#env_airsim_}"
GT_JSON="${3:-${REPO}/data_curated/train_curated.json}"
GT_LOG="${4:-${REPO}/gt_frame_dump_train_curated_env${SUFFIX}_full.log}"
SESS_TAG="${5:-full}"
SESSION="openfly_gt_curated_env${SUFFIX}_${SESS_TAG}"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd \"${REPO}\" && \
export OPENFLY_AIRSIM_ENV=\"${ENV}\" \
OPENFLY_GT_JSON_PATH=\"${GT_JSON}\" \
OPENFLY_NFS_GT_ROOT=/nfs/np/mnt/xtb/vln/train_curated \
OPENFLY_GT_ENV_PREFIXES=\"${ENV}/\" \
OPENFLY_GT_TRAIN_LOG=\"${GT_LOG}\" \
OPENFLY_GT_AFTER_POSE_SLEEP_SEC=0.1 \
OPENFLY_GT_START_INDEX=${START} && \
exec bash \"${REPO}/scripts/run_gt_train_eval_tmux.sh\"'"

echo "Session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
tmux list-sessions
