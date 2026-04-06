#!/usr/bin/env bash
# Download pose.jsonl from Hugging Face (see scripts/download_pose_jsonl.py) inside tmux
# so the run survives SSH disconnects. Progress is visible when you attach, and is also
# appended to a log file for tail -f.
#
# Usage:
#   bash scripts/run_download_pose_jsonl_tmux.sh
#
# Watch live in another terminal:
#   tail -f logs/download_pose_jsonl.log
#
# Attach to the tmux session (scroll with Ctrl-b then [):
#   tmux attach -t openfly_pose_dl
#
# Env overrides:
#   OPENFLY_POSE_SESSION   tmux session name (default: openfly_pose_dl)
#   OPENFLY_POSE_LOG         log file path (default: REPO_ROOT/logs/download_pose_jsonl.log)
#   OPENFLY_POSE_DEST        destination dir (default: REPO_ROOT/data_wo_annotation)
#   OPENFLY_POSE_EXTRA_ARGS  extra args passed to download_pose_jsonl.py (quoted string)
#   OPENFLY_POSE_SKIP_EXISTING  if not 0, pass --skip-existing (default 1; good for resume)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${OPENFLY_POSE_SESSION:-openfly_pose_dl}"
LOG_FILE="${OPENFLY_POSE_LOG:-${REPO_ROOT}/logs/download_pose_jsonl.log}"
DEST="${OPENFLY_POSE_DEST:-${REPO_ROOT}/data_wo_annotation}"
EXTRA="${OPENFLY_POSE_EXTRA_ARGS:-}"
SKIP_FLAG=()
if [[ "${OPENFLY_POSE_SKIP_EXISTING:-1}" != "0" ]]; then
  SKIP_FLAG=(--skip-existing)
fi

mkdir -p "$(dirname "$LOG_FILE")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "  attach:  tmux attach -t $SESSION" >&2
  echo "  log:     tail -f $LOG_FILE" >&2
  echo "  kill:    tmux kill-session -t $SESSION" >&2
  exit 1
fi

# -d: detached (keeps running after this script exits)
# bash -lc: one shell so cd + pipeline work; tee appends so reruns keep history
tmux new-session -d -s "$SESSION" bash -lc "cd '$REPO_ROOT' && echo \"Logging to: $LOG_FILE\" | tee -a '$LOG_FILE' && exec python -u scripts/download_pose_jsonl.py --dest '$DEST' ${SKIP_FLAG[*]} $EXTRA 2>&1 | tee -a '$LOG_FILE'"

echo "Started tmux session: $SESSION"
echo "  attach (live):  tmux attach -t $SESSION"
echo "  tail log:       tail -f $LOG_FILE"
echo "  stop:           tmux kill-session -t $SESSION"
