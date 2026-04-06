#!/usr/bin/env bash
# Detached tmux session running the full seen_curated GT dump (all 6 envs, sequential).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${OPENFLY_SEEN_TMUX_SESSION:-openfly_gt_seen_curated}"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "bash -lc 'exec bash \"${REPO}/scripts/run_seen_curated_gt_all_envs.sh\"'"
echo "Session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
tmux list-sessions
