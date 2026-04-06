#!/usr/bin/env bash
# Sequential strict redump: one trajectory in env_airsim_16, then one in env_airsim_26.
# Uses two tmux sessions (one after the other; two different AirSim binaries, not concurrent).
#
# Usage: bash scripts/tmux_strict_redump_env16_then_env26.sh
#
# Requires: data_curated/train_curated_env16_strict_redump1.json (1 row)
#           data_curated/train_curated_env26_strict_redump1.json (1 row)

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

S1="openfly_gt_curated_env16_strict_redump"
S2="openfly_gt_curated_env26_strict_redump"
LOG16="${REPO}/gt_frame_dump_train_curated_env16_strict_redump.log"
LOG26="${REPO}/gt_frame_dump_train_curated_env26_strict_redump.log"

kill_sims() {
  pkill -TERM -f 'python train/eval.py' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'python train/eval.py' 2>/dev/null || true
  pkill -TERM -f 'AirVLN-Linux-Shipping' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'AirVLN-Linux-Shipping' 2>/dev/null || true
}

echo "[1/2] Starting env_airsim_16 strict redump in tmux: ${S1}"
kill_sims
tmux kill-session -t "${S1}" 2>/dev/null || true
bash "${REPO}/scripts/tmux_curated_gt_resume.sh" env_airsim_16 0 \
  "${REPO}/data_curated/train_curated_env16_strict_redump1.json" \
  "${LOG16}" \
  strict_redump

echo "Waiting for env_airsim_16 job to finish (log: ${LOG16})..."
for _i in $(seq 1 720); do
  if grep -q "Evaluation complete!" "${LOG16}" 2>/dev/null && \
     grep -q "Completed evaluation of environment env_airsim_16" "${LOG16}" 2>/dev/null; then
    echo "env_airsim_16 redump finished."
    break
  fi
  sleep 30
done
if ! grep -q "Evaluation complete!" "${LOG16}" 2>/dev/null; then
  echo "Timeout waiting for env_airsim_16 completion. Check ${LOG16}" >&2
  exit 1
fi

echo "[2/2] Starting env_airsim_26 strict redump in tmux: ${S2}"
kill_sims
tmux kill-session -t "${S2}" 2>/dev/null || true
bash "${REPO}/scripts/tmux_curated_gt_resume.sh" env_airsim_26 0 \
  "${REPO}/data_curated/train_curated_env26_strict_redump1.json" \
  "${LOG26}" \
  strict_redump

echo "Done. Attach: tmux attach -t ${S2}"
echo "Logs: ${LOG16}  ${LOG26}"
