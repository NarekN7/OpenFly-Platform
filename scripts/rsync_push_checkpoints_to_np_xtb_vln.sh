#!/usr/bin/env bash
# Push selected Qwen3-VL checkpoint dirs from Weka to np (SSH host "np").
#
# Usage:
#   bash scripts/rsync_push_checkpoints_to_np_xtb_vln.sh
#
# Optional env:
#   CHECKPOINTS_SRC   default: /mnt/weka/nnurijanyan/checkpoints
#   NP_DEST           default: np:/mnt/xtb/vln
#
# Syncs only:
#   qwen3-vl-2b-vln-1frame-defaultsys-frozenvision-full-8gpu-b8
#   qwen3-vl-2b-vln-1frame-defaultsys-unfrozenvision-full-8gpu-b8
#
# Run inside tmux/screen; large transfers take a long time.

set -euo pipefail

CHECKPOINTS_SRC="${CHECKPOINTS_SRC:-/mnt/weka/nnurijanyan/checkpoints}"
NP_DEST="${NP_DEST:-np:/mnt/xtb/vln}"

DIRS=(
  "qwen3-vl-2b-vln-1frame-defaultsys-frozenvision-full-8gpu-b8"
  "qwen3-vl-2b-vln-1frame-defaultsys-unfrozenvision-full-8gpu-b8"
)

echo "Ensuring remote directory exists..."
ssh -o BatchMode=yes np "mkdir -p /mnt/xtb/vln"

for name in "${DIRS[@]}"; do
  echo ""
  echo "=== ${name} ==="
  echo "Source:  ${CHECKPOINTS_SRC}/${name}/"
  echo "Target:  ${NP_DEST}/${name}/"
  rsync -a --mkpath --info=progress2 -e ssh \
    "${CHECKPOINTS_SRC}/${name}/" \
    "${NP_DEST}/${name}/"
done

echo "Done."
