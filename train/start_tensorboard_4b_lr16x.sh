#!/usr/bin/env bash
# TensorBoard for 4B lr16x run (LR 3.2e-4, bs1-ga4).
set -euo pipefail

PORT="${TB_PORT:-18669}"
LOGDIR="/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr16x/runs"
RUN_DIR="${LOGDIR}/May20_13-52-23_gpu05"

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate

if ! [[ -d "${RUN_DIR}" ]]; then
  echo "Missing run dir: ${RUN_DIR}" >&2
  exit 1
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

echo "Starting TensorBoard (lr16x) on 127.0.0.1:${PORT}"
echo "Logdir: ${RUN_DIR}"
echo ""
echo "Open (after Cursor forwards port ${PORT}):"
echo "  http://localhost:${PORT}/"
echo ""
echo "Train + val on ONE chart:"
echo "  A) Custom Scalars tab → loss → 'train + validation CE loss'"
echo "  B) Time Series tab → filter 'loss/' → check loss/train AND loss/validation"
echo ""

exec tensorboard --logdir "${RUN_DIR}" --host 127.0.0.1 --port "${PORT}" --reload_interval 30 --load_fast=false
