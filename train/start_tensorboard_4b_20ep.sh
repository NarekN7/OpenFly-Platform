#!/usr/bin/env bash
# TensorBoard for 4B baseline LR 2e-5, 20 epochs.
set -euo pipefail

PORT="${TB_PORT:-18670}"
RUN_DIR="/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-20ep/runs/May22_14-55-55_gpu07"

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate

if ! [[ -d "${RUN_DIR}" ]]; then
  echo "Missing run dir: ${RUN_DIR}" >&2
  exit 1
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

echo "Starting TensorBoard (baseline 20ep) on 127.0.0.1:${PORT}"
echo "Logdir: ${RUN_DIR}"
echo "Open: http://localhost:${PORT}/"
echo "Train + val: Custom Scalars → loss → train + validation CE loss"
echo ""

exec tensorboard --logdir "${RUN_DIR}" --host 127.0.0.1 --port "${PORT}" --reload_interval 30 --load_fast=false
