#!/usr/bin/env bash
# TensorBoard: baseline 2e-5, 10 epochs — port 18667.
set -euo pipefail

PORT="${TB_PORT:-18667}"
RUN_DIR="/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep/runs/May20_05-06-33_gpu05"

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate

[[ -d "${RUN_DIR}" ]] || { echo "Missing: ${RUN_DIR}" >&2; exit 1; }
command -v fuser >/dev/null 2>&1 && fuser -k "${PORT}/tcp" 2>/dev/null || true

echo "TensorBoard baseline 10ep → http://127.0.0.1:${PORT}/"
echo "Custom Scalars → loss → train + validation CE loss"
exec tensorboard --logdir "${RUN_DIR}" --host 127.0.0.1 --port "${PORT}" --reload_interval 30 --load_fast=false
