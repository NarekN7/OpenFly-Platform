#!/usr/bin/env bash
# Start 4 TensorBoards for the h16 LR sweep (4B singleturn, bs1-ga4, 10ep).
#
# Opens four separate TensorBoard instances (one per run) so you can compare:
#   Custom Scalars → loss → train + validation CE loss
#
# Default ports (override with env vars if needed):
#   TB_PORT_BASELINE=18672
#   TB_PORT_LR4X=18673
#   TB_PORT_LR16X=18674
#   TB_PORT_LR025X=18675
#
# Usage:
#   bash train/start_tensorboard_4b_h16_all.sh
#
set -euo pipefail

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate

CKPT_BASE="/mnt/weka/nnurijanyan/checkpoints"

ROOT_BASELINE="${CKPT_BASE}/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h16-10ep"
ROOT_LR4X="${CKPT_BASE}/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h16-10ep-lr4x"
ROOT_LR16X="${CKPT_BASE}/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h16-10ep-lr16x"
ROOT_LR025X="${CKPT_BASE}/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h16-10ep-lr0.25x"

PORT_BASELINE="${TB_PORT_BASELINE:-18672}"
PORT_LR4X="${TB_PORT_LR4X:-18673}"
PORT_LR16X="${TB_PORT_LR16X:-18674}"
PORT_LR025X="${TB_PORT_LR025X:-18675}"

latest_tb_run_dir() {
  local root="$1"
  local tb_dir="${root}/tb"
  [[ -d "${tb_dir}" ]] || return 1
  # Pick newest run-*/ directory if present, else fall back to tb/ directly.
  local newest
  newest="$(ls -1dt "${tb_dir}"/run-* 2>/dev/null | head -n 1 || true)"
  if [[ -n "${newest}" && -d "${newest}" ]]; then
    echo "${newest}"
  else
    echo "${tb_dir}"
  fi
}

start_tb() {
  local name="$1"
  local root="$2"
  local port="$3"
  local logdir
  logdir="$(latest_tb_run_dir "${root}")"

  [[ -d "${logdir}" ]] || { echo "Missing logdir for ${name}: ${logdir}" >&2; exit 1; }

  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" 2>/dev/null || true
  fi

  echo "[${name}] TensorBoard → http://127.0.0.1:${port}/"
  echo "[${name}] Logdir: ${logdir}"
  nohup tensorboard --logdir "${logdir}" --host 127.0.0.1 --port "${port}" --reload_interval 30 --load_fast=false \
    >"/tmp/tb_${name}_${port}.log" 2>&1 &
}

start_tb "h16_baseline" "${ROOT_BASELINE}" "${PORT_BASELINE}"
start_tb "h16_lr4x" "${ROOT_LR4X}" "${PORT_LR4X}"
start_tb "h16_lr16x" "${ROOT_LR16X}" "${PORT_LR16X}"
start_tb "h16_lr0.25x" "${ROOT_LR025X}" "${PORT_LR025X}"

echo ""
echo "Ports:"
echo "  baseline : ${PORT_BASELINE}"
echo "  lr4x     : ${PORT_LR4X}"
echo "  lr16x    : ${PORT_LR16X}"
echo "  lr0.25x  : ${PORT_LR025X}"
echo ""
echo "In TensorBoard:"
echo "  Custom Scalars → loss → train + validation CE loss"

