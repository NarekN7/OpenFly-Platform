#!/usr/bin/env bash
# Five separate TensorBoard servers (one training run per port).
set -euo pipefail

ROOT="/home/nnurijanyan/OpenFly-Platform"
LOG="${ROOT}/slurm_logs/tensorboard"
mkdir -p "${LOG}"
chmod +x "${ROOT}"/train/start_tensorboard_4b_*.sh
source "${ROOT}/TrainOF/bin/activate"

for run_dir in \
  "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep/runs/May20_05-06-33_gpu05" \
  "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr4x/runs/May20_13-49-39_gpu01" \
  "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr16x/runs/May20_13-52-23_gpu05" \
  "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-20ep/runs/May22_14-55-55_gpu07" \
  "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr0.25x/runs/May22_20-02-19_gpu01"; do
  python "${ROOT}/scripts/backfill_tensorboard_combined_loss.py" --logdir "${run_dir}" 2>/dev/null || true
done

for p in 18667 18668 18669 18670 18671; do
  fuser -k "${p}/tcp" 2>/dev/null || true
done
sleep 1

start_one() {
  nohup bash "$1" >> "${LOG}/tb_$2.log" 2>&1 &
  echo "  $2 PID $! log ${LOG}/tb_$2.log"
  sleep 1
}

echo "Starting 5 TensorBoard instances..."
start_one "${ROOT}/train/start_tensorboard_4b_baseline.sh" "18667_baseline_10ep"
start_one "${ROOT}/train/start_tensorboard_4b_lr4x.sh" "18668_lr4x"
start_one "${ROOT}/train/start_tensorboard_4b_lr16x.sh" "18669_lr16x"
start_one "${ROOT}/train/start_tensorboard_4b_20ep.sh" "18670_baseline_20ep"
start_one "${ROOT}/train/start_tensorboard_4b_lr0.25x.sh" "18671_lr0.25x"

echo ""
echo "Open in browser (forward all 5 ports in Cursor → Ports panel if needed):"
echo "  http://localhost:18667/  baseline 2e-5, 10ep"
echo "  http://localhost:18668/  lr4x 8e-5"
echo "  http://localhost:18669/  lr16x 3.2e-4"
echo "  http://localhost:18670/  baseline 2e-5, 20ep"
echo "  http://localhost:18671/  lr0.25x 5e-6"
echo "Each: Custom Scalars → loss → train + validation CE loss"
