#!/bin/bash
#
# Qwen3-VL-4B trajectory-crop SFT (single frame, no temporal history):
#   --temporal_history_past 0  => user turn has 1 image (current view only).
#   8 GPUs, per-GPU bs1, grad accum 4 => global batch 32 (1*8*4), 10 epochs,
#   chat_window_turns 1, max_crop_length 4, weighted loss, TensorBoard, best_last.
#
# Submit (base LR 2e-5):
#   cd /home/nnurijanyan/OpenFly-Platform && mkdir -p slurm_logs && sbatch train/slurm_trainof_qwen3vl_4b_singleturn_8gpu_bs1_ga4_w1_h0_10ep.sbatch
#
# Higher LR (override CKPT_ROOT + LEARNING_RATE; optional --job-name):
#   LEARNING_RATE=8e-5 CKPT_ROOT=/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr4x \
#     sbatch --job-name=OF-qwen3vl4b-bs1-ga4-lr4x train/slurm_trainof_qwen3vl_4b_singleturn_8gpu_bs1_ga4_w1_h0_10ep.sbatch
#   LEARNING_RATE=3.2e-4 CKPT_ROOT=/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr16x \
#     sbatch --job-name=OF-qwen3vl4b-bs1-ga4-lr16x train/slurm_trainof_qwen3vl_4b_singleturn_8gpu_bs1_ga4_w1_h0_10ep.sbatch
#
# Baseline LR 2e-5, 20 epochs (longer walltime):
#   NUM_TRAIN_EPOCHS=20 CKPT_ROOT=.../qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-20ep \
#     sbatch --time=40:00:00 --job-name=OF-qwen3vl4b-bs1-ga4-20ep train/slurm_trainof_qwen3vl_4b_singleturn_8gpu_bs1_ga4_w1_h0_10ep.sbatch
# LR / 4 (5e-6), 10 epochs:
#   LEARNING_RATE=5e-6 CKPT_ROOT=.../qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep-lr0.25x \
#     sbatch --job-name=OF-qwen3vl4b-bs1-ga4-lr0.25x train/slurm_trainof_qwen3vl_4b_singleturn_8gpu_bs1_ga4_w1_h0_10ep.sbatch
#

#SBATCH --job-name=OF-qwen3vl4b-bs1-ga4-w1-h0-10ep
#SBATCH --partition=research
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

cd /home/nnurijanyan/OpenFly-Platform
mkdir -p slurm_logs

if [[ ! -f "/home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate" ]]; then
  echo "Missing TrainOF venv at /home/nnurijanyan/OpenFly-Platform/TrainOF" >&2
  exit 1
fi

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CKPT_ROOT="${CKPT_ROOT:-/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h0-10ep}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
INIT_MODEL="${INIT_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
export INIT_MODEL
RESUME="${RESUME:-0}"
FORCE_HF_MODEL_REFRESH="${FORCE_HF_MODEL_REFRESH:-0}"

MODEL_REF="${INIT_MODEL}"
RESUME_ARG=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  MODEL_REF="${INIT_MODEL}"
  RESUME_ARG=(--resume_from_checkpoint "${CKPT_ROOT}/checkpoint-last")
fi

echo "==== CKPT_ROOT (read/write) ===="
echo "${CKPT_ROOT}"
echo "==== MODEL_REF ===="
echo "${MODEL_REF}"
echo "==== RESUME ===="
echo "${RESUME}"
echo "==== Global batch size ===="
echo "per_device_train_batch_size=1 x num_gpus=8 x gradient_accumulation_steps=4 => 32"
echo "==== LEARNING_RATE ===="
echo "${LEARNING_RATE}"
echo "==== NUM_TRAIN_EPOCHS ===="
echo "${NUM_TRAIN_EPOCHS}"

HF_HUB_DIR="${HF_HOME}/hub"
mkdir -p "${HF_HUB_DIR}"
HUB_REPO_DIR="${HF_HUB_DIR}/models--${INIT_MODEL//\//--}"
# Hub id (e.g. Qwen/Qwen3-VL-4B-Instruct): prefetch once before 8-way DDP from_pretrained.
# Local paths (resume checkpoints under /mnt/...) skip this block.
if [[ "${INIT_MODEL}" != /* ]]; then
  if [[ "${FORCE_HF_MODEL_REFRESH}" == "1" || "${FORCE_HF_MODEL_REFRESH}" == "true" ]]; then
    if [[ -d "${HUB_REPO_DIR}" ]]; then
      echo "==== Removing possibly corrupted HF hub snapshot ===="
      echo "${HUB_REPO_DIR}"
      rm -rf "${HUB_REPO_DIR}"
    fi
  fi
  echo "==== huggingface_hub.snapshot_download ${INIT_MODEL} (one process before DDP) ===="
  python - <<PY
import os
from huggingface_hub import snapshot_download
repo = os.environ["INIT_MODEL"]
print("snapshot_download:", repo)
path = snapshot_download(repo_id=repo)
print("snapshot_download done:", path)
PY
else
  echo "==== Local INIT_MODEL (${INIT_MODEL}); skipping hub snapshot_download ===="
fi

nvidia-smi || true

accelerate launch --num_processes 8 /home/nnurijanyan/OpenFly-Platform/scripts/qwen3_vl_sft.py \
  --model_name_or_path "${MODEL_REF}" \
  --train_json "/home/nnurijanyan/OpenFly-Platform/data_curated/train_curated.json" \
  --eval_json "/home/nnurijanyan/OpenFly-Platform/data_curated/validation_curated.json" \
  --frames_root "/mnt/weka/nnurijanyan/data/vln/train_curated" \
  --output_dir "${CKPT_ROOT}" \
  --chat_window_turns 1 \
  --max_crop_length 4 \
  --temporal_history_past 0 \
  --crop_shift_sampling uniform \
  --loss_type weighted \
  --use_default_vln_system_prompt \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --eval_steps 500 \
  --checkpoint_layout best_last \
  --report_to tensorboard \
  --logging_steps 10 \
  --max_length 16384 \
  --dtype bf16 \
  --gradient_checkpointing \
  "${RESUME_ARG[@]}"
