#!/bin/bash
#
# 1×GPU (recommended first run) for Qwen3-VL-2B + LoRA SFT.
#
# Submit with:
#   cd /home/nnurijanyan/OpenFly-Platform
#   mkdir -p slurm_logs
#   sbatch train/slurm_trainof_qwen3vl_1gpu.sbatch
#

#SBATCH --job-name=TrainOF-1GPU
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

cd /home/nnurijanyan/OpenFly-Platform
mkdir -p slurm_logs

if [[ ! -f "/home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate" ]]; then
  echo "Missing TrainOF venv at /home/nnurijanyan/OpenFly-Platform/TrainOF" >&2
  echo "Create it first (see train/README_QWEN3_VL_SFT.md)." >&2
  exit 1
fi

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate

export TOKENIZERS_PARALLELISM=false

# Optional: keep HF caches on shared storage
export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

echo "==== nvidia-smi ===="
nvidia-smi

echo "==== torch cuda ===="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

accelerate launch --num_processes 1 /home/nnurijanyan/OpenFly-Platform/scripts/qwen3_vl_sft.py \
  --model_name_or_path "Qwen/Qwen3-VL-2B-Instruct" \
  --train_json "/home/nnurijanyan/OpenFly-Platform/data_curated/train_curated.json" \
  --frames_root "/mnt/weka/nnurijanyan/data/vln/train_curated" \
  --output_dir "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-2b-vln-simple-1gpu" \
  --temporal_history_past 16 \
  --loss_type weighted \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --max_length 16384 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 4 \
  --dtype bf16 \
  --gradient_checkpointing \
  --verify_images_exist

