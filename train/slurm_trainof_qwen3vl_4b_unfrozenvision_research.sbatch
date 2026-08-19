#!/bin/bash
#
# Qwen3-VL 4B, trajectory-crop SFT, default system prompt, trainable vision encoder.
# Same hyperparameters as slurm_trainof_qwen3vl_4b_frozenvision.sbatch except no --freeze_vision_encoder.
# Effective batch size: 8 GPUs * 1 per GPU * grad_accum 8 = 64.
#
# Submit with:
#   cd /home/nnurijanyan/OpenFly-Platform
#   mkdir -p slurm_logs
#   sbatch train/slurm_trainof_qwen3vl_4b_unfrozenvision_research.sbatch
#

#SBATCH --job-name=TrainOF-4B-1fSysUnfrz
#SBATCH --partition=research
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

cd /home/nnurijanyan/OpenFly-Platform
mkdir -p slurm_logs

if [[ ! -f "/home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate" ]]; then
  echo "Missing TrainOF venv at /home/nnurijanyan/OpenFly-Platform/TrainOF" >&2
  echo "Create it first on a node with internet access (see train/README_QWEN3_VL_SFT.md)." >&2
  exit 1
fi

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate

export TOKENIZERS_PARALLELISM=false

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

accelerate launch --num_processes 8 /home/nnurijanyan/OpenFly-Platform/scripts/qwen3_vl_sft.py \
  --model_name_or_path "Qwen/Qwen3-VL-4B-Instruct" \
  --train_json "/home/nnurijanyan/OpenFly-Platform/data_curated/train_curated.json" \
  --frames_root "/mnt/weka/nnurijanyan/data/vln/train_curated" \
  --output_dir "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-4b-vln-1frame-defaultsys-unfrozenvision-full-8gpu-b4" \
  --temporal_history_past 16 \
  --loss_type weighted \
  --use_default_vln_system_prompt \
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
