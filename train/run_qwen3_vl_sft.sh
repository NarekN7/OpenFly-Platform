#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash train/run_qwen3_vl_sft.sh
#
# This assumes you created the venv at repo root:
#   python -m venv --system-site-packages TrainOF
#   source TrainOF/bin/activate
#   pip install -U pip transformers accelerate peft trl datasets pillow safetensors

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/TrainOF/bin/activate" ]]; then
  echo "Missing ${ROOT_DIR}/TrainOF venv. Create it first (see comments at top of this script)." >&2
  exit 1
fi

source "${ROOT_DIR}/TrainOF/bin/activate"

export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 1 "${ROOT_DIR}/scripts/qwen3_vl_sft.py" \
  --model_name_or_path "Qwen/Qwen3-VL-2B-Instruct" \
  --train_json "/home/nnurijanyan/OpenFly-Platform/data_curated/train_curated.json" \
  --eval_json "/home/nnurijanyan/OpenFly-Platform/data_curated/validation_curated.json" \
  --frames_root "/mnt/weka/nnurijanyan/data/vln/train_curated" \
  --output_dir "/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-2b-vln-1frame-sys-frozen" \
  --max_crop_length 17 \
  --temporal_history_past 16 \
  --crop_shift_sampling uniform \
  --loss_type weighted \
  --use_default_vln_system_prompt \
  --freeze_vision_encoder \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --max_length 16384 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 500 \
  --eval_steps 500 \
  --save_total_limit 4 \
  --dtype bf16 \
  --gradient_checkpointing \
  --verify_images_exist
