#!/usr/bin/env bash
set -euo pipefail

# General SFT launcher. Set DATA_DIR and a fresh OUTPUT_DIR; optional INIT_MODEL/NPROC.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/scripts/openfly_vln_env.sh"
: "${DATA_DIR:?set DATA_DIR}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new experiment directory}"
python - "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if p.exists() and (not p.is_dir() or any(p.iterdir())):
    raise SystemExit(f"Use a new output directory for EOS training: {p}")
PY

accelerate launch --num_processes "${NPROC:-1}" "${ROOT_DIR}/scripts/qwen3_vl_sft.py" \
  --model_name_or_path "${INIT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}" \
  --train_json "${DATA_DIR}/train_curated.json" \
  --eval_json "${DATA_DIR}/validation_curated.json" \
  --frames_root "${DATA_DIR}/train_curated" \
  --output_dir "${OUTPUT_DIR}" \
  --temporal_history_past 16 \
  --loss_type weighted --supervise_eos \
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
  --verify_images_exist "$@"
