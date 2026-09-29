#!/usr/bin/env bash
# Current tier-2 h16 skill-mix baseline, with EOS training and action-only eval loss.
# Set OUTPUT_DIR to a new directory. OPENFLY_DRY_RUN=1 prints the resolved command.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/openfly_vln_env.sh"
: "${DATA_DIR:?set DATA_DIR to the canonical training data directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new experiment directory}"
EVAL_DATA_DIR="${OPENFLY_EVAL_DATA_DIR:-$DATA_DIR}"
FRAMES_ROOT="${FRAMES_ROOT:-$DATA_DIR/train_curated}"
INIT_MODEL="${INIT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
SKILL_MIX_RATE="${SKILL_MIX_RATE:-0.40}"
STOP_MIX_RATE="${STOP_MIX_RATE:-0.20}"
NPROC="${NPROC:-8}"
for path in "$DATA_DIR/trainx9_curated_0.json" "$DATA_DIR/trainx9_left_curated.json" \
  "$DATA_DIR/trainx9_right_curated.json" "$DATA_DIR/trainx9_stop_curated.json" \
  "$EVAL_DATA_DIR/left_evaluation_skill_validation.json" \
  "$EVAL_DATA_DIR/right_evaluation_skill_validation.json" \
  "$EVAL_DATA_DIR/stop_evaluation_skill_validation.json"; do
  [[ -f "$path" ]] || { echo "Missing dataset: $path" >&2; exit 1; }
done
[[ -d "$FRAMES_ROOT" ]] || { echo "Missing frames root: $FRAMES_ROOT" >&2; exit 1; }
python - "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if p.exists() and (not p.is_dir() or any(p.iterdir())):
    raise SystemExit(f"Use a new output directory for EOS training: {p}")
PY
command=(accelerate launch --num_processes "$NPROC" "$ROOT/scripts/qwen3_vl_sft.py"
  --model_name_or_path "$INIT_MODEL"
  --train_json "$DATA_DIR/trainx9_curated_0.json"
  --skill_json_left "$DATA_DIR/trainx9_left_curated.json"
  --skill_json_right "$DATA_DIR/trainx9_right_curated.json" --skill_mix_rate "$SKILL_MIX_RATE"
  --skill_json_stop "$DATA_DIR/trainx9_stop_curated.json" --skill_mix_rate_stop "$STOP_MIX_RATE"
  --eval_skill_json_left "$EVAL_DATA_DIR/left_evaluation_skill_validation.json"
  --eval_skill_json_right "$EVAL_DATA_DIR/right_evaluation_skill_validation.json"
  --eval_skill_json_stop "$EVAL_DATA_DIR/stop_evaluation_skill_validation.json"
  --frames_root "$FRAMES_ROOT" --output_dir "$OUTPUT_DIR"
  --chat_window_turns 1 --temporal_history_past 16 --min_pixels 784 --max_pixels 57600
  --loss_type weighted --supervise_eos --use_default_vln_system_prompt
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4
  --learning_rate "${LEARNING_RATE:-2e-5}" --num_train_epochs "${NUM_TRAIN_EPOCHS:-20}"
  --eval_steps 500 --checkpoint_layout best_last --report_to tensorboard --logging_steps 10
  --max_length 16384 --dtype bf16 --gradient_checkpointing --verify_images_exist "$@")
printf 'Training command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${OPENFLY_DRY_RUN:-0}" != "1" ]]; then
  exec "${command[@]}"
fi
