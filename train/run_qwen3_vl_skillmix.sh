#!/usr/bin/env bash
# Tier-2 h16 skill-mix baseline using the training-computer TrainOF env.
# Prefer the full train/slurm_trainof_*_lrb*_stop20_20ep_*.sbatch launchers for
# the old Slurm config banner. This wrapper remains for manual/ad-hoc runs.
#
# Required: DATA_DIR, OUTPUT_DIR (must be a new/empty directory).
# Optional: SKILL_MIX_RATE (default 0.40), STOP_MIX_RATE (default 0.20),
#           FRAMES_ROOT, LEARNING_RATE, NUM_TRAIN_EPOCHS, NPROC, OPENFLY_DRY_RUN=1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAINOF_ACTIVATE="${TRAINOF_ACTIVATE:-$ROOT/TrainOF/bin/activate}"
if [[ ! -f "$TRAINOF_ACTIVATE" ]]; then
  echo "Missing TrainOF venv activate script: $TRAINOF_ACTIVATE" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$TRAINOF_ACTIVATE"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

: "${DATA_DIR:?set DATA_DIR to the canonical training data directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new experiment directory}"
EVAL_DATA_DIR="${OPENFLY_EVAL_DATA_DIR:-$DATA_DIR}"
FRAMES_ROOT="${FRAMES_ROOT:-/mnt/weka/nnurijanyan/data/vln/train_curated}"
INIT_MODEL="${INIT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
SKILL_MIX_RATE="${SKILL_MIX_RATE:-0.40}"
STOP_MIX_RATE="${STOP_MIX_RATE:-0.20}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
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
    raise SystemExit(f"Use a new output directory: {p}")
PY

echo "==== OUTPUT_DIR (read/write) ===="
echo "$OUTPUT_DIR"
echo "==== MODEL_REF ===="
echo "$INIT_MODEL"
echo "==== Data (x9 tier-2, interleave-uniformt-actonly, L/R ${SKILL_MIX_RATE}, stop ${STOP_MIX_RATE}) ===="
echo "train general: $DATA_DIR/trainx9_curated_0.json"
echo "train skill L: $DATA_DIR/trainx9_left_curated.json"
echo "train skill R: $DATA_DIR/trainx9_right_curated.json"
echo "train skill stop: $DATA_DIR/trainx9_stop_curated.json"
echo "skill_mix_rate: $SKILL_MIX_RATE"
echo "skill_mix_rate_stop: $STOP_MIX_RATE"
echo "eval:  skill-combined left+right+stop validation"
echo "  left:  $EVAL_DATA_DIR/left_evaluation_skill_validation.json"
echo "  right: $EVAL_DATA_DIR/right_evaluation_skill_validation.json"
echo "  stop:  $EVAL_DATA_DIR/stop_evaluation_skill_validation.json"
echo "==== frames_root ===="
echo "$FRAMES_ROOT"
echo "==== Vision encoder ===="
echo "UNFROZEN (no --freeze_vision_encoder)"
echo "==== temporal_history_past ===="
echo "16 (interleaved frame→action history; no prompt suffix)"
echo "==== image pixel budget ===="
echo "min_pixels=784 (28*28) max_pixels=57600 (320*180) → ~320x160"
echo "==== Global batch size ===="
echo "per_device_train_batch_size=1 x num_gpus=${NPROC} x gradient_accumulation_steps=4"
echo "==== LEARNING_RATE ===="
echo "$LEARNING_RATE"
echo "==== NUM_TRAIN_EPOCHS ===="
echo "$NUM_TRAIN_EPOCHS"
echo "==== eval-best ===="
echo "lowest skill eval_loss (eval every 500 steps) → \${OUTPUT_DIR}/eval-best (+ checkpoint-last)"

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
  --loss_type weighted --use_default_vln_system_prompt
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4
  --learning_rate "$LEARNING_RATE" --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --eval_steps 500 --checkpoint_layout best_last --report_to tensorboard --logging_steps 10
  --max_length 16384 --dtype bf16 --gradient_checkpointing "$@")
printf 'Training command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${OPENFLY_DRY_RUN:-0}" != "1" ]]; then
  exec "${command[@]}"
fi
