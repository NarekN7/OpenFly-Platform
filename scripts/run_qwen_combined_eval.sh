#!/usr/bin/env bash
# Run all Qwen3-VL eval suites for one checkpoint (sequential):
#   1. Closed-loop AirSim (all envs) — scripts/run_qwen_eval_per_airsim_env.sh
#   2. Skill left/right validation — scripts/run_skill_eval_left_right.sh
#   3. Skill stop validation — scripts/run_skill_eval_stop.sh
#   4. Skill left/right test — scripts/run_skill_eval_left_right.sh
#   5. Skill stop test — scripts/run_skill_eval_stop.sh
#   6. In-train crop fit — scripts/run_skill_eval_intrain_crops.sh
#
# Usage (from repo root):
#   export OPENFLY_EVAL_QWEN3_CHECKPOINT=/path/to/checkpoint-last
#   bash scripts/run_qwen_combined_eval.sh
#
# Optional env:
#   OPENFLY_EVAL_BATCH_TAG          short name for run dir (default: combined_ckptlast)
#   OPENFLY_COMBINED_EVAL_ROOT       override parent output dir (default: eval_runs/<tag>_<timestamp>)
#   OPENFLY_EVAL_JSON               closed-loop eval JSON (default: data_curated/seen_curated.json)
#   OPENFLY_EVAL_ENVS               comma-separated AirSim env keys (default: all six)
#   OPENFLY_SKILL_LEFT_JSON         default: skill_eval/left_evaluation_skill_validation.json
#   OPENFLY_SKILL_RIGHT_JSON        default: skill_eval/right_evaluation_skill_validation.json
#   OPENFLY_SKILL_STOP_JSON         default: skill_eval/stop_evaluation_skill_validation.json
#   OPENFLY_SKILL_LEFT_TEST_JSON    default: skill_eval/left_evaluation_skill_test.json
#   OPENFLY_SKILL_RIGHT_TEST_JSON   default: skill_eval/right_evaluation_skill_test.json
#   OPENFLY_SKILL_STOP_TEST_JSON    default: skill_eval/stop_evaluation_skill_test.json
#   OPENFLY_INTRAIN_EVAL_JSON       default: skill_eval/trainx9_intrain_eval_500.json
#   OPENFLY_SKILL_IMAGE_ROOT        default: /mnt/xtb/vln/train_curated
#   OPENFLY_QWEN_DEVICE             default: cuda:0
#   OPENFLY_QWEN_TEMPORAL_HISTORY_PAST  default: 16
#   OPENFLY_SKILL_EVAL_LIMIT        cap offline skill samples (0 = all)
#   OPENFLY_EVAL_MAX_STEPS          closed-loop step cap per trajectory
#   OPENFLY_EVAL_MAX_TRAJECTORIES   closed-loop trajectory cap (0 = unlimited)
#   OPENFLY_COMBINED_USE_TMUX=1     launch in detached tmux (long runs)
#
# Uses conda env OF3 (scripts/openfly_of3_env.sh). Requires: conda activate OF3
# or let this script activate it. Sets USE_TORCH=1 USE_TF=0 so transformers
# does not import OF3's broken TF/numpy combo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/openfly_of3_env.sh"

CKPT="${OPENFLY_EVAL_QWEN3_CHECKPOINT:-}"
if [[ -z "$CKPT" ]]; then
  echo "ERROR: set OPENFLY_EVAL_QWEN3_CHECKPOINT to the model checkpoint-last path." >&2
  exit 1
fi
if [[ ! -d "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

_RUN_TAG="${OPENFLY_EVAL_BATCH_TAG:-combined_ckptlast}"
_TS="$(date +%Y%m%d_%H%M%S)"
COMBINED_ROOT="${OPENFLY_COMBINED_EVAL_ROOT:-$ROOT/eval_runs/${_RUN_TAG}_${_TS}}"
mkdir -p "$COMBINED_ROOT"

LOG="$COMBINED_ROOT/combined_eval.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "Combined Qwen3-VL eval"
echo "  checkpoint=$CKPT"
echo "  combined_root=$COMBINED_ROOT"
echo "  temporal_past=${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-16}"
echo "  device=${OPENFLY_QWEN_DEVICE:-cuda:0}"
echo "=============================================="

_run_skill_lr() {
  local out_dir="$1"
  local tag="$2"
  local left_json="$3"
  local right_json="$4"
  OPENFLY_EVAL_QWEN3_CHECKPOINT="$CKPT" \
  OPENFLY_EVAL_BATCH_TAG="$tag" \
  OPENFLY_SKILL_EVAL_OUT_DIR="$out_dir" \
  OPENFLY_SKILL_LEFT_JSON="$left_json" \
  OPENFLY_SKILL_RIGHT_JSON="$right_json" \
  OPENFLY_SKILL_IMAGE_ROOT="$image_root" \
  OPENFLY_QWEN_TEMPORAL_HISTORY_PAST="${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-16}" \
  OPENFLY_QWEN_DEVICE="${OPENFLY_QWEN_DEVICE:-cuda:0}" \
  OPENFLY_QWEN_DEVICE_MAP="${OPENFLY_QWEN_DEVICE_MAP:-}" \
  OPENFLY_SKILL_EVAL_LIMIT="${OPENFLY_SKILL_EVAL_LIMIT:-0}" \
  bash "$ROOT/scripts/run_skill_eval_left_right.sh"
}

_run_skill_stop() {
  local out_dir="$1"
  local tag="$2"
  local stop_json="$3"
  OPENFLY_EVAL_QWEN3_CHECKPOINT="$CKPT" \
  OPENFLY_EVAL_BATCH_TAG="$tag" \
  OPENFLY_SKILL_EVAL_OUT_DIR="$out_dir" \
  OPENFLY_SKILL_STOP_JSON="$stop_json" \
  OPENFLY_SKILL_IMAGE_ROOT="$image_root" \
  OPENFLY_QWEN_TEMPORAL_HISTORY_PAST="${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-16}" \
  OPENFLY_QWEN_DEVICE="${OPENFLY_QWEN_DEVICE:-cuda:0}" \
  OPENFLY_QWEN_DEVICE_MAP="${OPENFLY_QWEN_DEVICE_MAP:-}" \
  OPENFLY_SKILL_EVAL_LIMIT="${OPENFLY_SKILL_EVAL_LIMIT:-0}" \
  bash "$ROOT/scripts/run_skill_eval_stop.sh"
}

run_stages() {
  local image_root="${OPENFLY_SKILL_IMAGE_ROOT:-/mnt/xtb/vln/train_curated}"
  local closed_root="$COMBINED_ROOT/closed_loop"
  local lr_val_root="$COMBINED_ROOT/skill_lr_validation"
  local stop_val_root="$COMBINED_ROOT/skill_stop_validation"
  local lr_test_root="$COMBINED_ROOT/skill_lr_test"
  local stop_test_root="$COMBINED_ROOT/skill_stop_test"
  local intrain_root="$COMBINED_ROOT/skill_intrain"

  echo ""
  echo "===== [1/6] Closed-loop AirSim (all envs) ====="
  OPENFLY_EVAL_QWEN3_CHECKPOINT="$CKPT" \
  OPENFLY_EVAL_BATCH_ROOT="$closed_root" \
  OPENFLY_EVAL_JSON="${OPENFLY_EVAL_JSON:-data_curated/seen_curated.json}" \
  OPENFLY_EVAL_ENVS="${OPENFLY_EVAL_ENVS:-}" \
  OPENFLY_QWEN_TEMPORAL_HISTORY_PAST="${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-16}" \
  OPENFLY_QWEN_DEVICE="${OPENFLY_QWEN_DEVICE:-cuda:0}" \
  OPENFLY_QWEN_DEVICE_MAP="${OPENFLY_QWEN_DEVICE_MAP:-}" \
  OPENFLY_EVAL_MAX_STEPS="${OPENFLY_EVAL_MAX_STEPS:-}" \
  OPENFLY_EVAL_MAX_TRAJECTORIES="${OPENFLY_EVAL_MAX_TRAJECTORIES:-}" \
  OPENFLY_EVAL_TIMING="${OPENFLY_EVAL_TIMING:-}" \
  OPENFLY_QWEN_EVAL_IMAGE_WIDTH="${OPENFLY_QWEN_EVAL_IMAGE_WIDTH:-}" \
  OPENFLY_QWEN_EVAL_IMAGE_HEIGHT="${OPENFLY_QWEN_EVAL_IMAGE_HEIGHT:-}" \
  OPENFLY_EVAL_DISABLE_EARLY_STOP="${OPENFLY_EVAL_DISABLE_EARLY_STOP:-}" \
  bash "$ROOT/scripts/run_qwen_eval_per_airsim_env.sh"

  echo ""
  echo "===== [2/6] Skill left/right validation ====="
  _run_skill_lr "$lr_val_root" "${_RUN_TAG}_skill_lr_validation" \
    "${OPENFLY_SKILL_LEFT_JSON:-$ROOT/skill_eval/left_evaluation_skill_validation.json}" \
    "${OPENFLY_SKILL_RIGHT_JSON:-$ROOT/skill_eval/right_evaluation_skill_validation.json}"

  echo ""
  echo "===== [3/6] Skill stop validation ====="
  _run_skill_stop "$stop_val_root" "${_RUN_TAG}_skill_stop_validation" \
    "${OPENFLY_SKILL_STOP_JSON:-$ROOT/skill_eval/stop_evaluation_skill_validation.json}"

  echo ""
  echo "===== [4/6] Skill left/right test ====="
  _run_skill_lr "$lr_test_root" "${_RUN_TAG}_skill_lr_test" \
    "${OPENFLY_SKILL_LEFT_TEST_JSON:-$ROOT/skill_eval/left_evaluation_skill_test.json}" \
    "${OPENFLY_SKILL_RIGHT_TEST_JSON:-$ROOT/skill_eval/right_evaluation_skill_test.json}"

  echo ""
  echo "===== [5/6] Skill stop test ====="
  _run_skill_stop "$stop_test_root" "${_RUN_TAG}_skill_stop_test" \
    "${OPENFLY_SKILL_STOP_TEST_JSON:-$ROOT/skill_eval/stop_evaluation_skill_test.json}"

  echo ""
  echo "===== [6/6] In-train crop fit ====="
  OPENFLY_EVAL_QWEN3_CHECKPOINT="$CKPT" \
  OPENFLY_EVAL_BATCH_TAG="${_RUN_TAG}_skill_intrain" \
  OPENFLY_SKILL_EVAL_OUT_DIR="$intrain_root" \
  OPENFLY_INTRAIN_EVAL_JSON="${OPENFLY_INTRAIN_EVAL_JSON:-$ROOT/skill_eval/trainx9_intrain_eval_500.json}" \
  OPENFLY_SKILL_IMAGE_ROOT="$image_root" \
  OPENFLY_QWEN_TEMPORAL_HISTORY_PAST="${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-16}" \
  OPENFLY_QWEN_DEVICE="${OPENFLY_QWEN_DEVICE:-cuda:0}" \
  OPENFLY_QWEN_DEVICE_MAP="${OPENFLY_QWEN_DEVICE_MAP:-}" \
  OPENFLY_SKILL_EVAL_LIMIT="${OPENFLY_SKILL_EVAL_LIMIT:-0}" \
  bash "$ROOT/scripts/run_skill_eval_intrain_crops.sh"

  python3 "$ROOT/tools/aggregate_combined_eval.py" \
    --root "$COMBINED_ROOT" \
    --checkpoint "$CKPT" \
    --write-json "$COMBINED_ROOT/combined_summary.json"

  echo ""
  echo "=============================================="
  echo "Combined eval complete."
  echo "  root: $COMBINED_ROOT"
  echo "  summary: $COMBINED_ROOT/combined_summary.json"
  echo "  log: $LOG"
  echo "=============================================="
}

if [[ "${OPENFLY_COMBINED_USE_TMUX:-}" == "1" && -z "${OPENFLY_COMBINED_INNER:-}" ]]; then
  SESSION="combined_eval_${_RUN_TAG}_${_TS}"
  tmux new-session -d -s "$SESSION" bash -lc "
    set -euo pipefail
    cd '$ROOT'
    export OPENFLY_COMBINED_INNER=1
    export OPENFLY_COMBINED_USE_TMUX=0
    export OPENFLY_EVAL_QWEN3_CHECKPOINT='$CKPT'
    export OPENFLY_COMBINED_EVAL_ROOT='$COMBINED_ROOT'
    export OPENFLY_EVAL_BATCH_TAG='$_RUN_TAG'
    bash '$ROOT/scripts/run_qwen_combined_eval.sh'
    echo '[tmux combined eval done]'
    sleep 3600
  "
  echo "Launched tmux session: $SESSION"
  echo "  tail log: tail -f $LOG"
else
  run_stages
fi
