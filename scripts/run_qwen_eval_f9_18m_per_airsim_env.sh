#!/usr/bin/env bash
# Closed-loop Qwen3-VL eval with action 10 = 18 m forward (copy of eval path).
# Uses train/eval_f9_18m.py — does NOT modify train/eval.py or the default
# combined-eval pipeline. History still stores predicted action id 10.
#
# Usage (from repo root):
#   export OPENFLY_EVAL_QWEN3_CHECKPOINT=/path/to/checkpoint
#   bash scripts/run_qwen_eval_f9_18m_per_airsim_env.sh
#
# Or via combined eval:
#   OPENFLY_EVAL_PY=train/eval_f9_18m.py bash scripts/run_qwen_combined_eval.sh
#
# Optional env: same as scripts/run_qwen_eval_per_airsim_env.sh
#   OPENFLY_EVAL_DATA_DIR (default: repo data_curated/)
#   OPENFLY_EVAL_JSON (default: OPENFLY_EVAL_DATA_DIR/seenx9.json)
#   OPENFLY_EVAL_BATCH_ROOT / OPENFLY_EVAL_BATCH_TAG
#   OPENFLY_EVAL_ENVS
#   OPENFLY_QWEN_TEMPORAL_HISTORY_PAST
#   OPENFLY_EVAL_MAX_STEPS / OPENFLY_EVAL_MAX_TRAJECTORIES
#   OPENFLY_EVAL_TIMING
#   OPENFLY_QWEN_EVAL_IMAGE_WIDTH/HEIGHT
#   OPENFLY_EVAL_DISABLE_EARLY_STOP
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EVAL_DATA_DIR="${OPENFLY_EVAL_DATA_DIR:-$ROOT/data_curated}"
_TAG="${OPENFLY_EVAL_BATCH_TAG:-qwen_f9_18m}"
echo "F9 18m closed-loop eval (action 10 → 18 m, history keeps 10)"
export OPENFLY_EVAL_PY="${OPENFLY_EVAL_PY:-$ROOT/train/eval_f9_18m.py}"
export OPENFLY_EVAL_JSON="${OPENFLY_EVAL_JSON:-$EVAL_DATA_DIR/seenx9.json}"
export OPENFLY_EVAL_BATCH_ROOT="${OPENFLY_EVAL_BATCH_ROOT:-$ROOT/eval_runs/${_TAG}_$(date +%Y%m%d_%H%M%S)}"
exec bash "$ROOT/scripts/run_qwen_eval_per_airsim_env.sh"
