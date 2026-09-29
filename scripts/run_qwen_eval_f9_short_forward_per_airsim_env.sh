#!/usr/bin/env bash
# Closed-loop Qwen3-VL eval with F9→short-forward substitution (copy of eval path).
# Uses train/eval_f9_short_forward.py — does NOT modify train/eval.py or the default
# combined-eval pipeline.
#
# Usage (from repo root):
#   export OPENFLY_EVAL_QWEN3_CHECKPOINT=/path/to/checkpoint
#   bash scripts/run_qwen_eval_f9_short_forward_per_airsim_env.sh
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
_TAG="${OPENFLY_EVAL_BATCH_TAG:-qwen_f9_short_forward}"
echo "F9 short-forward closed-loop eval"
export OPENFLY_EVAL_PY="${OPENFLY_EVAL_PY:-$ROOT/train/eval_f9_short_forward.py}"
export OPENFLY_EVAL_JSON="${OPENFLY_EVAL_JSON:-$EVAL_DATA_DIR/seenx9.json}"
export OPENFLY_EVAL_BATCH_ROOT="${OPENFLY_EVAL_BATCH_ROOT:-$ROOT/eval_runs/${_TAG}_$(date +%Y%m%d_%H%M%S)}"
exec bash "$ROOT/scripts/run_qwen_eval_per_airsim_env.sh"
