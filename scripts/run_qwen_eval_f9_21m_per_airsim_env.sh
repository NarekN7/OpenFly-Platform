#!/usr/bin/env bash
# Closed-loop Qwen3-VL eval with action 10 = 21 m forward (copy of eval path).
# Uses train/eval_f9_21m.py — does NOT modify train/eval.py or the default
# combined-eval pipeline. History still stores predicted action id 10.
#
# Usage (from repo root):
#   export OPENFLY_EVAL_QWEN3_CHECKPOINT=/path/to/checkpoint
#   bash scripts/run_qwen_eval_f9_21m_per_airsim_env.sh
#
# Or via combined eval:
#   OPENFLY_EVAL_PY=train/eval_f9_21m.py bash scripts/run_qwen_combined_eval.sh
#
# Optional env: same as scripts/run_qwen_eval_per_airsim_env.sh
#   OPENFLY_EVAL_JSON (default: data_curated/seenx9.json)
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
# shellcheck disable=SC1091
source "$ROOT/scripts/openfly_of3_env.sh"

CKPT="${OPENFLY_EVAL_QWEN3_CHECKPOINT:-}"
if [[ -z "$CKPT" ]]; then
  echo "ERROR: set OPENFLY_EVAL_QWEN3_CHECKPOINT" >&2
  exit 1
fi
JSON="${OPENFLY_EVAL_JSON:-data_curated/seenx9.json}"
_TAG="${OPENFLY_EVAL_BATCH_TAG:-qwen_f9_21m}"
RUN_ROOT="${OPENFLY_EVAL_BATCH_ROOT:-$ROOT/eval_runs/${_TAG}_$(date +%Y%m%d_%H%M%S)}"
EVAL_PY="${OPENFLY_EVAL_PY:-$ROOT/train/eval_f9_21m.py}"
mkdir -p "$RUN_ROOT"

if [[ -n "${OPENFLY_EVAL_ENVS:-}" ]]; then
  _tmp="${OPENFLY_EVAL_ENVS//,/ }"
  read -r -a ENVS <<< "${_tmp}"
else
  ENVS=(env_airsim_16 env_airsim_18 env_airsim_23 env_airsim_26 env_airsim_gz env_airsim_sh)
fi

echo "F9 21m closed-loop eval (action 10 → 21 m, history keeps 10)"
echo "  checkpoint=$CKPT"
echo "  json=$JSON"
echo "  run_root=$RUN_ROOT"
echo "  script=$EVAL_PY"

for env in "${ENVS[@]}"; do
  echo "========== ${env} =========="
  OUT="$RUN_ROOT/$env"
  mkdir -p "$OUT"
  OPENFLY_GT_DUMP=0 \
  OPENFLY_EVAL_QWEN3_CHECKPOINT="$CKPT" \
  OPENFLY_EVAL_JSON="$JSON" \
  OPENFLY_GT_ENV_PREFIXES="${env}/" \
  OPENFLY_QWEN_TEMPORAL_HISTORY_PAST="${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-16}" \
  OPENFLY_EVAL_MAX_STEPS="${OPENFLY_EVAL_MAX_STEPS:-}" \
  OPENFLY_EVAL_MAX_TRAJECTORIES="${OPENFLY_EVAL_MAX_TRAJECTORIES:-}" \
  OPENFLY_EVAL_TIMING="${OPENFLY_EVAL_TIMING:-}" \
  OPENFLY_QWEN_EVAL_IMAGE_WIDTH="${OPENFLY_QWEN_EVAL_IMAGE_WIDTH:-}" \
  OPENFLY_QWEN_EVAL_IMAGE_HEIGHT="${OPENFLY_QWEN_EVAL_IMAGE_HEIGHT:-}" \
  OPENFLY_EVAL_DISABLE_EARLY_STOP="${OPENFLY_EVAL_DISABLE_EARLY_STOP:-}" \
  OPENFLY_EVAL_OUT_DIR="$OUT" \
  OPENFLY_EVAL_PY="$EVAL_PY" \
  python3 -u "$EVAL_PY" 2>&1 | tee "$OUT/console.log"
  echo "Artifacts: $OUT/predictions.json $OUT/metrics.json"
done
echo "Batch root: $RUN_ROOT"
