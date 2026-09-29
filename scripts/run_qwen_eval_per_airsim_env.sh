#!/usr/bin/env bash
# Run Qwen3-VL closed-loop eval once per AirSim environment (interleaved tier-2 alignment).
# Usage (from repo root):
#   bash scripts/run_qwen_eval_per_airsim_env.sh
# Optional env:
#   OPENFLY_EVAL_QWEN3_CHECKPOINT
#   OPENFLY_EVAL_DATA_DIR         closed-loop JSON root (default: repo data_curated/)
#   OPENFLY_EVAL_JSON             (default: OPENFLY_EVAL_DATA_DIR/seenx9.json)
#   OPENFLY_EVAL_BATCH_ROOT
#   OPENFLY_EVAL_ENVS             comma-separated env keys (default: all six AirSim envs)
#   OPENFLY_QWEN_TEMPORAL_HISTORY_PAST  default 16 (interleaved window, no left-pad)
#   OPENFLY_EVAL_MAX_STEPS
#   OPENFLY_EVAL_MAX_TRAJECTORIES
#   OPENFLY_EVAL_TIMING
#   OPENFLY_QWEN_EVAL_IMAGE_WIDTH/HEIGHT  optional cv2 pre-resize (omit = native + processor budget)
#   OPENFLY_EVAL_DISABLE_EARLY_STOP
#   OPENFLY_EVAL_PY               python entry (default: train/eval.py); e.g. train/eval_f9_short_forward.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/openfly_vln_env.sh"
EVAL_DATA_DIR="${OPENFLY_EVAL_DATA_DIR:-$ROOT/data_curated}"
python -c "import airsim, unrealcv, msgpackrpc"
CKPT="${OPENFLY_EVAL_QWEN3_CHECKPOINT:?set OPENFLY_EVAL_QWEN3_CHECKPOINT}"
JSON="${OPENFLY_EVAL_JSON:-$EVAL_DATA_DIR/seenx9.json}"
RUN_ROOT="${OPENFLY_EVAL_BATCH_ROOT:-$ROOT/eval_runs/qwen7741_per_env_$(date +%Y%m%d_%H%M%S)}"
EVAL_PY="${OPENFLY_EVAL_PY:-$ROOT/train/eval.py}"
mkdir -p "$RUN_ROOT"
if [[ -n "${OPENFLY_EVAL_ENVS:-}" ]]; then
  _tmp="${OPENFLY_EVAL_ENVS//,/ }"
  read -r -a ENVS <<< "${_tmp}"
else
  ENVS=(env_airsim_16 env_airsim_18 env_airsim_23 env_airsim_26 env_airsim_gz env_airsim_sh)
fi
echo "Closed-loop eval_py=$EVAL_PY"
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
  python3 -u "$EVAL_PY" 2>&1 | tee "$OUT/console.log"
  echo "Artifacts: $OUT/predictions.json $OUT/metrics.json"
done
echo "Batch root: $RUN_ROOT"
