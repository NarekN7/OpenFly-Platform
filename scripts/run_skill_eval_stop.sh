#!/usr/bin/env bash
# Offline stop skill eval (tier-2 / x9), does not use AirSim.
# Aligned with interleaved training: frame→action chat, no left-pad, GT history.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/openfly_of3_env.sh"

_DEFAULT_CKPT="/nfs/np/mnt/xtb/vln/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h16-x9-0-lrb20-10ep/checkpoint-last"
CKPT="${OPENFLY_EVAL_QWEN3_CHECKPOINT:-${_DEFAULT_CKPT}}"
IMAGE_ROOT="${OPENFLY_SKILL_IMAGE_ROOT:-/mnt/xtb/vln/train_curated}"
STOP_JSON="${OPENFLY_SKILL_STOP_JSON:-$ROOT/skill_eval/stop_evaluation_skill_validation.json}"
_RUN_TAG="${OPENFLY_EVAL_BATCH_TAG:-skill_stop_validation_4b_lrb20_ckptlast}"
RUN_ROOT="${OPENFLY_SKILL_EVAL_OUT_DIR:-$ROOT/eval_runs/${_RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

echo "Skill stop batch root: $RUN_ROOT"
OPENFLY_EVAL_QWEN3_CHECKPOINT="$CKPT" \
OPENFLY_SKILL_STOP_JSON="$STOP_JSON" \
OPENFLY_SKILL_IMAGE_ROOT="$IMAGE_ROOT" \
OPENFLY_QWEN_TEMPORAL_HISTORY_PAST="${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-16}" \
OPENFLY_QWEN_DEVICE="${OPENFLY_QWEN_DEVICE:-cuda:0}" \
OPENFLY_QWEN_DEVICE_MAP="${OPENFLY_QWEN_DEVICE_MAP:-}" \
OPENFLY_SKILL_EVAL_LIMIT="${OPENFLY_SKILL_EVAL_LIMIT:-0}" \
python3 -u "$ROOT/train/skill_eval_stop.py" \
  --checkpoint "$CKPT" \
  --json "$STOP_JSON" \
  --image-root "$IMAGE_ROOT" \
  --out-dir "$RUN_ROOT"

echo "Done. Summary: $RUN_ROOT/summary.json"
