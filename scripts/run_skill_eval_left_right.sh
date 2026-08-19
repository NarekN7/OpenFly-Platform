#!/usr/bin/env bash
# Offline left/right skill eval (tier-2 / x9), does not use AirSim.
# Aligned with interleaved training: frame→action chat, no left-pad, GT history.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/openfly_of3_env.sh"

_DEFAULT_CKPT="/nfs/np/mnt/xtb/vln/qwen3-vl-4b-vln-singleturn-8gpu-bs1-ga4-w1-h16-x9-0-lrb20-10ep/checkpoint-last"
CKPT="${OPENFLY_EVAL_QWEN3_CHECKPOINT:-${_DEFAULT_CKPT}}"
IMAGE_ROOT="${OPENFLY_SKILL_IMAGE_ROOT:-/mnt/xtb/vln/train_curated}"
_RUN_TAG="${OPENFLY_EVAL_BATCH_TAG:-skill_lr_4b_lrb20_ckptlast}"
RUN_ROOT="${OPENFLY_SKILL_EVAL_OUT_DIR:-$ROOT/eval_runs/${_RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

LEFT="${OPENFLY_SKILL_LEFT_JSON:-$ROOT/skill_eval/left_evaluation_skill_validation.json}"
RIGHT="${OPENFLY_SKILL_RIGHT_JSON:-$ROOT/skill_eval/right_evaluation_skill_validation.json}"

echo "Skill left/right batch root: $RUN_ROOT"
OPENFLY_EVAL_QWEN3_CHECKPOINT="$CKPT" \
OPENFLY_SKILL_IMAGE_ROOT="$IMAGE_ROOT" \
OPENFLY_QWEN_TEMPORAL_HISTORY_PAST="${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-16}" \
OPENFLY_QWEN_DEVICE="${OPENFLY_QWEN_DEVICE:-cuda:0}" \
OPENFLY_QWEN_DEVICE_MAP="${OPENFLY_QWEN_DEVICE_MAP:-}" \
OPENFLY_SKILL_EVAL_LIMIT="${OPENFLY_SKILL_EVAL_LIMIT:-0}" \
python3 -u "$ROOT/train/skill_eval_left_right.py" \
  --checkpoint "$CKPT" \
  --image-root "$IMAGE_ROOT" \
  --out-dir "$RUN_ROOT" \
  --json "$LEFT" \
  --json "$RIGHT"

echo "Done. Summary: $RUN_ROOT/summary.json"
