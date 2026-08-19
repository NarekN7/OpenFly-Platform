#!/usr/bin/env bash
# Run Qwen3-VL closed-loop eval once per AirSim environment (filters seen_curated by prefix).
# Usage (from repo root):
#   bash scripts/run_qwen_eval_per_airsim_env.sh
# Optional env:
#   OPENFLY_EVAL_QWEN3_CHECKPOINT  (default: checkpoint-7741 on /mnt/xtb/vln/...)
#   OPENFLY_EVAL_JSON             (default: data_curated/seen_curated.json)
#   OPENFLY_EVAL_BATCH_ROOT       (default: eval_runs/qwen7741_per_env_<timestamp>)
#   OPENFLY_EVAL_ENVS             comma-separated env keys, e.g. env_airsim_23,env_airsim_26
#                                 (default: all six AirSim envs)
#   OPENFLY_QWEN_TEMPORAL_HISTORY_PAST  past frames per step (default 8; images/step = past+1)
#   OPENFLY_QWEN_CHAT_WINDOW_TURNS     rolling (user->assistant) pairs (default 4)
#   OPENFLY_QWEN_INSTRUCTION_ONCE      1/0; instruction+suffix only on first user in window (default 1)
# Speed / profiling (Qwen closed-loop):
#   OPENFLY_EVAL_MAX_STEPS           max inferences per trajectory (default 40 in eval.py)
#   OPENFLY_EVAL_MAX_TRAJECTORIES    after prefix filter + start_index, run at most N episodes (0=unlimited).
#                                     Example: ONE_TRAJ= OPENFLY_GT_ENV_PREFIXES=env_airsim_16/ + MAX_TRAJECTORIES=1
#                                      for baseline wall time ~40 steps before resizing (Run 0), then Run 1+ with resize.
#   OPENFLY_EVAL_TIMING              1/true — per-step EVAL_TIMING lines, timing_steps.jsonl, timing_summary.json
#   OPENFLY_QWEN_EVAL_IMAGE_WIDTH    paired with HEIGHT: cv2.resize (INTER_AREA) before PIL (omit both = full-res)
#   OPENFLY_QWEN_EVAL_IMAGE_HEIGHT   e.g. 224 + WIDTH 224 for 224² inputs to the HF processor.
#   OPENFLY_EVAL_DISABLE_EARLY_STOP   1/true — ignore predicted action 0 for loop exit (always OPENFLY_EVAL_MAX_STEPS infer steps).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CKPT="${OPENFLY_EVAL_QWEN3_CHECKPOINT:-/mnt/xtb/vln/qwen3-vl-2b-vln-1frame-defaultsys-frozenvision-full-8gpu-b8/checkpoint-7741}"
JSON="${OPENFLY_EVAL_JSON:-data_curated/seen_curated.json}"
RUN_ROOT="${OPENFLY_EVAL_BATCH_ROOT:-$ROOT/eval_runs/qwen7741_per_env_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
if [[ -n "${OPENFLY_EVAL_ENVS:-}" ]]; then
  _tmp="${OPENFLY_EVAL_ENVS//,/ }"
  read -r -a ENVS <<< "${_tmp}"
else
  ENVS=(env_airsim_16 env_airsim_18 env_airsim_23 env_airsim_26 env_airsim_gz env_airsim_sh)
fi
for env in "${ENVS[@]}"; do
  echo "========== ${env} =========="
  OUT="$RUN_ROOT/$env"
  mkdir -p "$OUT"
  OPENFLY_GT_DUMP=0 \
  OPENFLY_EVAL_QWEN3_CHECKPOINT="$CKPT" \
  OPENFLY_EVAL_JSON="$JSON" \
  OPENFLY_GT_ENV_PREFIXES="${env}/" \
  OPENFLY_QWEN_TEMPORAL_HISTORY_PAST="${OPENFLY_QWEN_TEMPORAL_HISTORY_PAST:-}" \
  OPENFLY_QWEN_CHAT_WINDOW_TURNS="${OPENFLY_QWEN_CHAT_WINDOW_TURNS:-}" \
  OPENFLY_QWEN_INSTRUCTION_ONCE="${OPENFLY_QWEN_INSTRUCTION_ONCE:-}" \
  OPENFLY_EVAL_MAX_STEPS="${OPENFLY_EVAL_MAX_STEPS:-}" \
  OPENFLY_EVAL_MAX_TRAJECTORIES="${OPENFLY_EVAL_MAX_TRAJECTORIES:-}" \
  OPENFLY_EVAL_TIMING="${OPENFLY_EVAL_TIMING:-}" \
  OPENFLY_QWEN_EVAL_IMAGE_WIDTH="${OPENFLY_QWEN_EVAL_IMAGE_WIDTH:-}" \
  OPENFLY_QWEN_EVAL_IMAGE_HEIGHT="${OPENFLY_QWEN_EVAL_IMAGE_HEIGHT:-}" \
  OPENFLY_EVAL_DISABLE_EARLY_STOP="${OPENFLY_EVAL_DISABLE_EARLY_STOP:-}" \
  OPENFLY_EVAL_OUT_DIR="$OUT" \
  python3 -u "$ROOT/train/eval.py" 2>&1 | tee "$OUT/console.log"
  echo "Artifacts: $OUT/predictions.json $OUT/metrics.json (timing: timing_steps.jsonl timing_summary.json if OPENFLY_EVAL_TIMING)"
done
echo "Batch root: $RUN_ROOT"
