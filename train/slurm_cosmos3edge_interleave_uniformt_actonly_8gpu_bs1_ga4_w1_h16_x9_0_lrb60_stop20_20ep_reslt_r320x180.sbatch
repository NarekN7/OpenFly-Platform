#!/bin/bash
#
# Cosmos3-Edge reasoner, exact OpenFly Qwen LRB60 training semantics:
# general + 60% L/R skill mix + 20% stop; interleaved h16; action+EOS train;
# final-action-only skill validation; full model including SigLIP2 is unfrozen.
#
# Full run:
#   sbatch train/slurm_cosmos3edge_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb60_stop20_20ep_reslt_r320x180.sbatch
#
# Bounded smoke (separate output, includes a resume step):
#   CKPT_ROOT=/mnt/weka/nnurijanyan/checkpoints/cosmos3-edge-lrb60-smoke \
#   MAX_STEPS=2 DEBUG_SAMPLES=16 EVAL_STEPS=1 SMOKE_RESUME_AFTER=1 \
#   sbatch train/slurm_cosmos3edge_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb60_stop20_20ep_reslt_r320x180.sbatch

#SBATCH --job-name=OF-cosmos3edge-interleave-lrb60-stop20
#SBATCH --partition=research8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

ROOT="/mnt/weka/nnurijanyan/OpenFly-Platform"
COSMOS_FRAMEWORK_ROOT="${COSMOS_FRAMEWORK_ROOT:-/mnt/weka/nnurijanyan/cosmos-framework}"
COSMOS_FRAMEWORK_COMMIT="${COSMOS_FRAMEWORK_COMMIT:-cf5d68c00d97ccd2480a2320ed652b92dec63102}"
MODEL_REVISION="${COSMOS3_EDGE_MODEL_REVISION:-344d602b128d1bbdacb43b08d0a3626f46343e29}"

cd "${ROOT}"
mkdir -p slurm_logs

if [[ ! -f "${COSMOS_FRAMEWORK_ROOT}/.venv/bin/activate" ]]; then
  echo "Missing Cosmos environment. Run train/setup_cosmos3_edge_env.sh first." >&2
  exit 1
fi
if [[ "$(git -C "${COSMOS_FRAMEWORK_ROOT}" rev-parse HEAD)" != "${COSMOS_FRAMEWORK_COMMIT}" ]]; then
  echo "Cosmos framework checkout is not pinned to ${COSMOS_FRAMEWORK_COMMIT}" >&2
  exit 1
fi

source "${COSMOS_FRAMEWORK_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH=
export PYTHONPATH="${ROOT}/scripts:${COSMOS_FRAMEWORK_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COSMOS3_EDGE_MODEL_REVISION="${MODEL_REVISION}"
export COSMOS3_EDGE_ATTN_IMPLEMENTATION="${COSMOS3_EDGE_ATTN_IMPLEMENTATION:-sdpa}"

export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

MODEL_REF="${INIT_MODEL:-nvidia/Cosmos3-Edge}"
export MODEL_REF
CKPT_ROOT="${CKPT_ROOT:-/mnt/weka/nnurijanyan/checkpoints/cosmos3-edge-vln-interleave-uniformt-actonly-8gpu-bs1-ga4-w1-h16-x9-0-lrb60-stop20-20ep-reslt-r320x180}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
MAX_STEPS="${MAX_STEPS:--1}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-0}"
EVAL_STEPS="${EVAL_STEPS:-500}"
RESUME="${RESUME:-0}"

RESUME_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${CKPT_ROOT}/checkpoint-last")
fi
DEBUG_ARGS=()
if [[ "${DEBUG_SAMPLES}" -gt 0 ]]; then
  DEBUG_ARGS=(--debug_samples "${DEBUG_SAMPLES}" --verify_images_exist)
fi

echo "==== Cosmos3-Edge OpenFly LRB60 ===="
echo "framework_commit: ${COSMOS_FRAMEWORK_COMMIT}"
echo "model: ${MODEL_REF}"
echo "model_revision: ${MODEL_REVISION}"
echo "attention: ${COSMOS3_EDGE_ATTN_IMPLEMENTATION}"
echo "checkpoint: ${CKPT_ROOT}"
echo "train general: data_curated/trainx9_curated_0.json (52,832)"
echo "train skill L/R: data_curated/trainx9_{left,right}_curated.json"
echo "train skill stop: data_curated/trainx9_stop_curated.json"
echo "skill_mix_rate: 0.60 (15,849 left + 15,849 right)"
echo "skill_mix_rate_stop: 0.20 (10,566 stop; epoch_len=95,096)"
echo "eval: left+right+stop skill validation (300), final action only"
echo "vision: UNFROZEN (SigLIP2 + projector + LM + head trainable)"
echo "history: 16 past pairs + current (up to 17 interleaved turns)"
echo "pixels: min=784 max=57,600; Cosmos processor enforces factor 32"
echo "batch: 1/GPU x 8 GPUs x accumulation 4 = 32"
echo "lr=${LEARNING_RATE}; epochs=${NUM_TRAIN_EPOCHS}; max_steps=${MAX_STEPS}"
echo "schedule=cosine; warmup_ratio=0.05; seed=7; bf16; gradient checkpointing"
echo "eval_steps=${EVAL_STEPS}; eval-best + checkpoint-last"

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
assert transformers.__version__.split(".", 1)[0] == "4"
PY
nvidia-smi

echo "==== huggingface_hub.snapshot_download ${MODEL_REF} (one process before DDP) ===="
python - <<'PY'
import os
from cosmos3_edge_vl_sft import _prefetch_cosmos3_edge
_prefetch_cosmos3_edge(os.environ["MODEL_REF"])
PY

run_train() {
  local max_steps="$1"
  shift
  accelerate launch --num_processes 8 "${ROOT}/scripts/cosmos3_edge_vl_sft.py" \
    --model_name_or_path "${MODEL_REF}" \
    --train_json "${ROOT}/data_curated/trainx9_curated_0.json" \
    --skill_json_left "${ROOT}/data_curated/trainx9_left_curated.json" \
    --skill_json_right "${ROOT}/data_curated/trainx9_right_curated.json" \
    --skill_mix_rate 0.60 \
    --skill_json_stop "${ROOT}/data_curated/trainx9_stop_curated.json" \
    --skill_mix_rate_stop 0.20 \
    --eval_skill_json_left "${ROOT}/data_curated/left_evaluation_skill_validation.json" \
    --eval_skill_json_right "${ROOT}/data_curated/right_evaluation_skill_validation.json" \
    --eval_skill_json_stop "${ROOT}/data_curated/stop_evaluation_skill_validation.json" \
    --frames_root "/mnt/weka/nnurijanyan/data/vln/train_curated" \
    --output_dir "${CKPT_ROOT}" \
    --chat_window_turns 1 \
    --temporal_history_past 16 \
    --min_pixels 784 \
    --max_pixels 57600 \
    --loss_type weighted \
    --use_default_vln_system_prompt \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${max_steps}" \
    --eval_steps "${EVAL_STEPS}" \
    --checkpoint_layout best_last \
    --report_to tensorboard \
    --logging_steps 10 \
    --max_length 16384 \
    --dtype bf16 \
    --gradient_checkpointing \
    "${DEBUG_ARGS[@]}" \
    "$@"
}

run_train "${MAX_STEPS}" "${RESUME_ARGS[@]}"

if [[ "${SMOKE_RESUME_AFTER:-0}" == "1" ]]; then
  echo "==== smoke resume from checkpoint-last ===="
  if [[ ! -d "${CKPT_ROOT}/checkpoint-last" ]]; then
    echo "Missing ${CKPT_ROOT}/checkpoint-last after smoke train" >&2
    exit 1
  fi
  NEXT_STEPS=$((MAX_STEPS + 1))
  run_train "${NEXT_STEPS}" --resume_from_checkpoint "${CKPT_ROOT}/checkpoint-last"
fi
