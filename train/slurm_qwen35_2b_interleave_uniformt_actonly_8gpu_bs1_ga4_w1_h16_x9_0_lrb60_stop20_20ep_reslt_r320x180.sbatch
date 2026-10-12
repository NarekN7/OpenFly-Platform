#!/bin/bash
#
# Qwen3.5-2B interleave + uniform-t + actonly supervision, x9 tier-2:
#   general + 60% L/R skill mix + 20% stop skill
#   skill_mix_rate: 0.60 => R_each=15849; skill_mix_rate_stop: 0.20 => N_stop=10566 (epoch_len=95096)
#   TrainOF env + same LRB60 hyperparameters as Qwen3-VL / Cosmos3-Edge.
#   Thinking disabled (enable_thinking=False); vision unfrozen.
#
# Full run:
#   sbatch train/slurm_qwen35_2b_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb60_stop20_20ep_reslt_r320x180.sbatch
#
# Bounded smoke (separate output, includes a resume step):
#   CKPT_ROOT=/mnt/weka/nnurijanyan/checkpoints/qwen35-2b-lrb60-smoke \
#   MAX_STEPS=2 DEBUG_SAMPLES=16 EVAL_STEPS=1 SMOKE_RESUME_AFTER=1 \
#   sbatch train/slurm_qwen35_2b_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb60_stop20_20ep_reslt_r320x180.sbatch
#

#SBATCH --job-name=OF-qwen35-2b-interleave-actonly-lrb60-stop20-20ep
#SBATCH --partition=research8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

ROOT="/mnt/weka/nnurijanyan/OpenFly-Platform"
cd "${ROOT}"
mkdir -p slurm_logs

if [[ ! -f "/home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate" ]]; then
  echo "Missing TrainOF venv at /home/nnurijanyan/OpenFly-Platform/TrainOF" >&2
  exit 1
fi

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate
PY="/home/nnurijanyan/OpenFly-Platform/TrainOF/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "TrainOF python is not executable: ${PY}" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

CKPT_ROOT="${CKPT_ROOT:-/mnt/weka/nnurijanyan/checkpoints/qwen35-2b-vln-interleave-uniformt-actonly-8gpu-bs1-ga4-w1-h16-x9-0-lrb60-stop20-20ep-reslt-r320x180}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
INIT_MODEL="${INIT_MODEL:-Qwen/Qwen3.5-2B}"
export INIT_MODEL
MAX_STEPS="${MAX_STEPS:--1}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-0}"
EVAL_STEPS="${EVAL_STEPS:-500}"
RESUME="${RESUME:-0}"
FORCE_HF_MODEL_REFRESH="${FORCE_HF_MODEL_REFRESH:-0}"

RESUME_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${CKPT_ROOT}/checkpoint-last")
fi
DEBUG_ARGS=()
if [[ "${DEBUG_SAMPLES}" -gt 0 ]]; then
  DEBUG_ARGS=(--debug_samples "${DEBUG_SAMPLES}" --verify_images_exist)
fi

echo "==== CKPT_ROOT (read/write) ===="
echo "${CKPT_ROOT}"
echo "==== INIT_MODEL ===="
echo "${INIT_MODEL}"
echo "==== RESUME ===="
echo "${RESUME}"
echo "==== Data (x9 tier-2, interleave-uniformt-actonly, L/R 60%, stop 20%) ===="
echo "train general: data_curated/trainx9_curated_0.json (52,832)"
echo "train skill L: data_curated/trainx9_left_curated.json"
echo "train skill R: data_curated/trainx9_right_curated.json"
echo "train skill stop: data_curated/trainx9_stop_curated.json"
echo "skill_mix_rate: 0.60 (R_each=15849 left + 15849 right)"
echo "skill_mix_rate_stop: 0.20 (N_stop=10566; epoch_len=95096)"
echo "eval:  skill-combined left+right+stop validation (300), last_token loss"
echo "  left:  data_curated/left_evaluation_skill_validation.json"
echo "  right: data_curated/right_evaluation_skill_validation.json"
echo "  stop:  data_curated/stop_evaluation_skill_validation.json"
echo "==== Vision encoder ===="
echo "UNFROZEN (no --freeze_vision_encoder)"
echo "==== thinking ===="
echo "enable_thinking=False (collator); last-turn empty think tags are unlabeled"
echo "==== temporal_history_past ===="
echo "16 (interleaved frame→action history; no prompt suffix)"
echo "==== image pixel budget ===="
echo "min_pixels=784 (28*28) max_pixels=57600 (320*180) → ~320x160"
echo "==== Global batch size ===="
echo "per_device_train_batch_size=1 x num_gpus=8 x gradient_accumulation_steps=4 => 32"
echo "==== LEARNING_RATE ===="
echo "${LEARNING_RATE}"
echo "==== NUM_TRAIN_EPOCHS ===="
echo "${NUM_TRAIN_EPOCHS}"
echo "==== sequence cap ===="
echo "sequences above 4096 tokens shorten the instruction only; 17-frame samples are ~1200 tokens"
echo "==== max_steps / debug_samples / eval_steps ===="
echo "${MAX_STEPS} / ${DEBUG_SAMPLES} / ${EVAL_STEPS}"
echo "==== eval-best ===="
echo "lowest skill eval_loss (eval every ${EVAL_STEPS} steps) → \${CKPT_ROOT}/eval-best (+ checkpoint-last)"

"${PY}" - <<'PY'
import transformers
from transformers import Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration
print("transformers", transformers.__version__)
print("Qwen3VL", Qwen3VLForConditionalGeneration)
print("Qwen3_5", Qwen3_5ForConditionalGeneration)
assert transformers.__version__.startswith("5.6")
PY
nvidia-smi || true

echo "==== Qwen3.5 DeltaNet fast path ===="
"${PY}" - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

print("cuda", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("qwen35_fast_path", is_fast_path_available)
if not is_fast_path_available:
    raise SystemExit(
        "Qwen3.5 is still on the slow PyTorch DeltaNet path. "
        "causal-conv1d and flash-linear-attention must both import on this GPU."
    )
x = torch.randn(1, 16, 32, device="cuda", dtype=torch.bfloat16)
weight = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16)
y = causal_conv1d_fn(x, weight, activation="silu")
if not torch.isfinite(y).all():
    raise SystemExit("causal_conv1d produced non-finite output")
print("causal_conv1d ok", tuple(y.shape))

import triton
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

print("triton", triton.__version__)
if tuple(int(p) for p in triton.__version__.split(".")[:3]) < (3, 7, 1):
    raise SystemExit(
        f"Triton {triton.__version__} is refused by flash-linear-attention on Hopper "
        "(gated chunk_bwd_dqkwg). Need >= 3.7.1."
    )
q = torch.randn(1, 64, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.nn.functional.normalize(
    torch.randn(1, 64, 2, 16, device="cuda", dtype=torch.bfloat16), p=2, dim=-1
)
v = torch.randn(1, 64, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
beta = torch.rand(1, 64, 2, device="cuda", dtype=torch.bfloat16).sigmoid()
g = torch.nn.functional.logsigmoid(torch.rand(1, 64, 2, device="cuda", dtype=torch.bfloat16))
out, _ = chunk_gated_delta_rule(q, k, v, g, beta)
out.sum().backward()
if q.grad is None or not torch.isfinite(q.grad).all():
    raise SystemExit("gated delta-rule backward did not produce finite grads")
print("gated_delta_bwd ok", tuple(out.shape))
PY

HF_HUB_DIR="${HF_HOME}/hub"
mkdir -p "${HF_HUB_DIR}"
HUB_REPO_DIR="${HF_HUB_DIR}/models--${INIT_MODEL//\//--}"
if [[ "${INIT_MODEL}" != /* ]]; then
  if [[ "${FORCE_HF_MODEL_REFRESH}" == "1" || "${FORCE_HF_MODEL_REFRESH}" == "true" ]]; then
    if [[ -d "${HUB_REPO_DIR}" ]]; then
      echo "==== Removing possibly corrupted HF hub snapshot ===="
      echo "${HUB_REPO_DIR}"
      rm -rf "${HUB_REPO_DIR}"
    fi
  fi
  echo "==== huggingface_hub.snapshot_download ${INIT_MODEL} (one process before DDP) ===="
  "${PY}" - <<'PYINNER'
import os
from huggingface_hub import snapshot_download
repo = os.environ["INIT_MODEL"]
print("snapshot_download:", repo)
path = snapshot_download(repo_id=repo)
print("snapshot_download done:", path)
PYINNER
else
  echo "==== Local INIT_MODEL (${INIT_MODEL}); skipping hub snapshot_download ===="
fi

run_train() {
  local max_steps="$1"
  shift
  accelerate launch --num_processes 8 "${ROOT}/scripts/qwen35_2b_vl_sft.py" \
    --model_name_or_path "${INIT_MODEL}" \
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
