#!/bin/bash
#
# Qwen3.5-2B overfit-8 pipeline check (1 GPU):
#   4 general crops (weighted) + 2 left + 2 right (last_token)
#   same collator and WeightedTrainer as the LRB60 Qwen3.5 run
#   eval_loss on the same 8 crops every epoch
#
# Submit:
#   cd /home/nnurijanyan/OpenFly-Platform && sbatch train/slurm_qwen35_2b_overfit8_lrb.sbatch
#

#SBATCH --job-name=OF-qwen35-2b-overfit8-lrb
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
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

CKPT_ROOT="${CKPT_ROOT:-/mnt/weka/nnurijanyan/checkpoints/qwen35-2b-vln-overfit8-lrb}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2000}"
INIT_MODEL="${INIT_MODEL:-Qwen/Qwen3.5-2B}"
export INIT_MODEL
RESUME="${RESUME:-0}"
FORCE_HF_MODEL_REFRESH="${FORCE_HF_MODEL_REFRESH:-0}"

RESUME_ARG=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  RESUME_ARG=(--resume_from_checkpoint "${CKPT_ROOT}/checkpoint-last")
fi

echo "==== CKPT_ROOT (read/write) ===="
echo "${CKPT_ROOT}"
echo "==== INIT_MODEL ===="
echo "${INIT_MODEL}"
echo "==== RESUME ===="
echo "${RESUME}"
echo "==== Data (overfit-8 fixed crops) ===="
echo "train general: data_curated/overfit8/train_4.json (4, weighted)"
echo "train skill L: data_curated/overfit8/skill_left_2.json (2, last_token)"
echo "train skill R: data_curated/overfit8/skill_right_2.json (2, last_token)"
echo "epoch_len: 8 (concat-all, always last-t)"
echo "eval: same 8 crops and loss modes (every 8 steps = 1 epoch)"
echo "==== Vision encoder ===="
echo "UNFROZEN (no --freeze_vision_encoder)"
echo "==== thinking ===="
echo "enable_thinking=False (Qwen3.5 collator)"
echo "==== temporal_history_past ===="
echo "16"
echo "==== image pixel budget ===="
echo "min_pixels=784 max_pixels=57600"
echo "==== Global batch size ===="
echo "per_device_train_batch_size=1 x num_gpus=1 x gradient_accumulation_steps=1 => 1"
echo "==== LEARNING_RATE ===="
echo "${LEARNING_RATE}"
echo "==== NUM_TRAIN_EPOCHS ===="
echo "${NUM_TRAIN_EPOCHS}"
echo "==== eval-best ===="
echo "lowest eval_loss (eval every 8 steps) → \${CKPT_ROOT}/eval-best (+ checkpoint-last)"

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

accelerate launch --num_processes 1 "${ROOT}/scripts/qwen35_2b_vl_sft_overfit.py" \
  --overfit_concat \
  --model_name_or_path "${INIT_MODEL}" \
  --train_json "${ROOT}/data_curated/overfit8/train_4.json" \
  --skill_json_left "${ROOT}/data_curated/overfit8/skill_left_2.json" \
  --skill_json_right "${ROOT}/data_curated/overfit8/skill_right_2.json" \
  --eval_json "${ROOT}/data_curated/overfit8/eval_8.json" \
  --frames_root "/mnt/weka/nnurijanyan/data/vln/train_curated" \
  --output_dir "${CKPT_ROOT}" \
  --chat_window_turns 1 \
  --temporal_history_past 16 \
  --min_pixels 784 \
  --max_pixels 57600 \
  --loss_type weighted \
  --use_default_vln_system_prompt \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --eval_steps 8 \
  --logging_steps 1 \
  --checkpoint_layout best_last \
  --report_to tensorboard \
  --max_length 16384 \
  --dtype bf16 \
  --gradient_checkpointing \
  "${RESUME_ARG[@]}"
