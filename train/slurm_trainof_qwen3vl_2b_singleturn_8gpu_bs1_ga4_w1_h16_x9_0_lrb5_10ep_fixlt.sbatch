#!/bin/bash
#
# Qwen3-VL-2B x9 tier-2, end-with-0 general + 5% balanced L/R skill mix:
#   train general: data_curated/trainx9_curated_0.json (52,832)
#   train skill L: data_curated/trainx9_left_curated.json (last action 2)
#   train skill R: data_curated/trainx9_right_curated.json (last action 3)
#   skill_mix_rate: 0.05 => R_each=1320 each side, +2640 skill samples/epoch (epoch len 55472)
#   eval:  data_curated/validationx9.json (996)
#
# Submit:
#   cd /home/nnurijanyan/OpenFly-Platform && sbatch train/slurm_trainof_qwen3vl_2b_singleturn_8gpu_bs1_ga4_w1_h16_x9_0_lrb5_10ep_fixlt.sbatch
#

#SBATCH --job-name=OF-qwen3vl2b-bs1-ga4-h16-x9-0-lrb5-10ep-fixlt
#SBATCH --partition=research
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

cd /home/nnurijanyan/OpenFly-Platform
mkdir -p slurm_logs

if [[ ! -f "/home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate" ]]; then
  echo "Missing TrainOF venv at /home/nnurijanyan/OpenFly-Platform/TrainOF" >&2
  exit 1
fi

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

CKPT_ROOT="${CKPT_ROOT:-/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-2b-vln-singleturn-8gpu-bs1-ga4-w1-h16-x9-0-lrb5-10ep-fixlt}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
INIT_MODEL="${INIT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
export INIT_MODEL
RESUME="${RESUME:-0}"
FORCE_HF_MODEL_REFRESH="${FORCE_HF_MODEL_REFRESH:-0}"

MODEL_REF="${INIT_MODEL}"
RESUME_ARG=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" ]]; then
  MODEL_REF="${INIT_MODEL}"
  RESUME_ARG=(--resume_from_checkpoint "${CKPT_ROOT}/checkpoint-last")
fi

echo "==== CKPT_ROOT (read/write) ===="
echo "${CKPT_ROOT}"
echo "==== MODEL_REF ===="
echo "${MODEL_REF}"
echo "==== RESUME ===="
echo "${RESUME}"
echo "==== Data (x9 tier-2, end-with-0 + balanced L/R skill mix 5%) ===="
echo "train general: data_curated/trainx9_curated_0.json (52,832)"
echo "train skill L: data_curated/trainx9_left_curated.json"
echo "train skill R: data_curated/trainx9_right_curated.json"
echo "skill_mix_rate: 0.05 (R_each=1320 left + 1320 right)"
echo "eval:  data_curated/validationx9.json"
echo "==== Vision encoder ===="
echo "UNFROZEN (no --freeze_vision_encoder)"
echo "==== temporal_history_past ===="
echo "16 (17 images per user turn)"
echo "==== Global batch size ===="
echo "per_device_train_batch_size=1 x num_gpus=8 x gradient_accumulation_steps=4 => 32"
echo "==== LEARNING_RATE ===="
echo "${LEARNING_RATE}"
echo "==== NUM_TRAIN_EPOCHS ===="
echo "${NUM_TRAIN_EPOCHS}"
echo "==== eval-best ===="
echo "lowest eval_loss (eval every 500 steps) → \${CKPT_ROOT}/eval-best"

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
  python - <<PY
import os
from huggingface_hub import snapshot_download
repo = os.environ["INIT_MODEL"]
print("snapshot_download:", repo)
path = snapshot_download(repo_id=repo)
print("snapshot_download done:", path)
PY
else
  echo "==== Local INIT_MODEL (${INIT_MODEL}); skipping hub snapshot_download ===="
fi

nvidia-smi || true

accelerate launch --num_processes 8 /home/nnurijanyan/OpenFly-Platform/scripts/qwen3_vl_sft.py \
  --model_name_or_path "${MODEL_REF}" \
  --train_json "/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_curated_0.json" \
  --skill_json_left "/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_left_curated.json" \
  --skill_json_right "/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_right_curated.json" \
  --skill_mix_rate 0.05 \
  --eval_json "/home/nnurijanyan/OpenFly-Platform/data_curated/validationx9.json" \
  --frames_root "/mnt/weka/nnurijanyan/data/vln/train_curated" \
  --output_dir "${CKPT_ROOT}" \
  --chat_window_turns 1 \
  --temporal_history_past 16 \
  --loss_type weighted \
  --use_default_vln_system_prompt \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --eval_steps 500 \
  --checkpoint_layout best_last \
  --report_to tensorboard \
  --logging_steps 10 \
  --max_length 16384 \
  --dtype bf16 \
  --gradient_checkpointing \
  "${RESUME_ARG[@]}"
