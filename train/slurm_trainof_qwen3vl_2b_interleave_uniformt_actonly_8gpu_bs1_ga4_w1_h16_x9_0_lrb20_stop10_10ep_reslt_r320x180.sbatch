#!/bin/bash
#
# Qwen3-VL-2B interleave + uniform-t + actonly supervision, x9 tier-2:
#   general + 20% L/R skill mix + 10% stop skill
#   skill_mix_rate: 0.20 => R_each=5283; skill_mix_rate_stop: 0.10 => N_stop=5283 (epoch_len=68681)
#   no last-turn prompt suffix (USER[frame] -> ASSISTANT[action])
#
# Submit:
#   cd /mnt/weka/nnurijanyan/OpenFly-Platform && sbatch train/slurm_trainof_qwen3vl_2b_interleave_uniformt_actonly_8gpu_bs1_ga4_w1_h16_x9_0_lrb20_stop10_10ep_reslt_r320x180.sbatch
#

#SBATCH --job-name=OF-qwen3vl2b-interleave-actonly-lrb20-stop10-10ep
#SBATCH --partition=research
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

cd /mnt/weka/nnurijanyan/OpenFly-Platform
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

CKPT_ROOT="${CKPT_ROOT:-/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-2b-vln-interleave-uniformt-actonly-8gpu-bs1-ga4-w1-h16-x9-0-lrb20-stop10-10ep-reslt-r320x180}"
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
echo "==== Data (x9 tier-2, interleave-uniformt-actonly, L/R 20%, stop 10%) ===="
echo "train general: data_curated/trainx9_curated_0.json (52,832)"
echo "train skill L: data_curated/trainx9_left_curated.json"
echo "train skill R: data_curated/trainx9_right_curated.json"
echo "train skill stop: data_curated/trainx9_stop_curated.json"
echo "skill_mix_rate: 0.20 (R_each=5283 left + 5283 right)"
echo "skill_mix_rate_stop: 0.10 (N_stop=5283; epoch_len=68681)"
echo "eval:  skill-combined left+right+stop validation (300), last_token loss"
echo "  left:  data_curated/left_evaluation_skill_validation.json"
echo "  right: data_curated/right_evaluation_skill_validation.json"
echo "  stop:  data_curated/stop_evaluation_skill_validation.json"
echo "==== Vision encoder ===="
echo "UNFROZEN (no --freeze_vision_encoder)"
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
echo "==== eval-best ===="
echo "lowest skill eval_loss (eval every 500 steps) → \${CKPT_ROOT}/eval-best (+ checkpoint-last)"

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
  python - <<'PYINNER'
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

nvidia-smi || true

accelerate launch --num_processes 8 /home/nnurijanyan/OpenFly-Platform/scripts/qwen3_vl_sft.py \
  --model_name_or_path "${MODEL_REF}" \
  --train_json "/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_curated_0.json" \
  --skill_json_left "/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_left_curated.json" \
  --skill_json_right "/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_right_curated.json" \
  --skill_mix_rate 0.20 \
  --skill_json_stop "/home/nnurijanyan/OpenFly-Platform/data_curated/trainx9_stop_curated.json" \
  --skill_mix_rate_stop 0.10 \
  --eval_skill_json_left "/home/nnurijanyan/OpenFly-Platform/data_curated/left_evaluation_skill_validation.json" \
  --eval_skill_json_right "/home/nnurijanyan/OpenFly-Platform/data_curated/right_evaluation_skill_validation.json" \
  --eval_skill_json_stop "/home/nnurijanyan/OpenFly-Platform/data_curated/stop_evaluation_skill_validation.json" \
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
  --eval_steps 500 \
  --checkpoint_layout best_last \
  --report_to tensorboard \
  --logging_steps 10 \
  --max_length 16384 \
  --dtype bf16 \
  --gradient_checkpointing \
  "${RESUME_ARG[@]}"
