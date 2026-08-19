#!/bin/bash
#
# Qwen3-VL-2B trajectory-crop SFT: 8 GPUs, per-GPU batch 1, grad accum 1, global batch 8*1=8, 10 epochs,
# linear weighted loss, TensorBoard under ${CKPT_ROOT}/tb/, and two curated checkpoints:
#   checkpoint-best/  (lowest logged train loss, model + processor)
#   checkpoint-last/  (full mirror of the latest HF epoch checkpoint for resume)
#
# First run: loads base weights from ${INIT_MODEL}. To continue training from the same tree:
#   export RESUME=1
#   (script switches to --resume_from_checkpoint ${CKPT_ROOT}/checkpoint-last)
#
# Submit:
#   cd /home/nnurijanyan/OpenFly-Platform && mkdir -p slurm_logs && sbatch train/slurm_trainof_qwen3vl_4b_8x8_20ep_bestlast.sbatch
#

#SBATCH --job-name=OF-qwen3vl2b-bs1-w1-h16-10ep
#SBATCH --partition=research
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=2-00:00:00
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
# Helps fragmentation when sequences are very long (Qwen3-VL trajectory windows).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

# Single hub for artifacts (TensorBoard + checkpoints). Reads for resume come from the same tree.
CKPT_ROOT="${CKPT_ROOT:-/mnt/weka/nnurijanyan/checkpoints/qwen3-vl-2b-vln-singleturn-8gpu-bs1-w1-h16-10ep}"
INIT_MODEL="${INIT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
export INIT_MODEL
RESUME="${RESUME:-0}"
# Set to 0 after a successful run to skip re-download (saves time). Default 1 fixes poisoned/mixed HF hub trees.
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

# Optional cache purge + prefetch for selected INIT_MODEL.
HF_HUB_DIR="${HF_HOME}/hub"
mkdir -p "${HF_HUB_DIR}"
HUB_REPO_DIR="${HF_HUB_DIR}/models--${INIT_MODEL//\//--}"
if [[ "${FORCE_HF_MODEL_REFRESH}" == "1" || "${FORCE_HF_MODEL_REFRESH}" == "true" ]]; then
  if [[ -d "${HUB_REPO_DIR}" ]]; then
    echo "==== Removing possibly corrupted HF hub snapshot ===="
    echo "${HUB_REPO_DIR}"
    rm -rf "${HUB_REPO_DIR}"
  else
    echo "==== No existing hub dir (fresh download) ===="
    echo "${HUB_REPO_DIR}"
  fi
  echo "==== huggingface_hub.snapshot_download ${INIT_MODEL} (one process before DDP) ===="
  python - <<PY
import os
from huggingface_hub import snapshot_download
repo = os.environ["INIT_MODEL"]
print("snapshot_download:", repo)
snapshot_download(repo_id=repo)
print("snapshot_download done")
PY
else
  echo "==== FORCE_HF_MODEL_REFRESH=${FORCE_HF_MODEL_REFRESH}: skipping hub purge/prefetch ===="
fi

nvidia-smi || true

accelerate launch --num_processes 8 /home/nnurijanyan/OpenFly-Platform/scripts/qwen3_vl_sft.py \
  --model_name_or_path "${MODEL_REF}" \
  --train_json "/home/nnurijanyan/OpenFly-Platform/data_curated/train_curated.json" \
  --frames_root "/mnt/weka/nnurijanyan/data/vln/train_curated" \
  --output_dir "${CKPT_ROOT}" \
  --chat_window_turns 1 \
  --temporal_history_past 16 \
  --loss_type weighted \
  --use_default_vln_system_prompt \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --checkpoint_layout best_last \
  --report_to tensorboard \
  --logging_steps 10 \
  --max_length 16384 \
  --dtype bf16 \
  --gradient_checkpointing \
  "${RESUME_ARG[@]}"
