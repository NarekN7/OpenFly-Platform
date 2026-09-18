#!/usr/bin/env bash
# Arguments: immutable run directory, Conda root, optional per-split smoke limit.
# The submitter supplies account, partition, resources, array, and log paths.
set -eo pipefail
run_root="${1:?pass run directory}"
conda_root="${2:?pass Conda root}"
sample_limit="${3:-0}"
source "${conda_root}/etc/profile.d/conda.sh"
conda activate vln
set -u
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export USE_TORCH=1
export USE_TF=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENFLY_QWEN_DEVICE_MAP=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${run_root}"
task_index="${SLURM_ARRAY_TASK_ID:-1}"
IFS=$'\t' read -r mix checkpoint < <(sed -n "$((task_index + 1))p" checkpoints.tsv)
if [[ -z "${mix}" || -z "${checkpoint}" ]]; then
  echo "No checkpoint for task index ${task_index}" >&2
  exit 1
fi
if (( sample_limit > 0 )); then
  result_dir="${run_root}/smoke-${SLURM_JOB_ID}/lrb${mix}"
else
  result_dir="${run_root}/results/lrb${mix}"
fi
sha256sum --check source.sha256
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "job=${SLURM_JOB_ID} task=${task_index} mix=${mix} checkpoint=${checkpoint}"
python -u code/train/skill_eval_lr_diagnostics.py \
  --checkpoint "${checkpoint}" \
  --train-json "${run_root}/inputs/train_lr.json" \
  --validation-json "${run_root}/inputs/validation_lr.json" \
  --image-root "$(cat frames_root.txt)" \
  --out-dir "${result_dir}" \
  --device cuda:0 \
  --temporal-past 16 \
  --max-new-tokens 16 \
  --limit-per-split "${sample_limit}" \
  --check-reference
