#!/bin/bash
#SBATCH --job-name=OF-qwen35-2b-contract
#SBATCH --partition=research8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

ROOT="/mnt/weka/nnurijanyan/OpenFly-Platform"
cd "${ROOT}"
mkdir -p slurm_logs

source /home/nnurijanyan/OpenFly-Platform/TrainOF/bin/activate
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi
python "${ROOT}/scripts/qwen35_2b_contract_test.py" \
  --output_json "${ROOT}/slurm_logs/qwen35_2b_contract_${SLURM_JOB_ID}.json"
