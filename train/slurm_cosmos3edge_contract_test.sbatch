#!/bin/bash
#SBATCH --job-name=OF-cosmos3edge-contract
#SBATCH --partition=research8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/slurm_%j.out

set -euo pipefail

ROOT="/mnt/weka/nnurijanyan/OpenFly-Platform"
COSMOS_FRAMEWORK_ROOT="${COSMOS_FRAMEWORK_ROOT:-/mnt/weka/nnurijanyan/cosmos-framework}"
cd "${ROOT}"
mkdir -p slurm_logs
source "${COSMOS_FRAMEWORK_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH=
export PYTHONPATH="${ROOT}/scripts:${COSMOS_FRAMEWORK_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/weka/nnurijanyan/hf}"
export COSMOS3_EDGE_MODEL_REVISION="${COSMOS3_EDGE_MODEL_REVISION:-344d602b128d1bbdacb43b08d0a3626f46343e29}"
export COSMOS3_EDGE_ATTN_IMPLEMENTATION="${COSMOS3_EDGE_ATTN_IMPLEMENTATION:-sdpa}"

nvidia-smi
python "${ROOT}/scripts/cosmos3_edge_contract_test.py" \
  --output_json "${ROOT}/slurm_logs/cosmos3_edge_contract_${SLURM_JOB_ID}.json"
