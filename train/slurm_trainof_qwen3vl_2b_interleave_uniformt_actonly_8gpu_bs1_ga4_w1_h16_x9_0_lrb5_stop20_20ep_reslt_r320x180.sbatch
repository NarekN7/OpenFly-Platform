#!/usr/bin/env bash
# Updated baseline: action + EOS training; filenames retain the historical run IDs.
# Submit from this checkout with OUTPUT_DIR set to a new directory.
#SBATCH --job-name=of-q3vl2b-lrb5-eos
#SBATCH --account=research
#SBATCH --partition=research
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
ROOT="${OPENFLY_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
[[ -f "$ROOT/train/run_qwen3_vl_skillmix.sh" ]] || { echo "Submit from the updated repository root" >&2; exit 1; }
export SKILL_MIX_RATE="${SKILL_MIX_RATE:-0.05}"
exec bash "$ROOT/train/run_qwen3_vl_skillmix.sh" "$@"
