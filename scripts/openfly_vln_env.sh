#!/usr/bin/env bash
# Source from the selected checkout. No package installation or cache mutation.
_openfly_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$_openfly_repo/.openfly.env" ]]; then
  source "$_openfly_repo/.openfly.env" || return 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "vln" ]]; then
  _openfly_conda_root="${OPENFLY_CONDA_ROOT:-}"
  if [[ -z "$_openfly_conda_root" ]] && command -v conda >/dev/null 2>&1; then
    _openfly_conda_root="$(conda info --base)"
  fi
  _openfly_conda_root="${_openfly_conda_root:-$HOME/miniconda3}"
  if [[ ! -f "$_openfly_conda_root/etc/profile.d/conda.sh" ]]; then
    echo "Set OPENFLY_CONDA_ROOT to the Conda installation containing env vln." >&2
    return 1
  fi
  # Conda activation scripts may reference unset variables under Slurm's set -u.
  _openfly_restore_nounset=0
  if [[ $- == *u* ]]; then
    _openfly_restore_nounset=1
    set +u
  fi
  source "$_openfly_conda_root/etc/profile.d/conda.sh" || return 1
  conda activate vln || return 1
  if (( _openfly_restore_nounset )); then
    set -u
  fi
fi
export USE_TORCH=1 USE_TF=0 TOKENIZERS_PARALLELISM=false
export OPENFLY_QWEN_STOP_AFTER_ACTION="${OPENFLY_QWEN_STOP_AFTER_ACTION:-1}"
python "$_openfly_repo/tools/check_qwen_runtime.py" || return 1
unset _openfly_repo _openfly_conda_root _openfly_restore_nounset
