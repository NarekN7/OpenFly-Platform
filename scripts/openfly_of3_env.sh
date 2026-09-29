#!/usr/bin/env bash
# Activate the OF3 conda env and set transformers flags for VLM eval.
# Source from eval launchers (do not execute directly):
#   source "$(dirname "$0")/openfly_of3_env.sh"
#
# Why USE_TORCH=1 / USE_TF=0:
# OF3 has TensorFlow installed for other tooling, but its TF build is binary-
# incompatible with the numpy used by VLM deps. transformers AUTO-enables TF
# when both USE_TORCH and USE_TF are unset, which breaks AutoImageProcessor.
# Forcing torch-only avoids that path. AirSim / unrealcv stay on OF3.

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "OF3" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate OF3
fi

export USE_TORCH="${USE_TORCH:-1}"
export USE_TF="${USE_TF:-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/runtime-openfly-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR"

if [[ "${OPENFLY_OF3_ENV_QUIET:-}" != "1" ]]; then
  echo "OF3 env: python=$(command -v python3) USE_TORCH=$USE_TORCH USE_TF=$USE_TF" >&2
fi
