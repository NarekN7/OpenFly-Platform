#!/usr/bin/env bash
# Prepare NVIDIA's pinned Cosmos3-Edge training runtime without changing TrainOF.
set -euo pipefail

COSMOS_FRAMEWORK_COMMIT="${COSMOS_FRAMEWORK_COMMIT:-cf5d68c00d97ccd2480a2320ed652b92dec63102}"
COSMOS_FRAMEWORK_ROOT="${COSMOS_FRAMEWORK_ROOT:-/mnt/weka/nnurijanyan/cosmos-framework}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/weka/nnurijanyan/uv-cache}"
export UV_CACHE_DIR

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  else
    echo "uv is required; install it from https://docs.astral.sh/uv/" >&2
    exit 1
  fi
fi

if [[ ! -d "${COSMOS_FRAMEWORK_ROOT}/.git" ]]; then
  mkdir -p "$(dirname "${COSMOS_FRAMEWORK_ROOT}")"
  git clone https://github.com/NVIDIA/cosmos-framework.git "${COSMOS_FRAMEWORK_ROOT}"
fi

current_commit="$(git -C "${COSMOS_FRAMEWORK_ROOT}" rev-parse HEAD)"
if [[ "${current_commit}" != "${COSMOS_FRAMEWORK_COMMIT}" ]]; then
  if ! git -C "${COSMOS_FRAMEWORK_ROOT}" diff --quiet ||
     ! git -C "${COSMOS_FRAMEWORK_ROOT}" diff --cached --quiet; then
    echo "Refusing to change dirty Cosmos framework checkout: ${COSMOS_FRAMEWORK_ROOT}" >&2
    exit 1
  fi
  git -C "${COSMOS_FRAMEWORK_ROOT}" fetch origin "${COSMOS_FRAMEWORK_COMMIT}"
  git -C "${COSMOS_FRAMEWORK_ROOT}" checkout --detach "${COSMOS_FRAMEWORK_COMMIT}"
fi

if [[ "$(git -C "${COSMOS_FRAMEWORK_ROOT}" rev-parse HEAD)" != "${COSMOS_FRAMEWORK_COMMIT}" ]]; then
  echo "Cosmos framework commit verification failed" >&2
  exit 1
fi

cd "${COSMOS_FRAMEWORK_ROOT}"
uv sync --all-extras --group=cu130-train

source "${COSMOS_FRAMEWORK_ROOT}/.venv/bin/activate"
export LD_LIBRARY_PATH=
python - <<'PY'
import torch
import transformers
import cosmos_framework

assert tuple(map(int, transformers.__version__.split(".")[:1])) < (5,)
print("cosmos_framework:", cosmos_framework.__file__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
PY

echo "Cosmos3-Edge environment ready: ${COSMOS_FRAMEWORK_ROOT}/.venv"
