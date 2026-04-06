#!/usr/bin/env bash
# Start only the AirSim UE binary (headless flags are in envs/airsim/.../LinuxNoEditor/start.sh).
# Run from repo root is not required; script cds to repo.
#
# Usage:
#   bash scripts/run_airsim_sim_only.sh
#   OPENFLY_AIRSIM_ENV=env_airsim_18 bash scripts/run_airsim_sim_only.sh
#
# Then in another terminal (after ~40s): run GT eval with OPENFLY_SKIP_AIRSIM_LAUNCH=1

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${OPENFLY_AIRSIM_ENV:-env_airsim_16}"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/runtime-openfly-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR"

START_SH="${REPO_ROOT}/envs/airsim/${ENV_NAME}/LinuxNoEditor/start.sh"
if [[ ! -f "$START_SH" ]]; then
  echo "Missing $START_SH" >&2
  exit 1
fi

echo "Starting ${ENV_NAME} (also appending to ${REPO_ROOT}/airsim_start.log)"
bash "$START_SH" 2>&1 | tee -a "${REPO_ROOT}/airsim_start.log"
