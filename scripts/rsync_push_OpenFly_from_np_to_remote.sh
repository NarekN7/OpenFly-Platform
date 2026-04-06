#!/usr/bin/env bash
# Run on **np** (source). Copies this repo to another machine over SSH.
#
# Usage:
#   bash scripts/rsync_push_OpenFly_from_np_to_remote.sh <REMOTE_USER> <REMOTE_HOST> <REMOTE_DEST>
#
# Example:
#   bash scripts/rsync_push_OpenFly_from_np_to_remote.sh alice workstation /home/alice/OpenFly-Platform
#
# Optional env:
#   OPENFLY_REPO   source dir (default: /auto/home/nareknurijanyan/OpenFly-Platform)
# Add excludes by editing the rsync line or running rsync manually, e.g.:
#   --exclude '.cache/' --exclude 'data_wo_annotation/'
#
# Run inside tmux on np if the transfer is large.

set -euo pipefail

REMOTE_USER="${1:?Usage: $0 <REMOTE_USER> <REMOTE_HOST> <REMOTE_DEST_DIR>}"
REMOTE_HOST="${2:?}"
REMOTE_DEST="${3:?}"

REPO="${OPENFLY_REPO:-/auto/home/nareknurijanyan/OpenFly-Platform}"

echo "Source:  ${REPO}/"
echo "Target:  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/"
echo "Starting rsync (SSH)..."
exec rsync -a --mkpath --info=progress2 -e ssh \
  "${REPO}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/"
