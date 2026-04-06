#!/usr/bin/env bash
# Run on the **other computer** (destination). Pulls the repo from np over SSH.
#
# Prerequisites on this machine: rsync, ssh (key or agent to log in to np).
#
# Usage:
#   bash rsync_pull_OpenFly_from_np.sh <NP_USER> <NP_HOSTNAME> <LOCAL_DEST_DIR>
#
# Example:
#   bash rsync_pull_OpenFly_from_np.sh nareknurijanyan np /home/me/OpenFly-Platform
#
# If np’s full hostname works better than the short name:
#   bash rsync_pull_OpenFly_from_np.sh nareknurijanyan np.cs.university.edu /home/me/OpenFly-Platform
#
# Optional env:
#   NP_REPO_PATH  path on np (default: /auto/home/nareknurijanyan/OpenFly-Platform)
#
# Run inside tmux/screen if the transfer is large.

set -euo pipefail

NP_USER="${1:?Usage: $0 <NP_USER> <NP_HOST> <LOCAL_DEST_DIR>}"
NP_HOST="${2:?}"
LOCAL_DEST="${3:?}"

NP_PATH="${NP_REPO_PATH:-/auto/home/nareknurijanyan/OpenFly-Platform}"
REMOTE_SRC="${NP_USER}@${NP_HOST}:${NP_PATH}/"

echo "Remote:  ${REMOTE_SRC}"
echo "Local:   ${LOCAL_DEST}/"
echo "Starting rsync (pull)..."
exec rsync -a --mkpath --info=progress2 -e ssh \
  "${REMOTE_SRC}" \
  "${LOCAL_DEST}/"
