#!/usr/bin/env bash
# Usage:
#   ./deploy.sh ssh root@<host> -p <port> -i <key>
#
# Example:
#   ./deploy.sh ssh root@103.196.86.17 -p 43895 -i ~/.ssh/id_ed25519

set -euo pipefail

# Parse the pasted `ssh root@HOST -p PORT -i KEY` command.
# Accepts with or without the leading "ssh".
SSH_ARGS=("$@")
[[ "${SSH_ARGS[0]}" == "ssh" ]] && SSH_ARGS=("${SSH_ARGS[@]:1}")

POD_HOST=""
POD_PORT=22
SSH_KEY=""

i=0
while [[ $i -lt ${#SSH_ARGS[@]} ]]; do
    arg="${SSH_ARGS[$i]}"
    case "$arg" in
        root@*) POD_HOST="${arg#root@}" ;;
        -p)     i=$(( i+1 )); POD_PORT="${SSH_ARGS[$i]}" ;;
        -i)     i=$(( i+1 )); SSH_KEY="${SSH_ARGS[$i]}" ;;
    esac
    i=$(( i+1 ))
done

[[ -z "$POD_HOST" ]] && { echo "Error: could not parse host from: $*" >&2; exit 1; }

KEY_ARGS=()
if [[ -n "$SSH_KEY" ]]; then
    KEY_ARGS=(-i "$SSH_KEY")
fi

echo "→ Pod  : $POD_HOST  (port $POD_PORT)"
[[ -n "$SSH_KEY" ]] && echo "→ Key  : $SSH_KEY"
echo ""

# ── Remote: clone repo, install deps and start server ────────────────────────
REPO="https://github.com/vedtam/firered-aio.git"
REPO_DIR="/workspace/firered-aio"

ssh -T \
    "${KEY_ARGS[@]}" \
    -p "$POD_PORT" \
    -o StrictHostKeyChecking=no \
    -o BatchMode=yes \
    "root@$POD_HOST" \
    REPO_DIR="$REPO_DIR" REPO="$REPO" 'bash -s' <<'REMOTE'
set -e

if [[ -d "$REPO_DIR" ]]; then
    echo "[runpod] Updating repo..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[runpod] Cloning repo..."
    git clone "$REPO" "$REPO_DIR"
fi

cd "$REPO_DIR"

# Ensure CUDA_VISIBLE_DEVICES=0 is set for all future sessions on this pod.
if ! grep -q 'CUDA_VISIBLE_DEVICES' ~/.bashrc 2>/dev/null; then
    echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
    echo "[runpod] Set CUDA_VISIBLE_DEVICES=0 in ~/.bashrc"
fi
