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
    # Ensure the key has the _runpod suffix RunPod requires.
    [[ "$SSH_KEY" != *_runpod ]] && SSH_KEY="${SSH_KEY}_runpod"
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
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=60 \
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

# Keep the venv on the container disk (30 GB, 5 GB used) — not on
# /workspace — so model-cache has the full 50 GB volume to itself.
VENV_DIR="/root/.venvs/firered"

# Remove any stale venv that may have ended up on /workspace from
# a previous deploy (it ate disk space needed for model weights).
if [[ -d "/workspace/.venvs/firered" ]]; then
    echo "[runpod] Removing old /workspace venv to free volume space..."
    rm -rf /workspace/.venvs
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[runpod] Creating virtualenv at $VENV_DIR..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# Ensure CUDA_VISIBLE_DEVICES=0 and HF_HOME (model cache) point to persistent
# storage on the 50 GB /workspace volume, not the container disk.
if ! grep -q 'CUDA_VISIBLE_DEVICES' ~/.bashrc 2>/dev/null; then
    echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
    echo "[runpod] Set CUDA_VISIBLE_DEVICES=0 in ~/.bashrc"
fi
if ! grep -q 'HF_HOME' ~/.bashrc 2>/dev/null; then
    echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
    echo "[runpod] Set HF_HOME=/workspace/.cache/huggingface in ~/.bashrc"
fi
mkdir -p /workspace/.cache/huggingface

# Remove any model cache that landed on the container (ephemeral) disk.
if [[ -d ~/.cache/huggingface ]]; then
    echo "[runpod] Cleaning container-disk HF cache (~/.cache/huggingface)..."
    rm -rf ~/.cache/huggingface
fi

# Start the app in the background (nohup so it survives session end).
echo "[runpod] Installing requirements..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo "[runpod] Starting server..."
pkill -f "python app.py" 2>/dev/null || true
nohup env CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$VENV_DIR/bin/python" app.py > ~/firered.log 2>&1 &
echo "[runpod] Server PID: $! — logs at ~/firered.log"
REMOTE

# ── Open SSH tunnel in background, then tail the live log ────────────────────
echo ""
echo "Opening tunnel on http://127.0.0.1:7860/ ..."

ssh "${KEY_ARGS[@]}" \
    -p "$POD_PORT" \
    -o StrictHostKeyChecking=no \
    -o ExitOnForwardFailure=no \
    -L 7860:127.0.0.1:7860 \
    "root@$POD_HOST" \
    -N &
TUNNEL_PID=$!
trap "kill $TUNNEL_PID 2>/dev/null; exit" INT TERM

echo "Following server log — open http://127.0.0.1:7860/ once you see 'Running on'."
echo "Press Ctrl-C to disconnect."
echo ""

ssh "${KEY_ARGS[@]}" \
    -p "$POD_PORT" \
    -o StrictHostKeyChecking=no \
    "root@$POD_HOST" \
    "tail -f ~/firered.log"
