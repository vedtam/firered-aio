#!/usr/bin/env bash
# Usage:
#   ./runpod.sh <runpod_host> <tunnel_target>
#
# Example:
#   ./runpod.sh \
#     "wjf92tey1a58k8-64410d49@ssh.runpod.io" \
#     "root@157.157.221.29 -p 20665"

set -euo pipefail

SSH_HOST="${1:?Usage: $0 \"user@ssh.runpod.io\" \"user@host -p port\"}"
POD_TARGET="${2:?Usage: $0 \"user@ssh.runpod.io\" \"user@host -p port\"}"

RUNPOD_KEY="${HOME}/.ssh/id_ed25519_runpod"

if [[ ! -f "$RUNPOD_KEY" ]]; then
    echo "Error: key not found: $RUNPOD_KEY" >&2
    exit 1
fi

# ── Parse pod host / port ─────────────────────────────────────────────────────
POD_HOST=$(echo "$POD_TARGET" | awk '{print $1}')
POD_PORT=$(echo "$POD_TARGET" | sed -nE 's/.*-p[[:space:]]+([0-9]+).*/\1/p')

echo "→ Jump host  : $SSH_HOST"
echo "→ Pod        : $POD_HOST  (port $POD_PORT)"
echo "→ Key        : $RUNPOD_KEY"
echo ""

# ── Remote: clone repo, install deps and start server ────────────────────────
REPO="https://github.com/vedtam/firered-aio.git"
REPO_DIR="/workspace/firered-aio"

ssh -T \
    -i "$RUNPOD_KEY" \
    -p "$POD_PORT" \
    -o StrictHostKeyChecking=no \
    -o BatchMode=yes \
    "$POD_HOST" \
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
# echo "[runpod] Installing pre-requirements..."
# pip install -q -r pre-requirements.txt
# echo "[runpod] Installing requirements..."
# pip install -q -r requirements.txt
# echo "[runpod] Starting server (balanced profile)..."
# nohup env FIRERED_EXECUTION_PROFILE=balanced python app.py \
#     > ~/firered.log 2>&1 &
# echo "[runpod] Server PID: $!"
REMOTE

# echo ""
# echo "Waiting for server to initialise..."
# sleep 5

# ── Local: open SSH tunnel ────────────────────────────────────────────────────
# echo ""
# echo "Tunnel open — open in browser:"
# echo ""
# echo "    http://127.0.0.1:7860/"
# echo ""
# echo "Press Ctrl-C to close the tunnel."
# echo ""

# ssh -L 7860:127.0.0.1:7860 \
#     "$TUNNEL_HOST" \
#     -p "$TUNNEL_PORT" \
#     -i "$RUNPOD_KEY" \
#     -N
