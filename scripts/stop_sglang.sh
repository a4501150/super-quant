#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

PIDFILE="${PROJECT_DIR}/.sglang.pid"
STOP_TIMEOUT="${SGLANG_STOP_TIMEOUT:-360}"

if [[ ! -f "$PIDFILE" ]]; then
    echo "No SGLang PID file found. Server may not be running."
    exit 0
fi

PID=$(cat "$PIDFILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "SGLang process $PID is not running. Cleaning up PID file."
    rm -f "$PIDFILE"
    exit 0
fi

echo "Stopping SGLang server (PID $PID)..."
kill "$PID"

for i in $(seq 1 "${STOP_TIMEOUT}"); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "SGLang stopped."
        rm -f "$PIDFILE"
        exit 0
    fi
    sleep 1
done

echo "SGLang did not stop within ${STOP_TIMEOUT}s; force killing PID $PID..."
kill -9 "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "SGLang stopped."
