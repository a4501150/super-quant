#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

PIDFILE="${PROJECT_DIR}/.vllm.pid"
STOP_TIMEOUT="${VLLM_STOP_TIMEOUT:-360}"

if [[ ! -f "${PIDFILE}" ]]; then
    echo "No vLLM PID file found. Server may not be running."
    exit 0
fi

PID="$(<"${PIDFILE}")"

if ! kill -0 "${PID}" 2>/dev/null; then
    echo "vLLM process ${PID} is not running. Cleaning up PID file."
    rm -f "${PIDFILE}"
    exit 0
fi

if [[ -r "/proc/${PID}/cmdline" ]] &&
    ! tr '\0' ' ' < "/proc/${PID}/cmdline" | grep -q "vllm.entrypoints.openai.api_server"; then
    echo "ERROR: PID ${PID} does not belong to the vLLM API server. Removing stale PID file." >&2
    rm -f "${PIDFILE}"
    exit 1
fi

echo "Stopping vLLM server (PID ${PID})..."
kill "${PID}"

for _ in $(seq 1 "${STOP_TIMEOUT}"); do
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "vLLM stopped."
        rm -f "${PIDFILE}"
        exit 0
    fi
    sleep 1
done

echo "vLLM did not stop within ${STOP_TIMEOUT}s; force killing PID ${PID}..."
kill -9 "${PID}" 2>/dev/null || true
rm -f "${PIDFILE}"
echo "vLLM stopped."
