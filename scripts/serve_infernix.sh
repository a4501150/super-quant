#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

PIDFILE="${PROJECT_DIR}/.infernix.pid"
LOGFILE="${PROJECT_DIR}/.infernix.log"
PORT="${INFERNIX_PORT}"
SERVED_NAME="${MODEL_ALIAS:-${MODEL_NAME:-}}"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Infernix is already running (PID $(cat "$PIDFILE")). Run 'make stop-infernix' first."
    exit 1
fi
rm -f "$PIDFILE"
if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
    echo "ERROR: port ${PORT} is already in use (SGLang? run 'make stop')."
    exit 1
fi
for f in "$INFERNIX_BIN" "$INFERNIX_ARTIFACT" "$INFERNIX_NGRAM_VOLUME" "$INFERNIX_DRAFT_MODEL"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

ARGS=(
    "$INFERNIX_ARTIFACT"
    --ngram-volume "$INFERNIX_NGRAM_VOLUME"
    --kv-dtype "$INFERNIX_KV_DTYPE"
    --max-context "$INFERNIX_MAX_CONTEXT"
    --max-concurrency "$INFERNIX_MAX_CONCURRENCY"
    --spec dflash2 --draft-model "$INFERNIX_DRAFT_MODEL" --draft-tokens auto --lm-head-draft
    --tolerant-tool-calls
    --host "$INFERNIX_HOST" --port "$PORT"
)
[[ -n "$INFERNIX_HOST_CONTEXT_MIB" ]] && ARGS+=(--host-context-mib "$INFERNIX_HOST_CONTEXT_MIB")
[[ -n "$INFERNIX_PREFIX_CACHE_DISK" ]] && ARGS+=(--prefix-cache-disk "$INFERNIX_PREFIX_CACHE_DISK" --prefix-cache-disk-gib "$INFERNIX_PREFIX_CACHE_DISK_GIB")
[[ -n "$SERVED_NAME" ]] && ARGS+=(--served-model-name "$SERVED_NAME")
[[ -n "$INFERNIX_EXPERTS" ]] && ARGS+=(--experts "$INFERNIX_EXPERTS")
[[ "${INFERNIX_VISION}" == "true" ]] && ARGS+=(--vision)
# shellcheck disable=SC2206
[[ -n "$INFERNIX_EXTRA_ARGS" ]] && ARGS+=($INFERNIX_EXTRA_ARGS)

echo "=== Infernix ==="
echo "Binary:   $INFERNIX_BIN ($(git -C "$(dirname "$INFERNIX_BIN")/../.." rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Artifact: $INFERNIX_ARTIFACT"
echo "Port:     $PORT"
echo "Log:      $LOGFILE"

# setsid + nohup: the server survives the shell or harness that started it.
cd "$INFERNIX_MODEL_DIR"
setsid nohup "$INFERNIX_BIN" "${ARGS[@]}" > "$LOGFILE" 2>&1 < /dev/null &
SERVER_PID=$!
echo "$SERVER_PID" > "$PIDFILE"
echo "Server PID: $SERVER_PID"

for i in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server exited prematurely. Check $LOGFILE"
        tail -30 "$LOGFILE"
        rm -f "$PIDFILE"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "=== Infernix Ready after ${i}s ==="
        grep -E 'expert cache|VRAM:|capacity|fast paths' "$LOGFILE" | tail -4 || true
        echo "API: http://localhost:${PORT}/v1/chat/completions (model: ${SERVED_NAME:-see /v1/models})"
        exit 0
    fi
    sleep 1
done
echo "ERROR: Infernix did not become ready within 300s. Check $LOGFILE"
tail -30 "$LOGFILE"
exit 1
