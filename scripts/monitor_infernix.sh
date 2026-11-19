#!/usr/bin/env bash
# Start or stop infernix-monitor, ninferpro's status page for the server. It reads the server (its
# /metrics, /admin/expert-placement, request log and console log), switches its expert placement,
# and starts and stops it through infernix_control.sh; it can run before, during and after a
# server restart.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../configs/model.env"

PIDFILE="${PROJECT_DIR}/.infernix-monitor.pid"
LOGFILE="${PROJECT_DIR}/logs/infernix/monitor.log"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-start}" in
start)
    if running; then
        echo "The monitor is already running (PID $(cat "$PIDFILE")): http://${INFERNIX_MONITOR_HOST}:${INFERNIX_MONITOR_PORT}"
        exit 0
    fi
    [[ -x "$INFERNIX_MONITOR_BIN" ]] || { echo "ERROR: missing $INFERNIX_MONITOR_BIN (build ninferpro's infernix-monitor target)"; exit 1; }
    mkdir -p "$(dirname "$LOGFILE")"
    ARGS=(--engine "http://127.0.0.1:${INFERNIX_PORT}" --log-file "${PROJECT_DIR}/.infernix.log"
          --control "${SCRIPT_DIR}/infernix_control.sh"
          --host "$INFERNIX_MONITOR_HOST" --port "$INFERNIX_MONITOR_PORT")
    [[ -n "$INFERNIX_REQUEST_LOG" ]] && ARGS+=(--request-log "$INFERNIX_REQUEST_LOG")
    setsid nohup "$INFERNIX_MONITOR_BIN" "${ARGS[@]}" > "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 1
    if ! running; then
        echo "ERROR: the monitor exited:"; tail -5 "$LOGFILE"; rm -f "$PIDFILE"; exit 1
    fi
    echo "Monitor: http://${INFERNIX_MONITOR_HOST}:${INFERNIX_MONITOR_PORT} (PID $(cat "$PIDFILE"))"
    ;;
stop)
    if ! running; then
        echo "The monitor is not running."; rm -f "$PIDFILE"; exit 0
    fi
    kill "$(cat "$PIDFILE")"
    rm -f "$PIDFILE"
    echo "Monitor stopped."
    ;;
*)
    echo "usage: $0 start|stop"; exit 2 ;;
esac
