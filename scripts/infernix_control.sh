#!/usr/bin/env bash
# The control command of infernix-monitor's Start and Stop (its --control; ninferpro
# tools/monitor/README.md). `start [EXPERTS]` runs serve_infernix.sh with the page's shares as
# INFERNIX_EXPERTS (without them, model.env's); `stop` runs stop_infernix.sh. MODEL_DIR comes
# from the monitor's environment, so the page starts the model the monitor was started for.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
start)
    [[ -n "${2:-}" ]] && export INFERNIX_EXPERTS="$2"
    exec bash "${SCRIPT_DIR}/serve_infernix.sh"
    ;;
stop)
    exec bash "${SCRIPT_DIR}/stop_infernix.sh"
    ;;
*)
    echo "usage: $0 start [EXPERTS] | stop" >&2
    exit 2
    ;;
esac
