#!/usr/bin/env bash
# Read-only, bounded-tail observation; never stops/restarts a trainer.
set -Eeuo pipefail
MODE=${1:---once}
[[ "$MODE" == --once || "$MODE" == --follow ]] || { echo 'Usage: watch_logs.sh [--once|--follow] LOG...' >&2; exit 2; }
shift
(( $# > 0 )) || { echo 'Provide one or more node log paths' >&2; exit 2; }
INTERVAL_SECONDS=${INTERVAL_SECONDS:-30}
[[ "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid interval' >&2; exit 2; }
if [[ "$MODE" == --follow ]]; then
  : "${MONITOR_LOCK:?Set a unique lock file path for this run}"
  exec 9>"$MONITOR_LOCK"
  flock -n 9 || { echo 'Monitor for this lock is already running' >&2; exit 2; }
fi
while true; do
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  for logfile in "$@"; do
    printf 'Log: %s\n' "$logfile"
    if [[ ! -f "$logfile" ]]; then echo 'Waiting for log creation'; continue; fi
    # Bound I/O even when NCCL logs grow to GB. A tail may miss older metric rows.
    tail -c 262144 -- "$logfile" | { grep -E 'Step [0-9]+:|Traceback|Error:|RESOURCE_EXHAUSTED|OutOfMemory|Progress on:' || true; } | tail -n 12
  done
  [[ "$MODE" == --follow ]] || break
  sleep "$INTERVAL_SECONDS"
done
