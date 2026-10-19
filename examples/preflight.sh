#!/usr/bin/env bash
# Read-only configuration check. --node additionally inspects allocated GPUs/IB.
set -Eeuo pipefail
MODE=${1:---config}
[[ "$MODE" == --config || "$MODE" == --node ]] || { echo 'Usage: preflight.sh [--config|--node]' >&2; exit 2; }
if [[ -n ${TRAINING_ENV_FILE:-} ]]; then source "$TRAINING_ENV_FILE"; fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SKILL_ROOT=${TRAINING_SKILL_ROOT:-$(dirname -- "$HERE")}
: "${TRAINING_ROOT:?Set TRAINING_ROOT}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON}"
: "${TRAIN_SCRIPT:?Set TRAIN_SCRIPT}"
: "${PLAN_PATH:?Set PLAN_PATH}"
[[ -d "$TRAINING_ROOT" && -x "$TRAIN_PYTHON" && -r "$TRAIN_SCRIPT" && -r "$PLAN_PATH" ]] || { echo 'Missing code, interpreter or plan' >&2; exit 2; }
# A linked interpreter must also resolve within the mounted environment.
"$TRAIN_PYTHON" -c 'import sys; print("Python:", sys.version.split()[0])'
"$TRAIN_PYTHON" "$SKILL_ROOT/multinode-training/scripts/check_plan.py" "$PLAN_PATH"
"$TRAIN_PYTHON" - "$PLAN_PATH" <<'PYTHON'
import json, os, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
for variable, field in [('NNODES', 'nodes'), ('GPUS_PER_NODE', 'gpus_per_node')]:
    if variable in os.environ and int(os.environ[variable]) != plan[field]:
        raise SystemExit(f'{variable} conflicts with the validated plan')
PYTHON
if [[ "$MODE" == --node ]]; then
  : "${MASTER_ADDR:?Set current allocation MASTER_ADDR}"
  : "${GPUS_PER_NODE:?Set GPUS_PER_NODE}"
  ARGS=(--expect-gpus "$GPUS_PER_NODE" --coordinator-host "$MASTER_ADDR"
    --coordinator-port "${MASTER_PORT:-29500}" --dns-timeout "${DNS_TIMEOUT:-180}")
  case ${REQUIRE_IB:-false} in
    true) ARGS+=(--require-ib) ;;
    false) ;;
    *) echo 'REQUIRE_IB must be true or false' >&2; exit 2 ;;
  esac
  "$TRAIN_PYTHON" "$SKILL_ROOT/multinode-training/scripts/node_preflight.py" "${ARGS[@]}"
fi
# This does not import training modules, touch GPUs in config mode, or prove training.
echo 'Requested preflight checks passed. Native train/save/restore validation remains separate.'
