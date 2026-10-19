#!/usr/bin/env bash
# Platform must run this ONCE PER NODE, not once per GPU.
# Usage: TRAINING_ENV_FILE=/private/training.env bash torchrun_multinode.sh [trainer arguments...]
set -Eeuo pipefail
if [[ -n ${TRAINING_ENV_FILE:-} ]]; then source "$TRAINING_ENV_FILE"; fi
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${NNODES:?Set node count}"
: "${GPUS_PER_NODE:?Set GPUs per node}"
: "${NODE_RANK:?Use current platform node rank}"
: "${MASTER_ADDR:?Use current platform coordinator host}"
MASTER_PORT=${MASTER_PORT:-29500}
for variable in NNODES GPUS_PER_NODE MASTER_PORT; do
  [[ ${!variable} =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $variable" >&2; exit 2; }
done
[[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] && (( NODE_RANK < NNODES && MASTER_PORT <= 65535 )) || { echo 'Invalid rank/port' >&2; exit 2; }
[[ -z ${LOCAL_RANK+x} ]] || { echo 'Nested per-GPU launch detected; use a per-node platform command' >&2; exit 2; }
if (( NNODES > 1 )); then
  case "$MASTER_ADDR" in localhost|127.*|::1) echo 'Multi-node coordinator cannot be loopback' >&2; exit 2 ;; esac
fi
export PYTHONUNBUFFERED=1
# Debug INFO is useful for diagnosis but can produce multi-GB logs during training.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Inherit verified NIC/HCA settings. Do not hardcode bond0 or mlx5 names here.
# No automatic network fallback, checkpoint deletion, or job resubmission.
export TRAINING_ROOT TRAIN_PYTHON TRAIN_SCRIPT PLAN_PATH NNODES GPUS_PER_NODE MASTER_ADDR MASTER_PORT
bash "$HERE/preflight.sh" --node
cd -- "$TRAINING_ROOT"
exec "$TRAIN_PYTHON" -u -m torch.distributed.run \
  --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  "$TRAIN_SCRIPT" "$@"
