#!/usr/bin/env bash
set -euo pipefail

# Local-only defaults for macOS dev. Safe because world_size=1 uses loopback.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

# If you hit hostname/interface issues on macOS, uncomment:
# export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo0}"

python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-1}" -m src.train_ddp "$@"
