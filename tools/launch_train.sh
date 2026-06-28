#!/usr/bin/env bash
set -euo pipefail

RECIPE="${1:-batches/main.toml}"
OUT="${2:-runs/main}"
NPROC="${NPROC:-8}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    -m parison.pipe train --recipe "${RECIPE}" --out "${OUT}"
