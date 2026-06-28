#!/usr/bin/env bash
set -euo pipefail

RECIPE="${1:-batches/main.toml}"
CKPT="${2:-runs/main/best.ckpt}"

python -m parison.pipe evaluate --recipe "${RECIPE}" --ckpt "${CKPT}"
