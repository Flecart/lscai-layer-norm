#!/bin/bash
set -euo pipefail

EDF_DIR="$HOME/.edf"

ENVFILE="$EDF_DIR/ngc_tf1_jan.toml"

mkdir -p "$EDF_DIR"

# remove if already exists
[ -f "$ENVFILE" ] && rm "$ENVFILE"

cat > "$ENVFILE" << EOF
image = "/users/$USER/scratch/nematus_tf1.sqsh"

mounts = [
  "/capstor",
  "/iopsstor",
  "/users",
]

writable = true

workdir = "/users/$USER/scratch/lscai-layer-norm"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
EOF

echo "[envfile] Created $ENVFILE"
