#!/bin/bash
set -euo pipefail

EDF_DIR="$HOME/.edf"

# ? maybe we should also choose a different name than ngc_pt_jan.toml
# ? to avoid confusion with the original environment file provided by cscs
ENVFILE="$EDF_DIR/ngc_pt_jan.toml"

mkdir -p "$EDF_DIR"

# remove if already exists
[ -f "$ENVFILE" ] && rm "$ENVFILE"

cat > "$ENVFILE" << EOF
image = "/users/$USER/scratch/my_pytorch.sqsh"

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
