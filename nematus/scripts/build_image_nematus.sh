#!/bin/bash
set -euo pipefail

cd "$HOME/scratch/lscai-layer-norm/nematus"

echo "[nematus] Building TensorFlow 1.x image"

# Build podman image from Dockerfile.nematus
podman build -t nematus_tf1 -f Dockerfile.gpu .

echo "[nematus] Exporting to nematus_tf1.sqsh..."
SQSH="$HOME/scratch/nematus_tf1.sqsh"

# enroot import
set +e
enroot import -o nematus_tf1.sqsh podman://nematus_tf1
st=$?
set -e

[[ -f nematus_tf1.sqsh ]] || exit $st

echo "[nematus] Moving nematus_tf1.sqsh to $SQSH"
mv nematus_tf1.sqsh "$SQSH"

echo "[nematus] Listing $SQSH:"
ls -lh "$SQSH"

echo "[nematus] Done building and exporting image."
