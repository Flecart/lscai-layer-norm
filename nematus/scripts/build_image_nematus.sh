#!/bin/bash
set -euo pipefail

cd "$HOME/scratch/lscai-layer-norm/nematus"

echo "[nematus] Building TensorFlow 2.x image"

# Build podman image from Dockerfile.nematus
podman build -t nematus_tf2 -f Dockerfile.gpu .

echo "[nematus] Exporting to nematus_tf2.sqsh..."
SQSH="$HOME/scratch/nematus_tf2.sqsh"

# enroot import
set +e
enroot import -o nematus_tf2.sqsh podman://nematus_tf2
st=$?
set -e

[[ -f nematus_tf2.sqsh ]] || exit $st
echo "[nematus] Moving nematus_tf2.sqsh to $SQSH"
mv nematus_tf2.sqsh "$SQSH"

echo "[nematus] Listing $SQSH:"
ls -lh "$SQSH"

echo "[nematus] Done building and exporting image."
