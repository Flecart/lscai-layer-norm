#!/bin/bash
set -euo pipefail

cd "$HOME/scratch/lscai-layer-norm/nematus"

echo "[init_venv_nematus] running on $(hostname)"

# 0. Check TF & CUDA in the container environment
echo "[init_venv_nematus] Checking TensorFlow and GPU..."
python - << 'EOF'
import tensorflow as tf
from tensorflow.python.client import device_lib

print("TF version:", tf.__version__)

devices = device_lib.list_local_devices()
gpus = [d for d in devices if d.device_type == "GPU"]
print("GPUs found:", [g.name for g in gpus])
EOF

# 1. Remove existing venv if present
if [ -d ".venv" ]; then
    echo "[init_venv_nematus] Removing existing .venv..."
    rm -rf .venv
fi

# 2. Recreate venv with system site-packages (to reuse TF from the container)
echo "[init_venv_nematus] Creating .venv (system-site-packages)..."
uv venv --system-site-packages --python "$(which python)"

# 3. Activate venv
source .venv/bin/activate
echo "[init_venv_nematus] Python in venv: $(which python)"

# 4. Install Nematus and Python deps
# Nematus repo is the current directory (has setup.py)
echo "[init_venv_nematus] Installing Nematus (editable)..."
uv pip install --no-deps -e .

# Recommended extras (similar to your Nanochat set-up)
echo "[init_venv_nematus] Installing extra Python utilities..."
uv pip install --no-deps \
    subword-nmt \
    sacrebleu \
    pandas \
    jupyter \
    tqdm \
    matplotlib

# (Optional) if you want mosesdecoder without cloning manually:
#   git clone https://github.com/moses-smt/mosesdecoder ../mosesdecoder
# But typically moses is just called as a binary from scripts.

# 5. Final sanity check: can we import TF?
python - << 'EOF'
import tensorflow as tf
print("TF in venv:", tf.__version__)
EOF


echo "[init_venv_nematus] Done."
