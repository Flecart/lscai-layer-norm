#!/bin/bash
set -euo pipefail

cd "$HOME/scratch/lscai-layer-norm"

echo "[init_venv] running on $(hostname)"

# 0. Check that CUDA is available in this environment
echo "[init_venv] Checking CUDA availability..."
python -c "import torch; print('Initial CUDA available:', torch.cuda.is_available())"

# 1. Remove existing venv if present
if [ -d ".venv" ]; then
    echo "[init_venv] Removing existing .venv..."
    rm -rf .venv
fi

# 2. Recreate venv with system-site-packages
echo "[init_venv] Creating .venv (system-site-packages)..."

# use python interpreter of the container
uv venv --system-site-packages --python "$(which python)"

# 3. Install Python dependencies according to pyproject/uv.lock
echo "[init_venv] Running uv sync..."
uv sync

# activate the venv
source .venv/bin/activate

# ? probably we should consider adding these dependencies to pyproject.toml
# ? pip installing accellerate and deepspeed here seems to pull in incompatible torch versions
# 4) install everything EXCEPT the packages that pull in torch, with a safe numpy
uv pip install \
    datasets \
    transformers \
    wandb \
    dacite \
    pyyaml \
    packaging \
    safetensors \
    tqdm \
    sentencepiece \
    tensorboard \
    pandas \
    jupyter \
    seaborn

# 5 install the torch-hungry ones without dependencies, so they don’t drag in torch
uv pip install --no-deps accelerate deepspeed torchvision


# Check again that cuda is available
echo "[init_venv] Final CUDA availability check..."
python -c "import torch; print('Final CUDA available:', torch.cuda.is_available())"


echo "[init_venv] Done. .venv and rustbpe are ready."
