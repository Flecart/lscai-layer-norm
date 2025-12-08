#!/bin/bash
#SBATCH --job-name=init_venv_nematus
#SBATCH --account=large-sc-2
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --container-writable
#SBATCH --output=logs/init_venv_nematus_%j.out
#SBATCH --error=logs/init_venv_nematus_%j.err

set -euo pipefail

echo "[init_venv_nematus.sbatch] running on $(hostname)"

PROJECT_DIR="$HOME/scratch/lscai-layer-norm/nematus"
SCRIPT_PATH="$PROJECT_DIR/scripts/init_venv_nematus.sh"

echo "[init_venv_nematus.sbatch] PROJECT_DIR:  $PROJECT_DIR"
echo "[init_venv_nematus.sbatch] SCRIPT_PATH:  $SCRIPT_PATH"

# Just to be sure:
ls -l "$SCRIPT_PATH" || { echo "Script not found on host"; exit 1; }

srun \
  --ntasks=1 \
  --environment=ngc_tf2_jan \
  bash "$SCRIPT_PATH"
