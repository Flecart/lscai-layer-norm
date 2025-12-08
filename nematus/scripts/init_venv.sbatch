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

cd "$HOME/scratch/lscai-layer-norm/nematus"

# run the .init_venv.sh script inside the sbatch job
srun --ntasks=1 --environment=ngc_tf1_jan bash scripts/init_venv_nematus.sh