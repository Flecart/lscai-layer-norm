#!/bin/bash
#SBATCH --job-name=init_venv
#SBATCH --account=large-sc-2
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --container-writable
#SBATCH --output=logs/init_venv_%j.out
#SBATCH --error=logs/init_venv_%j.err

set -euo pipefail

echo "[init_venv.sbatch] running on $(hostname)"

cd "$HOME/scratch/lscai-layer-norm"

# run the .init_venv.sh script inside the sbatch job
srun --ntasks=1 --environment=ngc_pt_jan bash init_venv.sh