#!/bin/bash
#SBATCH --job-name=build_img_nematus
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --container-writable
#SBATCH --output=logs/build_nematus_img_%j.out
#SBATCH --error=logs/build_nematus_img_%j.err

cd "$HOME/scratch/lscai-layer-norm/nematus/scripts"

./build_image_nematus.sh