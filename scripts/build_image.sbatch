#!/bin/bash
#SBATCH --job-name=build_img
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --container-writable
#SBATCH --output=logs/build_img_%j.out
#SBATCH --error=logs/build_img_%j.err

./build_image.sh