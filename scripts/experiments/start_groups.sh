#! /bin/bash

# Just a script to start all the different normalization experiments with one command

sbatch scripts/experiments/rms_column.sbatch
sbatch scripts/experiments/rms_full.sbatch
sbatch scripts/experiments/rms_row.sbatch
sbatch scripts/experiments/rms_patched.sbatch
sbatch scripts/experiments/baseline.sbatch