#! /bin/bash

# Just a script to start all the different normalization experiments with one command

sbatch scripts/experiments/baseline_layer_norm.sbatch
sbatch scripts/experiments/baseline_no_learnable_norm.sbatch

sbatch scripts/experiments/rms_patched.sbatch
sbatch scripts/experiments/run_nanochat.sbatch
sbatch scripts/experiments/torus_norm_everywhere.sbatch
sbatch scripts/experiments/torus_norm_pre_mlp.sbatch