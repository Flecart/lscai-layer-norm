# Normalization Layers Are Not What You Need

Research project investigating normalization strategies and MLP fusion techniques in transformer models for AI training, done as part of the Large-Scale AI Engineering course in Fall 2025. Built on Karpathy's nanochat framework, this repository contains experimentation with different normalization approaches including RMSNorm variants and LayerNorm.

## Quick Start

### Bootstrap environment

Set up the training environment with PyTorch container and virtual environment:

```bash
./bootstrap.sh
```

This script:
- Creates a virtual environment in `.venv` matching the container's PyTorch installation
- Installs dependencies outside the Docker image but inside the container
- Follows [CSCS documentation guidelines](https://docs.cscs.ch/software/ml/pytorch/#optionally-extend-container-with-virtual-environment)

### Interactive Development

Get a compute node with the configured environment:

```bash
srun --account=large-sc-2 -p debug --time=60:00 --environment=ngc_pt_jan --pty bash
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Verify CUDA availability:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Configuration

#### WandB Setup
Create a `.wandb.env` file for experiment tracking:

```bash
cp wandb.env.example .wandb.env
# Edit .wandb.env with your WandB API key
```

## Running Experiments

### Test the Tokenizer
```bash
sbatch run_tok_train.sbatch
```

### Train Base Model - nanochat

Submit a training job using one of the experiment configurations:

```bash
# Baselines
sbatch scripts/experiments/baseline_layer_norm.sbatch        # Standard LayerNorm
sbatch scripts/experiments/baseline_no_learnable_norm.sbatch # Parameter-free RMSNorm

# MLP Fusion Strategies
sbatch scripts/experiments/rms_column.sbatch   # Column-wise scaling (input dimension)
sbatch scripts/experiments/rms_row.sbatch      # Row-wise scaling (output dimension)
sbatch scripts/experiments/rms_full.sbatch     # Full bidirectional scaling
sbatch scripts/experiments/rms_patched.sbatch  # Block-wise/patch scaling

# TorusNorm Experiments
sbatch scripts/experiments/torus_norm_everywhere.sbatch  # TorusNorm at all positions
sbatch scripts/experiments/torus_norm_pre_mlp.sbatch     # TorusNorm only before MLPs
```

### View Logs

All training logs are saved to `/logs/` with job-specific filenames containing the SLURM job ID.


## Nematus (Alternative Backend)

Bootstrap Nematus environment (takes ~30 minutes):

```bash
cd nematus
./scripts/bootstrap_nematus.sh
```

Train with Nematus:

```bash
cd nematus
sbatch scripts/experiments/run_nematus.sbatch
```

Logs are saved to `nematus/logs/`.
