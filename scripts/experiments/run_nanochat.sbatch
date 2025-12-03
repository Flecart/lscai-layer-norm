#!/bin/bash
#SBATCH --job-name=nanochat_train
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=08:00:00
#SBATCH --environment=ngc_pt_jan
#SBATCH --container-writable
#SBATCH --output=logs/nanochat_%j.out
#SBATCH --error=logs/nanochat_%j.err

# DIVIDE THE CONTEXT OF YOUR RUN SCRIPTS! USE BASELINE FOR BASELINE EXPERIMENT RUNS.

export TORCH_SHOW_FULL_TRACEBACK=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=1


set -eo pipefail

echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES:   $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID:   $SLURM_NODEID"

echo "[sbatch-master] setting DDP env vars"

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345         # pick a free port
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

NPROC_PER_NODE=4                 # == gpus-per-node
WANDB_RUN=speedrun               # adjust as you like

# Normalization hyperparameters
NORM_TYPE=none                  # choose from: layernorm, rms, learnable_rms, none
QK_NORM_TYPE=none               # choose from: layernorm, rms, learnable_rms, none


echo "[sbatch-master] launching srun/torchrun"
CMD='
echo "[srun] rank=$SLURM_PROCID host=$(hostname) node_rank=$SLURM_NODEID local_rank=$SLURM_LOCALID"

# 1. Explicitly CD to the correct directory inside the srun task
echo "[srun] Changing to work directory"
cd "$HOME/scratch/lscai-layer-norm"

# 2. Activate the virtual environment *inside this shell*
echo "[srun] Activating .venv"
source .venv/bin/activate
echo "[srun] Python executable: $(which python)"
# ----------------------------------------------------

# 3. source wandb env vars
if [[ -f ".wandb.env" ]]; then
    echo "[srun] Sourcing .wandb.env"
    source .wandb.env
else
    echo "[srun] .wandb.env not found, proceeding without it and setting WANDB_RUN to empty"
    export WANDB_RUN=""
fi

# 4 verify if WANDB_API_KEY is set
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[srun] WARNING: WANDB_API_KEY is not set. WandB logging will be disabled."
    export WANDB_DISABLED=true
fi

cd karpathy
python3 -m torch.distributed.run \
  --nnodes='"$SLURM_NNODES"' \
  --node_rank=$SLURM_NODEID \
  --nproc_per_node='"$NPROC_PER_NODE"' \
  --master_addr='"$MASTER_ADDR"' \
  --master_port='"$MASTER_PORT"' \
  -m scripts.base_train -- \
    --depth=20 \
    --run='"$WANDB_RUN"' \
    --norm_type='"$NORM_TYPE"' \
    --qk_norm_type='"$QK_NORM_TYPE"'
'

srun bash -c "$CMD"

echo "[sbatch-master] task finished"