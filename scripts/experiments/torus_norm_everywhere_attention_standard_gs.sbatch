#!/bin/bash
#SBATCH --job-name=nanochat_torus_group_size_sweep_attention_standard
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --time=10:00:00
#SBATCH --environment=ngc_pt_jan
#SBATCH --container-writable
#SBATCH --output=logs/torus_group_size_%A_%a.out
#SBATCH --error=logs/torus_group_size_%A_%a.err
#SBATCH --array=0-6

export TORCH_SHOW_FULL_TRACEBACK=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=1

set -eo pipefail

echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES:   $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID:   $SLURM_NODEID"
echo "[sbatch-master] SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# -------------------- CONFIG GRID --------------------


# choosing numbers that divide both n_embed and n_embed // n_head (n_embed=1280, n_head=10)
GROUP_SIZES=(2 4 8 16 32 64 128)   # <- sweep these
ID=${SLURM_ARRAY_TASK_ID}
GROUP_SIZE_TORUS_NORM=${GROUP_SIZES[$ID]}

# Fixed experiment settings (same as your single-run script)
NPROC_PER_NODE=4
WANDB_RUN_PREFIX="torus-norm-attention-default-gs"
WANDB_NOTES_PREFIX="Torus normalization everywhere + default attention; sweep group_size_torus_norm"

NORM_TYPE=torus
NORM_EPS=1e-6
MLP_TYPE=default
EMBED_NORM_TYPE=torus
FINAL_NORM_TYPE=torus
ATTENTION_STRATEGY=default
QK_NORM_TYPE=torus
PRE_ATTN_NORM_TYPE=torus
USE_MUON=false
DEPTH=20

WANDB_RUN="${WANDB_RUN_PREFIX}${GROUP_SIZE_TORUS_NORM}"
WANDB_NOTES="${WANDB_NOTES_PREFIX}, group_size_torus_norm ${GROUP_SIZE_TORUS_NORM}"

echo "[sbatch-master] depth=${DEPTH}, group_size_torus_norm=${GROUP_SIZE_TORUS_NORM}, attention_strategy=${ATTENTION_STRATEGY}, use_muon=${USE_MUON}"

# -------------------- DDP ENV --------------------
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=12345
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "[sbatch-master] launching srun/torchrun"

CMD='
echo "[srun] rank=$SLURM_PROCID host=$(hostname) node_rank=$SLURM_NODEID local_rank=$SLURM_LOCALID"

cd "$HOME/scratch/lscai-layer-norm"

source .venv/bin/activate
echo "[srun] Python executable: $(which python)"

if [[ -f ".wandb.env" ]]; then
    echo "[srun] Sourcing .wandb.env"
    source .wandb.env
else
    echo "[srun] .wandb.env not found, proceeding without it and setting WANDB_RUN to empty"
    export WANDB_RUN=""
fi

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
    --depth='"$DEPTH"' \
    --run='"$WANDB_RUN"' \
    --notes="'"$WANDB_NOTES"'" \
    --norm_type='"$NORM_TYPE"' \
    --qk_norm_type='"$QK_NORM_TYPE"' \
    --pre_attn_norm_type='"$PRE_ATTN_NORM_TYPE"' \
    --embed_norm_type='"$EMBED_NORM_TYPE"' \
    --final_norm_type='"$FINAL_NORM_TYPE"' \
    --norm_eps='"$NORM_EPS"' \
    --mlp_type='"$MLP_TYPE"' \
    --use_muon='"$USE_MUON"' \
    --attention_strategy='"$ATTENTION_STRATEGY"' \
    --group_size_torus_norm='"$GROUP_SIZE_TORUS_NORM"'
'

srun bash -c "$CMD"
echo "[sbatch-master] task finished"
