#!/bin/bash
#SBATCH --job-name=nanochat_nembd_ablation
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=70
#SBATCH --time=10:00:00
#SBATCH --environment=ngc_pt_jan
#SBATCH --container-writable
#SBATCH --output=logs/nembd_abl_%A_%a.out
#SBATCH --error=logs/nembd_abl_%A_%a.err
#SBATCH --array=0-5

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
DEPTH=20                           # fix depth
NEMBDS=(512 768 1024)
NORMS=("rms" "learnable_rms")

ID=${SLURM_ARRAY_TASK_ID}
EMB_IDX=$(( ID / 2 ))
NORM_IDX=$(( ID % 2 ))

N_EMBD=${NEMBDS[$EMB_IDX]}
NORM_TYPE=${NORMS[$NORM_IDX]}

USE_MUON=false

echo "[sbatch-master] depth=${DEPTH}, n_embd=${N_EMBD}, norm_type=${NORM_TYPE}, use_muon=${USE_MUON}"

NPROC_PER_NODE=4

# Unique WandB name
WANDB_RUN="d${DEPTH}_e${N_EMBD}_${NORM_TYPE}"
WANDB_NOTES="depth ${DEPTH}, n_embd ${N_EMBD}, norm ${NORM_TYPE}, use_muon ${USE_MUON}, mlp_type default, qk_norm_type rms, pre_attn_norm_type rms"

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
    echo "[srun] .wandb.env not found, disabling WandB"
    export WANDB_RUN=""
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
    --n_embd='"$N_EMBD"' \
    --run='"$WANDB_RUN"' \
    --notes="'"$WANDB_NOTES"'" \
    --norm_type='"$NORM_TYPE"' \
    --qk_norm_type=rms \
    --pre_attn_norm_type=rms \
    --norm_eps=1e-6 \
    --mlp_type=default \
    --num_iterations=20000 \
    --use_muon='"$USE_MUON"'
'

srun bash -c "$CMD"
echo "[sbatch-master] task finished"
