#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:08:00
#SBATCH --output=result_%j.out
#SBATCH --error=result_%j.err
#SBATCH --environment=ngc_pt_jan

# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail

# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"

echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"

echo "[sbatch-master] define some env vars that will be passed to the compute nodes"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=12345   # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

echo "[sbatch-master] execute command on compute nodes"

# The command that will run on each process
CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID wrong_host=$(hostname)\"
cd /users/$USER/scratch/lscai-layer-norm
uv run --active python train.py
# run your script which we create in the next step

    # torchrun \
    #     --nnodes="${SLURM_NNODES}" \
    #     --node_rank=\$SLURM_NODEID \
    #     --nproc_per_node=1 \
    #     --master_addr="${MASTER_ADDR}" \
    #     --master_port="${MASTER_PORT}" \
    #      /users/$USER/scratch/lscai-layer-norm/train.py
"

# Submits the CMD to all the processes on all the nodes.
srun bash -c "$CMD"

echo "[sbatch-master] task finished"