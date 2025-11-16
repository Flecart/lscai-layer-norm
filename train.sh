#!/bin/bash
#SBATCH --job-name=LSAIE_a4
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:08:00
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/result_%j.err
#SBATCH --environment=/capstor/store/cscs/ethz/large-sc-2/environment/ngc_pt_jan.toml


export OMP_NUM_THREADS=1
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

# Submits the CMD to all the processes on all the nodes.
echo "task started"
NPROC_PER_NODE=4
WANDB_RUN=speedrun

CMD="
# print current environment variables
echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID wrong_host=$(hostname)\"

# Change to project directory
cd /users/\$USER/project/lscai-layer-norm || cd /users/\$USER/scratch/lscai-layer-norm

echo \"Checking CUDA availability with container Python...\"
python3 -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'PyTorch version: {torch.__version__}')\" || echo \"Failed to check CUDA\"

# Add project to PYTHONPATH and use container's python3 directly
export PYTHONPATH=/users/\$USER/project/lscai-layer-norm:/users/\$USER/scratch/lscai-layer-norm:\$PYTHONPATH
python3 -m torch.distributed.run \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m scripts.base_train -- --depth=20 --run=$WANDB_RUN --device_type=cuda
"

srun bash -c "$CMD"

echo "task finished"