#!/bin/bash


export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p logs

podman build -t my_env .
enroot import -o my_pytorch.sqsh podman://my_env
mv my_pytorch.sqsh ~/scratch/
mkdir -p ~/.edf
ln -s ~/scratch/lscai-layer-norm/ngc_pt_jan.toml ~/.edf

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies

# git clone https://github.com/karpathy/nanochat chat
# mv chat/nanochat .
# mv chat/scripts .

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
uv run python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
uv run python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!

# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
# uv run python -m scripts.tok_train --max_chars=2000000000

# ANGELO: if I run this directly it fails, it probably its compute intensive and gets killed in the cluster.
# If I add ANY custom environment, it does work with this logs.
# slurmstepd: error: couldn't chdir to `/app': No such file or directory: going to /tmp instead
# slurmstepd: error: couldn't chdir to `/app': No such file or directory: going to /tmp instead
# slurmstepd: error: pyxis: container start failed with error code: 1
# slurmstepd: error: pyxis: container exited too soon
# slurmstepd: error: pyxis: printing engine log file:
# slurmstepd: error: pyxis:     [ERROR] Command not found: ldconfig
# slurmstepd: error: pyxis:     [ERROR] /etc/enroot/hooks.d/90-aws-ofi-nccl.sh exited with return code 1
# slurmstepd: error: pyxis: couldn't start container
# slurmstepd: error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1
# slurmstepd: error: Failed to invoke spank plugin stack
# srun: error: nid006959: task 0: Exited with exit code 1
# srun: Terminating StepId=1093346.0

# But removing the thing works.
# The above is compute intensive, put it in a slurm job:
srun \
    --job-name=test \
    --account=large-sc-2 \
    --partition=normal \
    --nodes=1 \
    --cpus-per-task=50 \
    --ntasks-per-node=1 \
    --time=00:08:00 \
    --output=logs/result_%j.out \
    --error=logs/result_%j.err \
    uv run python -m scripts.tok_train --max_chars=2000000000

# evaluate the tokenizer (report compression ratio etc.)
uv run python -m scripts.tok_eval

# Should return
# Vocab sizes:
# GPT-2: 50257
# GPT-4: 100277
# Ours: 65536

# Comparison with GPT-2:
# ===============================================================================================
# Text Type  Bytes    GPT-2           Ours            Relative     Better    
#                     Tokens  Ratio   Tokens  Ratio   Diff %      
# -----------------------------------------------------------------------------------------------
# news       1819     404     4.50    375     4.85       +7.2%     Ours      
# korean     893      745     1.20    721     1.24       +3.2%     Ours      
# code       1259     576     2.19    493     2.55      +14.4%     Ours      
# math       1834     936     1.96    966     1.90       -3.2%     GPT-2     
# science    1112     260     4.28    225     4.94      +13.5%     Ours      
# fwe-train  4208518  900364  4.67    856901  4.91       +4.8%     Ours      
# fwe-val    4908443  1059062 4.63    1010356 4.86       +4.6%     Ours      

# Comparison with GPT-4:
# ===============================================================================================
# Text Type  Bytes    GPT-4           Ours            Relative     Better    
#                     Tokens  Ratio   Tokens  Ratio   Diff %      
# -----------------------------------------------------------------------------------------------
# news       1819     387     4.70    375     4.85       +3.1%     Ours      
# korean     893      364     2.45    721     1.24      -98.1%     GPT-4     
# code       1259     309     4.07    493     2.55      -59.5%     GPT-4     
# math       1834     832     2.20    966     1.90      -16.1%     GPT-4     
# science    1112     249     4.47    225     4.94       +9.6%     Ours      
# fwe-train  4208518  874799  4.81    856901  4.91       +2.0%     Ours      
# fwe-val    4908443  1029691 4.77    1010356 4.86       +1.9%     Ours 