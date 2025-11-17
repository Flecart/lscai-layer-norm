#!/bin/bash
#SBATCH --job-name=nanochat_tok
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --time=00:30:00
#SBATCH --environment=ngc_pt_jan
#SBATCH --container-writable
#SBATCH --output=logs/tok_%j.out
#SBATCH --error=logs/tok_%j.err

export OMP_NUM_THREADS=1
set -euo pipefail

cd "$HOME/scratch/lscai-layer-norm"


# 1) Download a small number of shards (blocking)
uv run python3 -m nanochat.dataset -n 8

# 2) Kick off a larger download in the background
uv run python3 -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!

# 3) Train the tokenizer in this allocation (heavy step)
uv run python3 -m scripts.tok_train --max_chars=2000000000

# 4) Evaluate the tokenizer
uv run python3 -m scripts.tok_eval


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