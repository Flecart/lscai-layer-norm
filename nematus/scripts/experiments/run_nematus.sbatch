#!/bin/bash
#SBATCH --job-name=nematus_wmt17
#SBATCH --account=large-sc-2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --environment=ngc_tf1_jan
#SBATCH --container-writable
#SBATCH --output=logs/nematus_wmt17_%j.out
#SBATCH --error=logs/nematus_wmt17_%j.err

set -euo pipefail

echo "[nematus-sbatch] running on $(hostname)"
echo "[nematus-sbatch] SLURM_NODELIST: $SLURM_NODELIST"
echo "[nematus-sbatch] SLURM_NTASKS:   $SLURM_NTASKS"

# Where your WMT17 scripts live (the dir with download_files.sh / preprocess.sh / validate.sh)
SCRIPTS_DIR="$HOME/scratch/lscai-layer-norm/nematus/scripts/wmt17"

# Launch one shell per task; only rank 0 will actually train
srun bash -lc '
  echo "[nematus-sbatch] task: SLURM_PROCID=$SLURM_PROCID on $(hostname)"

  echo "[nematus-sbatch] rank 0: starting training"

  cd "'"$SCRIPTS_DIR"'"
  echo "[nematus-sbatch] In SCRIPTS_DIR: $(pwd)"

  # Recreate variables from train.sh
  script_dir=$(pwd)
  main_dir="$script_dir/../.."      # contains data/, model/, vars
  data_dir="$main_dir/data"
  working_dir="$main_dir/model"

  echo "[nematus-sbatch] main_dir:    $main_dir"
  echo "[nematus-sbatch] data_dir:    $data_dir"
  echo "[nematus-sbatch] working_dir: $working_dir"

  # Activate Nematus venv
  source "$HOME/scratch/lscai-layer-norm/nematus/.venv/bin/activate"
  echo "[nematus-sbatch] Python: $(which python)"

  # Make sure Python can see both the nematus package and this training tree
  export PYTHONPATH="$HOME/scratch/lscai-layer-norm/nematus:$main_dir:$PYTHONPATH"

  # language-independent variables (toolkit locations)
  # e.g. nematus_home, moses_scripts, subword_nmt_dir, device
  . "$main_dir/vars"

  # Fallback for nematus_home if not defined in vars
  nematus_home="$HOME/scratch/lscai-layer-norm/nematus"

  echo "[nematus-sbatch] nematus_home: $nematus_home"
  echo "[nematus-sbatch] src=$src trg=$trg"
  echo "[nematus-sbatch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  # This is literally what their train.sh does:
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "$nematus_home/nematus/train.py" \
      --model "$working_dir/model" \
      --datasets "$data_dir/corpus.bpe.$src" "$data_dir/corpus.bpe.$trg" \
      --valid_datasets "$data_dir/newstest2013.bpe.$src" "$data_dir/newstest2013.bpe.$trg" \
      --dictionaries "$data_dir/corpus.bpe.$src.json" "$data_dir/corpus.bpe.$trg.json" \
      --valid_script "$script_dir/validate.sh" \
      --reload latest_checkpoint \
      --dim_word 512 \
      --dim 1024 \
      --lrate 0.0005 \
      --optimizer adam \
      --maxlen 50 \
      --batch_size 128 \
      --valid_batch_size 128 \
      --token_batch_size 100000 \
      --valid_token_batch_size 60000 \
      --validFreq 10000 \
      --dispFreq 1000 \
      --saveFreq 30000 \
      --sampleFreq 10000 \
      --tie_decoder_embeddings \
      --layer_normalisation \
      --dec_base_recurrence_transition_depth 8 \
      --enc_recurrence_transition_depth 4
'

echo "[nematus-sbatch] job finished"
