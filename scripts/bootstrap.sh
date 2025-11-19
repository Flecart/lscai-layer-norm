#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/scratch/lscai-layer-norm"

echo "[bootstrap] Starting nanochat bootstrap..."
echo "[bootstrap] Project dir: $PROJECT_DIR"

cd "$PROJECT_DIR"

##########################################################
# Step 0 – Clean uv.lock (optional)
##########################################################

# the old uv.lock might have incompatible packages, better to refresh it
# also we don't want any torch version pinned in uv.lock (as it is in the official nanochat repo)

if command -v uv &> /dev/null; then
    echo "[bootstrap] Detected uv → refreshing uv.lock"
    rm -f uv.lock
    uv lock || echo "[bootstrap] Warning: uv lock failed (not fatal)"
else
    echo "[bootstrap] uv not available, skipping uv.lock cleanup"
fi

##########################################################
# Step 1 – Ensure logs directory exists
##########################################################

mkdir -p logs
echo "[bootstrap] Created logs/"

##########################################################
# Step 2 – Submit build job for image → .sqsh
##########################################################

echo "[bootstrap] Submitting image build job..."
echo "[bootstrap] Check logs/build_img_*.out for progress (tail -f logs/build_img_<jobid>.out)"

srun \
    --job-name=build_img \
    --account=large-sc-2 \
    --partition=normal \
    --nodes=1 \
    --cpus-per-task=8 \
    --ntasks-per-node=1 \
    --time=00:15:00 \
    --container-writable \
    --output=logs/build_img_%j.out \
    --error=logs/build_img_%j.err \
    bash build_image.sh


echo "[bootstrap] Build job completed."

BUILD_JOB=$(squeue --me --name=build_img --noheader --format="%i")
echo "[bootstrap] Build job ID: $BUILD_JOB"

##########################################################
# Step 3 – Call create_envfile.sh
##########################################################

# this creates the .edf/ngc_pt_jan.toml environment file and uses
# the personal $USER to point to the .sqsh image created in step 2

if [[ ! -x "create_envfile.sh" ]]; then
    echo "[bootstrap] ERROR: create_envfile.sh not found or not executable"
    exit 1
fi

echo "[bootstrap] Creating EDF environment file via create_envfile.sh..."
./create_envfile.sh


##########################################################
# Step 4 – Create virtual environment via init_venv.sh
##########################################################

# I thought it would be better to create the virtual environment
# after the image is built, we run the script inside the container and we first
# initialize it with the system-site-packages, so that cuda is then automatically detected.
# thanks to using the correct pytorch version from the container.
# the we uv sync to install the rest of the packages.
# This seems to be the preffered approach for hpc, as described in the CSCS docs:
# https://docs.cscs.ch/software/ml/pytorch/#optionally-extend-container-with-virtual-environment

# we have the advantage that we don't have to rebuild the docker image every time we want to change
# the python dependencies, we can just modify pyproject.toml and re-run init_venv.sh inside the container

if [[ ! -x "$PROJECT_DIR/init_venv.sh" ]]; then
    echo "[bootstrap] ERROR: init_venv.sh not found or not executable"
    exit 1
fi

echo "[bootstrap] Initializing virtual environment via init_venv.sh..."
echo "[bootstrap] Check logs/init_venv_*.out for progress (tail -f logs/init_venv_<jobid>.out)"
srun \
    --job-name=init_venv \
    --account=large-sc-2 \
    --partition=normal \
    --nodes=1 \
    --cpus-per-task=8 \
    --ntasks-per-node=1 \
    --time=00:15:00 \
    --container-writable \
    --environment=ngc_pt_jan \
    --output=logs/init_venv_%j.out \
    --error=logs/init_venv_%j.err \
    bash init_venv.sh


echo "[bootstrap] Virtual environment initialization job completed."

INIT_VENV_JOB=$(squeue --me --name=init_venv --noheader --format="%i")
echo "[bootstrap] init_venv job ID: $INIT_VENV_JOB"


##########################################################
# Step 5 – Info for the user
##########################################################

echo
echo "========================================================"
echo " Bootstrap complete!"
echo
echo " Job pipeline:"
echo "   1. build_img → job $BUILD_JOB"
echo "   2. init_venv → job $INIT_VENV_JOB"
echo
echo " EDF environment file created via:"
echo "   create_envfile.sh"
echo 
echo " Test the tokenizer: "
echo "   sbatch run_tok_train.sbatch"
echo
echo " Start training with:"
echo "   sbatch run_nanochat.sbatch"
echo
echo " Monitor jobs with:"
echo "   squeue --me"
echo "========================================================"
