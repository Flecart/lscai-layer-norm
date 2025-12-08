#!/bin/bash
set -euo pipefail

##########################################################
# Paths
##########################################################

PROJECT_DIR="$HOME/scratch/lscai-layer-norm/nematus"
SCRIPT_DIR="$PROJECT_DIR/scripts"
WMT17_DIR="$SCRIPT_DIR/wmt17"          # expects download_files.sh / preprocess.sh here
EDF_ENV_NAME="ngc_tf2_jan"

echo "[bootstrap-nematus] Starting Nematus/WMT17 bootstrap..."
echo "[bootstrap-nematus] Project dir: $PROJECT_DIR"
echo "[bootstrap-nematus] Scripts dir: $SCRIPT_DIR"
echo "[bootstrap-nematus] WMT17 scripts dir: $WMT17_DIR"

cd "$PROJECT_DIR"

##########################################################
# Step 0 – Ensure logs directory exists
##########################################################

mkdir -p logs
echo "[bootstrap-nematus] Created/verified logs/"

##########################################################
# Step 1 – Build container image → .sqsh (compute node)
##########################################################

BUILD_IMG_SCRIPT="$SCRIPT_DIR/build_image_nematus.sh"
if [[ ! -x "$BUILD_IMG_SCRIPT" ]]; then
    echo "[bootstrap-nematus] ERROR: $BUILD_IMG_SCRIPT not found or not executable"
    exit 1
fi

echo "[bootstrap-nematus] Submitting image build job..."
echo "[bootstrap-nematus] Check logs/build_nematus_img_*.out for progress (tail -f logs/build_nematus_img_<jobid>.out)"

srun \
    --job-name=build_nematus_img \
    --account=large-sc-2 \
    --partition=normal \
    --nodes=1 \
    --cpus-per-task=8 \
    --ntasks-per-node=1 \
    --time=00:20:00 \
    --container-writable \
    --output=logs/build_nematus_img_%j.out \
    --error=logs/build_nematus_img_%j.err \
    bash "$BUILD_IMG_SCRIPT"

echo "[bootstrap-nematus] Build job completed."

BUILD_JOB=$(squeue --me --name=build_nematus_img --noheader --format="%i" || true)
echo "[bootstrap-nematus] Build job ID: ${BUILD_JOB:-unknown}"

##########################################################
# Step 2 – Create EDF environment file
##########################################################

CREATE_ENV_SCRIPT="$SCRIPT_DIR/create_envfile.sh"
if [[ ! -x "$CREATE_ENV_SCRIPT" ]]; then
    echo "[bootstrap-nematus] ERROR: $CREATE_ENV_SCRIPT not found or not executable"
    exit 1
fi

echo "[bootstrap-nematus] Creating EDF environment file via create_envfile.sh..."
bash "$CREATE_ENV_SCRIPT"

##########################################################
# Step 3 – Initialize virtual environment inside container
##########################################################

INIT_VENV_SCRIPT="$SCRIPT_DIR/init_venv_nematus.sh"
if [[ ! -x "$INIT_VENV_SCRIPT" ]]; then
    echo "[bootstrap-nematus] ERROR: $INIT_VENV_SCRIPT not found or not executable"
    exit 1
fi

echo "[bootstrap-nematus] Initializing virtual environment via init_venv_nematus.sh..."
echo "[bootstrap-nematus] Check logs/init_venv_nematus_*.out for progress (tail -f logs/init_venv_nematus_<jobid>.out)"

srun \
    --job-name=init_venv_nematus \
    --account=large-sc-2 \
    --partition=normal \
    --nodes=1 \
    --cpus-per-task=8 \
    --ntasks-per-node=1 \
    --time=00:20:00 \
    --container-writable \
    --environment="$EDF_ENV_NAME" \
    --output=logs/init_venv_nematus_%j.out \
    --error=logs/init_venv_nematus_%j.err \
    bash "$INIT_VENV_SCRIPT"

echo "[bootstrap-nematus] Virtual environment initialization job completed."

INIT_VENV_JOB=$(squeue --me --name=init_venv_nematus --noheader --format="%i" || true)
echo "[bootstrap-nematus] init_venv_nematus job ID: ${INIT_VENV_JOB:-unknown}"

##########################################################
# Step 4 – Download WMT17 datasets (compute node)
##########################################################

DOWNLOAD_SCRIPT="$WMT17_DIR/download_files.sh"
if [[ ! -x "$DOWNLOAD_SCRIPT" ]]; then
    echo "[bootstrap-nematus] ERROR: $DOWNLOAD_SCRIPT not found or not executable"
    echo "[bootstrap-nematus] Expected at: $DOWNLOAD_SCRIPT"
    exit 1
fi

echo "[bootstrap-nematus] Submitting WMT17 download job..."
echo "[bootstrap-nematus] Check logs/wmt17_download_*.out for progress (tail -f logs/wmt17_download_<jobid>.out)"

srun \
    --job-name=wmt17_download \
    --account=large-sc-2 \
    --partition=normal \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --time=02:00:00 \
    --container-writable \
    --environment="$EDF_ENV_NAME" \
    --output=logs/wmt17_download_%j.out \
    --error=logs/wmt17_download_%j.err \
    bash -lc "
        cd '$WMT17_DIR'
        echo '[wmt17_download] running in $(pwd)'
        source '$PROJECT_DIR/.venv/bin/activate'
        ./download_files.sh
    "

WMT17_DOWNLOAD_JOB=$(squeue --me --name=wmt17_download --noheader --format="%i" || true)
echo "[bootstrap-nematus] wmt17_download job ID: ${WMT17_DOWNLOAD_JOB:-unknown}"

##########################################################
# Step 5 – Preprocess WMT17 data (compute node)
##########################################################

PREPROCESS_SCRIPT="$WMT17_DIR/preprocess.sh"
if [[ ! -x "$PREPROCESS_SCRIPT" ]]; then
    echo "[bootstrap-nematus] ERROR: $PREPROCESS_SCRIPT not found or not executable"
    echo "[bootstrap-nematus] Expected at: $PREPROCESS_SCRIPT"
    exit 1
fi

echo "[bootstrap-nematus] Submitting WMT17 preprocess job..."
echo "[bootstrap-nematus] Check logs/wmt17_preprocess_*.out for progress (tail -f logs/wmt17_preprocess_<jobid>.out)"

srun \
    --job-name=wmt17_preprocess \
    --account=large-sc-2 \
    --partition=normal \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --time=04:00:00 \
    --container-writable \
    --environment="$EDF_ENV_NAME" \
    --output=logs/wmt17_preprocess_%j.out \
    --error=logs/wmt17_preprocess_%j.err \
    bash -lc "
        cd '$WMT17_DIR'
        echo '[wmt17_preprocess] running in $(pwd)'
        source '$PROJECT_DIR/.venv/bin/activate'
        ./preprocess.sh
    "

WMT17_PREPROCESS_JOB=$(squeue --me --name=wmt17_preprocess --noheader --format="%i" || true)
echo "[bootstrap-nematus] wmt17_preprocess job ID: ${WMT17_PREPROCESS_JOB:-unknown}"

##########################################################
# Step 6 – Summary
##########################################################

echo
echo "========================================================"
echo " Nematus/WMT17 Bootstrap complete!"
echo
echo " Job pipeline:"
echo "   1. build_nematus_img   → job ${BUILD_JOB:-unknown}"
echo "   2. init_venv_nematus   → job ${INIT_VENV_JOB:-unknown}"
echo "   3. wmt17_download      → job ${WMT17_DOWNLOAD_JOB:-unknown}"
echo "   4. wmt17_preprocess    → job ${WMT17_PREPROCESS_JOB:-unknown}"
echo
echo " EDF environment file created via:"
echo "   $SCRIPT_DIR/create_envfile.sh"
echo
echo " Once this completes, you can start training with:"
echo "   sbatch $SCRIPT_DIR/experiments/run_nematus.sbatch"
echo
echo " Monitor jobs with:"
echo "   squeue --me"
echo "========================================================"
