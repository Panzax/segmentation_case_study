#!/bin/bash
# =============================================================================
# 03_run_all_experiments.sh
# =============================================================================
# Runs all SwinUNETR training experiments with different configurations.
#
# Prerequisites:
#   - Run 01_download_dataset.sh first
#   - Run 02_convert_to_32bit.sh first
#   - Activate conda environment with required packages
#
# Usage:
#   ./03_run_all_experiments.sh
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration - Paths relative to this script's location
# -----------------------------------------------------------------------------
REPRODUCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${REPRODUCE_DIR}/../.." && pwd)"

# Training script location
TRAIN_SCRIPT_DIR="${REPO_ROOT}/scripts/cellseg3d"

# Paths to local monai and cellseg3d
MONAI_PATH="${REPO_ROOT}/packages/MONAI"
CELLSEG3D_PATH="${REPO_ROOT}/packages/CellSeg3D"

# Dataset paths (from 01_download_dataset.sh)
DATA_DIR="${REPRODUCE_DIR}/data/CellSeg3D_mesoSPIM"
IMAGES_DIR="${DATA_DIR}/train/images"
LABELS_DIR="${DATA_DIR}/train/labels"
VALIDATION_IMAGES_DIR="${DATA_DIR}/val/images"
VALIDATION_LABELS_DIR="${DATA_DIR}/val/labels"

# Output directory for trained models
OUTPUT_DIR="${REPRODUCE_DIR}/outputs"

# Models to train
MODELS=("SwinUNetR_Mlp_LeakyReLU" "SwinUNetR_SwiGLU_LeakyReLU" "SwinUNetR_Mlp_ReLUSquared" "SwinUNetR_SwiGLU_ReLUSquared")

# Seeds for reproducibility
SEED=34936339

# -----------------------------------------------------------------------------
# Verify prerequisites
# -----------------------------------------------------------------------------
if [[ ! -d "${IMAGES_DIR}" ]]; then
    echo "Error: Dataset not found at ${DATA_DIR}"
    echo "Please run 01_download_dataset.sh first."
    exit 1
fi

if [[ ! -d "${MONAI_PATH}" ]]; then
    echo "Error: MONAI not found at ${MONAI_PATH}"
    exit 1
fi

if [[ ! -d "${CELLSEG3D_PATH}" ]]; then
    echo "Error: CellSeg3D not found at ${CELLSEG3D_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "============================================="
echo " Starting CellSeg3D SwinUNETR experiments"
echo "---------------------------------------------"
echo "Repository root:    ${REPO_ROOT}"
echo "Training script:    ${TRAIN_SCRIPT_DIR}"
echo "MONAI path:         ${MONAI_PATH}"
echo "CellSeg3D path:     ${CELLSEG3D_PATH}"
echo "Images directory:   ${IMAGES_DIR}"
echo "Labels directory:   ${LABELS_DIR}"
echo "Val images:         ${VALIDATION_IMAGES_DIR}"
echo "Val labels:         ${VALIDATION_LABELS_DIR}"
echo "Output directory:   ${OUTPUT_DIR}"
echo "Models to train:    ${MODELS[@]}"
echo "Seeds:              ${SEEDS[@]}"
echo "============================================="
echo ""

# Append monai and cellseg3d to python path
append_path() {
    case ":$PYTHONPATH:" in
        *":$1:"*) 
            echo "PYTHONPATH already contains: $1"
            ;;
        *) 
            PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$1"
            echo "Appended $1 to PYTHONPATH"
            ;;
    esac
}

append_path "$MONAI_PATH"
append_path "$CELLSEG3D_PATH"
export PYTHONPATH

echo ""
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# Change to training script directory
echo "Changing to script directory: $TRAIN_SCRIPT_DIR"
cd "$TRAIN_SCRIPT_DIR"

# Run all experiments
echo "---------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Running experiment with model: $MODEL"
    echo "---------------------------------------------"

    # Base model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python train_cellseg3d_swinunetr.py \
        --model $MODEL \
        --images_dir $IMAGES_DIR \
        --labels_dir $LABELS_DIR \
        --val_images_dir $VALIDATION_IMAGES_DIR \
        --val_labels_dir $VALIDATION_LABELS_DIR \
        --output_dir "$OUTPUT_DIR/base_model/" \
        --seed $SEED \
        --disable_wandb
    EXITCODE=$?
    if [ $EXITCODE -eq 0 ]; then
        echo "[OK] Successfully ran base model experiment for: $MODEL (seed: $SEED)"
    else
        echo "[FAIL] Error running experiment for model: $MODEL (exit code $EXITCODE)"
    fi

    # Model depths 1 1 1 1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python train_cellseg3d_swinunetr.py \
        --model $MODEL \
        --images_dir $IMAGES_DIR \
        --labels_dir $LABELS_DIR \
        --val_images_dir $VALIDATION_IMAGES_DIR \
        --val_labels_dir $VALIDATION_LABELS_DIR \
        --output_dir "$OUTPUT_DIR/model_depths_1_1_1_1/" \
        --depths 1 1 1 1 \
        --seed $SEED \
        --disable_wandb
    EXITCODE=$?
    if [ $EXITCODE -eq 0 ]; then
        echo "[OK] Successfully ran depths 1-1-1-1 experiment for: $MODEL (seed: $SEED)"
    else
        echo "[FAIL] Error running experiment for model: $MODEL (exit code $EXITCODE)"
    fi

    # Feature size 12 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    python train_cellseg3d_swinunetr.py \
        --model $MODEL \
        --images_dir $IMAGES_DIR \
        --labels_dir $LABELS_DIR \
        --val_images_dir $VALIDATION_IMAGES_DIR \
        --val_labels_dir $VALIDATION_LABELS_DIR \
        --output_dir "$OUTPUT_DIR/feature_size_12/" \
        --feature_size 12 \
        --seed $SEED \
        --disable_wandb
    EXITCODE=$?
    if [ $EXITCODE -eq 0 ]; then
        echo "[OK] Successfully ran feature_size 12 experiment for: $MODEL (seed: $SEED)"
    else
        echo "[FAIL] Error running experiment for model: $MODEL (exit code $EXITCODE)"
    fi


    echo "---------------------------------------------"
done

echo ""
echo "============================================="
echo "All experiments completed."
echo "Output directory: $OUTPUT_DIR"
echo "============================================="

