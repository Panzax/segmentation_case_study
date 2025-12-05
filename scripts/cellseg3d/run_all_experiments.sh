#! /bin/bash

# Script to run all experiments
SCRIPT_DIR="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/scripts/cellseg3d"

# Paths to local monai and cellseg3d
MONAI_PATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/monai"
CELLSEG3D_PATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/cellseg3d"

# Models to train
# MODELS=("SwinUNetR_Mlp_ReLUSquared" "SwinUNetR_SwiGLU_ReLUSquared")
MODELS=("SwinUNetR_Mlp_LeakyReLU" "SwinUNetR_SwiGLU_LeakyReLU" "SwinUNetR_Mlp_ReLUSquared" "SwinUNetR_SwiGLU_ReLUSquared")

# Paths to images and labels
IMAGES_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_combined_datasets/train/images"
LABELS_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_combined_datasets/train/labels"

VALIDATION_IMAGES_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_combined_datasets/val/images"
VALIDATION_LABELS_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_combined_datasets/val/labels"


# Path to output directory
OUTPUT_DIR="/clusterfs/nvme/segment_3d/tests/supervised_models/train_swin_unetr_cellseg3d_combined_datasets/feature_size_12"

echo "============================================="
echo " Starting CellSeg3D SwinUNETR experiments"
echo "---------------------------------------------"
echo "Script directory:   $SCRIPT_DIR"
echo "MONAI path:         $MONAI_PATH"
echo "CellSeg3D path:     $CELLSEG3D_PATH"
echo "Images directory:   $IMAGES_DIR"
echo "Labels directory:   $LABELS_DIR"
echo "Validation images directory:   $VALIDATION_IMAGES_DIR"
echo "Validation labels directory:   $VALIDATION_LABELS_DIR"
echo "Output directory:   $OUTPUT_DIR"
echo "Models to train:    ${MODELS[@]}"
echo "============================================="
echo ""

# Source bashrc and activate conda environment
source ~/.bashrc
conda activate segproj
echo "Conda environment activated: $(which python)"

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

# Change to script directory
echo "Changing to script directory: $SCRIPT_DIR"
cd $SCRIPT_DIR

# Run all experiments
echo "---------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Running experiment with model: $MODEL"
    echo "---------------------------------------------"
    python train_cellseg3d_swinunetr.py \
        --model $MODEL \
        --images_dir $IMAGES_DIR \
        --labels_dir $LABELS_DIR \
        --val_images_dir $VALIDATION_IMAGES_DIR \
        --val_labels_dir $VALIDATION_LABELS_DIR \
        --output_dir $OUTPUT_DIR \
        --depths 1 1 1 1
        # --feature_size 12
    EXITCODE=$?
    if [ $EXITCODE -eq 0 ]; then
        echo "✓ Successfully ran experiment for model: $MODEL"
    else
        echo "✗ Error running experiment for model: $MODEL (exit code $EXITCODE)"
    fi
    echo "---------------------------------------------"
done

echo ""
echo "============================================="
echo "All experiments completed."
echo "Output directory: $OUTPUT_DIR"
echo "============================================="

