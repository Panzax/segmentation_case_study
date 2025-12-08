#!/bin/bash
# Bash wrapper to run learning rate sweeps for each feature size

SCRIPT_DIR="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/scripts/cellseg3d"

# Paths to local monai and cellseg3d
MONAI_PATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/monai"
CELLSEG3D_PATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/cellseg3d"

# Dataset paths
IMAGES_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_mesoSPIM/train/images"
LABELS_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_mesoSPIM/train/labels"
VALIDATION_IMAGES_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_mesoSPIM/val/images"
VALIDATION_LABELS_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_mesoSPIM/val/labels"

# Output directory
OUTPUT_DIR="/clusterfs/nvme/segment_3d/tests/supervised_models/sweep_lr_per_feature_size"

# Feature sizes to test
FEATURE_SIZES=(12 24)

# Learning rates to test
LEARNING_RATES=(1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 1e-1)

# Training parameters
MAX_EPOCHS=50
BATCH_SIZE=1
MODEL_NAME="SwinUNetR_Mlp_LeakyReLU"
LOSS_FUNCTION="Generalized Dice"

echo "============================================="
echo " Learning Rate Sweeps per Feature Size"
echo "---------------------------------------------"
echo "Script directory:   $SCRIPT_DIR"
echo "Images directory:   $IMAGES_DIR"
echo "Labels directory:   $LABELS_DIR"
echo "Validation images:  $VALIDATION_IMAGES_DIR"
echo "Validation labels:  $VALIDATION_LABELS_DIR"
echo "Output directory:   $OUTPUT_DIR"
echo "Feature sizes:      ${FEATURE_SIZES[@]}"
echo "Learning rates:     ${LEARNING_RATES[@]}"
echo "Model:              $MODEL_NAME"
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

# Build learning rates string
LR_STRING="${LEARNING_RATES[@]}"

# Run the sweep script
echo "---------------------------------------------"
echo "Initializing sweeps..."
echo "---------------------------------------------"
python sweep_lr_per_feature_size.py \
    --images_dir "$IMAGES_DIR" \
    --labels_dir "$LABELS_DIR" \
    --val_images_dir "$VALIDATION_IMAGES_DIR" \
    --val_labels_dir "$VALIDATION_LABELS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --feature_sizes ${FEATURE_SIZES[@]} \
    --learning_rates ${LEARNING_RATES[@]} \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --method grid
    # --loss_function "$LOSS_FUNCTION" \

EXITCODE=$?
if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "Sweeps initialized successfully!"
    echo "============================================="
    echo ""
    echo "To run agents, use the commands printed above,"
    echo "or run agents in parallel:"
    echo ""
    echo "  # In separate terminals/sessions:"
    echo "  wandb agent <project>/<sweep_id_for_feature_size_12>"
    echo "  wandb agent <project>/<sweep_id_for_feature_size_24>"
    echo "  wandb agent <project>/<sweep_id_for_feature_size_48>"
else
    echo "Error initializing sweeps (exit code $EXITCODE)"
    exit $EXITCODE
fi
