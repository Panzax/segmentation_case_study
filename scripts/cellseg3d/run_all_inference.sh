#! /bin/bash

# Script to run inference for a set of checkpoints
# using infer_cellseg3d_swinunetr.py.
#
# Example usage:
#   ./run_all_inference.sh \
#       --checkpoints_dir /path/to/checkpoints \
#       --images_dir /path/to/images \
#       --output_dir /path/to/base_output \
#       --checkpoint_pattern "*.pth" \
#       --device cuda:0 \
#       --model_name SwinUNetR
#
# Any arguments after "--" are passed directly to
# infer_cellseg3d_swinunetr.py for every run.

SCRIPT_DIR="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/scripts/cellseg3d"

# Paths to local monai and cellseg3d
MONAI_PATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/monai"
CELLSEG3D_PATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/cellseg3d"

CHECKPOINTS_DIR="/clusterfs/nvme/segment_3d/tests/supervised_models/train_swin_unetr_cellseg3d_combined_datasets/base_model"
OUTPUT_DIR="/clusterfs/nvme/segment_3d/tests/supervised_models/infer_swinunetr_cellseg3d_combined_datasets/base_model"
IMAGES_DIR="/clusterfs/nvme/segment_3d/tests/datasets/CellSeg3D_combined_datasets/val/images"
CHECKPOINT_PATTERN="*best_metric.pth"
DEVICE="cuda:0"
FILETYPE=".tif"
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 --checkpoints_dir DIR --images_dir DIR --output_dir DIR [options] [-- extra_args...]" >&2
    echo "" >&2
    echo "Required arguments:" >&2
    echo "  --checkpoints_dir DIR   Directory containing .pth checkpoint files" >&2
    echo "  --images_dir DIR        Directory with images to run inference on" >&2
    echo "  --output_dir DIR        Base directory to save inference results" >&2
    echo "" >&2
    echo "Optional arguments:" >&2
    echo "  --checkpoint_pattern PATTERN   Glob for checkpoint files within checkpoints_dir (default: '*best_metric.pth')" >&2
\    echo "  --device DEVICE                Device string, e.g. cuda:0 or cpu (default: cuda:0)" >&2
    echo "  --filetype EXT                 Output file extension (default: .tif)" >&2
    echo "" >&2
    echo "Anything after '--' is passed to infer_cellseg3d_swinunetr.py for every run." >&2
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints_dir)
            CHECKPOINTS_DIR="$2"; shift 2 ;;
        --images_dir)
            IMAGES_DIR="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --checkpoint_pattern)
            CHECKPOINT_PATTERN="$2"; shift 2 ;;
        --model_name)
            MODEL_NAME="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --filetype)
            FILETYPE="$2"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1 ;;
    esac
done

# Validate required arguments
if [[ -z "$CHECKPOINTS_DIR" || -z "$IMAGES_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --checkpoints_dir, --images_dir, and --output_dir are required." >&2
    usage
    exit 1
fi

if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
    echo "Error: checkpoints_dir does not exist or is not a directory: $CHECKPOINTS_DIR" >&2
    exit 1
fi

if [[ ! -d "$IMAGES_DIR" ]]; then
    echo "Error: images_dir does not exist or is not a directory: $IMAGES_DIR" >&2
    exit 1
fi

# Discover checkpoints
shopt -s nullglob
CHECKPOINTS=("$CHECKPOINTS_DIR"/$CHECKPOINT_PATTERN)
shopt -u nullglob

if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "Error: no checkpoints found in $CHECKPOINTS_DIR matching pattern $CHECKPOINT_PATTERN" >&2
    exit 1
fi

# Create base output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo " Starting CellSeg3D SwinUNETR inference runs"
echo "---------------------------------------------"
echo "Script directory:       $SCRIPT_DIR"
echo "MONAI path:             $MONAI_PATH"
echo "CellSeg3D path:         $CELLSEG3D_PATH"
echo "Checkpoints directory:  $CHECKPOINTS_DIR"
echo "Images directory:       $IMAGES_DIR"
echo "Output directory:       $OUTPUT_DIR"
echo "Checkpoint pattern:     $CHECKPOINT_PATTERN"
echo "Device:                 $DEVICE"
echo "Output filetype:        $FILETYPE"
echo "Number of checkpoints:  ${#CHECKPOINTS[@]}"
echo "============================================="
echo ""

# Source bashrc and activate conda environment
source ~/.bashrc
conda activate segproj
echo "Conda environment activated: $(which python)"

# Append monai and cellseg3d to python path
append_path() {
    case ":$PYTHONPATH:" in
        *":$1:")
            echo "PYTHONPATH already contains: $1" ;;
        *)
            PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$1"
            echo "Appended $1 to PYTHONPATH" ;;
    esac
}

append_path "$MONAI_PATH"
append_path "$CELLSEG3D_PATH"
export PYTHONPATH

echo ""
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# Change to script directory (where infer_cellseg3d_swinunetr.py lives)
echo "Changing to script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

echo "---------------------------------------------"
for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_BASENAME="$(basename "$CKPT")"
    CKPT_NAME="${CKPT_BASENAME%.*}"
    MODEL_NAME_FROM_CKPT="$CKPT_NAME"

    # Strip known suffixes to recover model name, e.g.
    #   SwinUNetR_Mlp_LeakyReLU_best_metric.pth -> SwinUNetR_Mlp_LeakyReLU
    #   SwinUNetR_Mlp_LeakyReLU_latest.pth      -> SwinUNetR_Mlp_LeakyReLU
    if [[ "$MODEL_NAME_FROM_CKPT" == *_best_metric ]]; then
        MODEL_NAME_FROM_CKPT="${MODEL_NAME_FROM_CKPT%_best_metric}"
    elif [[ "$MODEL_NAME_FROM_CKPT" == *_latest ]]; then
        MODEL_NAME_FROM_CKPT="${MODEL_NAME_FROM_CKPT%_latest}"
    fi

    # Allow explicit override via --model_name; otherwise use auto-detected name
    if [[ -n "$MODEL_NAME" ]]; then
        EFFECTIVE_MODEL_NAME="$MODEL_NAME"
    else
        EFFECTIVE_MODEL_NAME="$MODEL_NAME_FROM_CKPT"
    fi

    MODEL_OUTPUT_DIR="$OUTPUT_DIR/$CKPT_NAME"

    echo ""
    echo "Running inference for checkpoint: $CKPT_BASENAME"
    echo "Using model name: $EFFECTIVE_MODEL_NAME"
    echo "Results will be saved to: $MODEL_OUTPUT_DIR"
    echo "---------------------------------------------"

    mkdir -p "$MODEL_OUTPUT_DIR"

    # Build command, only passing window arguments when explicitly set so that
    # infer_cellseg3d_swinunetr.py's own defaults are respected otherwise.
    CMD=(python infer_cellseg3d_swinunetr.py
        --checkpoint "$CKPT"
        --images_dir "$IMAGES_DIR"
        --output_dir "$MODEL_OUTPUT_DIR"
        --model_name "$EFFECTIVE_MODEL_NAME"
        --device "$DEVICE"
        --filetype "$FILETYPE"
    )

    CMD+=("${EXTRA_ARGS[@]}")

    "${CMD[@]}"

    EXITCODE=$?
    if [ $EXITCODE -eq 0 ]; then
        echo "✓ Successfully ran inference for checkpoint: $CKPT_BASENAME"
    else
        echo "✗ Error running inference for checkpoint: $CKPT_BASENAME (exit code $EXITCODE)" >&2
    fi
    echo "---------------------------------------------"
done

echo ""
echo "============================================="
echo "All inference runs completed."
echo "Base output directory: $OUTPUT_DIR"
echo "============================================="
