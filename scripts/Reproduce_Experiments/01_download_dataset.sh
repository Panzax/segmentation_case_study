#!/bin/bash
# =============================================================================
# 01_download_dataset.sh
# =============================================================================
# Downloads the CellSeg3D mesoSPIM dataset from Zenodo and organizes it into
# train/val splits for reproducible experiments.
#
# Dataset: 3D ground truth annotations of cleared whole mouse brain nuclei
#          imaged with a mesoSPIM system
# Source:  https://zenodo.org/records/11095111
# DOI:     10.5281/zenodo.11095111
# License: CC-BY-4.0
#
# Usage:
#   ./01_download_dataset.sh [--output-dir /path/to/output]
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ZENODO_URL="https://zenodo.org/records/11095111/files/DATASET_WITH_GT.zip?download=1"
ZENODO_DOI="10.5281/zenodo.11095111"
ZIP_FILENAME="DATASET_WITH_GT.zip"
EXPECTED_MD5="fbcaf6d4e65b99eac7caa082f6798542"

# Default output directory (relative to this script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/data/CellSeg3D_mesoSPIM"

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--output-dir /path/to/output]"
            echo ""
            echo "Downloads the CellSeg3D mesoSPIM dataset from Zenodo."
            echo ""
            echo "Options:"
            echo "  --output-dir    Directory to save the dataset (default: ${DEFAULT_OUTPUT_DIR})"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Dataset source: https://zenodo.org/records/11095111"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
print_header() {
    echo ""
    echo "============================================="
    echo " $1"
    echo "============================================="
}

print_step() {
    echo ""
    echo ">>> $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: '$1' is required but not installed."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
print_header "CellSeg3D mesoSPIM Dataset Download"

echo "Source:     ${ZENODO_URL}"
echo "DOI:        ${ZENODO_DOI}"
echo "Output:     ${OUTPUT_DIR}"

# Check if dataset already exists
if [[ -d "${OUTPUT_DIR}/train/images" ]] && [[ -d "${OUTPUT_DIR}/val/images" ]]; then
    TRAIN_COUNT=$(find "${OUTPUT_DIR}/train/images" -name "*.tif" 2>/dev/null | wc -l)
    VAL_COUNT=$(find "${OUTPUT_DIR}/val/images" -name "*.tif" 2>/dev/null | wc -l)
    if [[ ${TRAIN_COUNT} -ge 5 ]] && [[ ${VAL_COUNT} -ge 1 ]]; then
        echo ""
        echo "Dataset already exists at ${OUTPUT_DIR}"
        echo "  Train images: ${TRAIN_COUNT}"
        echo "  Val images:   ${VAL_COUNT}"
        echo ""
        read -p "Re-download and overwrite? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping download. Dataset is ready."
            exit 0
        fi
    fi
fi

# Check dependencies
print_step "Checking dependencies..."
check_command "unzip"

# Check for wget or curl
DOWNLOAD_CMD=""
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl"
else
    echo "Error: Either 'wget' or 'curl' is required."
    exit 1
fi
echo "Using ${DOWNLOAD_CMD} for download"

# Create output directories
print_step "Creating directory structure..."
mkdir -p "${OUTPUT_DIR}/train/images"
mkdir -p "${OUTPUT_DIR}/train/labels"
mkdir -p "${OUTPUT_DIR}/val/images"
mkdir -p "${OUTPUT_DIR}/val/labels"

# Create temp directory for download
TEMP_DIR=$(mktemp -d)
echo "Temp directory: ${TEMP_DIR}"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

# Download dataset
print_step "Downloading dataset from Zenodo (30.6 MB)..."
cd "${TEMP_DIR}"

if [[ "${DOWNLOAD_CMD}" == "wget" ]]; then
    wget --progress=bar:force -O "${ZIP_FILENAME}" "${ZENODO_URL}"
else
    curl -L --progress-bar -o "${ZIP_FILENAME}" "${ZENODO_URL}"
fi

# Verify download
if [[ ! -f "${ZIP_FILENAME}" ]]; then
    echo "Error: Download failed - file not found"
    exit 1
fi

FILE_SIZE=$(stat -f%z "${ZIP_FILENAME}" 2>/dev/null || stat -c%s "${ZIP_FILENAME}" 2>/dev/null)
echo "Downloaded: ${ZIP_FILENAME} (${FILE_SIZE} bytes)"

# Optional: verify MD5 checksum
print_step "Verifying checksum..."
if command -v md5sum &> /dev/null; then
    ACTUAL_MD5=$(md5sum "${ZIP_FILENAME}" | cut -d' ' -f1)
    if [[ "${ACTUAL_MD5}" == "${EXPECTED_MD5}" ]]; then
        echo "Checksum verified: ${ACTUAL_MD5}"
    else
        echo "Warning: Checksum mismatch!"
        echo "  Expected: ${EXPECTED_MD5}"
        echo "  Actual:   ${ACTUAL_MD5}"
        echo "Continuing anyway..."
    fi
elif command -v md5 &> /dev/null; then
    ACTUAL_MD5=$(md5 -q "${ZIP_FILENAME}")
    if [[ "${ACTUAL_MD5}" == "${EXPECTED_MD5}" ]]; then
        echo "Checksum verified: ${ACTUAL_MD5}"
    else
        echo "Warning: Checksum mismatch!"
        echo "  Expected: ${EXPECTED_MD5}"
        echo "  Actual:   ${ACTUAL_MD5}"
        echo "Continuing anyway..."
    fi
else
    echo "Skipping checksum verification (md5sum/md5 not available)"
fi

# Extract dataset
print_step "Extracting dataset..."
unzip -q "${ZIP_FILENAME}"

# Check extraction
if [[ ! -d "DATASET_WITH_GT" ]]; then
    echo "Error: Expected 'DATASET_WITH_GT' directory not found after extraction"
    ls -la
    exit 1
fi

# Reorganize files into train/val structure
print_step "Reorganizing files into train/val structure..."

# Train images (c1, c2, c3, c4, visual->v1)
echo "  Copying train images..."
cp "DATASET_WITH_GT/c1image.tif" "${OUTPUT_DIR}/train/images/c1image.tif"
cp "DATASET_WITH_GT/c2image.tif" "${OUTPUT_DIR}/train/images/c2image.tif"
cp "DATASET_WITH_GT/c3image.tif" "${OUTPUT_DIR}/train/images/c3image.tif"
cp "DATASET_WITH_GT/c4image.tif" "${OUTPUT_DIR}/train/images/c4image.tif"
cp "DATASET_WITH_GT/visual.tif" "${OUTPUT_DIR}/train/images/v1image.tif"

# Train labels (c1, c2, c3, c4, visual->v1) with renaming
echo "  Copying train labels..."
cp "DATASET_WITH_GT/labels/c1labels_new_label.tif" "${OUTPUT_DIR}/train/labels/c1label.tif"
cp "DATASET_WITH_GT/labels/c2labels_new_label.tif" "${OUTPUT_DIR}/train/labels/c2label.tif"
cp "DATASET_WITH_GT/labels/c3labels_new_label.tif" "${OUTPUT_DIR}/train/labels/c3label.tif"
cp "DATASET_WITH_GT/labels/c4labels_new_label.tif" "${OUTPUT_DIR}/train/labels/c4label.tif"
cp "DATASET_WITH_GT/labels/visual_gt_new_label.tif" "${OUTPUT_DIR}/train/labels/v1label.tif"

# Validation images (c5)
echo "  Copying validation images..."
cp "DATASET_WITH_GT/c5image.tif" "${OUTPUT_DIR}/val/images/c5image.tif"

# Validation labels (c5) with renaming
echo "  Copying validation labels..."
cp "DATASET_WITH_GT/labels/c5labels_new_label.tif" "${OUTPUT_DIR}/val/labels/c5label.tif"

# Print final structure
print_header "Dataset Ready"
echo ""
echo "Location: ${OUTPUT_DIR}"
echo ""
echo "Directory structure:"

# Use tree if available, otherwise use find
if command -v tree &> /dev/null; then
    tree "${OUTPUT_DIR}"
else
    echo "${OUTPUT_DIR}/"
    echo "+-- train/"
    echo "|   +-- images/"
    for f in "${OUTPUT_DIR}/train/images/"*.tif; do
        echo "|   |   +-- $(basename "$f")"
    done
    echo "|   +-- labels/"
    for f in "${OUTPUT_DIR}/train/labels/"*.tif; do
        echo "|       +-- $(basename "$f")"
    done
    echo "+-- val/"
    echo "    +-- images/"
    for f in "${OUTPUT_DIR}/val/images/"*.tif; do
        echo "    |   +-- $(basename "$f")"
    done
    echo "    +-- labels/"
    for f in "${OUTPUT_DIR}/val/labels/"*.tif; do
        echo "        +-- $(basename "$f")"
    done
fi

echo ""
echo "============================================="
echo " Download complete!"
echo "============================================="
echo ""
echo "To use this dataset for training, run:"
echo ""
echo "  python scripts/cellseg3d/train_cellseg3d_swinunetr.py \\"
echo "      --images_dir ${OUTPUT_DIR}/train/images \\"
echo "      --labels_dir ${OUTPUT_DIR}/train/labels \\"
echo "      --val_images_dir ${OUTPUT_DIR}/val/images \\"
echo "      --val_labels_dir ${OUTPUT_DIR}/val/labels \\"
echo "      --output_dir ./outputs"
echo ""
echo "Citation:"
echo "  Achard et al. (2024). CellSeg3D: Self-supervised 3D cell segmentation"
echo "  for fluorescence microscopy. eLife. DOI: 10.7554/eLife.99848"
echo ""
echo "Dataset DOI: ${ZENODO_DOI}"
echo ""

