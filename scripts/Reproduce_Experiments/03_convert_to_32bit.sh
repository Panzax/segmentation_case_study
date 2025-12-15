#!/bin/bash
# =============================================================================
# 03_convert_to_32bit.sh
# =============================================================================
# Converts all downloaded dataset images to 32-bit floating point format.
#
# This script verifies that all expected files from 02_download_dataset.sh
# exist, then uses convert_images_to_32bit.py to convert them in place.
#
# Prerequisites:
#   - Run 01_setup_environment.sh first
#   - Run 02_download_dataset.sh first
#   - Activate conda environment: conda activate segproj
#
# Usage:
#   ./03_convert_to_32bit.sh
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data/CellSeg3D_mesoSPIM"
CONVERT_SCRIPT="${REPO_ROOT}/scripts/cellseg3d/convert_images_to_32bit.py"

# Convert paths to Windows format if running in Git Bash/MINGW
if command -v cygpath &> /dev/null; then
    DATA_DIR_WIN=$(cygpath -w "${DATA_DIR}")
    CONVERT_SCRIPT_WIN=$(cygpath -w "${CONVERT_SCRIPT}")
else
    DATA_DIR_WIN="${DATA_DIR}"
    CONVERT_SCRIPT_WIN="${CONVERT_SCRIPT}"
fi

# Expected files (12 total)
EXPECTED_TRAIN_IMAGES=(
    "c1image.tif"
    "c2image.tif"
    "c3image.tif"
    "c4image.tif"
    "v1image.tif"
)

EXPECTED_TRAIN_LABELS=(
    "c1label.tif"
    "c2label.tif"
    "c3label.tif"
    "c4label.tif"
    "v1label.tif"
)

EXPECTED_VAL_IMAGES=(
    "c5image.tif"
)

EXPECTED_VAL_LABELS=(
    "c5label.tif"
)

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

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
print_header "Convert Dataset to 32-bit"

echo "Data directory:   ${DATA_DIR}"
echo "Convert script:   ${CONVERT_SCRIPT}"

# Check if convert script exists
if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo ""
    echo "Error: Convert script not found at ${CONVERT_SCRIPT}"
    exit 1
fi

# Check if data directory exists
if [[ ! -d "${DATA_DIR}" ]]; then
    echo ""
    echo "Error: Data directory not found at ${DATA_DIR}"
    echo ""
    echo "Please run 02_download_dataset.sh first:"
    echo "  ./02_download_dataset.sh"
    exit 1
fi

# -----------------------------------------------------------------------------
# Verify all expected files exist
# -----------------------------------------------------------------------------
print_step "Verifying downloaded files..."

MISSING_FILES=()
FOUND_COUNT=0

# Check train images
echo "  Checking train/images/..."
for file in "${EXPECTED_TRAIN_IMAGES[@]}"; do
    filepath="${DATA_DIR}/train/images/${file}"
    if [[ -f "${filepath}" ]]; then
        FOUND_COUNT=$((FOUND_COUNT + 1))
    else
        MISSING_FILES+=("train/images/${file}")
    fi
done

# Check train labels
echo "  Checking train/labels/..."
for file in "${EXPECTED_TRAIN_LABELS[@]}"; do
    filepath="${DATA_DIR}/train/labels/${file}"
    if [[ -f "${filepath}" ]]; then
        FOUND_COUNT=$((FOUND_COUNT + 1))
    else
        MISSING_FILES+=("train/labels/${file}")
    fi
done

# Check val images
echo "  Checking val/images/..."
for file in "${EXPECTED_VAL_IMAGES[@]}"; do
    filepath="${DATA_DIR}/val/images/${file}"
    if [[ -f "${filepath}" ]]; then
        FOUND_COUNT=$((FOUND_COUNT + 1))
    else
        MISSING_FILES+=("val/images/${file}")
    fi
done

# Check val labels
echo "  Checking val/labels/..."
for file in "${EXPECTED_VAL_LABELS[@]}"; do
    filepath="${DATA_DIR}/val/labels/${file}"
    if [[ -f "${filepath}" ]]; then
        FOUND_COUNT=$((FOUND_COUNT + 1))
    else
        MISSING_FILES+=("val/labels/${file}")
    fi
done

EXPECTED_TOTAL=12
echo ""
echo "Found ${FOUND_COUNT}/${EXPECTED_TOTAL} expected files"

# Report missing files
if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo ""
    echo "Error: Missing files detected!"
    echo ""
    for missing in "${MISSING_FILES[@]}"; do
        echo "  - ${missing}"
    done
    echo ""
    echo "Please run 02_download_dataset.sh first to download the dataset."
    exit 1
fi

echo "All expected files found!"

# -----------------------------------------------------------------------------
# Check if already converted
# -----------------------------------------------------------------------------
print_step "Checking current data type..."

SAMPLE_FILE="${DATA_DIR_WIN}/train/images/c1image.tif"

CURRENT_DTYPE=$(python -c "
import tifffile
arr = tifffile.imread(r'${SAMPLE_FILE}')
print(arr.dtype)
")

echo "  Sample file: c1image.tif"
echo "  Current dtype: ${CURRENT_DTYPE}"

if [[ "${CURRENT_DTYPE}" == "float32" ]]; then
    echo ""
    echo "Images appear to be already 32-bit floating point."
    echo ""
    echo "Verifying all files..."
    echo ""
    
    # Verify all files are float32
    python -c "
import tifffile
import numpy as np
from pathlib import Path

data_dir = Path(r'${DATA_DIR_WIN}')
all_files = [
    'train/images/c1image.tif',
    'train/images/c2image.tif',
    'train/images/c3image.tif',
    'train/images/c4image.tif',
    'train/images/v1image.tif',
    'train/labels/c1label.tif',
    'train/labels/c2label.tif',
    'train/labels/c3label.tif',
    'train/labels/c4label.tif',
    'train/labels/v1label.tif',
    'val/images/c5image.tif',
    'val/labels/c5label.tif',
]

success_count = 0

for rel_path in all_files:
    filepath = data_dir / rel_path
    try:
        arr = tifffile.imread(filepath)
        dtype = arr.dtype
        if dtype == np.float32:
            status = 'OK'
            success_count += 1
        else:
            status = f'NOT float32 ({dtype})'
        print(f'  [{status}] {rel_path}')
    except Exception as e:
        print(f'  [ERROR] {rel_path}: {e}')

print('')
print(f'Results: {success_count}/12 files are already float32')
"
    
    echo ""
    read -p "Re-convert anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Skipping conversion. Images are already float32."
        echo ""
        echo "Next step: Run ./04_run_all_experiments.sh"
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Convert to 32-bit
# -----------------------------------------------------------------------------
print_step "Converting images to 32-bit floating point..."

echo "This will convert all .tif files in place."
echo ""

# Run the conversion script
python "${CONVERT_SCRIPT_WIN}" \
    --input_dir "${DATA_DIR_WIN}" \
    --in_place \
    --ext .tif

# -----------------------------------------------------------------------------
# Verify conversion
# -----------------------------------------------------------------------------
print_step "Verifying conversion of all files..."

# Check all files to confirm they are now float32
VERIFICATION_RESULT=$(python -c "
import tifffile
import numpy as np
from pathlib import Path
import sys

data_dir = Path(r'${DATA_DIR_WIN}')
all_files = [
    'train/images/c1image.tif',
    'train/images/c2image.tif',
    'train/images/c3image.tif',
    'train/images/c4image.tif',
    'train/images/v1image.tif',
    'train/labels/c1label.tif',
    'train/labels/c2label.tif',
    'train/labels/c3label.tif',
    'train/labels/c4label.tif',
    'train/labels/v1label.tif',
    'val/images/c5image.tif',
    'val/labels/c5label.tif',
]

success_count = 0
fail_count = 0

print('  Checking dtypes of all converted files:')
print('')

for rel_path in all_files:
    filepath = data_dir / rel_path
    try:
        arr = tifffile.imread(filepath)
        dtype = arr.dtype
        if dtype == np.float32:
            status = 'OK'
            success_count += 1
        else:
            status = f'FAIL (got {dtype})'
            fail_count += 1
        print(f'    [{status}] {rel_path}: {dtype}')
    except Exception as e:
        print(f'    [ERROR] {rel_path}: {e}')
        fail_count += 1

print('')
print(f'  Results: {success_count}/12 files are float32')

if fail_count > 0:
    print(f'  WARNING: {fail_count} files failed verification!')
    sys.exit(1)
else:
    print('  All files successfully converted to float32!')
    sys.exit(0)
")

VERIFY_EXIT_CODE=$?

echo "${VERIFICATION_RESULT}"

if [[ ${VERIFY_EXIT_CODE} -ne 0 ]]; then
    echo ""
    echo "Error: Verification failed! Some files were not converted properly."
    exit 1
fi

print_header "Conversion Complete"
echo ""
echo "All ${EXPECTED_TOTAL} files have been converted to 32-bit floating point."
echo ""
echo "Data location: ${DATA_DIR}"
echo ""
echo "Next step: Run ./04_run_all_experiments.sh"
echo ""

