#!/bin/bash
# =============================================================================
# 01_setup_environment.sh
# =============================================================================
# Creates a conda environment with all required dependencies for the
# CellSeg3D experiments.
#
# Prerequisites:
#   - Conda or Miniconda/Miniforge installed
#   - CUDA-capable GPU (recommended)
#
# Usage:
#   ./01_setup_environment.sh
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
ENV_NAME="segproj"

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
print_header "Setup Conda Environment"

echo "Environment file: ${ENV_FILE}"
echo "Environment name: ${ENV_NAME}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo ""
    echo "Error: conda is not installed or not in PATH."
    echo ""
    echo "Please install Miniconda or Miniforge:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo "  https://github.com/conda-forge/miniforge"
    exit 1
fi

# Check if environment file exists
if [[ ! -f "${ENV_FILE}" ]]; then
    echo ""
    echo "Error: environment.yml not found at ${ENV_FILE}"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "Environment '${ENV_NAME}' already exists."
    echo ""
    read -p "Remove and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo ""
        echo "Skipping environment creation."
        echo "To activate the existing environment, run:"
        echo "  conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create environment from yml file
print_step "Creating conda environment '${ENV_NAME}'..."
echo "This may take 10-30 minutes depending on your internet connection."
echo ""

# Create a temporary yml file without the prefix line (for portability)
# Note: Must have .yml extension for conda to accept it
TEMP_DIR=$(mktemp -d)
TEMP_ENV_FILE="${TEMP_DIR}/environment.yml"
grep -v "^prefix:" "${ENV_FILE}" > "${TEMP_ENV_FILE}"

# Create the environment
conda env create -f "${TEMP_ENV_FILE}" -n "${ENV_NAME}"

# Clean up temp files
rm -rf "${TEMP_DIR}"

# Verify installation
print_step "Verifying installation..."

# Source conda for this shell session
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Activate environment
conda activate "${ENV_NAME}"

# Check key packages
python -c "
import sys
print(f'Python version: {sys.version}')

packages = [
    ('torch', 'PyTorch'),
    ('monai', 'MONAI'),
    ('tifffile', 'tifffile'),
    ('numpy', 'NumPy'),
    ('napari_cellseg3d', 'CellSeg3D'),
]

print('')
print('Checking installed packages:')
missing = []
for module, name in packages:
    try:
        pkg = __import__(module)
        version = getattr(pkg, '__version__', 'installed')
        print(f'  [OK] {name}: {version}')
    except ImportError:
        print(f'  [MISSING] {name}')
        missing.append(name)

if missing:
    print('')
    print(f'Warning: {len(missing)} package(s) not found.')
    print('You may need to install MONAI and CellSeg3D manually:')
    print('  pip install -e ./monai')
    print('  pip install -e ./cellseg3d')
else:
    print('')
    print('All key packages installed successfully!')
"

# Check for CUDA
print_step "Checking CUDA availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA is available!')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU count: {torch.cuda.device_count()}')
else:
    print('  CUDA is NOT available.')
    print('  Training will be slow without GPU acceleration.')
"

print_header "Environment Setup Complete"
echo ""
echo "To activate this environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Then proceed to the next step:"
echo "  ./02_download_dataset.sh"
echo ""
