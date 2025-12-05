#!/bin/bash
# Setup script to install local editable MONAI and CellSeg3D in Apptainer container
#
# This script should be run inside your Apptainer container to ensure
# that the local editable versions of MONAI and CellSeg3D are used.
#
# Usage:
#   apptainer exec --nv --bind /clusterfs/nvme:/clusterfs/nvme your_image.sif \
#     bash /clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/scripts/setup_cellseg3d_env.sh

set -eux

MONAI_PATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/monai"
CELLSEG3D_PATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/cellseg3d"

echo "=========================================="
echo "Setting up CellSeg3D environment"
echo "=========================================="

# Check paths exist
if [ ! -d "$MONAI_PATH" ]; then
    echo "ERROR: MONAI path does not exist: $MONAI_PATH"
    exit 1
fi

if [ ! -d "$CELLSEG3D_PATH" ]; then
    echo "ERROR: CellSeg3D path does not exist: $CELLSEG3D_PATH"
    exit 1
fi

# Install MONAI in editable mode
echo ""
echo "Installing MONAI in editable mode..."
cd "$MONAI_PATH"
pip install -e . --no-deps || pip install -e .

# Install CellSeg3D in editable mode
echo ""
echo "Installing CellSeg3D in editable mode..."
cd "$CELLSEG3D_PATH"
pip install -e . --no-deps || pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python /clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/scripts/cellseg3d/verify_cellseg3d_setup.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="

