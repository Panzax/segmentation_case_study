#!/usr/bin/env python3
"""
Verify that CellSeg3D and MONAI are properly installed and accessible.

This script checks that:
1. MONAI can be imported and points to the local editable install
2. CellSeg3D (napari_cellseg3d) can be imported and points to the local editable install
3. Key components (SupervisedTrainingWorker, configs) are accessible
4. Model list includes SwinUNETR

Run this inside your Apptainer container after installing editable packages.
"""

import sys
from pathlib import Path

print("=" * 60)
print("CellSeg3D Setup Verification")
print("=" * 60)

# Check MONAI
print("\n1. Checking MONAI installation...")
try:
    import monai

    monai_path = Path(monai.__file__).resolve()
    print(f"   ✓ MONAI imported successfully")
    print(f"   Location: {monai_path}")
    print(f"   Version: {getattr(monai, '__version__', 'unknown')}")

    # Check if it's the local editable install
    expected_monai_path = Path("/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/monai")
    if str(monai_path).startswith(str(expected_monai_path)):
        print(f"   ✓ MONAI is from local editable install")
    else:
        print(f"   ⚠ MONAI may not be from local editable install")
        print(f"   Expected path to contain: {expected_monai_path}")
except ImportError as e:
    print(f"   ✗ Failed to import MONAI: {e}")
    sys.exit(1)

# Check CellSeg3D
print("\n2. Checking CellSeg3D (napari_cellseg3d) installation...")
try:
    import napari_cellseg3d

    cellseg3d_path = Path(napari_cellseg3d.__file__).resolve()
    print(f"   ✓ napari_cellseg3d imported successfully")
    print(f"   Location: {cellseg3d_path}")

    # Check if it's the local editable install
    expected_cellseg3d_path = Path("/clusterfs/nvme/martinalvarez/GitHub/CellSeg3D")
    if str(cellseg3d_path).startswith(str(expected_cellseg3d_path)):
        print(f"   ✓ CellSeg3D is from local editable install")
    else:
        print(f"   ⚠ CellSeg3D may not be from local editable install")
        print(f"   Expected path to contain: {expected_cellseg3d_path}")
except ImportError as e:
    print(f"   ✗ Failed to import napari_cellseg3d: {e}")
    print("   Make sure CellSeg3D is installed in editable mode:")
    print("     pip install -e /clusterfs/nvme/martinalvarez/GitHub/CellSeg3D")
    sys.exit(1)

# Check key components
print("\n3. Checking key CellSeg3D components...")
try:
    from napari_cellseg3d import config
    from napari_cellseg3d.code_models.worker_training import SupervisedTrainingWorker
    from napari_cellseg3d.code_models.models.model_SwinUNetR import SwinUNETR_

    print("   ✓ config imported")
    print("   ✓ SupervisedTrainingWorker imported")
    print("   ✓ SwinUNETR_ imported")
except ImportError as e:
    print(f"   ✗ Failed to import key components: {e}")
    sys.exit(1)

# Check model list
print("\n4. Checking available models...")
try:
    model_list = config.MODEL_LIST
    print(f"   ✓ Found {len(model_list)} models in MODEL_LIST:")
    for model_name in sorted(model_list.keys()):
        print(f"     - {model_name}")

    if "SwinUNetR" in model_list:
        print("   ✓ SwinUNetR is available")
    else:
        print("   ✗ SwinUNetR not found in MODEL_LIST")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error checking model list: {e}")
    sys.exit(1)

# Check config classes
print("\n5. Checking config classes...")
try:
    model_info = config.ModelInfo(name="SwinUNetR")
    weights_info = config.WeightsInfo()
    det_config = config.DeterministicConfig()
    train_config = config.SupervisedTrainingWorkerConfig()

    print("   ✓ ModelInfo created")
    print("   ✓ WeightsInfo created")
    print("   ✓ DeterministicConfig created")
    print("   ✓ SupervisedTrainingWorkerConfig created")
except Exception as e:
    print(f"   ✗ Error creating config objects: {e}")
    sys.exit(1)

# Check MONAI components used by CellSeg3D
print("\n6. Checking MONAI components used by CellSeg3D...")
try:
    from monai.data import CacheDataset, DataLoader, PatchDataset
    from monai.transforms import (
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        SpatialPadd,
    )
    from monai.losses import DiceCELoss, DiceLoss
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference

    print("   ✓ MONAI data components imported")
    print("   ✓ MONAI transforms imported")
    print("   ✓ MONAI losses imported")
    print("   ✓ MONAI metrics imported")
    print("   ✓ MONAI inferers imported")
except ImportError as e:
    print(f"   ✗ Failed to import MONAI components: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All checks passed! CellSeg3D setup is correct.")
print("=" * 60)

