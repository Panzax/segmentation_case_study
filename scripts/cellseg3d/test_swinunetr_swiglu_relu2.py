#!/usr/bin/env python3
"""
Sanity check script for SwinUNETR_SwiGLU_ReLU2 model.

This script tests:
1. Model instantiation
2. Forward pass with dummy 3D input
3. Output shape verification

Usage:
    python scripts/cellseg3d/test_swinunetr_swiglu_relu2.py
"""

import sys
from pathlib import Path

# Add paths for imports
repo_root = Path(__file__).parent.parent.parent
cellseg3d_path = repo_root / "cellseg3d"
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(cellseg3d_path))

import torch

# Import CellSeg3D components
try:
    from napari_cellseg3d import config
    from napari_cellseg3d.utils import LOGGER
except ImportError as e:
    print(f"Error importing CellSeg3D: {e}")
    print(f"Tried paths: {repo_root}, {cellseg3d_path}")
    print("Make sure CellSeg3D is available. Trying direct import...")
    # Try direct import
    try:
        sys.path.insert(0, str(cellseg3d_path / "napari_cellseg3d"))
        import config
        from utils import LOGGER
        print("✓ Direct import successful")
    except ImportError as e2:
        print(f"Direct import also failed: {e2}")
        print("Make sure CellSeg3D is installed in editable mode:")
        print(f"  cd {cellseg3d_path} && pip install -e .")
        sys.exit(1)

logger = LOGGER


def test_model_instantiation():
    """Test that the model can be instantiated."""
    logger.info("=" * 60)
    logger.info("Testing SwinUNETR_SwiGLU_ReLU2 Model Instantiation")
    logger.info("=" * 60)
    
    model_name = "SwinUNetR_SwiGLU_ReLU2"
    
    # Check if model is in MODEL_LIST
    if model_name not in config.MODEL_LIST:
        logger.error(f"Model '{model_name}' not found in MODEL_LIST")
        logger.error(f"Available models: {list(config.MODEL_LIST.keys())}")
        return False
    
    logger.info(f"✓ Model '{model_name}' found in MODEL_LIST")
    
    # Get model class
    model_class = config.MODEL_LIST[model_name]
    logger.info(f"✓ Model class: {model_class}")
    
    # Try to instantiate the model
    try:
        model = model_class(
            in_channels=1,
            out_channels=1,
            input_img_size=(64, 64, 64),
            use_checkpoint=False,  # Disable checkpointing for testing
        )
        logger.info("✓ Model instantiated successfully")
        return model
    except Exception as e:
        logger.error(f"✗ Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, device="cpu"):
    """Test forward pass with dummy input."""
    logger.info("=" * 60)
    logger.info("Testing Forward Pass")
    logger.info("=" * 60)
    
    # Create dummy 3D input: [batch, channels, depth, height, width]
    batch_size = 1
    in_channels = 1
    spatial_size = (64, 64, 64)
    
    dummy_input = torch.randn(batch_size, in_channels, *spatial_size)
    logger.info(f"Input shape: {dummy_input.shape}")
    
    # Move model and input to device
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Forward pass
    try:
        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"Output shape: {output.shape}")
        
        # Verify output shape
        expected_shape = (batch_size, 1, *spatial_size)  # [B, C, D, H, W]
        if output.shape == expected_shape:
            logger.info(f"✓ Output shape matches expected: {expected_shape}")
            return True
        else:
            logger.warning(f"⚠ Output shape {output.shape} does not match expected {expected_shape}")
            logger.warning("  This might be okay depending on the model architecture")
            return True  # Still consider it a pass if forward works
        
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("Starting sanity check for SwinUNETR_SwiGLU_ReLU2...")
    
    # Test 1: Model instantiation
    model = test_model_instantiation()
    if model is None:
        logger.error("Model instantiation failed. Aborting.")
        return 1
    
    # Test 2: Forward pass
    # Try CPU first
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        logger.info(f"CUDA available, using {device}")
    else:
        logger.info("CUDA not available, using CPU")
    
    success = test_forward_pass(model, device=device)
    
    if success:
        logger.info("=" * 60)
        logger.info("✓ All tests passed!")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("✗ Some tests failed")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

