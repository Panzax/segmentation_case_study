#!/usr/bin/env python3
"""
Lightweight inference script for Swin-UNETR model.

Bypasses the complex inference infrastructure and directly:
1. Loads DeepSpeed ZeRO checkpoint
2. Reads input Zarr files
3. Performs sliding window inference
4. Saves predictions as NumPy arrays
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cell_observatory_finetune.models.meta_arch.swin_unetr import FinetuneSwinUNETR
from cell_observatory_platform.data.io import read_zarr

from monai.inferers import sliding_window_inference as monai_sliding_window_inference


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with Swin-UNETR model on Zarr input files"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (parent of latest_model)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Zarr file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prediction .npy file(s)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=128,
        help="Sliding window size (default: 128)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Stride for sliding window (default: 64 for 50%% overlap)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--checkpoint_tag",
        type=str,
        default="latest_model",
        help="Checkpoint tag to load (default: latest_model)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    return parser.parse_args()


def load_model(checkpoint_dir: str, checkpoint_tag: str, device: str) -> FinetuneSwinUNETR:
    """
    Initialize and load Swin-UNETR model from checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        checkpoint_tag: Checkpoint tag (e.g., 'latest_model')
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print("Initializing model...")
    model = FinetuneSwinUNETR(
        task="boundary_segmentation",
        output_channels=1,
        model_template="swin-unetr-base",
        input_fmt="ZYXC",
        input_shape=(128, 128, 128, 2),  # Z, Y, X, C
        patch_shape=(2, 2, 2, None),
        feature_size=48,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        qkv_bias=True,
        mlp_ratio=4.0,
        mlp_type="Mlp",
        norm_name="layer",
        drop_rate=0.5,
        attn_drop_rate=0.5,
        dropout_path_rate=0.0,
        normalize=True,
        patch_norm=False,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=True,
        loss_fn="generalized_dice",
    )
    
    print(f"Loading checkpoint from {checkpoint_dir} (tag: {checkpoint_tag})...")
    model.load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        tag=checkpoint_tag,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
        load_module_only=True,
    )
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    
    return model


def load_zarr_data(zarr_path: str) -> np.ndarray:
    """
    Load Zarr file as NumPy array.
    
    Args:
        zarr_path: Path to Zarr file
        
    Returns:
        NumPy array in ZYXC format (image channels only, mask dropped)
    """
    print(f"Loading Zarr file from {zarr_path}...")
    zarr_handle = read_zarr(zarr_path, dtype="float16")
    
    # Read the data
    if hasattr(zarr_handle, "read"):
        # Tensorstore handle
        data = zarr_handle.read().result()
    else:
        # Already a numpy array
        data = zarr_handle
    
    print(f"Loaded data shape: {data.shape}")
    
    # Ensure ZYXC format
    if data.ndim == 5:  # TZYXC
        data = data[0]  # Take first timepoint -> ZYXC
        print(f"Converted from TZYXC to ZYXC, new shape: {data.shape}")
    elif data.ndim == 4:  # ZYXC
        pass  # Already correct
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}, expected ZYXC or TZYXC")
    
    # Drop last channel (mask), keep only image channels
    if data.shape[-1] != 2:
        raise ValueError(
            f"Expected 2 channels (image + mask), got {data.shape[-1]}"
        )
    image = data[..., :-1]          # e.g. from (Z, Y, X, 2) -> (Z, Y, X, 1)
    print(f"Using image channels shape (mask dropped): {image.shape}")
    
    # Normalize image volume (global mean/std), like training Normalize wrapper
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    eps = 1e-4
    if std < eps:
        std = eps
    image = (image - mean) / std
    print(f"Applied mean/std normalization: mean={float(mean):.4f}, std={float(std):.4f}")
    
    return image


def sliding_window_inference(
    model: FinetuneSwinUNETR,
    data: np.ndarray,
    window_size: int,
    stride: int,
    device: str,
    batch_size: int = 1,
) -> np.ndarray:
    """
    Perform sliding window inference on 3D volume using MONAI's implementation
    with Gaussian blending to avoid grid artifacts.

    Args:
        model: Loaded model
        data: Input data in ZYXC format (Z, Y, X, C)
        window_size: Size of sliding window (Z=Y=X=window_size)
        stride: Stride for sliding window (controls overlap)
        device: Device to run inference on
        batch_size: Sliding-window batch size

    Returns:
        Predictions in ZYXC format (Z, Y, X, C_out)
    """
    Z, Y, X, C = data.shape
    if C != 1:
        raise ValueError(f"Expected 1 input channel after mask drop, got {C}")

    # MONAI expects NCHWD: (N, C, Z, Y, X)
    # Keep the large input volume on CPU; MONAI will move only window patches
    # to `sw_device` (GPU) for prediction. This dramatically reduces peak GPU
    # memory compared to storing the whole output/count maps on GPU.
    inputs = (
        torch.from_numpy(data)
        .permute(3, 0, 1, 2)  # (C, Z, Y, X)
        .unsqueeze(0)  # (1, C, Z, Y, X)
        .float()
    )

    # Convert stride -> overlap fraction (MONAI uses fractional overlap)
    if stride <= 0 or stride > window_size:
        raise ValueError(f"stride must be in (0, window_size], got stride={stride}, window_size={window_size}")
    overlap = 1.0 - float(stride) / float(window_size)

    roi_size = (window_size, window_size, window_size)
    sw_batch_size = batch_size

    def predictor(patch: torch.Tensor) -> torch.Tensor:
        """
        MONAI passes patch as (N, C, Z, Y, X).
        Our model.predict expects (N, Z, Y, X, C).
        """
        # (N, C, Z, Y, X) -> (N, Z, Y, X, C)
        patch_zyxc = patch.permute(0, 2, 3, 4, 1)
        out_zyxc = model.predict({"data_tensor": patch_zyxc})  # (N, Z, Y, X, C_out)
        # Back to (N, C_out, Z, Y, X) for MONAI stitching
        out_ncz = out_zyxc.permute(0, 4, 1, 2, 3)
        return out_ncz

    print(
        f"Running MONAI sliding_window_inference with roi_size={roi_size}, "
        f"overlap={overlap:.2f}, sw_batch_size={sw_batch_size}, mode='gaussian'"
    )

    # Use MONAI's Gaussian blending to avoid grid artifacts.
    # - inputs and stitched outputs live on CPU (`device="cpu"`)
    # - window patches are processed on GPU (`sw_device=device`)
    outputs_ncz = monai_sliding_window_inference(
        inputs=inputs,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
        mode="gaussian",   # key: smooth blending at patch edges
        progress=True,
        sw_device=device,
        device="cpu",
    )  # (1, C_out, Z, Y, X)

    # (1, C_out, Z, Y, X) -> (Z, Y, X, C_out)
    outputs_zyxc = outputs_ncz[0].permute(1, 2, 3, 0).cpu().numpy()
    return outputs_zyxc


def save_predictions(predictions: np.ndarray, output_path: str):
    """
    Save predictions as NumPy array.
    
    Args:
        predictions: Predictions array in ZYXC format
        output_path: Path to save .npy file
    """
    print(f"Saving predictions to {output_path}...")
    print(f"Output shape: {predictions.shape}")
    np.save(output_path, predictions)
    print("Predictions saved successfully")


def main():
    """Main inference pipeline."""
    args = parse_args()
    
    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Zarr file not found: {input_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(
        checkpoint_dir=str(checkpoint_path),
        checkpoint_tag=args.checkpoint_tag,
        device=args.device,
    )
    
    # Load data
    data = load_zarr_data(str(input_path))
    
    # Run inference
    predictions = sliding_window_inference(
        model=model,
        data=data,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # Construct output filename based on input name and current datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = input_path.stem  # e.g. '000x_000y_000z'
    output_path = output_dir / f"{base_name}_pred_{timestamp}.npy"
    
    # Save predictions
    save_predictions(predictions, str(output_path))
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()

