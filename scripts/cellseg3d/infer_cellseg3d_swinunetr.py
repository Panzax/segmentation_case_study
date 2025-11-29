#!/usr/bin/env python3
"""
Inference script for SwinUNETR using CellSeg3D's InferenceWorker.

This script uses CellSeg3D's existing inference pipeline (InferenceWorker)
to run inference on trained SwinUNETR models. It can be run inside an
Apptainer container with local editable installs of MONAI and CellSeg3D.

Usage:
    python infer_cellseg3d_swinunetr.py \
        --checkpoint /path/to/checkpoint.pth \
        --images_dir /path/to/images \
        --output_dir /path/to/output \
        [--config config.yaml]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CellSeg3D components
try:
    from napari_cellseg3d import config
    from napari_cellseg3d.code_models.worker_inference import InferenceWorker
    from napari_cellseg3d.utils import LOGGER
except ImportError as e:
    print(f"Error importing CellSeg3D: {e}")
    print("Make sure CellSeg3D is installed in editable mode:")
    print("  pip install -e /path/to/CellSeg3D")
    sys.exit(1)

logger = LOGGER


def get_image_filepaths(images_dir: Path, file_extensions: Optional[List[str]] = None) -> List[str]:
    """
    Get list of image file paths from directory.

    Args:
        images_dir: Directory containing image volumes
        file_extensions: List of file extensions to match

    Returns:
        List of image file paths
    """
    if file_extensions is None:
        file_extensions = [".tif", ".tiff", ".nii", ".nii.gz"]

    image_files = []
    for ext in file_extensions:
        image_files.extend([str(f) for f in images_dir.glob(f"*{ext}")])

    if not image_files:
        raise ValueError(
            f"No image files found in {images_dir} with extensions {file_extensions}"
        )

    # Sort for consistent ordering
    image_files = sorted(image_files)
    logger.info(f"Found {len(image_files)} image files")
    return image_files


def create_inference_config(
    checkpoint_path: str,
    images_filepaths: List[str],
    output_dir: Path,
    model_name: str = "SwinUNetR",
    device: str = "cuda:0",
    window_size: Optional[int] = 64,
    window_overlap: float = 0.25,
    model_input_size: Optional[int] = None,
    filetype: str = ".tif",
    keep_on_cpu: bool = False,
    compute_stats: bool = False,
    thresholding_enabled: bool = False,
    threshold_value: float = 0.5,
    instance_segmentation_enabled: bool = False,
    use_crf: bool = False,
) -> config.InferenceWorkerConfig:
    """
    Create an InferenceWorkerConfig with specified parameters.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        images_filepaths: List of image file paths to process
        output_dir: Directory to save results
        model_name: Name of the model (must be in config.MODEL_LIST)
        device: Device to use ('cuda:0', 'cpu', etc.)
        window_size: Sliding window size (None to disable sliding window)
        window_overlap: Sliding window overlap fraction
        model_input_size: Model input size (if None, uses model default)
        filetype: File extension for saved results
        keep_on_cpu: Keep results on CPU
        compute_stats: Compute cell statistics
        thresholding_enabled: Enable thresholding post-processing
        threshold_value: Threshold value for post-processing
        instance_segmentation_enabled: Enable instance segmentation
        use_crf: Use CRF post-processing

    Returns:
        Configured InferenceWorkerConfig
    """
    # Create model info
    model_info = config.ModelInfo(name=model_name)
    if model_input_size is not None:
        model_info.model_input_size = [model_input_size, model_input_size, model_input_size]

    # Create weights info
    weights_info = config.WeightsInfo(
        use_custom=True,
        path=checkpoint_path,
    )

    # Create sliding window config
    if window_size is not None:
        sliding_window_config = config.SlidingWindowConfig(
            window_size=window_size,
            window_overlap=window_overlap,
        )
    else:
        sliding_window_config = config.SlidingWindowConfig()

    # Create post-processing config
    post_process_config = config.PostProcessConfig()
    post_process_config.thresholding.enabled = thresholding_enabled
    post_process_config.thresholding.threshold_value = threshold_value
    post_process_config.instance.enabled = instance_segmentation_enabled

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create inference config
    worker_config = config.InferenceWorkerConfig(
        device=device,
        model_info=model_info,
        weights_config=weights_info,
        images_filepaths=images_filepaths,
        results_path=str(output_dir),
        filetype=filetype,
        keep_on_cpu=keep_on_cpu,
        compute_stats=compute_stats,
        post_process_config=post_process_config,
        sliding_window_config=sliding_window_config,
        use_crf=use_crf,
    )

    return worker_config


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with SwinUNETR using CellSeg3D's InferenceWorker"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing image volumes to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SwinUNetR",
        help="Model name (must be in CellSeg3D MODEL_LIST)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Sliding window size (set to 0 to disable)",
    )
    parser.add_argument(
        "--window_overlap",
        type=float,
        default=0.25,
        help="Sliding window overlap fraction",
    )
    parser.add_argument(
        "--model_input_size",
        type=int,
        default=None,
        help="Model input size (if None, uses model default)",
    )
    parser.add_argument(
        "--filetype",
        type=str,
        default=".tif",
        help="File extension for saved results",
    )
    parser.add_argument(
        "--keep_on_cpu",
        action="store_true",
        help="Keep results on CPU",
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help="Compute cell statistics",
    )
    parser.add_argument(
        "--thresholding",
        action="store_true",
        help="Enable thresholding post-processing",
    )
    parser.add_argument(
        "--threshold_value",
        type=float,
        default=0.5,
        help="Threshold value for post-processing",
    )
    parser.add_argument(
        "--instance_segmentation",
        action="store_true",
        help="Enable instance segmentation",
    )
    parser.add_argument(
        "--use_crf",
        action="store_true",
        help="Use CRF post-processing",
    )

    args = parser.parse_args()

    # Convert paths to Path objects
    checkpoint_path = Path(args.checkpoint)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    # Validate paths
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")

    logger.info("=" * 60)
    logger.info("CellSeg3D SwinUNETR Inference")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    if args.window_size > 0:
        logger.info(f"Sliding window: {args.window_size} (overlap: {args.window_overlap})")
    else:
        logger.info("Sliding window: disabled")
    logger.info("=" * 60)

    # Get image filepaths
    logger.info("Collecting image files...")
    images_filepaths = get_image_filepaths(images_dir)

    # Create inference config
    logger.info("Creating inference configuration...")
    worker_config = create_inference_config(
        checkpoint_path=str(checkpoint_path),
        images_filepaths=images_filepaths,
        output_dir=output_dir,
        model_name=args.model_name,
        device=args.device,
        window_size=args.window_size if args.window_size > 0 else None,
        window_overlap=args.window_overlap,
        model_input_size=args.model_input_size,
        filetype=args.filetype,
        keep_on_cpu=args.keep_on_cpu,
        compute_stats=args.compute_stats,
        thresholding_enabled=args.thresholding,
        threshold_value=args.threshold_value,
        instance_segmentation_enabled=args.instance_segmentation,
        use_crf=args.use_crf,
    )

    # Create and run inference worker
    logger.info("Initializing inference worker...")
    worker = InferenceWorker(worker_config=worker_config)

    logger.info("Starting inference...")
    logger.info("=" * 60)

    # Run inference (iterate over generator)
    try:
        results = []
        for result in worker.inference():
            results.append(result)
            logger.info(f"Processed image {result.image_id + 1}/{len(images_filepaths)}")
    except KeyboardInterrupt:
        logger.warning("Inference interrupted by user")
    except Exception as e:
        logger.error(f"Inference failed with error: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("Inference completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Processed {len(results)} images")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()