#!/usr/bin/env python3
"""
Training script for SwinUNETR using CellSeg3D's SupervisedTrainingWorker.

This script uses CellSeg3D's existing training pipeline (SupervisedTrainingWorker)
to train SwinUNETR models on CellSeg3D datasets. It can be run inside an
Apptainer container with local editable installs of MONAI and CellSeg3D.

Usage:
    python train_cellseg3d_swinunetr.py \
        --images_dir /path/to/images \
        --labels_dir /path/to/labels \
        --output_dir /path/to/output \
        [--config config.yaml]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CellSeg3D components
try:
    from napari_cellseg3d import config
    from napari_cellseg3d.code_models.worker_training import SupervisedTrainingWorker
    from napari_cellseg3d.utils import LOGGER, get_padding_dim
except ImportError as e:
    print(f"Error importing CellSeg3D: {e}")
    print("Make sure CellSeg3D is installed in editable mode:")
    print("  pip install -e /path/to/CellSeg3D")
    sys.exit(1)

logger = LOGGER


def create_train_dataset_dict(
    images_dir: Path,
    labels_dir: Path,
    file_extensions: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> List[dict]:
    """
    Create a training dataset dictionary compatible with CellSeg3D's format.

    Args:
        images_dir: Directory containing image volumes
        labels_dir: Directory containing label volumes
        file_extensions: List of file extensions to match (e.g., ['.tif', '.tiff', '.nii.gz'])
        exclude: List of strings/regexes to exclude from the dataset
        
    Returns:
        List of dicts with 'image' and 'label' keys, matching CellSeg3D's format
    """
    if file_extensions is None:
        file_extensions = [".tif", ".tiff", ".nii", ".nii.gz"]

    # Get all image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    if exclude is not None:
        image_files = [f for f in image_files if not any(re.match(pattern, f.name) for pattern in exclude)]
        logger.info(f"Excluded {len(image_files)} files from the dataset")
        for f in image_files:
            logger.info(f"  {f.name}")

    if not image_files:
        raise ValueError(
            f"No image files found in {images_dir} with extensions {file_extensions}"
        )

    # Sort to ensure consistent ordering
    image_files = sorted(image_files)

    # Match labels to images (assuming matching filenames)
    data_dicts = []
    for img_path in image_files:
        # Try to find matching label file
        label_path = None
        
        # Strategy 1: Try exact stem match (original behavior)
        for ext in file_extensions:
            candidate = labels_dir / (img_path.stem + ext)
            if candidate.exists():
                label_path = candidate
                break
        
        # Strategy 2: Try replacing "image" with "label" in stem
        if label_path is None:
            label_stem = img_path.stem.replace("image", "label")
            for ext in file_extensions:
                candidate = labels_dir / (label_stem + ext)
                if candidate.exists():
                    label_path = candidate
                    break
        
        # Strategy 3: Try numeric ID matching (e.g., c1image -> c1label)
        if label_path is None:
            # Strategy 3: Try numeric ID matching (e.g., c1image -> c1label)
            stem = img_path.stem
            img_id = "".join(filter(str.isdigit, stem))
            if img_id:
                for lab_file in labels_dir.iterdir():
                    if (
                        lab_file.is_file()
                        and lab_file.suffix in file_extensions
                    ):
                        lab_id = "".join(filter(str.isdigit, lab_file.stem))
                        if img_id == lab_id:
                            label_path = lab_file
                            break
        
        if label_path is None:
            logger.warning(f"No matching label found for {img_path.name}, skipping")
            continue        
        
        data_dicts.append({"image": str(img_path), "label": str(label_path)})

    if not data_dicts:
        raise ValueError(
            f"No matching image-label pairs found between {images_dir} and {labels_dir}"
        )

    logger.info(f"Created dataset dict with {len(data_dicts)} image-label pairs")
    for i, d in enumerate(data_dicts[:5]):  # Log first 5
        logger.info(f"  {i+1}: {Path(d['image']).name} <-> {Path(d['label']).name}")
    if len(data_dicts) > 5:
        logger.info(f"  ... and {len(data_dicts) - 5} more")

    return data_dicts


def create_training_config(
    train_data_dict: List[dict],
    val_data_dict: Optional[List[dict]],
    output_dir: Path,
    model_name: str = "SwinUNetR",
    model_feature_size: int = 24,
    model_depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
    device: str = "cuda:0",
    max_epochs: int = 50,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    training_percent: float = 0.8,
    validation_interval: int = 2,
    sampling: bool = True,
    num_samples: int = 2,
    sample_size: Optional[List[int]] = None,
    do_augmentation: bool = True,
    num_workers: int = 4,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 10,
    loss_function: str = "DiceCE",
    use_pretrained: bool = False,
    use_custom_weights: bool = False,
    custom_weights_path: Optional[str] = None,
    deterministic: bool = True,
    seed: int = 34936339,
    downsample_zoom: Optional[List[float]] = None,
) -> config.SupervisedTrainingWorkerConfig:
    """
    Create a SupervisedTrainingWorkerConfig with specified parameters.

    Args:
        train_data_dict: Dataset dictionary from create_train_dataset_dict
        val_data_dict: Optional dataset dictionary for explicit validation set.
            When provided, this is used as the validation set instead of
            creating an internal split from train_data_dict.
        output_dir: Directory to save checkpoints and logs
        model_name: Name of the model (must be in config.MODEL_LIST)
        device: Device to use ('cuda:0', 'cpu', etc.)
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        training_percent: Fraction of data to use for training (rest for validation)
        validation_interval: Validate every N epochs
        sampling: Whether to use patch-based sampling
        num_samples: Number of patches per image (if sampling=True)
        sample_size: Size of patches [Z, Y, X] (if sampling=True)
        do_augmentation: Whether to apply data augmentation
        num_workers: Number of data loader workers
        scheduler_factor: Factor for ReduceLROnPlateau scheduler
        scheduler_patience: Patience for ReduceLROnPlateau scheduler
        loss_function: Loss function name ('Dice', 'DiceCE', 'Generalized Dice', 'Tversky')
        use_pretrained: Whether to use pretrained weights from CellSeg3D
        use_custom_weights: Whether to use custom weights
        custom_weights_path: Path to custom weights file (if use_custom_weights=True)
        deterministic: Whether to use deterministic training
        seed: Random seed (if deterministic=True)
        downsample_zoom: Optional downsampling factors [Z, Y, X] applied before patch
            extraction/padding. For example, [1.0, 0.5, 0.5] keeps Z unchanged and
            downsamples Y and X by 2x. When None, no downsampling is applied.

    Returns:
        Configured SupervisedTrainingWorkerConfig
    """
    if sample_size is None:
        sample_size = [64, 64, 64]

    # Create model info
    model_kwargs = {}
    if model_feature_size is not None:
        # Forwarded to SwinUNETR wrappers, which in turn pass it to MONAI
        # SwinUNETR. This lets us sweep model size without touching the
        # worker logic.
        model_kwargs["feature_size"] = model_feature_size
    if model_depths is not None:
        model_kwargs["depths"] = model_depths

    model_info = config.ModelInfo(name=model_name, model_kwargs=model_kwargs)

    # Create weights info
    weights_info = config.WeightsInfo(
        use_pretrained=use_pretrained,
        use_custom=use_custom_weights,
        path=custom_weights_path if use_custom_weights else None,
    )

    # Create deterministic config
    det_config = config.DeterministicConfig(enabled=deterministic, seed=seed)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create supervised training config
    worker_config = config.SupervisedTrainingWorkerConfig(
        device=device,
        model_info=model_info,
        weights_info=weights_info,
        train_data_dict=train_data_dict,
        val_data_dict=val_data_dict,
        training_percent=training_percent,
        max_epochs=max_epochs,
        loss_function=loss_function,
        learning_rate=np.float64(learning_rate),
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        validation_interval=validation_interval,
        batch_size=batch_size,
        results_path_folder=str(output_dir),
        sampling=sampling,
        num_samples=num_samples,
        sample_size=sample_size,
        do_augmentation=do_augmentation,
        num_workers=num_workers,
        deterministic_config=det_config,
        downsample_zoom=downsample_zoom,
    )

    return worker_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train SwinUNETR using CellSeg3D's SupervisedTrainingWorker"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing image volumes",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Directory containing label volumes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--val_images_dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing validation image volumes. "
            "If set, this directory is used for validation and "
            "--images_dir is used for training."
        ),
    )
    parser.add_argument(
        "--val_labels_dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing validation label volumes. "
            "Must be provided when --val_images_dir is set."
        ),
    )
    parser.add_argument(
        "--model_name",
        "--model",
        dest="model_name",
        type=str,
        default="SwinUNetR",
        help="Model name (must be in CellSeg3D MODEL_LIST)",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=None,
        help=(
            "Base feature size for SwinUNETR variants "
            "(e.g. 24, 48, 96) to sweep model capacity."
        ),
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar="DEPTH",
        help="Depth of each stage in the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cpu, etc.)",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--training_percent",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest for validation)",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=2,
        help="Validate every N epochs",
    )
    parser.add_argument(
        "--sampling",
        action="store_true",
        default=True,
        help="Use patch-based sampling",
    )
    parser.add_argument(
        "--no_sampling",
        dest="sampling",
        action="store_false",
        help="Use full volumes (no patch sampling)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of patches per image (if sampling=True)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        metavar=("Z", "Y", "X"),
        help="Patch size [Z Y X] (if sampling=True)",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        default=True,
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--no_augmentation",
        dest="augmentation",
        action="store_false",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="Generalized Dice",
        choices=["Dice", "Generalized Dice", "DiceCE", "Tversky"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use pretrained weights from CellSeg3D",
    )
    parser.add_argument(
        "--custom_weights",
        type=str,
        default=None,
        help="Path to custom weights file",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=34936339,
        help="Random seed (if deterministic=True)",
    )
    parser.add_argument(
        "--downsample_zoom",
        type=float,
        nargs="+",
        default=None,
        metavar="FACTOR",
        help=(
            "Optional downsampling factors [Z Y X] applied before patch extraction/padding. "
            "For example, '1.0 0.5 0.5' keeps Z unchanged and downsamples Y and X by 2x. "
            "Can also provide a single value to apply to all axes. "
            "When not provided, no downsampling is applied."
        ),
    )
    # Internal validation is always enabled; explicit validation sets can be
    # provided via --val_images_dir/--val_labels_dir.

    args = parser.parse_args()

    # Convert paths to Path objects
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    val_images_dir = Path(args.val_images_dir) if args.val_images_dir else None
    val_labels_dir = Path(args.val_labels_dir) if args.val_labels_dir else None

    # Validate paths
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory does not exist: {labels_dir}")
    if (val_images_dir is not None) ^ (val_labels_dir is not None):
        raise ValueError(
            "Both --val_images_dir and --val_labels_dir must be provided "
            "together when specifying an explicit validation set."
        )
    if val_images_dir is not None and not val_images_dir.exists():
        raise ValueError(
            f"Validation images directory does not exist: {val_images_dir}"
        )
    if val_labels_dir is not None and not val_labels_dir.exists():
        raise ValueError(
            f"Validation labels directory does not exist: {val_labels_dir}"
        )

    logger.info("=" * 60)
    logger.info("CellSeg3D SwinUNETR Training")
    logger.info("=" * 60)
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Labels directory: {labels_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Training percent: {args.training_percent}")
    logger.info(f"Sampling: {args.sampling}")
    if args.sampling:
        logger.info(f"  Num samples: {args.num_samples}")
        logger.info(f"  Sample size: {args.sample_size}")
    logger.info(f"Augmentation: {args.augmentation}")
    logger.info(f"Loss function: {args.loss_function}")
    if val_images_dir is not None:
        logger.info(f"Validation images directory: {val_images_dir}")
        logger.info(f"Validation labels directory: {val_labels_dir}")

    # Parse downsample_zoom argument
    downsample_zoom = None
    if args.downsample_zoom is not None:
        if len(args.downsample_zoom) == 1:
            # Single value applied to all axes
            downsample_zoom = [args.downsample_zoom[0]] * 3
        elif len(args.downsample_zoom) == 3:
            downsample_zoom = list(args.downsample_zoom)
        else:
            raise ValueError(
                f"downsample_zoom must be 1 or 3 values, got {len(args.downsample_zoom)}"
            )
        logger.info(f"Downsample zoom: {downsample_zoom} [Z, Y, X]")
    else:
        logger.info("Downsample zoom: None (no downsampling)")

    logger.info("=" * 60)

    # Create dataset dictionary
    logger.info("Creating dataset dictionary...")
    train_data_dict = create_train_dataset_dict(images_dir, labels_dir)

    val_data_dict: Optional[List[dict]] = None
    if val_images_dir is not None and val_labels_dir is not None:
        logger.info("Creating validation dataset dictionary...")
        val_data_dict = create_train_dataset_dict(val_images_dir, val_labels_dir)

    # Create training config
    logger.info("Creating training configuration...")
    worker_config = create_training_config(
        train_data_dict=train_data_dict,
        val_data_dict=val_data_dict,
        output_dir=output_dir,
        model_name=args.model_name,
        model_feature_size=args.feature_size,
        model_depths=args.depths,
        device=args.device,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_percent=args.training_percent,
        validation_interval=args.validation_interval,
        sampling=args.sampling,
        num_samples=args.num_samples,
        sample_size=args.sample_size,
        do_augmentation=args.augmentation,
        num_workers=args.num_workers,
        loss_function=args.loss_function,
        use_pretrained=args.use_pretrained,
        use_custom_weights=args.custom_weights is not None,
        custom_weights_path=args.custom_weights,
        deterministic=args.deterministic,
        seed=args.seed,
        downsample_zoom=downsample_zoom,
    )

    # Create and run training worker
    logger.info("Initializing training worker...")
    worker = SupervisedTrainingWorker(worker_config=worker_config)

    logger.info("Starting training...")
    logger.info("=" * 60)

    # Run training (iterate over generator)
    try:
        for report in worker.train():
            # Optionally log progress here
            # The worker handles all logging internally
            pass
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

