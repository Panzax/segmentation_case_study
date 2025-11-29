#!/usr/bin/env python3
"""
Helper script to prepare CellSeg3D datasets for training.

This script helps organize datasets into the format expected by CellSeg3D:
- images/ directory with image volumes
- labels/ directory with corresponding label volumes
- Matching filenames between images and labels

Usage:
    python prepare_cellseg3d_dataset.py \
        --input_dir /path/to/raw/data \
        --output_dir /path/to/prepared/dataset \
        [--image_pattern "*.tif"] \
        [--label_pattern "*_label.tif"]
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional


def find_matching_files(
    source_dir: Path,
    image_pattern: str,
    label_pattern: str,
    image_keyword: Optional[str] = None,
    label_keyword: Optional[str] = None,
) -> List[tuple]:
    """
    Find matching image-label pairs based on patterns or keywords.

    Args:
        source_dir: Directory containing both images and labels
        image_pattern: Glob pattern for image files (e.g., "*.tif")
        label_pattern: Glob pattern for label files (e.g., "*_label.tif")
        image_keyword: Optional keyword that must be in image filename
        label_keyword: Optional keyword that must be in label filename

    Returns:
        List of (image_path, label_path) tuples
    """
    image_files = list(source_dir.glob(image_pattern))
    label_files = list(source_dir.glob(label_pattern))

    if image_keyword:
        image_files = [f for f in image_files if image_keyword in f.name]
    if label_keyword:
        label_files = [f for f in label_files if label_keyword in f.name]

    # Sort for consistent ordering
    image_files = sorted(image_files)
    label_files = sorted(label_files)

    # Try to match images and labels
    pairs = []
    for img_path in image_files:
        # Strategy 1: Look for label with same stem but different pattern
        label_candidates = [
            lab
            for lab in label_files
            if lab.stem == img_path.stem or lab.stem.startswith(img_path.stem)
        ]

        # Strategy 2: Look for label with matching numeric ID
        if not label_candidates:
            img_id = None
            for part in img_path.stem.split("_"):
                if part.isdigit():
                    img_id = part
                    break

            if img_id:
                label_candidates = [
                    lab for lab in label_files if img_id in lab.stem
                ]

        if label_candidates:
            pairs.append((img_path, label_candidates[0]))
        else:
            print(f"Warning: No matching label found for {img_path.name}")

    return pairs


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    image_pattern: str = "*.tif",
    label_pattern: str = "*_label.tif",
    image_keyword: Optional[str] = None,
    label_keyword: Optional[str] = None,
    copy_files: bool = True,
):
    """
    Prepare a dataset in CellSeg3D format.

    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to create prepared dataset
        image_pattern: Glob pattern for image files
        label_pattern: Glob pattern for label files
        image_keyword: Optional keyword filter for images
        label_keyword: Optional keyword filter for labels
        copy_files: If True, copy files; if False, create symlinks
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find matching pairs
    print(f"Searching for image-label pairs in {input_dir}...")
    pairs = find_matching_files(
        input_dir, image_pattern, label_pattern, image_keyword, label_keyword
    )

    if not pairs:
        raise ValueError(
            f"No matching image-label pairs found with patterns "
            f"'{image_pattern}' and '{label_pattern}'"
        )

    print(f"Found {len(pairs)} matching pairs")

    # Copy or link files
    operation = "Copying" if copy_files else "Linking"
    print(f"{operation} files to {output_dir}...")

    for i, (img_path, label_path) in enumerate(pairs, 1):
        # Use consistent naming: keep original name or rename to match
        img_dest = images_dir / img_path.name
        label_dest = labels_dir / label_path.name

        if copy_files:
            shutil.copy2(img_path, img_dest)
            shutil.copy2(label_path, label_dest)
        else:
            if img_dest.exists():
                img_dest.unlink()
            if label_dest.exists():
                label_dest.unlink()
            img_dest.symlink_to(img_path.resolve())
            label_dest.symlink_to(label_path.resolve())

        if i % 10 == 0:
            print(f"  Processed {i}/{len(pairs)} pairs...")

    print(f"\nDataset prepared successfully!")
    print(f"  Images: {images_dir} ({len(pairs)} files)")
    print(f"  Labels: {labels_dir} ({len(pairs)} files)")
    print(f"\nYou can now use this dataset with train_cellseg3d_swinunetr.py:")
    print(f"  --images_dir {images_dir}")
    print(f"  --labels_dir {labels_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare CellSeg3D dataset from raw data"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw image and label files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to create prepared dataset",
    )
    parser.add_argument(
        "--image_pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for image files",
    )
    parser.add_argument(
        "--label_pattern",
        type=str,
        default="*_label.tif",
        help="Glob pattern for label files",
    )
    parser.add_argument(
        "--image_keyword",
        type=str,
        default=None,
        help="Optional keyword that must be in image filename",
    )
    parser.add_argument(
        "--label_keyword",
        type=str,
        default=None,
        help="Optional keyword that must be in label filename",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files",
    )

    args = parser.parse_args()

    prepare_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        image_pattern=args.image_pattern,
        label_pattern=args.label_pattern,
        image_keyword=args.image_keyword,
        label_keyword=args.label_keyword,
        copy_files=not args.symlink,
    )


if __name__ == "__main__":
    main()

