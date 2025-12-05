#!/usr/bin/env python3
"""
Convert all image files in a directory tree to 32‑bit.

This script is intended for CellSeg3D-style volumetric datasets (e.g. .tif).
It walks an input directory, finds images, and rewrites them as 32‑bit
floating‑point volumes either in place or into a separate output tree.

Usage:
    python convert_images_to_32bit.py \
        --input_dir /path/to/images_root \
        [--output_dir /path/to/output_root] \
        [--ext .tif .tiff] \
        [--in_place]

If --output_dir is omitted and --in_place is set, files are overwritten
in place. If --output_dir is provided, the directory structure under
input_dir is mirrored into output_dir.
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    import tifffile
except ImportError as e:  # pragma: no cover - simple import guard
    raise ImportError(
        "tifffile is required for convert_images_to_32bit.py. "
        "Install it with `pip install tifffile`."
    ) from e


def find_image_files(
    root: Path,
    exts: Sequence[str],
) -> Iterable[Path]:
    """Yield image files under root matching the given extensions."""
    exts_set = {e.lower() for e in exts}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts_set:
            yield path


def convert_to_32bit(
    src: Path,
    dst: Path,
) -> None:
    """
    Load a volume from src, cast to float32, and save to dst.

    Existing dst is overwritten.
    """
    arr = tifffile.imread(src)

    # Cast to float32; keep dynamic range as-is.
    arr32 = np.asarray(arr, dtype=np.float32)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dst, arr32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all images in a directory tree to 32‑bit float."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing images to convert.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional root directory to write converted images. "
            "If omitted and --in_place is set, files are overwritten in place. "
            "If omitted and --in_place is not set, this flag is required."
        ),
    )
    parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=[".tif", ".tiff"],
        help="Image file extensions to include (default: .tif .tiff).",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Overwrite files in place under input_dir.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_dir is None and not args.in_place:
        raise ValueError(
            "Either --output_dir must be provided or --in_place must be set."
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    exts: List[str] = list(args.ext)

    print(f"Scanning for images under {input_dir} with extensions {exts}...")
    files = list(find_image_files(input_dir, exts))
    print(f"Found {len(files)} image files to convert.")

    for i, src in enumerate(files, 1):
        if args.in_place or output_dir is None:
            dst = src
        else:
            # Mirror directory structure under output_dir
            rel = src.relative_to(input_dir)
            dst = output_dir / rel

        convert_to_32bit(src, dst)

        if i % 10 == 0 or i == len(files):
            print(f"  Converted {i}/{len(files)} files...")

    print("Conversion to 32‑bit completed.")


if __name__ == "__main__":
    main()


