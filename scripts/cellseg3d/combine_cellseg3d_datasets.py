#!/usr/bin/env python3
"""
Combine multiple CellSeg3D-compatible datasets into a single master dataset.

This script is tailored to the dataset layout under:
    /clusterfs/nvme/segment_3d/tests/datasets

It will create a combined dataset directory like:
    CellSeg3D_combined_datasets/
        images/
        labels/

with filenames that encode their dataset of origin, following:
    {labels|images}/{datasetNameCamelCase}_{imageNumber}_{label|image}.tif

Examples:
    /labels/mesoSPIM_C1_label.tif
    /images/mesoSPIM_C1_image.tif

    /labels/platynereisISH_01_label.tif
    /images/platynereisISH_01_image.tif

It also writes a CSV mapping each processed file back to the original file
with metadata useful for tracking.
"""

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_DATASETS_ROOT = Path("/clusterfs/nvme/segment_3d/tests/datasets")
DEFAULT_COMBINED_NAME = "CellSeg3D_combined_datasets"


@dataclass
class FileMapping:
    """Represents a single processed file and its original source."""

    processed_path: Path
    original_path: Path
    dataset_name: str
    kind: str  # "image" or "label"
    image_id: str
    prefix: str


def ensure_output_dirs(output_dir: Path) -> Tuple[Path, Path]:
    """Create images/ and labels/ subdirectories under the output directory."""
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def check_output_clean(output_dir: Path, overwrite: bool) -> None:
    """
    Ensure the output directory is either empty or allowed to be overwritten.

    This guards against accidentally mixing incompatible runs.
    """
    if not output_dir.exists():
        return

    has_files = any(output_dir.rglob("*"))
    if has_files and not overwrite:
        raise RuntimeError(
            f"Output directory '{output_dir}' is not empty. "
            "Use --overwrite to allow reusing it."
        )


def generate_mesospim_pairs(
    datasets_root: Path,
) -> Iterable[Tuple[Path, Path, str, str]]:
    """
    Generate (image_path, label_path, prefix, image_id) for CellSeg3D_mesoSPIM.

    Uses the already-prepared dataset at:
        CellSeg3D_mesoSPIM/prepared/{images,labels}
    """
    dataset_name = "CellSeg3D_mesoSPIM"
    prefix = "mesoSPIM"
    prepared_dir = datasets_root / dataset_name / "prepared"
    images_dir = prepared_dir / "images"
    labels_dir = prepared_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        return

    for img_path in sorted(images_dir.glob("*.tif")):
        stem = img_path.stem  # e.g. "c1image"
        # Map "c1image" -> "C1", "v1image" -> "V1"
        image_id = stem
        if stem.startswith(("c", "v")) and "image" in stem:
            core = stem.split("image", maxsplit=1)[0]
            image_id = core.upper()

        # Find matching label: assume same leading token with "label"
        label_stem_candidate = stem.replace("image", "label")
        label_path = labels_dir / f"{label_stem_candidate}.tif"
        if not label_path.exists():
            # Fallback: first label whose stem starts with the image core token
            core = stem.split("image", maxsplit=1)[0]
            candidates = sorted(
                [p for p in labels_dir.glob("*.tif") if p.stem.startswith(core)]
            )
            if not candidates:
                print(
                    f"Warning: no label found for mesoSPIM image {img_path.name}",
                )
                continue
            label_path = candidates[0]

        yield img_path, label_path, prefix, image_id


def _parse_x_style_image_id(stem: str) -> str:
    """
    Convert stems like 'X01', 'X2_left', 'X02_test' into an image_id:
        '01', '02_left', '02_test'
    """
    if not stem.startswith("X"):
        return stem

    rest = stem[1:]
    num_part = ""
    suffix = ""
    for ch in rest:
        if ch.isdigit():
            num_part += ch
        else:
            suffix = rest[len(num_part) :]
            break

    if not num_part:
        return stem

    num_part = num_part.zfill(2)
    if suffix:
        # Drop leading underscores for cleanliness
        suffix = suffix.lstrip("_")
        return f"{num_part}_{suffix}"
    return num_part


def generate_xy_pairs(
    dataset_dir: Path,
    prefix: str,
) -> Iterable[Tuple[Path, Path, str, str]]:
    """
    Generate pairs for datasets where images are X*.tif and labels are Y*.tif.

    Assumes layout:
        dataset_dir/
            all/
                images/X*.tif
                masks/Y*.tif
    """
    all_dir = dataset_dir / "all"
    images_dir = all_dir / "images"
    masks_dir = all_dir / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        return

    label_index = {p.stem: p for p in masks_dir.glob("Y*.tif")}

    for img_path in sorted(images_dir.glob("X*.tif")):
        stem = img_path.stem  # e.g. "X1", "X2_left", "X01", "X02_train"
        base = stem[1:]  # strip leading "X"
        label_stem = f"Y{base}"
        label_path = label_index.get(label_stem)
        if label_path is None:
            print(
                f"Warning: no Y* label found for image {img_path} "
                f"(expected stem {label_stem})",
            )
            continue

        image_id = _parse_x_style_image_id(stem)
        yield img_path, label_path, prefix, image_id


def generate_platynereis_nuclei_pairs(
    datasets_root: Path,
) -> Iterable[Tuple[Path, Path, str, str]]:
    """
    Generate pairs for Platynereis-Nuclei-CBG (non X/Y naming).

    Layout:
        Platynereis-Nuclei-CBG/train/{images,masks}
    """
    dataset_name = "Platynereis-Nuclei-CBG"
    prefix = "platynereisNuclei"
    train_dir = datasets_root / dataset_name / "train"
    images_dir = train_dir / "images"
    masks_dir = train_dir / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        return

    label_index = {p.stem: p for p in masks_dir.glob("*.tif")}

    for img_path in sorted(images_dir.glob("*.tif")):
        stem = img_path.stem
        label_path = label_index.get(stem)
        if label_path is None:
            print(
                f"Warning: no label found for Platynereis-Nuclei image "
                f"{img_path.name}",
            )
            continue

        # Use suffix after "dataset_hdf5_" as the image id if present
        if stem.startswith("dataset_hdf5_"):
            image_id = stem.replace("dataset_hdf5_", "", 1)
        else:
            image_id = stem

        yield img_path, label_path, prefix, image_id


def collect_all_pairs(
    datasets_root: Path,
) -> List[Tuple[Path, Path, str, str, str]]:
    """
    Collect all (image_path, label_path, prefix, image_id, dataset_name) pairs
    from the known datasets under datasets_root.
    """
    pairs: List[Tuple[Path, Path, str, str, str]] = []

    # 1) CellSeg3D_mesoSPIM
    for img, lab, prefix, image_id in generate_mesospim_pairs(datasets_root):
        pairs.append((img, lab, prefix, image_id, "CellSeg3D_mesoSPIM"))

    # 2) Mouse-Skull-Nuclei-CBG (X/Y pairs)
    mouse_dir = datasets_root / "Mouse-Skull-Nuclei-CBG"
    if mouse_dir.exists():
        for img, lab, prefix, image_id in generate_xy_pairs(
            mouse_dir,
            prefix="mouseSkullNuclei",
        ):
            pairs.append((img, lab, prefix, image_id, "Mouse-Skull-Nuclei-CBG"))

    # 3) Platynereis-ISH-Nuclei-CBG (X/Y pairs)
    plat_ish_dir = datasets_root / "Platynereis-ISH-Nuclei-CBG"
    if plat_ish_dir.exists():
        for img, lab, prefix, image_id in generate_xy_pairs(
            plat_ish_dir,
            prefix="platynereisISH",
        ):
            pairs.append(
                (img, lab, prefix, image_id, "Platynereis-ISH-Nuclei-CBG")
            )

    # 4) Platynereis-Nuclei-CBG (standard images/masks)
    for img, lab, prefix, image_id in generate_platynereis_nuclei_pairs(
        datasets_root
    ):
        pairs.append((img, lab, prefix, image_id, "Platynereis-Nuclei-CBG"))

    if not pairs:
        raise RuntimeError(
            f"No image/label pairs found under '{datasets_root}'. "
            "Check that the expected datasets are present."
        )

    return pairs


def copy_or_link(
    src: Path,
    dst: Path,
    copy_files: bool,
) -> None:
    """Copy or symlink src -> dst, replacing existing dst if needed."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build_mappings(
    pairs: Iterable[Tuple[Path, Path, str, str, str]],
    images_dir: Path,
    labels_dir: Path,
    copy_files: bool,
) -> List[FileMapping]:
    """
    Materialize pairs into the combined dataset and return per-file mappings.
    """
    all_mappings: List[FileMapping] = []

    for img_path, lab_path, prefix, image_id, dataset_name in pairs:
        image_filename = f"{prefix}_{image_id}_image.tif"
        label_filename = f"{prefix}_{image_id}_label.tif"

        img_dst = images_dir / image_filename
        lab_dst = labels_dir / label_filename

        copy_or_link(img_path, img_dst, copy_files=copy_files)
        copy_or_link(lab_path, lab_dst, copy_files=copy_files)

        all_mappings.append(
            FileMapping(
                processed_path=img_dst,
                original_path=img_path,
                dataset_name=dataset_name,
                kind="image",
                image_id=image_id,
                prefix=prefix,
            )
        )
        all_mappings.append(
            FileMapping(
                processed_path=lab_dst,
                original_path=lab_path,
                dataset_name=dataset_name,
                kind="label",
                image_id=image_id,
                prefix=prefix,
            )
        )

    return all_mappings


def write_csv_mapping(
    mappings: Iterable[FileMapping],
    csv_path: Path,
    output_dir: Path,
    datasets_root: Path,
) -> None:
    """Write a CSV that maps processed files to their originals with metadata."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    processed_at = datetime.now(timezone.utc).isoformat()

    fieldnames = [
        "processed_path",
        "processed_relpath",
        "original_path",
        "original_relpath",
        "dataset_name",
        "kind",
        "image_id",
        "prefix",
        "datasets_root",
        "output_dir",
        "processed_at_utc",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in mappings:
            writer.writerow(
                {
                    "processed_path": str(m.processed_path),
                    "processed_relpath": str(
                        m.processed_path.relative_to(output_dir)
                    ),
                    "original_path": str(m.original_path),
                    "original_relpath": str(
                        m.original_path.relative_to(datasets_root)
                    ),
                    "dataset_name": m.dataset_name,
                    "kind": m.kind,
                    "image_id": m.image_id,
                    "prefix": m.prefix,
                    "datasets_root": str(datasets_root),
                    "output_dir": str(output_dir),
                    "processed_at_utc": processed_at,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple CellSeg3D datasets into a single dataset with "
            "standardized naming and a CSV provenance table."
        )
    )
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help=(
            "Root directory containing the individual datasets "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_DATASETS_ROOT / DEFAULT_COMBINED_NAME,
        help=(
            "Directory for the combined dataset "
            "(default: <datasets_root>/CellSeg3D_combined_datasets)"
        ),
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=None,
        help=(
            "Path to write the CSV mapping file. "
            "Defaults to <output_dir>/combined_mapping.csv."
        ),
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow reusing a non-empty output_dir by overwriting files. "
            "Use with care."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets_root: Path = args.datasets_root
    output_dir: Path = args.output_dir
    csv_path: Path = (
        args.csv_path
        if args.csv_path is not None
        else output_dir / "combined_mapping.csv"
    )
    copy_files = not args.symlink

    if not datasets_root.exists():
        raise RuntimeError(f"Datasets root does not exist: {datasets_root}")

    check_output_clean(output_dir, overwrite=args.overwrite)
    images_dir, labels_dir = ensure_output_dirs(output_dir)

    print(f"Collecting pairs from datasets under: {datasets_root}")
    pairs = collect_all_pairs(datasets_root)
    print(f"Found {len(pairs)} image/label pairs across datasets.")

    print(
        f"{'Copying' if copy_files else 'Symlinking'} "
        f"files into combined dataset at: {output_dir}"
    )
    mappings = build_mappings(
        pairs=pairs,
        images_dir=images_dir,
        labels_dir=labels_dir,
        copy_files=copy_files,
    )
    print(f"Wrote {len(mappings)} files to combined dataset.")

    print(f"Writing CSV mapping to: {csv_path}")
    write_csv_mapping(
        mappings=mappings,
        csv_path=csv_path,
        output_dir=output_dir,
        datasets_root=datasets_root,
    )
    print("Done.")


if __name__ == "__main__":
    main()


