# Reproduce Experiments

This directory contains scripts to reproduce the experiments from this case study.
Run each script sequentially to set up the environment, download data, and train models.

## Prerequisites

- Linux/macOS (or WSL on Windows)
- Python 3.10+
- CUDA-capable GPU (recommended)
- `wget` or `curl` for downloading
- `unzip` for extracting archives

## Scripts

| Script | Description |
|--------|-------------|
| `01_download_dataset.sh` | Downloads the CellSeg3D mesoSPIM dataset from Zenodo |
| `02_convert_to_32bit.sh` | Converts all images to 32-bit floating point format |

## Quick Start

```bash
# 1. Make scripts executable
chmod +x *.sh

# 2. Download the dataset
./01_download_dataset.sh

# 3. Convert images to 32-bit (required for training)
./02_convert_to_32bit.sh

# The dataset will be saved to: scripts/Reproduce_Experiments/data/CellSeg3D_mesoSPIM/
```

## Dataset Information

The mesoSPIM dataset is downloaded from [Zenodo (DOI: 10.5281/zenodo.11095111)](https://zenodo.org/records/11095111).

**Structure after download:**
```
scripts/Reproduce_Experiments/data/CellSeg3D_mesoSPIM/
├── train/
│   ├── images/
│   │   ├── c1image.tif
│   │   ├── c2image.tif
│   │   ├── c3image.tif
│   │   ├── c4image.tif
│   │   └── v1image.tif
│   └── labels/
│       ├── c1label.tif
│       ├── c2label.tif
│       ├── c3label.tif
│       ├── c4label.tif
│       └── v1label.tif
└── val/
    ├── images/
    │   └── c5image.tif
    └── labels/
        └── c5label.tif
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{10.7554/eLife.99848,
    title = {CellSeg3D, Self-supervised 3D cell segmentation for fluorescence microscopy},
    author = {Achard, Cyril and Kousi, Timokleia and Frey, Markus and others},
    journal = {eLife},
    volume = {13},
    year = {2025},
    doi = {10.7554/eLife.99848},
}
```

Dataset DOI: [10.5281/zenodo.11095111](https://doi.org/10.5281/zenodo.11095111)

