# Reproduce Experiments

This directory contains scripts to reproduce the SwinUNETR segmentation experiments from this case study.

## Requirements

- **Linux** (the environment.yml contains Linux-specific packages)
- Conda or Miniforge installed
- CUDA-capable GPU
- `wget` or `curl` for downloading
- `unzip` for extracting archives

> **Note:** These scripts are designed for Linux systems. The conda environment will not install correctly on Windows or macOS due to platform-specific dependencies.

## Scripts

| Script | Description |
|--------|-------------|
| `01_setup_environment.sh` | Creates conda environment `segproj` from `environment.yml` |
| `02_download_dataset.sh` | Downloads mesoSPIM dataset from [Zenodo](https://zenodo.org/records/11095111) |
| `03_convert_to_32bit.sh` | Converts all images to 32-bit float (required for training) |
| `04_run_all_experiments.sh` | Trains SwinUNETR with multiple model variants and seeds |

## Quick Start

```bash
# Navigate to this directory
cd scripts/Reproduce_Experiments

# Make all scripts executable
chmod +x *.sh

# Step 1: Create conda environment (~10-30 min)
./01_setup_environment.sh
conda activate segproj

# Step 2: Download dataset from Zenodo (~30 MB)
./02_download_dataset.sh

# Step 3: Convert images to 32-bit float
./03_convert_to_32bit.sh

# Step 4: Run all training experiments (requires GPU)
./04_run_all_experiments.sh
```

## Output Locations

- **Dataset:** `data/CellSeg3D_mesoSPIM/`
- **Trained models:** `outputs/`

## Dataset Structure

After running `02_download_dataset.sh`:

```
data/CellSeg3D_mesoSPIM/
├── train/
│   ├── images/   (5 files: c1-c4, v1)
│   └── labels/   (5 files: c1-c4, v1)
└── val/
    ├── images/   (1 file: c5)
    └── labels/   (1 file: c5)
```

## Experiments

The `04_run_all_experiments.sh` script trains:

- **Models:** SwinUNetR_Mlp_LeakyReLU, SwinUNetR_SwiGLU_LeakyReLU, SwinUNetR_Mlp_ReLUSquared, SwinUNetR_SwiGLU_ReLUSquared
- **Seeds:** 34936339, 42, 1, 123456789
- **Configurations:** base model, depths 1-1-1-1, feature_size 12

## Troubleshooting

### Environment fails to install
The `environment.yml` is Linux-specific. If you're on Windows/macOS, you'll need to manually install packages or use WSL/Docker.

### CUDA not available
Ensure you have NVIDIA drivers and CUDA toolkit installed. Check with:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Citation

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
