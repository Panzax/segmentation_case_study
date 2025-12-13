# Segmentation Case Study

A reproducible case study for 3D cell segmentation using SwinUNETR on mesoSPIM microscopy data.

## Quick Start: Reproduce Experiments

All experiment reproduction scripts are in `scripts/Reproduce_Experiments/`. 

**Requirements:** Linux with CUDA GPU, Conda installed.

```bash
cd scripts/Reproduce_Experiments
chmod +x *.sh

./01_setup_environment.sh    # Create conda environment
conda activate segproj
./02_download_dataset.sh     # Download mesoSPIM dataset from Zenodo
./03_convert_to_32bit.sh     # Convert images to float32
./04_run_all_experiments.sh  # Train all model variants
```

See [scripts/Reproduce_Experiments/README.md](scripts/Reproduce_Experiments/README.md) for details.

## Repository Structure

```
segmentation_case_study/
├── scripts/
│   └── Reproduce_Experiments/   # Reproducibility scripts
│       ├── 01_setup_environment.sh
│       ├── 02_download_dataset.sh
│       ├── 03_convert_to_32bit.sh
│       └── 04_run_all_experiments.sh
├── cellseg3d/                   # CellSeg3D library (local install)
├── monai/                       # MONAI library (local install)
├── cell_observatory_platform/   # Training platform
└── cell_observatory_finetune/   # Fine-tuning configs
```

## Citation

If you use this work, please cite:

```bibtex
@article{10.7554/eLife.99848,
    title = {CellSeg3D, Self-supervised 3D cell segmentation for fluorescence microscopy},
    author = {Achard, Cyril and Kousi, Timokleia and Frey, Markus and others},
    journal = {eLife},
    year = {2025},
    doi = {10.7554/eLife.99848},
}
```
