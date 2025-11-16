# segmentation_case_study


# Directory structure Rough Draft

segmentation_case_study/
├── configs/
│   ├── experiments/
│   │   └── segmentation/
│   │       └── baseline_segmentation.yaml
│   └── hydra/
│       └── config.yaml           # Your Hydra defaults
│
├── src/
│   ├── models/                   # segmentation models
│   ├── data/                     # segmentation data structures
│   └── evaluation/               # segmentation evaluation logic
│
└── cell_observatory_platform/    # Submodule (use as-is)
└── cell_observatory_finetune/    # Submodule (use as-is)

# Guidelines
- Point to the platform's code as much as possible in our config files to avoid duplicating code.
