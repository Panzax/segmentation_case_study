#!/bin/bash

source ~/.bashrc
conda activate segproj
cd /clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study

# Make both the repo root and the platform subdir importable
export PYTHONPATH="/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study:/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/cell_observatory_platform:/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/monai${PYTHONPATH}"

# (Optionally also keep the envs we discussed earlier)
export REPO_DIR=/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/cell_observatory_finetune
export PLATFORM_REPO_DIR=/clusterfs/nvme/martinalvarez/GitHub/segmentation_case_study/cell_observatory_platform
export REPO_NAME=cell_observatory_finetune
export PLATFORM_REPO_NAME=cell_observatory_platform

python cell_observatory_finetune/manager.py --config-name=experiments/abc/3D/swin-unetr/test_swin_unetr_reproducibility_11-24-25.yaml 
