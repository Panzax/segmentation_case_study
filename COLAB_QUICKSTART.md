# Quick Start: SwinUNETR_SwiGLU_ReLU2 in Colab

## Yes, it works! Here's how:

### Minimal Colab Setup (Copy-Paste Ready)

```python
# === Cell 1: Setup ===
!git clone https://github.com/YOUR_USERNAME/segmentation_case_study.git
%cd segmentation_case_study
!pip install torch torchvision monai einops wandb
%cd cellseg3d && pip install -e . && cd ..

# === Cell 2: WandB Login ===
import wandb
wandb.login()  # Enter your API key

# === Cell 3: Train with SwinUNETR_SwiGLU_ReLU2 ===
import sys
sys.path.insert(0, '/content/segmentation_case_study')

from napari_cellseg3d import config
from napari_cellseg3d.code_models.worker_training import SupervisedTrainingWorker

# Your data paths (adjust these)
images_dir = "/content/your_images"
labels_dir = "/content/your_labels"
output_dir = "/content/output"

# Create dataset
from scripts.cellseg3d.train_cellseg3d_swinunetr import create_train_dataset_dict
from pathlib import Path

train_data_dict = create_train_dataset_dict(
    Path(images_dir), 
    Path(labels_dir)
)

# Create config with SwinUNETR_SwiGLU_ReLU2
model_info = config.ModelInfo(name="SwinUNetR_SwiGLU_ReLU2")  # <-- The new model!

worker_config = config.SupervisedTrainingWorkerConfig(
    device="cuda:0",
    model_info=model_info,
    weights_info=config.WeightsInfo(use_pretrained=False),
    train_data_dict=train_data_dict,
    training_percent=0.8,
    max_epochs=50,
    loss_function="Generalized Dice",
    learning_rate=1e-3,
    batch_size=1,
    results_path_folder=output_dir,
    sampling=True,
    num_samples=2,
    sample_size=[64, 64, 64],
    do_augmentation=True,
    num_workers=2,
)

# WandB is automatically enabled if wandb is installed!
worker = SupervisedTrainingWorker(worker_config=worker_config)
worker.wandb_config.mode = "online"  # Enable WandB logging

# Start training (WandB logs automatically)
for report in worker.train():
    pass
```

## What Gets Logged to WandB?

Automatically logged:
- ✅ Training loss (per step and per epoch)
- ✅ Validation Dice score
- ✅ Learning rate
- ✅ Model architecture info
- ✅ All hyperparameters
- ✅ Best model checkpoint info

WandB run name: `SwinUNetR_SwiGLU_ReLU2_supervised_training - {timestamp}`

## That's It!

The model is already:
- ✅ Registered in CellSeg3D's MODEL_LIST
- ✅ Integrated with the training pipeline
- ✅ Compatible with WandB (automatic if wandb is installed)
- ✅ Ready to use - just change `model_name` to `"SwinUNetR_SwiGLU_ReLU2"`

No additional code changes needed! 🎉

