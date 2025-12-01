# Using SwinUNETR_SwiGLU_ReLU2 in Google Colab

This guide explains how to use the new `SwinUNETR_SwiGLU_ReLU2` model in Google Colab with WandB integration.

## Quick Start

The model is already integrated into CellSeg3D's training pipeline, so using it in Colab is straightforward!

## Step-by-Step Setup

### 1. Clone the Repository in Colab

```python
# In a Colab cell
!git clone https://github.com/YOUR_USERNAME/segmentation_case_study.git
%cd segmentation_case_study

# Initialize submodules if needed
!git submodule update --init --recursive
```

### 2. Install Dependencies

```python
# Install core dependencies
!pip install torch torchvision monai einops wandb

# Install CellSeg3D in editable mode
%cd cellseg3d
!pip install -e .
%cd ..
```

### 3. Set Up WandB (Optional but Recommended)

```python
import wandb
wandb.login()  # Enter your API key when prompted
```

### 4. Prepare Your Data

Upload your training data to Colab (or mount Google Drive):

```python
# Option A: Upload directly to Colab
from google.colab import files
# Upload your images and labels folders

# Option B: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Then point to your data in Drive
```

### 5. Run Training with SwinUNETR_SwiGLU_ReLU2

You can use the existing training script with the new model name:

```python
import sys
sys.path.insert(0, '/content/segmentation_case_study')

from scripts.cellseg3d.train_cellseg3d_swinunetr import main
import argparse

# Create arguments (or modify the script to accept them directly)
class Args:
    images_dir = "/content/your_images"
    labels_dir = "/content/your_labels"
    output_dir = "/content/output"
    model_name = "SwinUNetR_SwiGLU_ReLU2"  # <-- Use the new model!
    device = "cuda:0"
    max_epochs = 50
    batch_size = 1
    learning_rate = 1e-3
    training_percent = 0.8
    validation_interval = 2
    sampling = True
    num_samples = 2
    sample_size = [64, 64, 64]
    augmentation = True
    num_workers = 2
    loss_function = "Generalized Dice"
    use_pretrained = False
    custom_weights = None
    deterministic = True
    seed = 34936339

# Convert to argparse.Namespace for compatibility
args = argparse.Namespace(**vars(Args()))

# Run training
main()
```

### Alternative: Use CellSeg3D's SupervisedTrainingWorker Directly

For more control, you can use the training worker directly (similar to the Colab WNet notebook):

```python
import sys
sys.path.insert(0, '/content/segmentation_case_study')

from pathlib import Path
from napari_cellseg3d import config
from napari_cellseg3d.code_models.worker_training import SupervisedTrainingWorker
from napari_cellseg3d.utils import LOGGER

# Create dataset dictionary
train_data_dict = [
    {"image": "/content/images/img1.tif", "label": "/content/labels/lab1.tif"},
    {"image": "/content/images/img2.tif", "label": "/content/labels/lab2.tif"},
    # ... add more
]

# Create model config with SwinUNETR_SwiGLU_ReLU2
model_info = config.ModelInfo(name="SwinUNetR_SwiGLU_ReLU2")

# Create training config
worker_config = config.SupervisedTrainingWorkerConfig(
    device="cuda:0",
    model_info=model_info,  # <-- This will use SwinUNetR_SwiGLU_ReLU2
    weights_info=config.WeightsInfo(use_pretrained=False),
    train_data_dict=train_data_dict,
    training_percent=0.8,
    max_epochs=50,
    loss_function="Generalized Dice",
    learning_rate=1e-3,
    scheduler_factor=0.5,
    scheduler_patience=10,
    validation_interval=2,
    batch_size=1,
    results_path_folder="/content/output",
    sampling=True,
    num_samples=2,
    sample_size=[64, 64, 64],
    do_augmentation=True,
    num_workers=2,
    deterministic_config=config.DeterministicConfig(enabled=True, seed=34936339),
)

# WandB config (optional)
wandb_config = config.WandBConfig(
    mode="online",  # or "offline" or "disabled"
    save_model_artifact=False,
)

# Set WandB config in worker
worker = SupervisedTrainingWorker(worker_config=worker_config)
worker.wandb_config = wandb_config

# Start training
for report in worker.train():
    # Training progress is logged automatically
    # WandB metrics are logged if wandb is installed and configured
    pass
```

## WandB Integration

The `SupervisedTrainingWorker` already has WandB integration built-in! It will automatically log:

- **Training metrics**: Loss per step and per epoch
- **Validation metrics**: Dice score
- **Model info**: Model architecture, parameters
- **Hyperparameters**: Learning rate, batch size, etc.

To enable WandB:

1. Install wandb: `!pip install wandb`
2. Login: `wandb.login()`
3. The worker will automatically detect wandb and start logging

WandB runs are created with:
- **Project**: "CellSeg3D"
- **Name**: `{model_name}_supervised_training - {timestamp}`
- **Tags**: `[model_name, "supervised"]`

## Model-Specific Notes

### SwinUNETR_SwiGLU_ReLU2 Features

- **SwiGLU activation** in transformer MLP blocks
- **ReLU² activation** in convolutional encoder/decoder blocks
- Same architecture as SwinUNETR, just with different activations

### Configuration Options

The model accepts the same parameters as regular SwinUNETR:

```python
model_info = config.ModelInfo(
    name="SwinUNetR_SwiGLU_ReLU2",
    model_input_size=[64, 64, 64],  # Optional: specify input size
)
```

## Troubleshooting

### Issue: "Model 'SwinUNetR_SwiGLU_ReLU2' not found in MODEL_LIST"

**Solution**: Make sure you've cloned the repository with the latest changes and that the model is registered in `cellseg3d/napari_cellseg3d/config.py`.

### Issue: WandB not logging

**Solution**: 
1. Check if wandb is installed: `import wandb`
2. Check if you're logged in: `wandb.login()`
3. Check the worker's wandb_config mode (should be "online" or "offline", not "disabled")

### Issue: CUDA out of memory

**Solution**: Reduce batch size or patch size:
```python
batch_size = 1  # Reduce from 2 or 4
sample_size = [48, 48, 48]  # Reduce from [64, 64, 64]
```

## Example Colab Notebook Structure

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/segmentation_case_study.git
%cd segmentation_case_study
!pip install torch torchvision monai einops wandb
%cd cellseg3d && pip install -e . && cd ..

# Cell 2: WandB Login
import wandb
wandb.login()

# Cell 3: Import and Configure
from napari_cellseg3d import config
from napari_cellseg3d.code_models.worker_training import SupervisedTrainingWorker

# ... setup data paths and config ...

# Cell 4: Train
worker = SupervisedTrainingWorker(worker_config=worker_config)
for report in worker.train():
    pass
```

## Next Steps

Once training is complete:
- Checkpoints are saved in `results_path_folder`
- Best model: `{model_name}_best_metric.pth`
- Latest model: `{model_name}_latest.pth`
- WandB dashboard: View at https://wandb.ai

## Questions?

- Check the main `SETUP.md` for general setup instructions
- Check `scripts/cellseg3d/train_cellseg3d_swinunetr.py` for command-line usage
- The model implementation is in `cell_observatory_finetune/models/meta_arch/swin_unetr_swiglu_relu2.py`

