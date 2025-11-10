# CIFAR-10 Segmentation with DeepSpeed

This folder contains code for performing semantic segmentation on the CIFAR-10 dataset using a U-Net architecture with DeepSpeed for distributed training.

## Overview

- **Dataset**: CIFAR-10.1 (32x32 RGB images)
- **Task**: Semantic segmentation (pixel-wise classification into 3 classes)
- **Model**: Simplified U-Net architecture adapted for small images
- **Framework**: PyTorch + DeepSpeed for distributed training

## Files

- `cifar_segmentation_ds.py` - Main training script with DeepSpeed integration
- `unet_model.py` - U-Net model definition
- `unet_parts.py` - U-Net building blocks (DoubleConv, Down, Up, OutConv)
- `ds_config.json` - DeepSpeed configuration (optimizer, scheduler, batch size)
- `launch_job.sh` - SLURM batch script for cluster submission
- `README.md` - This file

## Segmentation Task

Since CIFAR-10 doesn't come with segmentation annotations, this implementation creates simple segmentation masks based on image brightness:
- **Class 0**: Dark regions (brightness < 0.33)
- **Class 1**: Medium regions (0.33 ≤ brightness < 0.66)
- **Class 2**: Bright regions (brightness ≥ 0.66)

This is a demonstration setup. For real segmentation tasks, you would use proper annotated segmentation masks.

## Model Architecture

The U-Net architecture is simplified for 32x32 CIFAR images:
- **Input**: 3 channels (RGB)
- **Output**: 3 channels (class logits per pixel)
- **Encoder**: 3 downsampling blocks (64→128→256→256)
- **Decoder**: 3 upsampling blocks with skip connections
- **Final**: 1x1 convolution to produce class logits

## Usage

### On the Cluster (SLURM)

1. Submit the job:
```bash
cd assignment3/ds_segmentation
sbatch launch_job.sh
```

2. Monitor the job:
```bash
squeue -u $USER
tail -f ds_segmentation.out
```

### Local/Interactive Run

1. Load required modules:
```bash
module load data/scikit-learn
module load vis/matplotlib
module load bio/Seaborn/0.13.2-gfbf-2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
```

2. Run with DeepSpeed on single GPU:
```bash
deepspeed cifar_segmentation_ds.py --deepspeed --deepspeed_config ds_config.json
```

3. For multi-GPU interactive session:
```bash
# Request interactive session with 2 GPUs
si-gpu -G 2 -N 1 -c 7

# Run with DeepSpeed
deepspeed --num_gpus=2 cifar_segmentation_ds.py --deepspeed --deepspeed_config ds_config.json
```

## Configuration

### DeepSpeed Config (`ds_config.json`)

- **Batch size**: 32
- **Optimizer**: Adam (lr=0.0001)
- **Scheduler**: WarmupLR (500 warmup steps)
- **ZeRO**: Stage 1 optimization
- **Mixed Precision**: Disabled (can be enabled by setting `"fp16": {"enabled": true}`)

### Training Parameters

In `cifar_segmentation_ds.py`:
- `num_epochs`: 50
- `validation_split`: 0.2 (20% of data for validation)
- `log_interval`: 10 (print loss every 10 batches)

## Output

The script will print:
- Training loss every 10 batches
- Validation loss every 5 epochs
- Final pixel accuracy on validation set

Example output:
```
Training samples: 1616
Validation samples: 404
Model: UNet
Total parameters: 1,942,851
[1,    10] loss: 1.098
[1,    20] loss: 1.045
...
Epoch 1: Val Loss: 0.987
--------------------------------------------------
Finished Training
Pixel Accuracy on validation set: 62.35%
```

## Performance Notes

- **Memory**: U-Net uses ~2GB GPU memory for batch size 32
- **Speed**: ~100 images/sec on single V100 GPU
- **Scaling**: Near-linear speedup with multiple GPUs using DeepSpeed ZeRO-1

## Customization

To adapt this code for your own segmentation task:

1. Replace `SegmentationDataset` to load your own images and masks
2. Adjust `n_classes` in UNet initialization to match your task
3. Modify `ds_config.json` to tune hyperparameters
4. Update loss function if needed (e.g., use Dice loss for imbalanced classes)

## Requirements

- PyTorch >= 2.0
- DeepSpeed
- NumPy
- scikit-learn
- matplotlib (optional, for visualization)

## References

- U-Net: https://arxiv.org/abs/1505.04597
- DeepSpeed: https://www.deepspeed.ai/
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
