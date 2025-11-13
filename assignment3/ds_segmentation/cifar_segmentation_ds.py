import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt

import os
import deepspeed
from deepspeed.accelerator import get_accelerator
from sklearn.model_selection import train_test_split
from unet_model import UNet



# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# Load CIFAR-10 data
X = np.load(os.path.join(data_dir, 'cifar10.1_v4_data.npy'))
y = np.load(os.path.join(data_dir, 'cifar10.1_v4_labels.npy'))

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")


class SegmentationDataset(Dataset):
    """
    Custom dataset for segmentation on CIFAR-10.
    Creates simple segmentation masks based on image regions.
    """
    def __init__(self, images, labels):
        """
        Args:
            images: numpy array of images (N, H, W, C)
            labels: numpy array of class labels (N,)
        """
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and convert to tensor (C, H, W)
        image = self.images[idx]
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # Create a simple segmentation mask
        # For demonstration: divide image into regions based on brightness
        # In practice, you'd have real segmentation annotations
        img_gray = image.mean(dim=0)  # Convert to grayscale
        
        # Create 3-class segmentation mask based on brightness thresholds
        mask = torch.zeros(image.shape[1], image.shape[2], dtype=torch.long)
        mask[img_gray < 0.33] = 0  # Dark regions
        mask[(img_gray >= 0.33) & (img_gray < 0.66)] = 1  # Medium regions
        mask[img_gray >= 0.66] = 2  # Bright regions
        
        return image, mask


# Convert to PyTorch datasets
def prepare_data(X, y, validation_split=0.2, random_state=42):
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=validation_split,
        stratify=y,
        random_state=random_state
    )

    train_dataset = SegmentationDataset(X_train, y_train)
    val_dataset = SegmentationDataset(X_val, y_val)

    return train_dataset, val_dataset


# Main execution
def main():
    # Initialize DeepSpeed distributed backend
    # When using srun with --gpus-per-task=1, CUDA_VISIBLE_DEVICES is set to show only one GPU
    # So LOCAL_RANK should always be 0, but we need to ensure OpenMPI variables don't conflict
    
    # Unset OpenMPI local rank to avoid conflicts with --gpus-per-task=1
    # This is necessary because CUDA_VISIBLE_DEVICES remaps GPUs to start from 0
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        del os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    
    if "LOCAL_RANK" not in os.environ:
        # With --gpus-per-task=1, each task only sees GPU 0 via CUDA_VISIBLE_DEVICES
        os.environ["LOCAL_RANK"] = "0"
    
    if "RANK" not in os.environ:
        if "OMPI_COMM_WORLD_RANK" in os.environ:
            os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        elif "SLURM_PROCID" in os.environ:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
        else:
            os.environ["RANK"] = "0"
    
    if "WORLD_SIZE" not in os.environ:
        if "OMPI_COMM_WORLD_SIZE" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        elif "SLURM_NTASKS" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        else:
            os.environ["WORLD_SIZE"] = "1"
    
    deepspeed.init_distributed()
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    get_accelerator().set_device(_local_rank)

    # Prepare data
    train_dataset, val_dataset = prepare_data(X, y, validation_split=0.2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create UNet model for segmentation
    # 3 input channels (RGB), 3 output classes (background, medium, foreground)
    model = UNet(n_channels=3, n_classes=3, bilinear=True)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        training_data=train_dataset,
        config="ds_config.json"
    )

    # Get the local device name (str) and local rank (int)
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Use CrossEntropyLoss for multi-class segmentation
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50
    log_interval = 10

    ########################################################################
    # Train the network
    ########################################################################
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model_engine.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader):
            # Get the inputs: images and segmentation masks
            inputs, masks = data[0].to(local_device), data[1].to(local_device)

            # Try to convert to target_dtype if needed
            if target_dtype is not None:
                inputs = inputs.to(target_dtype)

            # Forward pass
            outputs = model_engine(inputs)
            
            # outputs shape: (B, n_classes, H, W)
            # masks shape: (B, H, W) with integer class labels
            loss = criterion(outputs, masks)

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()

            # Print statistics
            running_loss += loss.item()
            if local_rank == 0 and i % log_interval == (log_interval - 1):
                avg_loss = running_loss / log_interval
                print(f"[{epoch + 1:d}, {i + 1:5d}] loss: {avg_loss:.3f}")
                train_losses.append(avg_loss)
                running_loss = 0.0

        ####################################################################
        # Validation phase
        ####################################################################
        if epoch % 5 == 0:  # Validate every 5 epochs
            model_engine.eval()
            val_loss = 0.0
            val_batches = 0
            
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            with torch.no_grad():
                for data in val_loader:
                    images, masks = data
                    images, masks = images.to(local_device), masks.to(local_device)
                    
                    if target_dtype is not None:
                        images = images.to(target_dtype)
                    
                    outputs = model_engine(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            if local_rank == 0:
                print(f"\nEpoch {epoch + 1}: Val Loss: {avg_val_loss:.3f}\n")
                print("-" * 50)

    print("Finished Training")

    ########################################################################
    # Test the network on validation data
    ########################################################################
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Calculate pixel accuracy
    correct_pixels = 0
    total_pixels = 0
    
    model_engine.eval()
    with torch.no_grad():
        for data in val_loader:
            images, masks = data
            images, masks = images.to(local_device), masks.to(local_device)
            
            if target_dtype is not None:
                images = images.to(target_dtype)
            
            outputs = model_engine(images)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class per pixel
            
            # Count correct pixels
            total_pixels += masks.numel()
            correct_pixels += (predicted == masks).sum().item()

    if model_engine.local_rank == 0:
        pixel_accuracy = 100 * correct_pixels / total_pixels
        print(f"Pixel Accuracy on validation set: {pixel_accuracy:.2f}%")
        print(f"Total pixels evaluated: {total_pixels:,}")


# Run the main function
if __name__ == "__main__":
    main()
