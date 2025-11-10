import time
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json

# Modules to load in the ULHPC cluster
#module load data/scikit-learn
#module load vis/matplotlib
#module load bio/Seaborn/0.13.2-gfbf-2023b
## For AION (to use CPU)
#module load ai/PyTorch/2.3.0-foss-2023b
## For IRIS (to use GPU)
#module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, indices, file_names, transform=None):
        """
        Custom dataset for segmentation that loads .npy files on-the-fly

        Args:
            image_dir: Directory containing image .npy files
            mask_dir: Directory containing mask .npy files  
            indices: List of indices to use from file_names
            file_names: List of all file names
            transform: Optional transforms to apply
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.indices = indices
        self.file_names = file_names
        self.transform = transform
        # Store file paths for quick access
        self.image_paths = [os.path.join(image_dir, file_names[i]) for i in indices]
        self.mask_paths = [os.path.join(mask_dir, file_names[i]) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Load image and mask from disk
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load .npy files
        image = np.load(image_path)
        mask = np.load(mask_path)

        image = image[..., [3, 2, 1]]  # Assuming RGB values are in the last three channels

        image[image > 1] = 1
        mask = mask.argmax(axis=-1).astype(np.uint8)

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()  # Use long for segmentation masks

        # Ensure correct shape: (C, H, W) for images, (H, W) or (C, H, W) for masks
        if len(image.shape) == 3 and image.shape[0] != 3:  # If channels last
            image = image.permute(2, 0, 1)

        if len(mask.shape) == 3:  # If mask has channel dimension
            mask = mask.squeeze(0)  # Remove channel dim or handle as needed

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            # You might want separate transforms for masks

        return image, mask


# Set paths
image_dir = '/work/projects/bigdata_sets/sentinel-2/subscenes/'
mask_dir = '/work/projects/bigdata_sets/sentinel-2/masks/'

# Get all file names (assuming same names in both folders)
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

# Verify files match
assert len(image_files) == len(mask_files), "Number of images and masks don't match"
assert all(img == msk for img, msk in zip(image_files, mask_files)), "File names don't match"

print(f"Found {len(image_files)} images and masks")

# Create train/test split
total_samples = len(image_files)
indices = np.random.permutation(total_samples)

# Adjust split ratio as needed
train_ratio = 0.8
train_size = int(total_samples * train_ratio)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

print(f"Train samples: {len(train_indices)}")
print(f"Test samples: {len(test_indices)}")

num_epochs = 100
batch_size = 8
num_workers = 16

# Create datasets
train_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    indices=train_indices,
    file_names=image_files
)

test_dataset = SegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    indices=test_indices,
    file_names=image_files
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    #pin_memory=True  # Faster GPU transfer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    #pin_memory=True
)

print(f"Train loader: {len(train_loader)} batches")
print(f"Test loader: {len(test_loader)} batches")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used: {device}')

for epoch in range(num_epochs):
    start_time = time.time()
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        #print(f"Images shape: {images.shape} masks shape: {masks.shape}")
    elapsed = time.time() - start_time
    print(f"Epoch: {epoch} Elapsed time on pass all train data: {elapsed} seconds")
