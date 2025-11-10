import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import os
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from sklearn.model_selection import train_test_split

#module load data/scikit-learn
#module load vis/matplotlib
#module load bio/Seaborn/0.13.2-gfbf-2023b

## For AION (to use CPU)
#module load ai/PyTorch/2.3.0-foss-2023b
## For IRIS (to use GPU)
#module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

# Load your data
X = np.load('/work/projects/bigdata_sets/cifar-10.1/cifar10.1_v4_data.npy')
y = np.load('/work/projects/bigdata_sets/cifar-10.1/cifar10.1_v4_labels.npy')

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")


# Convert to PyTorch tensors and preprocess
def prepare_data(X, y, validation_split=0.2, random_state=42):
    # Convert from NHWC to NCHW format and normalize
    X_tensor = torch.from_numpy(X).float().permute(0, 3, 1, 2) / 255.0
    y_tensor = torch.from_numpy(y).long()

    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor,
        test_size=validation_split,
        stratify=y_tensor,  # This ensures even class distribution
        random_state=random_state
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    return train_dataset, val_dataset


# Create a MobileNet-like CNN for CIFAR
class SimpleMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleMobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # Pointwise
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),      # 32x32 -> 16x16
            conv_dw(32, 64, 1),     # 16x16 -> 16x16
            conv_dw(64, 128, 2),    # 16x16 -> 8x8
            conv_dw(128, 128, 1),   # 8x8 -> 8x8
            conv_dw(128, 256, 2),   # 8x8 -> 4x4
            conv_dw(256, 256, 1),   # 4x4 -> 4x4
            conv_dw(256, 512, 2),   # 4x4 -> 2x2
            conv_dw(512, 512, 1),   # 2x2 -> 2x2

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Alternative simpler CNN (good for CIFAR)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Main execution
def main():
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed()
    _local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(_local_rank)

    # Prepare data
    train_dataset, val_dataset = prepare_data(X, y, validation_split=0.2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")

    # Choose your model
    #model = SimpleCNN(num_classes=num_classes)
    model = SimpleMobileNet(num_classes=num_classes)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        training_data=train_dataset,
        config="ds_config.json"
    )

    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed.
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters ()):,}")

    criterion = nn.CrossEntropyLoss()
    num_epochs = 50
    log_interval = 10
    ########################################################################
    # Train the network.
    ########################################################################
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # Get the inputs. ``data`` is a list of [inputs, labels].
            inputs, labels = data[0].to(local_device), data[1].to(local_device)

            # Try to convert to target_dtype if needed.
            if target_dtype is not None:
                inputs = inputs.to(target_dtype)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            # Print statistics
            running_loss += loss.item()
            if local_rank == 0 and i % log_interval == (
                log_interval - 1
            ):  # Print every log_interval mini-batches.
                print(
                    f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / log_interval : .3f}"
                )
                running_loss = 0.0
    print("Finished Training")

    # TODO: Complete the plotting (should be only on rank 1)
    # # Plot results
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss')

    # plt.subplot(1, 2, 2)
    # plt.plot(train_accs, label='Train Accuracy')
    # plt.plot(val_accs, label='Val Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.title('Training and Validation Accuracy')

    # plt.tight_layout()
    # plt.savefig('training_loss_cnn_test.jpg')

    ########################################################################
    # Test the network on the test data.
    ########################################################################
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # For total accuracy.
    correct, total = 0, 0
    # For accuracy per class.
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    test_batch_size = 4
    # Start testing.
    model_engine.eval()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            if target_dtype is not None:
                images = images.to(target_dtype)
            outputs = model_engine(images.to(local_device))
            _, predicted = torch.max(outputs.data, 1)
            # Count the total accuracy.
            total += labels.size(0)
            correct += (predicted == labels.to(local_device)).sum().item()

            # Count the accuracy per class.
            batch_correct = (predicted == labels.to(local_device)).squeeze()
            for i in range(test_batch_size):
                label = labels[i]
                class_correct[label] += batch_correct[i].item()
                class_total[label] += 1

    # The 10 classes for CIFAR10.
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    if model_engine.local_rank == 0:
        print(
            f"Accuracy of the network on the {total} test images: {100 * correct / total : .0f} %"
        )
        print(class_correct)
        print("================")
        print(class_total)
        # For all classes, print the accuracy.
        for i in range(10):
            print(
                f"Accuracy of {classes[i] : >5s} : {100 * class_correct[i] / class_total[i] : 2.0f} %"
            )


# Run the main function
if __name__ == "__main__":
    main()
