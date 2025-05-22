import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import timm

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to the image dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
args = parser.parse_args()

data_dir = args.data_dir
batch_size = args.batch_size

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names you want to include
class_names = ('aca_bd', 'aca_md', 'aca_pd')

# Add Gaussian noise transform
def add_gaussian_noise(x):
    return x + 0.01 * torch.randn_like(x)

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(224, scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Lambda(add_gaussian_noise)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Subset wrapper
class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# ViT model wrapper
class ViTModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ViTModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.7),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)

# Filter the dataset to include only the desired classes
def filter_dataset(dataset, target_classes):
    # Create a mask for the target classes
    idx = [i for i, target in enumerate(dataset.targets) 
           if dataset.classes[target] in target_classes]
    filtered_data = torch.utils.data.Subset(dataset, idx)
    return filtered_data

# Training function
def train_model(model, train_loader, val_loader, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_scheduler = ExponentialLR(optimizer, gamma=0.96)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        exp_scheduler.step()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        plateau_scheduler.step(val_loss / len(val_loader.dataset))

        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader.dataset):.4f}, Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss / len(val_loader.dataset):.4f}, Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved best model')

        if epoch > 20 and val_acc < best_val_acc:
            print('Early stopping')
            break

    return model

# Testing function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {correct / total:.4f}')

if __name__ == "__main__":
    torch.manual_seed(42)

    # Load full dataset first
    base_dataset = ImageFolder(root=data_dir)

    # Filter dataset by desired class names
    full_dataset = filter_dataset(base_dataset, class_names)

    # Access targets using the original base_dataset
    subset_targets = [base_dataset.targets[i] for i in full_dataset.indices]

    # First split: Train and Temp (Val+Test)
    train_idx, temp_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=0.3,
        stratify=subset_targets,
        random_state=42
    )

    # Get stratified targets for the temp split
    temp_targets = [subset_targets[i] for i in temp_idx]

    # Second split: Val and Test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_targets,
        random_state=42
    )

    # Subset datasets with transforms
    train_dataset = SubsetDataset(torch.utils.data.Subset(full_dataset, train_idx), transform=train_transform)
    val_dataset = SubsetDataset(torch.utils.data.Subset(full_dataset, val_idx), transform=val_test_transform)
    test_dataset = SubsetDataset(torch.utils.data.Subset(full_dataset, test_idx), transform=val_test_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = ViTModel(num_classes=3).to(device)

    # Train
    model = train_model(model, train_loader, val_loader, num_epochs=50)

    # Evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    test_model(model, test_loader)
