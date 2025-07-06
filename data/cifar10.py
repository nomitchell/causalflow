# data/cifar10.py
# PURPOSE: Handles all data loading and preprocessing for the CIFAR-10 dataset.
# This keeps the main training script clean and focused.
#
# WHERE TO GET CODE: This is standard PyTorch using torchvision.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, data_dir='./data'):
    """
    Returns the training and testing dataloaders for CIFAR-10.
    """
    # TO-DO: Define the standard transformations for CIFAR-10.
    # For training, it's common to use data augmentation like random crops and flips.
    # For testing, only normalization is needed.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader