# data/cifar10.py
# This file has been rewritten to provide a more flexible and robust data loading
# pipeline for the CIFAR-10 dataset. It separates training and testing transforms
# and provides a clean interface for the training scripts.

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
import numpy as np

class CIFAR10:
    """
    A wrapper class for the CIFAR-10 dataset that handles transformations
    and provides a consistent interface for training and testing.
    """
    def __init__(self, root='./data', train=True, download=True, transform=None):
        """
        Initializes the dataset.

        Args:
            root (str): The root directory where the dataset is stored.
            train (bool): If True, creates dataset from training set, otherwise from test set.
            download (bool): If True, downloads the dataset from the internet.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, `transforms.RandomCrop`.
        """
        if transform is None:
            # If no transform is provided, use the default testing transform.
            self.transform = self.get_test_transform()
        else:
            self.transform = transform
            
        self.dataset = PyTorchCIFAR10(
            root=root,
            train=train,
            download=download,
            transform=self.transform
        )

    @staticmethod
    def get_train_transform():
        """
        Returns the standard set of augmentations for training on CIFAR-10.
        Includes random cropping and horizontal flipping to improve model generalization.
        """
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Normalization is often done, but generative models like flows/diffusion
            # typically work better in the [0, 1] or [-1, 1] range without std normalization.
            # We will stick to [0, 1] for simplicity, as handled by ToTensor().
        ])

    @staticmethod
    def get_test_transform():
        """
        Returns the standard transform for testing/validation.
        No augmentations are applied, only conversion to a tensor.
        """
        return transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # The underlying PyTorch dataset object handles the __getitem__ logic.
        return self.dataset[idx]

# For convenience, you can still define a default transform if needed,
# but the training scripts will now explicitly call get_train_transform() or get_test_transform().
CIFAR10.transform = CIFAR10.get_test_transform()

