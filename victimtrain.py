# training/train_victim.py
# This script is dedicated to training a high-accuracy WideResNet classifier
# on the CIFAR-10 dataset. This will serve as the "victim model" for
# generating adversarial attacks in the subsequent defense pipeline.
# It includes data augmentation and a learning rate scheduler for optimal performance.

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Model and Data Imports ---
from models.networks.resnet.wideresnet import WideResNet
from data.cifar10 import CIFAR10

def get_config_and_setup(args):
    """Load configuration from YAML and set up device."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config_obj = type('Config', (), {})()
    for key, value in config.items():
        setattr(config_obj, key, value)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_obj.device = device
    
    print(f"--- Victim Model Training ---")
    print(f"Using device: {device}")
    
    return config_obj

def main():
    parser = argparse.ArgumentParser(description="Train Victim Classifier (WideResNet)")
    parser.add_argument('--config', type=str, default='configs/cifar10.yml', help='Path to the config file.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='Path to save checkpoints.')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    # --- Data Loading and Augmentation ---
    # We now explicitly get the correct transforms from our data module.
    train_transform = CIFAR10.get_train_transform()
    test_transform = CIFAR10.get_test_transform()

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model Initialization ---
    model = WideResNet(
        depth=config.wrn_depth, 
        widen_factor=config.wrn_widen_factor, 
        num_classes=config.num_classes,
        dropRate=0.3
    ).to(config.device)

    # --- Optimizer and Scheduler ---
    criterion = nn.CrossEntropyLoss()
    
    # --- KEY CORRECTION: Use appropriate hyperparameters for SGD ---
    # The previous learning rate of 1e-4 was far too low for SGD.
    # We now use a standard, higher initial learning rate and a multi-step scheduler.
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.1, # Standard initial LR for SGD on CIFAR-10
        momentum=0.9, 
        weight_decay=5e-4
    )
    # A multi-step scheduler that decays the LR is standard for WRN training.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    # This will decay the LR by 10x at epoch 100 and again at 150.
    # --- END KEY CORRECTION ---
    
    best_acc = 0.0
    os.makedirs(args.checkpoint_path, exist_ok=True)

    print(f"--- Starting Victim Model Training for {config.victim_train_epochs} epochs ---")
    for epoch in range(config.victim_train_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Victim Train Epoch {epoch+1}/{config.victim_train_epochs}]")
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{running_loss / (i + 1):.4f}", "LR": f"{scheduler.get_last_lr()[0]:.4f}"})
        
        # --- Validation Loop ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_acc = 100 * correct / total
        print(f"\n---===[ Validation Epoch {epoch+1} ]===--- Accuracy: {epoch_acc:.2f}%")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            print(f"*** New best accuracy: {best_acc:.2f}%. Saving victim model. ***")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, 'victim_wrn_best.pt'))
        
        scheduler.step()

    print("\n--- Victim Model Training Complete ---")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to {os.path.join(args.checkpoint_path, 'victim_wrn_best.pt')}")

if __name__ == '__main__':
    main()
