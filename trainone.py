# training/train_causal_encoder.py
# This is the training script for Stage 1 of the CausalFlow pipeline.
# Its sole purpose is to train the CausalEncoder and LatentClassifier to learn
# a robust, disentangled representation (s, z) from clean images.
# The UNet is NOT used in this stage.

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Model Imports ---
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier

# --- Module Imports ---
from modules.cib import CIBLoss, CLUB
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
    
    print(f"--- CausalEncoder Training Stage 1 ---")
    print(f"Using device: {device}")
    
    return config_obj

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train Causal Encoder and Latent Classifier")
    parser.add_argument('--config', type=str, default='configs/cifar10.yml', help='Path to the config file.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='Path to save checkpoints.')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    # --- Data Loading ---
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=CIFAR10.transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=CIFAR10.transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # --- Model Initialization ---
    encoder = CausalEncoder(s_dim=config.s_dim, z_dim=config.z_dim).to(config.device)
    latent_classifier = LatentClassifier(s_dim=config.s_dim, num_classes=config.num_classes).to(config.device)
    club_estimator = CLUB(config.s_dim, config.z_dim, config.s_dim).to(config.device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(latent_classifier.parameters()),
        lr=config.lr
    )
    club_optimizer = torch.optim.Adam(club_estimator.parameters(), lr=config.lr)

    # --- Loss Function ---
    cib_loss_fn = CIBLoss(
        gamma_ce=config.gamma_ce, 
        lambda_kl=config.lambda_kl, 
        eta_club=config.eta_club
    )
    
    best_val_acc = 0.0
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # --- Training Loop ---
    print("--- Starting Stage 1: Causal Representation Training ---")
    for epoch in range(config.causal_pretrain_epochs):
        encoder.train()
        latent_classifier.train()
        club_estimator.train()
        
        pbar = tqdm(train_loader, desc=f"[Encoder Train Epoch {epoch+1}]")
        
        for i, (x_clean, y_true) in enumerate(pbar):
            x_clean = x_clean.to(config.device)
            y_true = y_true.to(config.device)

            s, z, mu, logvar = encoder(x_clean)
            
            # Update CLUB estimator
            club_loss = club_estimator.learning_loss(s.detach(), z.detach())
            club_optimizer.zero_grad()
            club_loss.backward()
            club_optimizer.step()

            # Update Encoder and Classifier
            optimizer.zero_grad()
            y_pred = latent_classifier(s)
            
            loss_dict = cib_loss_fn(y_pred, y_true, mu, logvar, s, z, club_estimator)
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Pred Loss": f"{loss_dict['prediction_loss'].item():.4f}",
                "KL Loss": f"{loss_dict['kl_loss'].item():.4f}"
            })

        # --- Validation Loop ---
        encoder.eval()
        latent_classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(config.device)
                y_val = y_val.to(config.device)
                s_val, _, _, _ = encoder(x_val)
                outputs = latent_classifier(s_val)
                _, predicted = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        
        val_acc = 100 * correct / total
        print(f"---===[ Validation Epoch {epoch+1} ]===--- Clean Accuracy on S-vector: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"*** New best validation accuracy: {best_acc:.2f}%. Saving model. ***")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'latent_classifier_state_dict': latent_classifier.state_dict(),
            }, os.path.join(args.checkpoint_path, 'causal_encoder_best.pt'))

    print("--- Stage 1 Training Complete ---")

if __name__ == '__main__':
    main()
