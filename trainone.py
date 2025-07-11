# training/train_causal_encoder.py
# This is a new training script for Stage 1 of the CausalFlow pipeline.
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
# We only need the encoder and the classifier for this stage.
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier

# --- Module Imports ---
from modules.cib import CIBLoss, CLUB
from data.cifar10 import CIFAR10

def get_config_and_setup(args):
    """Load configuration from YAML and set up device."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a configuration object
    config_obj = type('Config', (), {})()
    for key, value in config.items():
        setattr(config_obj, key, value)
    
    # Set up device
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
    # We only need the encoder and the latent classifier for this stage.
    encoder = CausalEncoder(s_dim=config.s_dim, z_dim=config.z_dim).to(config.device)
    latent_classifier = LatentClassifier(s_dim=config.s_dim, num_classes=config.num_classes).to(config.device)
    club_estimator = CLUB(config.s_dim, config.z_dim, config.s_dim).to(config.device)

    # --- Optimizer ---
    # The optimizer will only update the parameters of the encoder, classifier, and CLUB estimator.
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(latent_classifier.parameters()),
        lr=config.lr
    )
    club_optimizer = torch.optim.Adam(club_estimator.parameters(), lr=config.lr)

    # --- Loss Function ---
    # The CIBLoss will now only focus on prediction, KL divergence, and disentanglement.
    # The reconstruction component is removed for this stage.
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
        
        total_loss_sum = 0
        pbar = tqdm(train_loader, desc=f"[Encoder Train Epoch {epoch+1}]")
        
        for i, (x_clean, y_true) in enumerate(pbar):
            x_clean = x_clean.to(config.device)
            y_true = y_true.to(config.device)

            # Forward pass through the encoder
            s, z, mu, logvar = encoder(x_clean)
            
            # --- CIB Loss Calculation ---
            # 1. Update CLUB estimator to maximize the MI lower bound
            club_loss = club_estimator.learning_loss(s.detach(), z.detach())
            club_optimizer.zero_grad()
            club_loss.backward()
            club_optimizer.step()

            # 2. Update Encoder and Classifier
            optimizer.zero_grad()
            
            # Get predictions from the latent space
            y_pred = latent_classifier(s)
            
            # Calculate the CIB loss components (without reconstruction)
            loss_dict = cib_loss_fn(
                y_pred=y_pred, 
                y_true=y_true, 
                mu=mu, 
                logvar=logvar,
                s=s,
                z=z,
                club_estimator=club_estimator
            )
            
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            optimizer.step()
            
            total_loss_sum += total_loss.item()
            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Pred Loss": f"{loss_dict['prediction_loss'].item():.4f}",
                "KL Loss": f"{loss_dict['kl_loss'].item():.4f}",
                "Disentangle": f"{loss_dict['disentanglement_loss'].item():.4f}"
            })

        # --- Validation Loop ---
        encoder.eval()
        latent_classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = x_val.to(config.device)
                y_val = y_val.to(config.device)
                
                s_val, _, _, _ = encoder(x_val)
                outputs = latent_classifier(s_val)
                _, predicted = torch.max(outputs.data, 1)
                
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        
        val_acc = 100 * correct / total
        print(f"---===[ Validation Epoch {epoch+1} ]===--- Clean Accuracy on S-vector: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"*** New best validation accuracy: {best_val_acc:.2f}%. Saving model. ***")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'latent_classifier_state_dict': latent_classifier.state_dict(),
                'epoch': epoch,
                'config': config
            }, os.path.join(args.checkpoint_path, 'causal_encoder_best.pt'))

    print("--- Stage 1 Training Complete ---")
    print(f"Best validation accuracy from latent space: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
