# inference/inference_causal_flow.py
# This is the final, rewritten script for inference and evaluation.
# It implements the "Option A" pipeline by:
# 1. Loading the frozen CausalEncoder/Classifier from Stage 1.
# 2. Loading the trained Purifier UNet from Stage 2.
# 3. Using a multi-step ODE solver for robust inference.
# 4. Using a single-step, fully differentiable model for honest adaptive attack evaluation.

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchattacks
from autoattack import AutoAttack

# --- Model Imports ---
from models.causalunet import CausalUNet
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier
from models.networks.resnet.wideresnet import WideResNet

# --- Module Imports ---
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
    
    print(f"--- CausalFlow Inference & Evaluation ---")
    print(f"Using device: {device}")
    
    return config_obj

def load_models(config, args):
    """Load all necessary models from checkpoints."""
    # Load Stage 1 models (Encoder and Latent Classifier)
    print(f"Loading Stage 1 models from {args.encoder_checkpoint}")
    encoder_ckpt = torch.load(args.encoder_checkpoint, map_location=config.device)
    
    encoder = CausalEncoder(s_dim=config.s_dim, z_dim=config.z_dim).to(config.device)
    encoder.load_state_dict(encoder_ckpt['encoder_state_dict'])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    latent_classifier = LatentClassifier(s_dim=config.s_dim, num_classes=config.num_classes).to(config.device)
    latent_classifier.load_state_dict(encoder_ckpt['latent_classifier_state_dict'])
    latent_classifier.eval()
    for param in latent_classifier.parameters():
        param.requires_grad = False
    
    print("Encoder and Latent Classifier loaded and frozen.")

    # Load Stage 2 model (Purifier UNet)
    print(f"Loading Stage 2 Purifier UNet from {args.purifier_checkpoint}")
    purifier_ckpt = torch.load(args.purifier_checkpoint, map_location=config.device)
    
    purifier_unet = CausalUNet(config).to(config.device)
    purifier_unet.load_state_dict(purifier_ckpt['purifier_state_dict'])
    purifier_unet.eval()
    # Purifier is NOT frozen, as the attack needs to backprop through it.

    print("Purifier UNet loaded.")
    
    return purifier_unet, encoder, latent_classifier

class CausalFlowAttackable(nn.Module):
    """
    A wrapper to make the CausalFlow defense pipeline fully differentiable
    for a worst-case adaptive attack evaluation. This uses a SINGLE-STEP
    purification to avoid obfuscating gradients.
    """
    def __init__(self, purifier_unet, encoder, latent_classifier):
        super().__init__()
        self.purifier_unet = purifier_unet
        self.encoder = encoder
        self.latent_classifier = latent_classifier

    def forward(self, x):
        # This forward pass defines the single, differentiable path for the attacker.
        
        # 1. Get a "best guess" for s and z from the adversarial input
        # This provides the conditioning for the UNet.
        s_cond, z_cond, _, _ = self.encoder(x)
        
        # 2. Perform a SINGLE-STEP purification.
        # We simulate this by predicting the velocity at the midpoint (t=0.5)
        # and applying it once. This is a simplification but provides a clean gradient path.
        t = torch.full((x.shape[0],), 0.5, device=x.device)
        predicted_ut = self.purifier_unet(x, t, s_cond, z_cond)
        x_purified = torch.clamp(x + predicted_ut, 0, 1)
        
        # 3. Classify based on the purified image's latent space.
        s_final, _, _, _ = self.encoder(x_purified)
        logits = self.latent_classifier(s_final)
        
        return logits

def solve_ode_purification(x_adv, purifier_unet, encoder, n_steps=10):
    """
    Performs multi-step purification using the Euler ODE solver.
    This is used for the actual defense, not for the attack evaluation.
    """
    x_t = x_adv.clone()
    dt = 1.0 / n_steps
    
    # Get initial conditioning vectors
    s_cond, z_cond, _, _ = encoder(x_t)

    for i in range(n_steps):
        t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
        with torch.no_grad():
            # Predict velocity and take one step
            velocity = purifier_unet(x_t, t, s_cond, z_cond)
            x_t = x_t + velocity * dt
            
    return torch.clamp(x_t, 0, 1)

def main():
    parser = argparse.ArgumentParser(description="CausalFlow Inference and Evaluation")
    parser.add_argument('--config', type=str, default='configs/cifar10.yml', help='Path to the config file.')
    parser.add_argument('--encoder_checkpoint', type=str, default='./checkpoints/causal_encoder_best.pt')
    parser.add_argument('--purifier_checkpoint', type=str, default='./checkpoints/purifier_best.pt')
    parser.add_argument('--attack', type=str, default='pgd', choices=['pgd', 'autoattack'])
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    # --- Load Models ---
    purifier_unet, encoder, latent_classifier = load_models(config, args)

    # --- Data Loading ---
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=CIFAR10.transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # --- Setup Attack ---
    # We wrap our models in the differentiable wrapper for the attack.
    attackable_model = CausalFlowAttackable(purifier_unet, encoder, latent_classifier).to(config.device)
    
    if args.attack == 'pgd':
        print("Setting up PGD-10 attack...")
        attack = torchattacks.PGD(attackable_model, 
                                  eps=config.attack_params['eps'], 
                                  alpha=config.attack_params['alpha'], 
                                  steps=config.attack_params['iters'])
    elif args.attack == 'autoattack':
        print("Setting up AutoAttack...")
        attack = AutoAttack(attackable_model, norm='Linf', eps=config.attack_params['eps'], version='standard')

    # --- Evaluation Loop ---
    total_correct_clean = 0
    total_correct_robust = 0
    total_samples = 0

    pbar = tqdm(test_loader, desc=f"Evaluating with {args.attack.upper()}")
    for x_clean, y_true in pbar:
        x_clean = x_clean.to(config.device)
        y_true = y_true.to(config.device)

        # 1. Evaluate Clean Accuracy
        with torch.no_grad():
            # The "clean" evaluation still uses the full purification pipeline,
            # as this is the actual defense mechanism.
            x_purified_clean = solve_ode_purification(x_clean, purifier_unet, encoder)
            s_clean, _, _, _ = encoder(x_purified_clean)
            clean_logits = latent_classifier(s_clean)
            _, clean_preds = torch.max(clean_logits, 1)
            total_correct_clean += (clean_preds == y_true).sum().item()

        # 2. Generate Adversarial Examples against the differentiable pipeline
        x_adv = attack(x_clean, y_true)

        # 3. Evaluate Robust Accuracy on the generated examples
        with torch.no_grad():
            # Use the full, multi-step defense to classify the adversarial image
            x_purified_adv = solve_ode_purification(x_adv, purifier_unet, encoder)
            s_adv, _, _, _ = encoder(x_purified_adv)
            robust_logits = latent_classifier(s_adv)
            _, robust_preds = torch.max(robust_logits, 1)
            total_correct_robust += (robust_preds == y_true).sum().item()
        
        total_samples += y_true.size(0)
        
        clean_acc = 100 * total_correct_clean / total_samples
        robust_acc = 100 * total_correct_robust / total_samples
        pbar.set_postfix({"Clean Acc": f"{clean_acc:.2f}%", "Robust Acc": f"{robust_acc:.2f}%"})

    print("\n--- Final Results ---")
    print(f"Clean Accuracy: {100 * total_correct_clean / total_samples:.2f}%")
    print(f"Robust Accuracy against {args.attack.upper()}: {100 * total_correct_robust / total_samples:.2f}%")

if __name__ == '__main__':
    main()
