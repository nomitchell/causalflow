# inference/inference_causal_flow.py
# This is the final, comprehensive script for evaluating the CausalFlow defense.
# It has been updated to be compatible with the new DDPM++ CausalUNet.
#
# Key Features:
# 1. SOTA Architecture: Loads and uses the new DDPM++ based CausalUNet.
# 2. Rigorous Evaluation: Uses a multi-step ODE solver for the actual defense
#    at inference time (`solve_ode_purification`).
# 3. Adaptive Attack Surface: Exposes a fully differentiable, multi-step
#    attackable model (`CausalFlowAttackable`) to the adversary.
# 4. Gold-Standard Attacks: Supports both PGD and AutoAttack for evaluation.

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
from models.causalunet import CausalUNet # The new DDPM++ model
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier

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
    
    print(f"--- CausalFlow SOTA Evaluation ---")
    print(f"Loading purifier from: {args.purifier_checkpoint}")
    print(f"Using device: {device}")
    
    return config_obj

def load_models(config, args):
    """Loads all necessary models from checkpoints."""
    # Load Stage 1 models (Encoder and Latent Classifier)
    print(f"Loading frozen CausalEncoder from {args.encoder_checkpoint}")
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
    print(f"Loading Purifier UNet from {args.purifier_checkpoint}")
    purifier_ckpt = torch.load(args.purifier_checkpoint, map_location=config.device)
    
    # Instantiate the new CausalUNet with the DDPM++ architecture
    purifier_unet = CausalUNet(config).to(config.device)
    purifier_unet.load_state_dict(purifier_ckpt['purifier_state_dict'])
    purifier_unet.eval() # Set to eval mode, but gradients will be enabled by the attacker

    print("Purifier UNet loaded.")
    
    return purifier_unet, encoder, latent_classifier

def solve_ode_purification(purifier_unet, encoder, x_in, n_steps=10):
    """
    Performs multi-step purification using the Euler ODE solver. This is the
    actual defense mechanism used at inference time. It runs with no_grad()
    for efficiency during evaluation.
    """
    with torch.no_grad():
        x_t = x_in.clone()
        dt = 1.0 / n_steps
        
        # Get initial conditioning vectors from the potentially adversarial input.
        # These vectors will guide the entire purification trajectory.
        s_cond, z_cond, _, _ = encoder(x_t)

        # The flow is learned from noisy (t=0) to clean (t=1).
        # To purify, we integrate from t=0 to t=1, starting at the adversarial image.
        for i in range(n_steps):
            t_integrate = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            velocity = purifier_unet(x_t, t_integrate, s_cond, z_cond)
            x_t = x_t + velocity * dt
            
    return torch.clamp(x_t, 0, 1)

class CausalFlowAttackable(nn.Module):
    """
    A wrapper to make the CausalFlow defense pipeline fully differentiable
    for a rigorous, worst-case adaptive attack evaluation.
    """
    def __init__(self, purifier_unet, encoder, latent_classifier, n_attack_steps=5):
        super().__init__()
        self.purifier_unet = purifier_unet
        self.encoder = encoder
        self.latent_classifier = latent_classifier
        self.n_attack_steps = n_attack_steps

    def forward(self, x):
        # This forward pass defines the differentiable path for the attacker.
        x_t = x
        dt = 1.0 / self.n_attack_steps
        
        # The attacker gets to see the initial conditioning.
        s_cond, z_cond, _, _ = self.encoder(x_t)

        # Differentiable ODE solve
        for i in range(self.n_attack_steps):
            t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            # Call the new CausalUNet with the correct signature
            velocity = self.purifier_unet(x_t, t, s_cond, z_cond)
            x_t = x_t + velocity * dt
        
        x_purified = torch.clamp(x_t, 0, 1)
        
        # Classify based on the purified image's latent space.
        s_final, _, _, _ = self.encoder(x_purified)
        logits = self.latent_classifier(s_final)
        
        return logits

def main():
    parser = argparse.ArgumentParser(description="CausalFlow SOTA Evaluation Script")
    parser.add_argument('--config', type=str, default='configs/cifar10_causalflow.yml')
    parser.add_argument('--encoder_checkpoint', type=str, default='./checkpoints/causal_encoder_best.pt')
    parser.add_argument('--purifier_checkpoint', type=str, required=True, help='Path to the trained purifier model (e.g., causal_purifier_ddpm_arch_final.pt).')
    parser.add_argument('--attack', type=str, default='autoattack', choices=['pgd', 'autoattack'])
    parser.add_argument('--n_eval', type=int, default=1000, help='Number of samples to evaluate on.')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    purifier_unet, encoder, latent_classifier = load_models(config, args)

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=CIFAR10.get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # --- Setup the full defense pipeline (for classification) ---
    def full_defense_classify(x):
        x_purified = solve_ode_purification(purifier_unet, encoder, x, n_steps=10)
        s_final, _, _, _ = encoder(x_purified)
        return latent_classifier(s_final)

    # --- Setup the attackable model (for the adversary) ---
    attackable_model = CausalFlowAttackable(purifier_unet, encoder, latent_classifier, n_attack_steps=5).to(config.device)
    
    # --- Run Evaluation ---
    x_test = torch.cat([x for i, (x, y) in enumerate(test_loader) if i * config.batch_size < args.n_eval], dim=0)
    y_test = torch.cat([y for i, (x, y) in enumerate(test_loader) if i * config.batch_size < args.n_eval], dim=0)
    x_test, y_test = x_test.to(config.device), y_test.to(config.device)

    # 1. Evaluate Clean Accuracy
    print("\nEvaluating Clean Accuracy...")
    clean_correct = 0
    with torch.no_grad():
        for i in tqdm(range(0, x_test.size(0), config.batch_size), desc="Clean Eval"):
            batch_x = x_test[i:i+config.batch_size]
            batch_y = y_test[i:i+config.batch_size]
            logits = full_defense_classify(batch_x)
            clean_correct += (logits.argmax(1) == batch_y).sum().item()
    
    clean_accuracy = 100 * clean_correct / x_test.size(0)
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")

    # 2. Evaluate Robust Accuracy
    print(f"\nSetting up {args.attack.upper()} attack...")
    if args.attack == 'pgd':
        attack = torchattacks.PGD(attackable_model, eps=config.attack_params['eps'], alpha=config.attack_params['alpha'], steps=20)
        x_adv = attack(x_test, y_test)
    else: # AutoAttack
        attack = AutoAttack(attackable_model, norm='Linf', eps=config.attack_params['eps'], version='standard', verbose=False)
        x_adv = attack.run_standard_evaluation(x_test, y_test)

    print("Evaluating Robust Accuracy...")
    robust_correct = 0
    with torch.no_grad():
        for i in tqdm(range(0, x_adv.size(0), config.batch_size), desc="Robust Eval"):
            batch_x_adv = x_adv[i:i+config.batch_size]
            batch_y = y_test[i:i+config.batch_size]
            logits = full_defense_classify(batch_x_adv)
            robust_correct += (logits.argmax(1) == batch_y).sum().item()

    robust_accuracy = 100 * robust_correct / x_adv.size(0)
    print(f"Robust Accuracy against {args.attack.upper()}: {robust_accuracy:.2f}%")

    print("\n--- SOTA Evaluation Complete ---")
    print(f"Model: {args.purifier_checkpoint}")
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")
    print(f"Robust Accuracy ({args.attack.upper()}): {robust_accuracy:.2f}%")

if __name__ == '__main__':
    main()
