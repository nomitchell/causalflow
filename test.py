# inference/inference_causal_flow.py
# This is the final, comprehensive script for evaluating the CausalFlow defense.
# It is designed to provide a rigorous, SOTA-level assessment by:
# 1. Loading the complete two-stage model (frozen encoder, trained purifier).
# 2. Using a multi-step ODE solver for the actual defense at inference time.
# 3. Exposing a fully differentiable, multi-step attackable model to the adversary.
# 4. Supporting both PGD and the gold-standard AutoAttack for evaluation.

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
    
    purifier_unet = CausalUNet(config).to(config.device)
    purifier_unet.load_state_dict(purifier_ckpt['purifier_state_dict'])
    # Set to eval mode, but we will need its gradients for the attack.
    purifier_unet.eval() 

    print("Purifier UNet loaded.")
    
    return purifier_unet, encoder, latent_classifier

def solve_ode_purification(x_in, purifier_unet, encoder, n_steps=10):
    """
    Performs multi-step purification using the Euler ODE solver.
    This is the actual defense mechanism used at inference time.
    It runs with no_grad() for efficiency.
    """
    with torch.no_grad():
        x_t = x_in.clone()
        dt = 1.0 / n_steps
        
        # Get initial conditioning vectors from the potentially adversarial input
        s_cond, z_cond, _, _ = encoder(x_t)

        for i in range(n_steps):
            t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            velocity = purifier_unet(x_t, t, s_cond, z_cond)
            x_t = x_t + velocity * dt
            
    return torch.clamp(x_t, 0, 1)

class CausalFlowAttackable(nn.Module):
    """
    A wrapper to make the CausalFlow defense pipeline fully differentiable
    for a rigorous, worst-case adaptive attack evaluation.
    This implements the "Evaluation Rigor" suggestion by using a small,
    but non-trivial, number of differentiable ODE steps.
    """
    def __init__(self, purifier_unet, encoder, latent_classifier, n_attack_steps=5):
        super().__init__()
        self.purifier_unet = purifier_unet
        self.encoder = encoder
        self.latent_classifier = latent_classifier
        self.n_attack_steps = n_attack_steps # Number of steps the attacker can see

    def forward(self, x):
        # This forward pass defines the differentiable path for the attacker.
        # It does NOT use torch.no_grad().
        
        x_t = x
        dt = 1.0 / self.n_attack_steps
        
        # The attacker gets to see the initial conditioning.
        s_cond, z_cond, _, _ = self.encoder(x_t)

        # Differentiable ODE solve
        for i in range(self.n_attack_steps):
            t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            velocity = self.purifier_unet(x_t, t, s_cond, z_cond)
            x_t = x_t + velocity * dt
        
        x_purified = torch.clamp(x_t, 0, 1)
        
        # Classify based on the purified image's latent space.
        s_final, _, _, _ = self.encoder(x_purified)
        logits = self.latent_classifier(s_final)
        
        return logits

def main():
    parser = argparse.ArgumentParser(description="CausalFlow SOTA Evaluation Script")
    parser.add_argument('--config', type=str, default='configs/cifar10.yml')
    parser.add_argument('--encoder_checkpoint', type=str, default='./checkpoints/causal_encoder_best.pt')
    parser.add_argument('--purifier_checkpoint', type=str, required=True, help='Path to the trained purifier model (e.g., control or z-agnostic).')
    parser.add_argument('--attack', type=str, default='autoattack', choices=['pgd', 'autoattack'])
    parser.add_argument('--n_eval', type=int, default=1000, help='Number of samples to evaluate on.')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    purifier_unet, encoder, latent_classifier = load_models(config, args)

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=CIFAR10.transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # --- Setup the full defense pipeline (for classification) ---
    # This is a simple wrapper for the final classification step.
    def full_defense_classify(x):
        x_purified = solve_ode_purification(x, purifier_unet, encoder, n_steps=10)
        s_final, _, _, _ = encoder(x_purified)
        return latent_classifier(s_final)

    # --- Setup the attackable model (for the adversary) ---
    attackable_model = CausalFlowAttackable(purifier_unet, encoder, latent_classifier, n_attack_steps=5).to(config.device)
    
    # --- Run Evaluation ---
    x_test = torch.cat([x for i, (x, y) in enumerate(test_loader) if i*config.batch_size < args.n_eval], dim=0)
    y_test = torch.cat([y for i, (x, y) in enumerate(test_loader) if i*config.batch_size < args.n_eval], dim=0)
    x_test, y_test = x_test.to(config.device), y_test.to(config.device)

    # 1. Evaluate Clean Accuracy
    print("\nEvaluating Clean Accuracy...")
    clean_correct = 0
    with torch.no_grad():
        for i in tqdm(range(0, args.n_eval, config.batch_size), desc="Clean Eval"):
            batch_x = x_test[i:i+config.batch_size]
            batch_y = y_test[i:i+config.batch_size]
            
            logits = full_defense_classify(batch_x)
            clean_correct += (logits.argmax(1) == batch_y).sum().item()
    
    clean_accuracy = 100 * clean_correct / args.n_eval
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")

    # 2. Evaluate Robust Accuracy
    print(f"\nSetting up {args.attack.upper()} attack...")
    if args.attack == 'pgd':
        attack = torchattacks.PGD(attackable_model, eps=config.attack_params['eps'], alpha=config.attack_params['alpha'], steps=20) # Use a stronger PGD
        x_adv = attack(x_test, y_test)
    else: # AutoAttack
        attack = AutoAttack(attackable_model, norm='Linf', eps=config.attack_params['eps'], version='standard', verbose=False)
        x_adv = attack.run_standard_evaluation(x_test, y_test)

    print("Evaluating Robust Accuracy...")
    robust_correct = 0
    with torch.no_grad():
        for i in tqdm(range(0, args.n_eval, config.batch_size), desc="Robust Eval"):
            batch_x_adv = x_adv[i:i+config.batch_size]
            batch_y = y_test[i:i+config.batch_size]
            
            logits = full_defense_classify(batch_x_adv)
            robust_correct += (logits.argmax(1) == batch_y).sum().item()

    robust_accuracy = 100 * robust_correct / args.n_eval
    print(f"Robust Accuracy against {args.attack.upper()}: {robust_accuracy:.2f}%")

    print("\n--- SOTA Evaluation Complete ---")
    print(f"Model: {args.purifier_checkpoint}")
    print(f"Clean Accuracy: {clean_accuracy:.2f}%")
    print(f"Robust Accuracy ({args.attack.upper()}): {robust_accuracy:.2f}%")

if __name__ == '__main__':
    main()
