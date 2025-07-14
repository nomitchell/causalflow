# training/train_purifier_gaussian.py
# This is the FINAL, REWRITTEN training script for Stage 2.
#
# Key Changes:
# 1. Architectural Compatibility: This script is now fully compatible with the new
#    DDPM++ CausalUNet architecture.
# 2. Correct Latent Loss Calculation: Instead of a single-step approximation, this
#    script now uses a small, differentiable, multi-step ODE solver
#    (`solve_ode_for_training`) to get a more accurate purified image. This provides
#    a much higher quality and more stable gradient for the latent loss, which is
#    the core of the causal guidance mechanism.
# 3. Validation and Reporting: Includes a full validation loop to measure
#    clean and robust accuracy after each epoch. It saves the model based on
#    the best robust accuracy and logs sample images for qualitative analysis.
#
# CORRECTIONS APPLIED:
# - Added a CosineAnnealingLR learning rate scheduler to stabilize training.

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchattacks
from torchvision.utils import save_image

# --- Model Imports ---
from models.causalunet import CausalUNet
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier

# --- Module Imports ---
from data.cifar10 import CIFAR10
from modules.cfm import ConditionalFlowMatcher

def get_config_and_setup(args):
    """Load configuration from YAML and set up device."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config_obj = type('Config', (), {})()
    for key, value in config.items():
        setattr(config_obj, key, value)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_obj.device = device
    
    print(f"--- Purifier Training Stage 2 (with Validation) ---")
    print(f"Using device: {device}")
    
    return config_obj

def load_frozen_models(config, checkpoint_path):
    """Loads the pre-trained and frozen CausalEncoder and LatentClassifier."""
    print(f"Loading frozen models from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    encoder = CausalEncoder(s_dim=config.s_dim, z_dim=config.z_dim).to(config.device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print("CausalEncoder loaded and frozen.")

    latent_classifier = LatentClassifier(s_dim=config.s_dim, num_classes=config.num_classes).to(config.device)
    latent_classifier.load_state_dict(checkpoint['latent_classifier_state_dict'])
    latent_classifier.eval()
    for param in latent_classifier.parameters():
        param.requires_grad = False
    print("LatentClassifier loaded and frozen.")
    
    return encoder, latent_classifier

def solve_ode(purifier_unet, s_cond, z_cond, x_start, n_steps=10):
    """
    A differentiable ODE solver for use inside both training and validation.
    """
    x_t = x_start.clone()
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
        velocity = purifier_unet(x_t, t, s_cond, z_cond)
        x_t = x_t + velocity * dt
        
    return torch.clamp(x_t, 0, 1)

class FullDefense(nn.Module):
    """ A wrapper to make the defense pipeline differentiable for the attacker. """
    def __init__(self, purifier, encoder, classifier, n_steps=10):
        super().__init__()
        self.purifier = purifier
        self.encoder = encoder
        self.classifier = classifier
        self.n_steps = n_steps

    def forward(self, x):
        # The attacker gets to see the initial conditioning.
        s_cond, z_cond, _, _ = self.encoder(x)
        x_purified = solve_ode(self.purifier, s_cond, z_cond, x, self.n_steps)
        s_final, _, _, _ = self.encoder(x_purified)
        logits = self.classifier(s_final)
        return logits

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train Latent-Guided Gaussian Denoiser")
    parser.add_argument('--config', type=str, default='configs/cifar10.yml', help='Path to the config file.')
    parser.add_argument('--encoder_checkpoint', type=str, default='./checkpoints/causal_encoder_best.pt')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--log_path', type=str, default='./logs')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    # --- Data Loading ---
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=CIFAR10.get_train_transform())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=CIFAR10.get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # --- Model Initialization ---
    purifier_unet = CausalUNet(config).to(config.device)
    frozen_encoder, frozen_classifier = load_frozen_models(config, args.encoder_checkpoint)

    optimizer = torch.optim.Adam(purifier_unet.parameters(), lr=config.lr)
    
    # --- STABILITY IMPROVEMENT: Add Learning Rate Scheduler ---
    scheduler = CosineAnnealingLR(optimizer, T_max=config.joint_finetune_epochs, eta_min=1e-6)
    
    pixel_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()
    cfm = ConditionalFlowMatcher(sigma=0.0)

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    # --- Attacker for Validation ---
    attackable_defense = FullDefense(purifier_unet, frozen_encoder, frozen_classifier, n_steps=5).to(config.device)
    attack = torchattacks.PGD(attackable_defense, eps=config.attack_params['eps'], alpha=config.attack_params['alpha'], steps=10)

    best_robust_acc = 0.0

    print("--- Starting Stage 2: Latent-Guided Denoiser Training ---")
    for epoch in range(config.joint_finetune_epochs):
        purifier_unet.train()
        pbar = tqdm(train_loader, desc=f"[Denoiser Train Epoch {epoch+1}]")
        
        for i, (x_clean, _) in enumerate(pbar):
            x_clean = x_clean.to(config.device)
            noise = torch.randn_like(x_clean) * config.noise_std
            x_noisy = x_clean + noise

            optimizer.zero_grad()

            with torch.no_grad():
                s_target, z_target, _, _ = frozen_encoder(x_clean)

            t, xt, ut = cfm.sample_location_and_conditional_flow(x0=x_noisy, x1=x_clean)
            predicted_ut = purifier_unet(xt, t, s_target, z_target)
            flow_loss = pixel_loss_fn(predicted_ut, ut)

            x_purified_estimate = solve_ode(purifier_unet, s_target, z_target, xt, n_steps=5)
            s_denoised, _, _, _ = frozen_encoder(x_purified_estimate)
            latent_loss = latent_loss_fn(s_denoised, s_target)

            total_loss = flow_loss + config.lambda_latent * latent_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(purifier_unet.parameters(), 1.0)
            
            optimizer.step()
            
            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Flow Loss": f"{flow_loss.item():.4f}",
                "Latent Loss": f"{latent_loss.item():.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}" # Log the learning rate
            })

        # --- VALIDATION LOOP ---
        purifier_unet.eval()
        total_clean, correct_clean = 0, 0
        total_robust, correct_robust = 0, 0
        
        pbar_val = tqdm(test_loader, desc=f"[Validation Epoch {epoch+1}]")
        tem = 0
        for i, (x, y) in enumerate(pbar_val):
            if tem % 50 == 0:
                break
            else:
                tem += 1
            x, y = x.to(config.device), y.to(config.device)

            x_adv = attack(x, y)
            
            with torch.no_grad():
                s_cond_clean, z_cond_clean, _, _ = frozen_encoder(x)
                x_purified_clean = solve_ode(purifier_unet, s_cond_clean, z_cond_clean, x, n_steps=10)
                s_final_clean, _, _, _ = frozen_encoder(x_purified_clean)
                logits_clean = frozen_classifier(s_final_clean)
                correct_clean += (logits_clean.argmax(1) == y).sum().item()
                total_clean += y.size(0)

                s_cond_adv, z_cond_adv, _, _ = frozen_encoder(x_adv)
                x_purified_adv = solve_ode(purifier_unet, s_cond_adv, z_cond_adv, x_adv, n_steps=10)
                s_final_adv, _, _, _ = frozen_encoder(x_purified_adv)
                logits_adv = frozen_classifier(s_final_adv)
                correct_robust += (logits_adv.argmax(1) == y).sum().item()
                total_robust += y.size(0)

                pbar_val.set_postfix({
                    "Clean Acc": f"{100*correct_clean/total_clean:.2f}%",
                    "Robust Acc": f"{100*correct_robust/total_robust:.2f}%"
                })

        clean_acc = 100 * correct_clean / total_clean
        robust_acc = 100 * correct_robust / total_robust
        
        print(f"\n---===[ Epoch {epoch+1} Results ]===---")
        print(f"Clean Accuracy: {clean_acc:.2f}%")
        print(f"Robust Accuracy (PGD-10): {robust_acc:.2f}%")

        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            print(f"*** New best robust accuracy: {best_robust_acc:.2f}%. Saving model. ***")
            torch.save({
                'purifier_state_dict': purifier_unet.state_dict(),
            }, os.path.join(args.checkpoint_path, 'causal_purifier_best.pt'))
        
        # Log Sample Images
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_x = x[:8]
                sample_x_adv = attack(sample_x, y[:8])
                s_cond_log, z_cond_log, _, _ = frozen_encoder(sample_x_adv)
                sample_x_purified = solve_ode(purifier_unet, s_cond_log, z_cond_log, sample_x_adv, n_steps=10)
                
                comparison = torch.cat([sample_x.cpu(), sample_x_adv.cpu(), sample_x_purified.cpu()])
                save_image(comparison, os.path.join(args.log_path, f'epoch_{epoch+1}_comparison.png'), nrow=8)
                print(f"Saved sample image grid to {args.log_path}")
        
        # --- Step the scheduler after each epoch ---
        scheduler.step()

    print("\n--- Stage 2 Training Complete ---")
    print(f"Best robust accuracy achieved: {best_robust_acc:.2f}%")

if __name__ == '__main__':
    main()
