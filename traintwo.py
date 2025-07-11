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

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Model Imports ---
from models.causalunet import CausalUNet # The new DDPM++ model
from models.encoder import CausalEncoder

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
    
    print(f"--- Purifier Training Stage 2 (Gaussian Denoising - Corrected) ---")
    print(f"Using device: {device}")
    
    return config_obj

def load_frozen_encoder(config, checkpoint_path):
    """Loads the pre-trained and frozen CausalEncoder from Stage 1."""
    print(f"Loading frozen CausalEncoder from {checkpoint_path}")
    encoder = CausalEncoder(s_dim=config.s_dim, z_dim=config.z_dim).to(config.device)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
        
    print("CausalEncoder loaded and frozen.")
    return encoder

def solve_ode_for_training(purifier_unet, s_cond, z_cond, x_start, n_steps=5):
    """
    A lightweight, differentiable ODE solver for use inside the training loop.
    This provides a more accurate target for the latent loss.
    """
    x_t = x_start.clone()
    dt = 1.0 / n_steps
    
    # This loop is differentiable
    for i in range(n_steps):
        t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
        velocity = purifier_unet(x_t, t, s_cond, z_cond)
        x_t = x_t + velocity * dt
        
    return torch.clamp(x_t, 0, 1)

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train Latent-Guided Gaussian Denoiser")
    parser.add_argument('--config', type=str, default='configs/cifar10_causalflow.yml', help='Path to the config file.')
    parser.add_argument('--encoder_checkpoint', type=str, default='./checkpoints/causal_encoder_best.pt')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    # --- Data Loading ---
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=CIFAR10.get_train_transform())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # --- Model Initialization ---
    purifier_unet = CausalUNet(config).to(config.device)
    frozen_encoder = load_frozen_encoder(config, args.encoder_checkpoint)

    # --- Optimizer and Loss ---
    optimizer = torch.optim.Adam(purifier_unet.parameters(), lr=config.lr)
    pixel_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()
    cfm = ConditionalFlowMatcher(sigma=0.0) # sigma=0.0 as noise is added manually

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # --- Training Loop ---
    print("--- Starting Stage 2: Latent-Guided Denoiser Training ---")
    for epoch in range(config.joint_finetune_epochs):
        purifier_unet.train()
        pbar = tqdm(train_loader, desc=f"[Denoiser Train Epoch {epoch+1}]")
        
        for i, (x_clean, y_true) in enumerate(pbar):
            x_clean = x_clean.to(config.device)
            
            # Corrupt the clean image with Gaussian noise
            noise = torch.randn_like(x_clean) * config.noise_std
            x_noisy = x_clean + noise

            optimizer.zero_grad()

            # 1. Get the ground-truth `s` and `z` vectors from the clean image.
            with torch.no_grad():
                s_target, z_target, _, _ = frozen_encoder(x_clean)

            # 2. Sample the flow path from the NOISY image to the CLEAN image.
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0=x_noisy, x1=x_clean)
            
            # 3. Predict the velocity field using the UNet, guided by the target latents.
            predicted_ut = purifier_unet(xt, t, s_target, z_target)
            
            # 4. Calculate the primary flow-matching loss in pixel space.
            flow_loss = pixel_loss_fn(predicted_ut, ut)

            # 5. --- KEY IMPROVEMENT ---
            # Use the lightweight ODE solver to get a better estimate of the purified image.
            # This provides a more stable and accurate target for the latent loss.
            # We start the process from `xt` (the point on the noisy-to-clean path).
            x_purified_estimate = solve_ode_for_training(purifier_unet, s_target, z_target, xt)
            
            # Now, get the `s` vector from this more accurate estimate.
            s_denoised, _, _, _ = frozen_encoder(x_purified_estimate)
            
            latent_loss = latent_loss_fn(s_denoised, s_target)

            # 6. Combine the losses.
            total_loss = flow_loss + config.lambda_latent * latent_loss
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Flow Loss": f"{flow_loss.item():.4f}",
                "Latent Loss": f"{latent_loss.item():.4f}"
            })

    print("--- Stage 2 Training Complete ---")
    torch.save({
        'purifier_state_dict': purifier_unet.state_dict(),
    }, os.path.join(args.checkpoint_path, 'causal_purifier_ddpm_arch_final.pt'))

if __name__ == '__main__':
    main()
