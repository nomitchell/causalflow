# training/train_purifier_gaussian_z_agnostic.py
# This is the FINAL, REWRITTEN script for the "Causal Purity" ablation study.
#
# Key Changes:
# 1. Architectural Compatibility: Fully compatible with the new DDPM++ CausalUNet.
# 2. Correct Latent Loss: Uses the multi-step ODE solver for a stable latent loss signal.
# 3. Correct z-Agnostic Logic: This is the core of the experiment. The purifier is
#    guided by the target image's `s` vector but is conditioned on a `z` vector
#    from a DIFFERENT, RANDOMLY SHUFFLED image in the same batch. This tests if
#    the model can learn to purify content (`s`) while being invariant to style (`z`).

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
    
    print(f"--- Purifier Training (z-Agnostic Ablation) ---")
    print(f"Guiding with target `s` and SHUFFLED `z`.")
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
    """
    x_t = x_start.clone()
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
        velocity = purifier_unet(x_t, t, s_cond, z_cond)
        x_t = x_t + velocity * dt
        
    return torch.clamp(x_t, 0, 1)

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Ablation: Train z-Agnostic Denoiser")
    parser.add_argument('--config', type=str, default='configs/cifar10_causalflow.yml', help='Path to the config file.')
    parser.add_argument('--encoder_checkpoint', type=str, default='./checkpoints/causal_encoder_best.pt')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=CIFAR10.get_train_transform())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    purifier_unet = CausalUNet(config).to(config.device)
    frozen_encoder = load_frozen_encoder(config, args.encoder_checkpoint)

    optimizer = torch.optim.Adam(purifier_unet.parameters(), lr=config.lr)
    pixel_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()
    cfm = ConditionalFlowMatcher(sigma=0.0)

    os.makedirs(args.checkpoint_path, exist_ok=True)

    print("--- Starting Stage 2 (z-Agnostic) Training ---")
    for epoch in range(config.joint_finetune_epochs):
        purifier_unet.train()
        pbar = tqdm(train_loader, desc=f"[z-Agnostic Train Epoch {epoch+1}]")
        
        for i, (x_clean, y_true) in enumerate(pbar):
            x_clean = x_clean.to(config.device)
            
            noise = torch.randn_like(x_clean) * config.noise_std
            x_noisy = x_clean + noise

            optimizer.zero_grad()

            with torch.no_grad():
                s_target, z_target, _, _ = frozen_encoder(x_clean)

            t, xt, ut = cfm.sample_location_and_conditional_flow(x0=x_noisy, x1=x_clean)
            
            # --- EXPERIMENTAL LOGIC: Guide with target `s` and SHUFFLED `z` ---
            # We shuffle the z_target vector across the batch. This forces the
            # UNet to learn to purify based on the content `s` while applying
            # a style `z` from a different image.
            z_shuffled = z_target[torch.randperm(z_target.shape[0])]
            predicted_ut = purifier_unet(xt, t, s_target, z_shuffled)
            # -----------------------------------------------------------------
            
            flow_loss = pixel_loss_fn(predicted_ut, ut)

            # Use the ODE solver for a stable latent loss target
            x_purified_estimate = solve_ode_for_training(purifier_unet, s_target, z_shuffled, xt)
            s_denoised, _, _, _ = frozen_encoder(x_purified_estimate)
            
            latent_loss = latent_loss_fn(s_denoised, s_target)

            total_loss = flow_loss + config.lambda_latent * latent_loss
            
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})

    print("--- Stage 2 (z-Agnostic) Training Complete ---")
    torch.save({
        'purifier_state_dict': purifier_unet.state_dict(),
    }, os.path.join(args.checkpoint_path, 'purifier_z_agnostic_final.pt'))

if __name__ == '__main__':
    main()
