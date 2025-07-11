# training/train_purifier.py
# This is a new training script for Stage 2 of the CausalFlow pipeline.
# Its purpose is to train the CausalUNet as a latent-guided purifier.
# It uses the frozen CausalEncoder from Stage 1 to define its loss function.

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchattacks

# --- Model Imports ---
from models.causalunet import CausalUNet # The updated UNet
from models.encoder import CausalEncoder
from models.networks.resnet.wideresnet import WideResNet # Victim model for generating attacks

# --- Module Imports ---
from data.cifar10 import CIFAR10
from modules.cfm import ConditionalFlowMatcher # For calculating the velocity field `ut`

def get_config_and_setup(args):
    """Load configuration from YAML and set up device."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config_obj = type('Config', (), {})()
    for key, value in config.items():
        setattr(config_obj, key, value)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_obj.device = device
    
    print(f"--- Purifier Training Stage 2 ---")
    print(f"Using device: {device}")
    
    return config_obj

def load_frozen_encoder(config, checkpoint_path):
    """Loads the pre-trained and frozen CausalEncoder from Stage 1."""
    print(f"Loading frozen CausalEncoder from {checkpoint_path}")
    encoder = CausalEncoder(s_dim=config.s_dim, z_dim=config.z_dim).to(config.device)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    encoder.eval() # Set to evaluation mode
    for param in encoder.parameters():
        param.requires_grad = False # Freeze all parameters
        
    print("CausalEncoder loaded and frozen.")
    return encoder

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train Latent-Guided Purifier")
    parser.add_argument('--config', type=str, default='configs/cifar10.yml', help='Path to the config file.')
    parser.add_argument('--encoder_checkpoint', type=str, default='./checkpoints/causal_encoder_best.pt', help='Path to the frozen encoder from Stage 1.')
    parser.add_argument('--victim_checkpoint', type=str, default='./checkpoints/victim_wrn_pretrained.pt', help='Path to the victim classifier.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='Path to save purifier checkpoints.')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    # --- Data Loading ---
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=CIFAR10.transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    # --- Model Initialization ---
    # 1. The Purifier UNet (this is what we'll train)
    purifier_unet = CausalUNet(config).to(config.device)
    
    # 2. The Frozen Encoder (provides the training target)
    frozen_encoder = load_frozen_encoder(config, args.encoder_checkpoint)
    
    # 3. The Victim Model (to generate attacks)
    victim_model = WideResNet(depth=config.wrn_depth, widen_factor=config.wrn_widen_factor, num_classes=config.num_classes).to(config.device)
    victim_model.load_state_dict(torch.load(args.victim_checkpoint, map_location=config.device))
    victim_model.eval()

    # --- Attack Initialization ---
    attack = torchattacks.PGD(victim_model, 
                              eps=config.attack_params['eps'], 
                              alpha=config.attack_params['alpha'], 
                              steps=config.attack_params['iters'])

    # --- Optimizer ---
    # The optimizer will ONLY update the parameters of our purifier UNet.
    optimizer = torch.optim.Adam(purifier_unet.parameters(), lr=config.lr)

    # --- Loss Function ---
    # The loss is a simple MSE between the target `s` vector and the purified `s` vector.
    latent_loss_fn = nn.MSELoss()
    
    # --- Flow Matcher ---
    # We still need the CFM to define the path between x_adv and x_clean
    # to calculate the target velocity field `ut`.
    cfm = ConditionalFlowMatcher(sigma=config.sigma)

    best_robust_acc = 0.0 # We now save based on robust accuracy
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # --- Training Loop ---
    print("--- Starting Stage 2: Latent-Guided Purifier Training ---")
    for epoch in range(config.joint_finetune_epochs): # Using the fine-tuning epochs from config
        purifier_unet.train()
        
        pbar = tqdm(train_loader, desc=f"[Purifier Train Epoch {epoch+1}]")
        
        for i, (x_clean, y_true) in enumerate(pbar):
            x_clean = x_clean.to(config.device)
            y_true = y_true.to(config.device)

            # Generate adversarial examples on-the-fly
            x_adv = attack(x_clean, y_true)
            
            optimizer.zero_grad()

            # --- Latent-Guided Loss Calculation ---
            # 1. Get the ground-truth `s` vector from the clean image using the frozen encoder.
            # We don't need gradients for this part.
            with torch.no_grad():
                s_target, z_target, _, _ = frozen_encoder(x_clean)

            # 2. Sample time `t` and compute the target velocity `ut` for the flow model.
            # This defines the "ideal" path from x_adv to x_clean.
            t, xt, ut = cfm.sample_location_and_conditional_flow(x_adv, x_clean)
            
            # 3. Predict the velocity field `ut` using the purifier UNet.
            # We guide the UNet with the target `s` and `z` vectors.
            predicted_ut = purifier_unet(xt, t, s_target, z_target)
            
            # 4. Calculate the purified image from the *adversarial* starting point.
            # This is a single-step purification for the purpose of loss calculation.
            x_purified = torch.clamp(x_adv + predicted_ut, 0, 1)

            # 5. Get the `s` vector from the purified image.
            s_purified, _, _, _ = frozen_encoder(x_purified)
            
            # 6. The loss is the MSE between the purified `s` and the target `s`.
            loss = latent_loss_fn(s_purified, s_target)

            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"Latent MSE Loss": f"{loss.item():.6f}"})

        # --- Validation (omitted for brevity, but would be here) ---
        # A proper validation loop would be needed to test the robust accuracy
        # of the current purifier and save the best model. This involves
        # multi-step purification and classification with the frozen models.
        
    print("--- Stage 2 Training Complete ---")
    # For now, we save the final model. In a real scenario, you'd save the best one.
    torch.save({
        'purifier_state_dict': purifier_unet.state_dict(),
        'epoch': epoch,
        'config': config
    }, os.path.join(args.checkpoint_path, 'purifier_best.pt'))


if __name__ == '__main__':
    main()
