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

    # Create a config object for dot notation access
    config_obj = type('Config', (), {})()
    for key, value in config.items():
        setattr(config_obj, key, value)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_obj.device = device

    print(f"--- Ablation 1: Preprocessor-Blind PGD Training ---")
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

# --- Victim Model for Adversarial Example Generation ---
class VictimModel(nn.Module):
    """
    A simple wrapper for the frozen encoder and classifier.
    This model is the "victim" for the PGD attack during training data generation.
    It is "blind" to the existence of the purifier.
    """
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        s, _, _, _ = self.encoder(x)
        logits = self.classifier(s)
        return logits

# --- Differentiable ODE Solvers ---
def solve_ode_for_training(purifier_unet, s_cond, z_cond, x_start, start_t, n_steps=5):
    """
    A lightweight, differentiable multi-step ODE solver for the training loop.
    Provides a stable estimate of the purified image for the latent loss.
    Integrates from the sampled time `start_t` to 1.0.
    """
    x_t = x_start
    # Calculate the time step for the remaining interval
    dt = (1.0 - start_t.item()) / n_steps

    for i in range(n_steps):
        current_t_val = start_t.item() + i * dt
        t = torch.full((x_t.shape[0],), current_t_val, device=x_t.device)
        velocity = purifier_unet(x_t, t, s_cond, z_cond)
        x_t = x_t + velocity * dt

    return torch.clamp(x_t, 0, 1)

def solve_ode_for_inference(purifier_unet, s_cond, z_cond, x_start, n_steps=20):
    """
    A more precise ODE solver for validation and inference. NOT used in the
    training backprop path. Integrates over the full [0, 1] interval.
    """
    x_t = x_start.clone()
    dt = 1.0 / n_steps

    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            velocity = purifier_unet(x_t, t, s_cond, z_cond)
            x_t = x_t + velocity * dt

    return torch.clamp(x_t, 0, 1)


# --- Full Defense Wrapper for Validation Attack ---
class FullDefense(nn.Module):
    """
    A wrapper to make the full defense pipeline differentiable for the VALIDATION
    attacker. This simulates a powerful, fully-adaptive white-box attack.
    """
    def __init__(self, purifier, encoder, classifier, n_steps=10):
        super().__init__()
        self.purifier = purifier
        self.encoder = encoder
        self.classifier = classifier
        self.n_steps = n_steps # The attacker knows the number of purification steps

    def forward(self, x):
        x_t = x
        dt = 1.0 / self.n_steps

        # Attacker gets the initial conditioning from the input
        s_cond, z_cond, _, _ = self.encoder(x_t)

        # Differentiable ODE solve that the attacker's gradients can flow through
        for i in range(self.n_steps):
            t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
            velocity = self.purifier(x_t, t, s_cond, z_cond)
            x_t = x_t + velocity * dt

        x_purified = torch.clamp(x_t, 0, 1)
        s_final, _, _, _ = self.encoder(x_purified)
        logits = self.classifier(s_final)
        return logits

def main():
    parser = argparse.ArgumentParser(description="Ablation 1: Pre-Blind PGD Training")
    parser.add_argument('--config', type=str, default='configs/cifar10_causalflow.yml', help='Path to the config file.')
    parser.add_argument('--encoder_checkpoint', type=str, default='./checkpoints/causal_encoder_best.pt', help='Path to the frozen encoder/classifier checkpoint.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_ablation')
    parser.add_argument('--log_path', type=str, default='./logs_ablation')
    args = parser.parse_args()

    config = get_config_and_setup(args)

    # --- Data Loading ---
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=CIFAR10.get_train_transform())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=CIFAR10.get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model Initialization ---
    purifier_unet = CausalUNet(config).to(config.device)
    frozen_encoder, frozen_classifier = load_frozen_models(config, args.encoder_checkpoint)

    optimizer = torch.optim.Adam(purifier_unet.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.joint_finetune_epochs, eta_min=1e-6)

    pixel_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()
    cfm = ConditionalFlowMatcher(sigma=0.0) # sigma=0 for rectified flow

    # Create directories for this ablation run
    ablation_name = "pgd_preblind"
    checkpoint_dir = os.path.join(args.checkpoint_path, ablation_name)
    log_dir = os.path.join(args.log_path, ablation_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Attacker for TRAINING DATA GENERATION (Preprocessor-Blind) ---
    victim_model = VictimModel(frozen_encoder, frozen_classifier).to(config.device)
    train_attack = torchattacks.PGD(victim_model, eps=config.attack_params['eps'], alpha=config.attack_params['alpha'], steps=10)

    # --- Attacker for VALIDATION (Fully-Adaptive) ---
    attackable_defense = FullDefense(purifier_unet, frozen_encoder, frozen_classifier, n_steps=10).to(config.device)
    validation_attack = torchattacks.PGD(attackable_defense, eps=config.attack_params['eps'], alpha=config.attack_params['alpha'], steps=20)

    best_robust_acc = 0.0

    print("--- Starting Training ---")
    for epoch in range(config.joint_finetune_epochs):
        purifier_unet.train()
        pbar = tqdm(train_loader, desc=f"[{ablation_name.upper()} Train Epoch {epoch+1}]")

        for i, (x_clean, y) in enumerate(pbar):
            x_clean, y = x_clean.to(config.device), y.to(config.device)

            optimizer.zero_grad()

            # --- Step 1: Generate Pre-Processor Blind Adversarial Data ---
            # The attacker only sees the classifier, not the purifier.
            x_adv = train_attack(x_clean, y)

            # --- Step 2: Calculate Losses for Purifier Training ---
            # Get the target causal factors from the *clean* image
            with torch.no_grad():
                s_target, z_target, _, _ = frozen_encoder(x_clean)

            # Sample a time `t`, an interpolated point `xt`, and the ground-truth velocity `ut`
            # The flow is from the adversarial image (x0) to the clean image (x1)
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0=x_adv, x1=x_clean)

            # Predict the velocity with the UNet, conditioned on the target latents
            predicted_ut = purifier_unet(xt, t, s_target, z_target)

            # LOSS 1: Flow Matching Loss (Pixel-space velocity)
            flow_loss = pixel_loss_fn(predicted_ut, ut)

            # LOSS 2: Latent Consistency Loss
            # Get a stable estimate of the purified image using a few ODE steps
            x_purified_approx = solve_ode_for_training(purifier_unet, s_target, z_target, xt, t, n_steps=5)
            # Get the causal latent of this estimated purified image
            s_denoised, _, _, _ = frozen_encoder(x_purified_approx)
            # The loss encourages the purified latent to match the target latent
            latent_loss = latent_loss_fn(s_denoised, s_target)

            # Combine the losses
            total_loss = flow_loss + config.lambda_latent * latent_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(purifier_unet.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Flow Loss": f"{flow_loss.item():.4f}",
                "Latent Loss": f"{latent_loss.item():.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}"
            })

        # --- VALIDATION LOOP ---
        purifier_unet.eval()
        total_clean, correct_clean = 0, 0
        total_robust, correct_robust = 0, 0

        pbar_val = tqdm(test_loader, desc=f"[{ablation_name.upper()} Validation Epoch {epoch+1}]")

        for i, (x, y) in enumerate(pbar_val):
            x, y = x.to(config.device), y.to(config.device)

            # Generate adversarial examples using the FULLY ADAPTIVE validation attacker
            x_adv_val = validation_attack(x, y)

            # --- Perform Purification and Classification (with no_grad) ---
            # On Clean images
            s_cond_clean, z_cond_clean, _, _ = frozen_encoder(x)
            x_purified_clean = solve_ode_for_inference(purifier_unet, s_cond_clean, z_cond_clean, x, n_steps=20)
            s_final_clean, _, _, _ = frozen_encoder(x_purified_clean)
            logits_clean = frozen_classifier(s_final_clean)
            correct_clean += (logits_clean.argmax(1) == y).sum().item()
            total_clean += y.size(0)

            # On Adversarial images
            s_cond_adv, z_cond_adv, _, _ = frozen_encoder(x_adv_val)
            x_purified_adv = solve_ode_for_inference(purifier_unet, s_cond_adv, z_cond_adv, x_adv_val, n_steps=20)
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
        print(f"Robust Accuracy (Adaptive PGD-20): {robust_acc:.2f}%")

        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            print(f"*** New best robust accuracy: {best_robust_acc:.2f}%. Saving model. ***")
            torch.save({
                'purifier_state_dict': purifier_unet.state_dict(),
            }, os.path.join(checkpoint_dir, f'{ablation_name}_best.pt'))

        # Log sample images for qualitative analysis
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_x = x[:8]
                sample_y = y[:8]
                sample_x_adv = validation_attack(sample_x, sample_y)
                s_cond_log, z_cond_log, _, _ = frozen_encoder(sample_x_adv)
                sample_x_purified = solve_ode_for_inference(purifier_unet, s_cond_log, z_cond_log, sample_x_adv, n_steps=20)

                comparison = torch.cat([sample_x.cpu(), sample_x_adv.cpu(), sample_x_purified.cpu()])
                save_image(comparison, os.path.join(log_dir, f'epoch_{epoch+1}_comparison.png'), nrow=8)
                print(f"Saved sample image grid to {log_dir}")

        scheduler.step()

    print(f"\n--- {ablation_name.upper()} Training Complete ---")
    print(f"Best robust accuracy achieved: {best_robust_acc:.2f}%")

if __name__ == '__main__':
    main()
