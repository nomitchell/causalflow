# train_causal_flow.py
# PURPOSE: This is the main orchestrator for the entire project. It brings together
# the models, the loss functions, and the data to train the CausalFlow model.
# The logic here is a hybrid of the training scripts from both repositories.

import torch
import torch.nn.functional as F
import yaml # Using YAML for configs is best practice
from pathlib import Path
from tqdm import tqdm

pretrained = True

# --- Model and Module Imports ---
from models.causalunet import UNetModel # Assuming the modified UNetModel is here
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier
from modules.cfm import ConditionalFlowMatcher
from modules.cib import CIBLoss
from data.cifar10 import get_cifar10_loaders

# --- Helper function for device management ---
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # --- 1. Configuration ---
    # It's best practice to load hyperparameters from a separate config file.
    # This makes experiments repeatable and easy to track.
    # For now, we define it as a dictionary.
    config = {
        'image_size': 32,
        'in_channels': 3,
        'model_channels': 128,
        'out_channels': 3,
        'num_res_blocks': 2,
        'attention_resolutions': "16,8",
        's_dim': 128,
        'z_dim': 128,
        'num_classes': 10,
        'lr': 1e-4,
        'batch_size': 1,
        'pretrain_epochs': 50,
        'main_train_epochs': 200,
        'sigma': 0.01,
        'lambda_kl': 1e-2,
        'gamma_ce': 1e-2,
        'eta_club': 1e-6, # maybe 7
        'wrn_depth': 28,
        'wrn_widen_factor': 10,
    }

    # --- 2. Setup (Device, Data, Models) ---
    device = get_device()
    print(f"Using device: {device}")

    # Data Loaders
    train_loader, test_loader = get_cifar10_loaders(batch_size=config['batch_size'])

    # Model Instantiation and moving them to the correct device
    attention_resolutions = [int(res) for res in config['attention_resolutions'].split(',')]
    unet = UNetModel(
        image_size=config['image_size'], in_channels=config['in_channels'],
        model_channels=config['model_channels'], out_channels=config['out_channels'],
        num_res_blocks=config['num_res_blocks'], attention_resolutions=attention_resolutions,
        s_dim=config['s_dim'], z_dim=config['z_dim']
    ).to(device)

    encoder = CausalEncoder(
        backbone_arch='WRN', s_dim=config['s_dim'], z_dim=config['z_dim'],
        wrn_depth=config['wrn_depth'], wrn_widen_factor=config['wrn_widen_factor']
    ).to(device)

    classifier = LatentClassifier(s_dim=config['s_dim'], num_classes=config['num_classes']).to(device)

    # Loss and Flow Matcher Instantiation
    cib_loss_fn = CIBLoss(
        lambda_kl=config['lambda_kl'], gamma_ce=config['gamma_ce'],
        eta_club=config['eta_club'], s_dim=config['s_dim'], z_dim=config['z_dim']
    ).to(device) # The CIB loss also has parameters (the CLUB estimator) that need to be on the GPU

    flow_matcher = ConditionalFlowMatcher(sigma=config['sigma'])

    # Optimizer for all models
    optimizer = torch.optim.Adam(
        list(unet.parameters()) + list(encoder.parameters()) +
        list(classifier.parameters()) + list(cib_loss_fn.parameters()), # Include loss parameters
        lr=config['lr']
    )

    # Create directories for saving models
    Path("checkpoints").mkdir(exist_ok=True)

    # --- 3. Stage 1: Pre-training (Reconstruction) ---
    print("--- Starting Stage 1: Pre-training for Reconstruction ---")
    if not pretrained:
        for epoch in range(config['pretrain_epochs']):
            for i, (x_clean, y) in tqdm(enumerate(train_loader)):
                x_clean = x_clean.to(device)
                optimizer.zero_grad()
                
                t, xt, ut = flow_matcher(x_clean)
                # Encoder must also run on the correct device
                s, z, _, _ = encoder(x_clean)
                
                predicted_ut = unet(xt, t, s, z)
                
                loss = F.mse_loss(predicted_ut, ut)
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(f"[Pre-train Epoch {epoch+1}/{config['pretrain_epochs']}] [Batch {i}/{len(train_loader)}] Loss: {loss.item():.4f}")
        print("Pre-training complete. Saving pre-trained models.")
        torch.save(unet.state_dict(), "checkpoints/unet_pretrained.pt")
        torch.save(encoder.state_dict(), "checkpoints/encoder_pretrained.pt")
    else:
        print("Loading pretrained models from checkpoints for Stage 2...")
        unet.load_state_dict(torch.load("checkpoints/unet_pretrained.pt", map_location=device))
        encoder.load_state_dict(torch.load("checkpoints/encoder_pretrained.pt", map_location=device))

    # --- 4. Stage 2: Joint Training (Full CIB Loss) ---
    print("--- Starting Stage 2: Joint Training with CIB Loss ---")
    for epoch in range(config['main_train_epochs']):
        for i, (x_clean, y) in tqdm(enumerate(train_loader)):
            x_clean, y = x_clean.to(device), y.to(device)
            optimizer.zero_grad()

            s, z, s_params, z_params = encoder(x_clean)
            t, xt, ut = flow_matcher(x_clean)
            predicted_ut = unet(xt, t, s, z)
            logits = classifier(s)
            
            # The full CIB loss calculation
            loss, loss_dict = cib_loss_fn(predicted_ut, ut, logits, y, s, z, s_params, z_params)
            
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                # Basic logging to console
                log_str = f"[Main Epoch {epoch+1}/{config['main_train_epochs']}] [Batch {i}/{len(train_loader)}] Total Loss: {loss.item():.4f}"
                for k, v in loss_dict.items():
                    log_str += f" | {k}: {v:.4f}"
                print(log_str)
                # TODO: Replace print with a proper logger like TensorBoard or W&B

    print("Main training complete. Saving final models.")
    torch.save(unet.state_dict(), "checkpoints/unet_final.pt")
    torch.save(encoder.state_dict(), "checkpoints/encoder_final.pt")
    torch.save(classifier.state_dict(), "checkpoints/classifier_final.pt")


if __name__ == "__main__":
    main()