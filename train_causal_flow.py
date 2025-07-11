# train_causal_flow.py
# Final, methodologically sound script for training the CausalFlow model.
# Implements a four-stage resumable pipeline with the correct training dynamics.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import os
import sys
import datetime
import numpy as np

# --- Model and Module Imports ---
from models.causalunet import UNetModel
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier
from modules.cfm import ConditionalFlowMatcher
from modules.cib import CIBLoss
from data.cifar10 import get_cifar10_loaders
from models.networks.resnet.wideresnet import WideResNet

# ======================================================================================
# 1. UTILITY AND HELPER FUNCTIONS
# ======================================================================================

class Logger(object):
    """Redirects print statements to both the console and a log file."""
    def __init__(self, filename="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_device():
    """Gets the best available device for PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pgd_attack(model, images, labels, eps, alpha, iters):
    """Performs the PGD attack on a given model."""
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        model.zero_grad()
        cost = loss_fn(outputs, labels)
        cost.backward()
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            perturbation = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + perturbation, min=0, max=1)
    return adv_images

def validate_victim(model, val_loader, loss_fn, device):
    """Calculates loss and accuracy for the victim model on a validation set."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    model.train()
    return total_loss / len(val_loader), 100 * correct / total

def validate_purifier(unet, encoder, classifier, victim_model, val_loader, device, attack_params):
    """Calculates Clean and Robust Accuracy for the CausalFlow model."""
    unet.eval()
    encoder.eval()
    classifier.eval()
    victim_model.eval()
    
    clean_correct, robust_correct, total = 0, 0, 0

    for x_clean, y in tqdm(val_loader, desc="Validating", leave=False, ncols=100):
        x_clean, y = x_clean.to(device), y.to(device)
        total += y.size(0)

        # ### BUG FIX: The attack requires gradients, so it must be outside the no_grad() block ###
        x_adv = pgd_attack(victim_model, x_clean, y, **attack_params)

        with torch.no_grad():
            # --- Clean Accuracy ---
            s_clean, _, _, _ = encoder(x_clean)
            logits_clean = classifier(s_clean)
            clean_correct += (torch.argmax(logits_clean, dim=1) == y).sum().item()

            # --- Robust Accuracy ---
            s_adv, _, _, _ = encoder(x_adv)
            logits_robust = classifier(s_adv)
            robust_correct += (torch.argmax(logits_robust, dim=1) == y).sum().item()

    clean_acc = (clean_correct / total) * 100
    robust_acc = (robust_correct / total) * 100
    
    unet.train()
    encoder.train()
    classifier.train()
    
    return clean_acc, robust_acc


# ======================================================================================
# 2. MAIN ORCHESTRATOR
# ======================================================================================

def main():
    # --- Configuration ---
    config = {
        'image_size': 32, 'in_channels': 3, 'model_channels': 128, 'out_channels': 3,
        'num_res_blocks': 2, 'attention_resolutions': "16,8", 's_dim': 128, 'z_dim': 128,
        'num_classes': 10, 'lr': 1e-4, 'batch_size': 128, 'finetune_lr': 1e-5,
        'victim_train_epochs': 200, 'recon_pretrain_epochs': 50,
        'causal_pretrain_epochs': 50, 'joint_finetune_epochs': 150,
        'sigma': 0.01,
        'alpha_recon': 5.0, 'gamma_ce': 20.0, 'lambda_kl': 0.5, 'eta_club': 1e-2,
        'wrn_depth': 28, 'wrn_widen_factor': 10,
        'attack_params': {'eps': 8/255, 'alpha': 2/255, 'iters': 10}
    }

    # --- Setup Logging and Device ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"training_log_{timestamp}.txt"
    sys.stdout = Logger(log_file_path)

    device = get_device()
    print(f"--- CausalFlow Training Run ({timestamp}) ---")
    print(f"Using device: {device}")
    print("Configuration:", config)
    print("-" * 80)

    Path("checkpoints").mkdir(exist_ok=True)
    full_train_loader, test_loader = get_cifar10_loaders(batch_size=config['batch_size'])

    # --- Model Instantiation ---
    unet = UNetModel(image_size=config['image_size'], in_channels=config['in_channels'], model_channels=config['model_channels'], out_channels=config['out_channels'], num_res_blocks=config['num_res_blocks'], attention_resolutions=[int(res) for res in config['attention_resolutions'].split(',')], s_dim=config['s_dim'], z_dim=config['z_dim']).to(device)
    encoder = CausalEncoder(backbone_arch='WRN', s_dim=config['s_dim'], z_dim=config['z_dim'], wrn_depth=config['wrn_depth'], wrn_widen_factor=config['wrn_widen_factor']).to(device)
    classifier = LatentClassifier(s_dim=config['s_dim'], num_classes=config['num_classes']).to(device)
    victim_model = WideResNet(depth=config['wrn_depth'], widen_factor=config['wrn_widen_factor'], num_classes=config['num_classes']).to(device)
    
    # === STAGE 0: VICTIM MODEL PRE-TRAINING ===
    print("\n--- Stage 0: Victim Classifier Setup ---")
    victim_checkpoint_path = "checkpoints/victim_wrn_pretrained.pt"
    if not os.path.exists(victim_checkpoint_path):
        print(f"Victim checkpoint not found. Training a new one for up to {config['victim_train_epochs']} epochs...")
        full_train_dataset = full_train_loader.dataset
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
        victim_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        victim_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        victim_optimizer = torch.optim.Adam(victim_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(victim_optimizer, 'min', patience=5, factor=0.5, verbose=True)
        loss_fn = nn.CrossEntropyLoss()
        best_val_loss, patience_counter = float('inf'), 0
        for epoch in range(config['victim_train_epochs']):
            victim_model.train()
            for x_clean, y in tqdm(victim_train_loader, desc=f"Training Victim Epoch {epoch+1}"):
                x_clean, y = x_clean.to(device), y.to(device)
                victim_optimizer.zero_grad()
                logits = victim_model(x_clean)
                loss = loss_fn(logits, y)
                loss.backward()
                victim_optimizer.step()
            val_loss, val_acc = validate_victim(victim_model, victim_val_loader, loss_fn, device)
            print(f"Victim Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(victim_model.state_dict(), victim_checkpoint_path)
                print(f"New best model saved to {victim_checkpoint_path}")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 15:
                print("Stopping early.")
                break
    print(f"Loading best pre-trained victim model from {victim_checkpoint_path}")
    victim_model.load_state_dict(torch.load(victim_checkpoint_path, map_location=device))
    victim_model.eval()

    # --- Optimizer for CausalFlow Models ---
    optimizer = torch.optim.Adam(list(unet.parameters()) + list(encoder.parameters()) + list(classifier.parameters()), lr=config['lr'])
    flow_matcher = ConditionalFlowMatcher(sigma=config['sigma'])

    # === STAGE 1: RECONSTRUCTION PRE-TRAINING ===
    print("\n--- Stage 1: Purifier Reconstruction Pre-training ---")
    recon_checkpoint_path = "checkpoints/recon_pretrained.pt"
    if not os.path.exists(recon_checkpoint_path):
        print(f"Starting reconstruction pre-training for {config['recon_pretrain_epochs']} epochs...")
        for epoch in range(config['recon_pretrain_epochs']):
            for x_clean, _ in tqdm(full_train_loader, desc=f"Recon Pre-train Epoch {epoch+1}"):
                x_clean = x_clean.to(device)
                optimizer.zero_grad()
                t, xt, ut = flow_matcher(x_clean, x_clean)
                s, z, _, _ = encoder(x_clean)
                predicted_ut = unet(xt, t, s, z)
                loss = F.mse_loss(predicted_ut, ut)
                loss.backward()
                optimizer.step()
        torch.save({'unet': unet.state_dict(), 'encoder': encoder.state_dict()}, recon_checkpoint_path)
    else:
        print("Loading pre-trained reconstruction models.")
        state_dict = torch.load(recon_checkpoint_path, map_location=device)
        unet.load_state_dict(state_dict['unet'])
        encoder.load_state_dict(state_dict['encoder'])
    
    # --- Loss function for main training stages ---
    cib_loss_fn = CIBLoss(alpha_recon=config['alpha_recon'], lambda_kl=config['lambda_kl'], gamma_ce=config['gamma_ce'], eta_club=config['eta_club'], s_dim=config['s_dim'], z_dim=config['z_dim']).to(device)
    optimizer.add_param_group({'params': cib_loss_fn.parameters()})

    # === STAGE 2 & 3: MAIN TRAINING PIPELINE ===
    print("\n--- Starting Main Training Pipeline ---")
    total_main_epochs = config['causal_pretrain_epochs'] + config['joint_finetune_epochs']
    
    causal_checkpoint_path = "checkpoints/causal_pretrained.pt"
    start_epoch = 0
    if os.path.exists(causal_checkpoint_path):
        print("Found existing causal pre-training checkpoint. Loading and moving to fine-tuning.")
        state_dict = torch.load(causal_checkpoint_path, map_location=device)
        encoder.load_state_dict(state_dict['encoder'])
        classifier.load_state_dict(state_dict['classifier'])
        start_epoch = config['causal_pretrain_epochs']

    best_robust_acc = 0
    patience_counter = 0
    
    for epoch in range(start_epoch, total_main_epochs):
        is_causal_pretrain_phase = epoch < config['causal_pretrain_epochs']

        if epoch == 0 and is_causal_pretrain_phase:
            print("--- Beginning Stage 2: Causal Representation Pre-training ---")
            unet.eval()
        
        if epoch == config['causal_pretrain_epochs']:
            print("\n" + "="*80)
            print("--- Causal Pre-training Complete ---")
            torch.save({'encoder': encoder.state_dict(), 'classifier': classifier.state_dict()}, causal_checkpoint_path)
            print(f"Saved causal pre-trained models to {causal_checkpoint_path}")
            print(f"--- Beginning Stage 3: Joint Fine-tuning with LR: {config['finetune_lr']} ---")
            print("="*80 + "\n")
            unet.train()
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['finetune_lr']

        phase_desc = f"Causal Pre-train Epoch {epoch+1}" if is_causal_pretrain_phase else f"Joint Fine-tune Epoch {epoch - config['causal_pretrain_epochs'] + 1}"

        for i, (x_clean, y) in enumerate(tqdm(full_train_loader, desc=phase_desc)):
            x_clean, y = x_clean.to(device), y.to(device)
            optimizer.zero_grad()
            
            batch_size = x_clean.shape[0]
            half_batch = batch_size // 2
            x_clean_adv, y_adv = x_clean[:half_batch], y[:half_batch]
            x_clean_benign, y_benign = x_clean[half_batch:], y[half_batch:]
            x_adv = pgd_attack(victim_model, x_clean_adv, y_adv, **config['attack_params'])
            x_input = torch.cat([x_adv, x_clean_benign], dim=0)
            x_target = torch.cat([x_clean_adv, x_clean_benign], dim=0)
            y_target = torch.cat([y_adv, y_benign], dim=0)
            
            s, z, s_params, z_params = encoder(x_input)
            logits = classifier(s)
            t, xt, ut = flow_matcher(x_input, x_target)
            predicted_ut = unet(xt, t, s, z)
            
            loss, loss_dict = cib_loss_fn(predicted_ut, ut, logits, y_target, s, z, s_params, z_params, include_recon=(not is_causal_pretrain_phase))
            
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                log_str = f"[{phase_desc}] [Batch {i}/{len(full_train_loader)}] Total Loss: {loss.item():.4f}"
                for k, v in loss_dict.items():
                    log_str += f" | {k}: {v:.4f}"
                print(log_str)


        # ### BUG FIX: Corrected argument order for validation function ###
        clean_acc, robust_acc = validate_purifier(unet, encoder, classifier, victim_model, test_loader, device, config['attack_params'])
        print(f"---===[ Validation Epoch {epoch+1} ]===--- Clean Acc: {clean_acc:.2f}% | Robust Acc: {robust_acc:.2f}%")
        if best_robust_acc < robust_acc:
                best_robust_acc = robust_acc
                torch.save({
                    'unet': unet.state_dict(),
                    'encoder': encoder.state_dict(),
                    'classifier': classifier.state_dict(),
                }, "checkpoints/causalflow_final.pt")
                print(f"New best model saved")
                patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 10:
            print("Stopping early.")
            break

    print("\nFull training pipeline complete.")
    torch.save({
        'unet': unet.state_dict(),
        'encoder': encoder.state_dict(),
        'classifier': classifier.state_dict(),
    }, "checkpoints/causalflow_final.pt")

if __name__ == "__main__":
    main()
