# test_causalflow_sota.py
# FINAL VERSION: A comprehensive script to rigorously evaluate and compare multiple defense
# configurations against a suite of attacks, including AutoAttack.
# CORRECTED: Fixed RuntimeError by removing torch.no_grad() from model evaluation calls.

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
import datetime
import pandas as pd

# --- Model and Data Imports ---
from models.causalunet import UNetModel
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier
from modules.cfm import ConditionalFlowMatcher
from models.networks.resnet.wideresnet import WideResNet
from data.cifar10 import get_cifar10_loaders
from autoattack import AutoAttack

# ======================================================================================
# 1. UTILITY AND HELPER FUNCTIONS
# ======================================================================================

def get_device():
    """Gets the final available device for PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_from_checkpoint(model_class, checkpoint_path, model_key, device, **kwargs):
    """A robust utility to load a model's state_dict from a composite or direct checkpoint file."""
    model = model_class(**kwargs).to(device)
    # Ensure the checkpoint is loaded correctly, especially in environments with strict loading.
    full_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(full_checkpoint, dict) and model_key in full_checkpoint:
        state_dict = full_checkpoint[model_key]
    else:
        state_dict = full_checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def pgd_attack(model, images, labels, eps, alpha, iters):
    """Performs a standard PGD attack against a given model."""
    images = images.clone().detach().to(images.device)
    labels = labels.clone().detach().to(labels.device)
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
            adv_images = torch.clamp(images + perturbation, min=0, max=1).detach()
    return adv_images

# ======================================================================================
# 2. SOTA PURIFICATION IMPLEMENTATION (from FlowPure)
# ======================================================================================

def purify_likelihood_maximization(x_adv, unet, encoder, flow_matcher, steps=40, lr=0.01, lambda_reg=0.1):
    """
    Purifies an image by optimizing it to maximize the likelihood under the flow model.
    MODIFIED FOR DIFFERENTIABILITY using BPDA (Backward Pass Differentiable Approximation).
    This resolves the "can't optimize a non-leaf Tensor" error while allowing gradient flow for attacks.
    """
    # Create a leaf tensor for the inner optimization loop. This is detached from x_adv's graph.
    x_purified_leaf = x_adv.detach().clone().requires_grad_(True)
    
    # The optimizer will update this leaf tensor.
    optimizer = torch.optim.Adam([x_purified_leaf], lr=lr)

    # s and z can be computed with no_grad as they are conditions, not optimized variables in this loop.
    with torch.no_grad():
        s, z, _, _ = encoder(x_adv)

    for _ in range(steps):
        optimizer.zero_grad()
        t_dummy = torch.zeros(x_purified_leaf.shape[0], device=x_purified_leaf.device)
        
        _, _, ut_zero = flow_matcher(x_purified_leaf, x_purified_leaf)
        
        # During the purification steps, s and z are detached constants.
        predicted_ut = unet(x_purified_leaf, t_dummy, s.detach(), z.detach())
        nll = F.mse_loss(predicted_ut, ut_zero)
        
        # The regularization term pulls the optimized leaf towards the original adversarial input.
        # We use x_adv.detach() because we are only optimizing x_purified_leaf.
        reg_loss = lambda_reg * F.mse_loss(x_purified_leaf, x_adv.detach())
        loss = nll + reg_loss
        
        # This backward call computes gradients for the *purification* loss w.r.t. x_purified_leaf.
        loss.backward()
        optimizer.step()
        
        # Clamp the data in-place.
        x_purified_leaf.data.clamp_(0, 1)

    # BPDA trick: In the forward pass, return the purified image.
    # In the backward pass, the gradient will flow through x_adv as if this function was an identity.
    # This approximates the gradient of the purification function as the identity matrix.
    return x_adv + (x_purified_leaf.detach() - x_adv.detach())

# ======================================================================================
# 3. DEFENSE MODEL WRAPPERS
# ======================================================================================

class VictimOnly(nn.Module):
    """Wrapper for the undefended victim model."""
    def __init__(self, victim_model):
        super().__init__()
        self.model = victim_model
    def forward(self, x):
        return self.model(x)

class EncoderOnlyDefense(nn.Module):
    """Defense using only the Causal Encoder and Latent Classifier."""
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    def forward(self, x):
        s, _, _, _ = self.encoder(x)
        return self.classifier(s)

class CausalFlowDefense(nn.Module):
    """Wraps the full defense pipeline for evaluation."""
    def __init__(self, unet, encoder, classifier, flow_matcher, use_sota_purify=False):
        super().__init__()
        self.unet = unet
        self.encoder = encoder
        self.classifier = classifier
        self.flow_matcher = flow_matcher
        self.use_sota_purify = use_sota_purify

    def forward(self, x):
        if self.use_sota_purify:
            # The purification function is now differentiable w.r.t x via BPDA
            x_purified = purify_likelihood_maximization(x, self.unet, self.encoder, self.flow_matcher)
        else:
            # This branch is naturally differentiable.
            s_init, z_init, _, _ = self.encoder(x)
            t_dummy = torch.zeros(x.shape[0], device=x.device)
            predicted_ut = self.unet(x, t_dummy, s_init, z_init)
            x_purified = torch.clamp(x + predicted_ut, 0, 1)

        # The final classification must allow gradients for the adaptive attack.
        s_final, _, _, _ = self.encoder(x_purified)
        logits = self.classifier(s_final)
        return logits

# ======================================================================================
# 4. EVALUATION RUNNER
# ======================================================================================

def run_full_evaluation(defense_name, defense_model, victim_model, test_loader, device, config):
    """Runs a full suite of tests on a given defense model."""
    print(f"\n--- Evaluating Defense: {defense_name} ---")
    defense_model.eval()
    '''
    # --- Clean Accuracy ---
    correct, total = 0, 0
    for images, labels in tqdm(test_loader, desc=f"1/4 Clean Eval ({defense_name})"):
        images, labels = images.to(device), labels.to(device)
        # SOTA purification requires grads in its forward pass, so we can't use torch.no_grad().
        outputs = defense_model(images)
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    clean_acc = 100 * correct / total

    # --- Preprocessor-Blind PGD ---
    correct, total = 0, 0
    for images, labels in tqdm(test_loader, desc=f"2/4 Blind PGD Eval ({defense_name})"):
        images, labels = images.to(device), labels.to(device)
        x_adv = pgd_attack(victim_model, images, labels, **config['attack_params'])
        # SOTA purification requires grads in its forward pass, so we can't use torch.no_grad().
        outputs = defense_model(x_adv)
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    blind_pgd_acc = 100 * correct / total
'''
    # --- Adaptive PGD ---
    correct, total = 0, 0
    for images, labels in tqdm(test_loader, desc=f"3/4 Adaptive PGD ({defense_name})"):
        images, labels = images.to(device), labels.to(device)
        x_adv = pgd_attack(defense_model, images, labels, **config['attack_params'])
        # SOTA purification requires grads in its forward pass, so we can't use torch.no_grad().
        outputs = defense_model(x_adv)
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        adaptive_pgd_acc = 100 * correct / total
        print("adapt", adaptive_pgd_acc)
    '''
    # --- AutoAttack ---
    # Limiting to a smaller subset for speed, as AutoAttack is very slow.
    test_subset_size = 256 
    x_test = torch.cat([x for i, (x, y) in enumerate(test_loader) if i * config['batch_size'] < test_subset_size], dim=0)
    y_test = torch.cat([y for i, (x, y) in enumerate(test_loader) if i * config['batch_size'] < test_subset_size], dim=0)

    adversary = AutoAttack(defense_model, norm='Linf', eps=config['attack_params']['eps'], version='standard', device=device)
    print(f"Running AutoAttack on {defense_name}... (on a subset of {test_subset_size} images)")
    
    # AutoAttack requires gradients to run, so it must not be in a no_grad block.
    x_adv_aa = adversary.run_standard_evaluation(x_test, y_test, bs=config['batch_size'])
    
    # Evaluating the results also requires running the forward pass without no_grad.
    outputs = defense_model(x_adv_aa.to(device))
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        autoattack_acc = 100 * (predicted.cpu() == y_test).sum().item() / len(y_test)
'''
    return {
        "Defense": defense_name,
        #"Clean Acc (%)": clean_acc,
        #"Blind PGD Acc (%)": blind_pgd_acc,
        "Adaptive PGD Acc (%)": adaptive_pgd_acc,
        #"AutoAttack Acc (%)": autoattack_acc
    }

# ======================================================================================
# 5. MAIN SCRIPT
# ======================================================================================

def main():
    # --- Config ---
    config = {
        'batch_size': 32, 's_dim': 128, 'z_dim': 128, 'num_classes': 10,
        'image_size': 32, 'in_channels': 3, 'model_channels': 128, 'out_channels': 3,
        'num_res_blocks': 2, 'attention_resolutions': "16,8",
        'wrn_depth': 28, 'wrn_widen_factor': 10,
        'attack_params': {'eps': 8/255, 'alpha': 2/255, 'iters': 10}
    }
    
    device = get_device()
    print(f"--- CausalFlow SOTA Evaluation ---")
    print(f"Using device: {device}")
    print("-" * 80)

    # --- Load Data ---
    _, test_loader = get_cifar10_loaders(batch_size=config['batch_size'])
    
    # --- Load All Models ---
    try:
        print("Loading trained models...")
        unet_kwargs = {'image_size':config['image_size'], 'in_channels':config['in_channels'], 'model_channels':config['model_channels'], 'out_channels':config['out_channels'], 'num_res_blocks':config['num_res_blocks'], 'attention_resolutions':[int(res) for res in config['attention_resolutions'].split(',')], 's_dim':config['s_dim'], 'z_dim':config['z_dim']}
        encoder_kwargs = {'backbone_arch':'WRN', 's_dim':config['s_dim'], 'z_dim':config['z_dim'], 'wrn_depth':config['wrn_depth'], 'wrn_widen_factor':config['wrn_widen_factor']}
        classifier_kwargs = {'s_dim':config['s_dim'], 'num_classes':config['num_classes']}
        
        unet = load_model_from_checkpoint(UNetModel, "checkpoints/causalflow_final.pt", 'unet', device, **unet_kwargs)
        encoder = load_model_from_checkpoint(CausalEncoder, "checkpoints/causalflow_final.pt", 'encoder', device, **encoder_kwargs)
        classifier = load_model_from_checkpoint(LatentClassifier, "checkpoints/causalflow_final.pt", 'classifier', device, **classifier_kwargs)
        victim_model = load_model_from_checkpoint(WideResNet, "checkpoints/victim_wrn_pretrained.pt", 'victim_model', device, depth=config['wrn_depth'], widen_factor=config['wrn_widen_factor'], num_classes=config['num_classes'])
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load model checkpoints. Please ensure they are trained and available.")
        print(f"Error details: {e}")
        print("Please check that 'checkpoints/causalflow_final.pt' and 'checkpoints/victim_wrn_pretrained.pt' exist.")
        return

    # --- Instantiate all defense configurations ---
    flow_matcher = ConditionalFlowMatcher(sigma=0.0)
    
    defense_configs = {
        #"Victim Model (No Defense)": VictimOnly(victim_model),
        #"Encoder-Only Defense": EncoderOnlyDefense(encoder, classifier),
        #"CausalFlow (Simple Purify)": CausalFlowDefense(unet, encoder, classifier, flow_matcher, use_sota_purify=False),
        "CausalFlow (SOTA Purify)": CausalFlowDefense(unet, encoder, classifier, flow_matcher, use_sota_purify=True)
    }

    # --- Run evaluations and collect results ---
    results = []
    for name, model in defense_configs.items():
        model.eval()
        result = run_full_evaluation(name, model, victim_model, test_loader, device, config)
        results.append(result)

    # --- Final Report ---
    df = pd.DataFrame(results)
    print("\n\n" + "="*80)
    print("                     CausalFlow Final Performance Summary")
    print("="*80)
    print(df.to_markdown(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
