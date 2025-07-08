# inference_causal_flow.py
# PURPOSE: Implements the three-stage defense pipeline for classifying a single
# (potentially adversarial) image at test time. This is where the model's
# robustness is actually realized.

import torch
import torch.nn.functional as F
from models.causalunet import UNetModel
from models.encoder import CausalEncoder
from models.classifier import LatentClassifier
from modules.cfm import ConditionalFlowMatcher
from autoattack import AutoAttack  # Requires 'pip install autoattack'

# --- Utility: Load model weights ---
def load_model(model_class, weight_path, device, **kwargs):
    model = model_class(**kwargs).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

# --- Purification: Reverse ODE (FlowPure baseline) ---
def purify_image_reverse_ode(adversarial_image, unet_model, flow_matcher, steps=20):
    """
    Stage 1: Adversarial Purification (AP).
    ### HARD RESEARCH AREA 3: Effective Purification ###
    # CausalDiff uses Likelihood Maximization (LM) for purification. This is an
    # iterative optimization process that finds a clean image `x*` that is close
    # to the adversarial input but has a high likelihood under the generative model.
    #
    # TO-DO:
    # 1. Implement LM for a flow-based model. This is non-trivial. It involves
    #    calculating the gradient of the log-likelihood with respect to the input `x`
    #    and performing gradient ascent.
    # 2. As a simpler placeholder/baseline, one could run the reverse ODE/SDE of the
    #    flow from t=1 to t=0, starting from the adversarial image. This is what
    #    FlowPure does, but it may be less effective than true LM.
    """
    device = adversarial_image.device
    x = adversarial_image.clone().detach().to(device)
    t_vals = torch.linspace(1, 0, steps, device=device)
    dt = -1.0 / (steps - 1)
    # Dummy s, z for unconditional purification (or use encoder if desired)
    s = torch.zeros(x.shape[0], unet_model.s_proj.in_features, device=device)
    z = torch.zeros(x.shape[0], unet_model.z_proj.in_features, device=device)
    for t in t_vals:
        t_batch = torch.full((x.shape[0],), t, device=device)
        with torch.no_grad():
            velocity = unet_model(x, t_batch, s, z)
        x = x + velocity * dt
    return x.detach()

# --- Purification: Likelihood Maximization (SOTA approach) ---
def purify_image_likelihood_max(adversarial_image, unet_model, flow_matcher, encoder_model=None, 
                               steps=100, lr=0.01, lambda_reg=0.1, noise_scale=0.01):
    """
    Stage 1: Adversarial Purification via Likelihood Maximization.
    
    This implements the SOTA purification approach from FlowPure:
    - Maximizes log-likelihood under the trained flow model
    - Uses L2 regularization to stay close to adversarial input
    - Optionally uses encoder to get better s, z estimates
    
    Args:
        adversarial_image: [B, C, H, W] potentially adversarial input
        unet_model: trained U-Net flow model
        flow_matcher: conditional flow matcher
        encoder_model: optional encoder for better s, z estimates
        steps: number of optimization steps
        lr: learning rate for optimization
        lambda_reg: L2 regularization strength
        noise_scale: noise scale for better exploration
    
    Returns:
        purified_image: [B, C, H, W] purified image
    """
    device = adversarial_image.device
    batch_size = adversarial_image.shape[0]
    
    # Initialize from adversarial image with small noise for better exploration
    x = adversarial_image.clone().detach().to(device)
    x = x + torch.randn_like(x) * noise_scale
    x.requires_grad_(True)
    
    # Get s, z estimates (either from encoder or zeros)
    if encoder_model is not None:
        with torch.no_grad():
            s, z, _, _ = encoder_model(adversarial_image)
    else:
        s = torch.zeros(batch_size, unet_model.s_proj.in_features, device=device)
        z = torch.zeros(batch_size, unet_model.z_proj.in_features, device=device)
    
    # Optimizer for the image
    optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.999))
    
    # Track optimization progress
    losses = []
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Sample multiple timesteps for better likelihood estimation
        num_samples = 5
        total_nll = 0
        
        for _ in range(num_samples):
            # Get flow samples
            t, xt, ut = flow_matcher(x)
            
            # Predict velocity field
            predicted_ut = unet_model(xt, t, s, z)
            
            # Negative log-likelihood: MSE between predicted and target velocity
            # This approximates -log p(x) under the flow model
            nll = F.mse_loss(predicted_ut, ut, reduction='mean')
            total_nll += nll
        
        # Average negative log-likelihood
        avg_nll = total_nll / num_samples
        
        # L2 regularization to stay close to adversarial input
        reg_loss = lambda_reg * F.mse_loss(x, adversarial_image, reduction='mean')
        
        # Total loss: minimize negative log-likelihood + regularization
        total_loss = avg_nll + reg_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
        
        optimizer.step()
        
        # Clamp to valid image range
        x.data.clamp_(0, 1)
        
        losses.append(total_loss.item())
        
        # Early stopping if loss plateaus
        if step > 20 and abs(losses[-1] - losses[-2]) < 1e-6:
            break
    
    return x.detach()

# --- Causal Factor Inference (optimize s, z) ---
def infer_causal_factors(purified_image, encoder_model, unet_model, flow_matcher, steps=100, lr=0.01):
    with torch.no_grad():
        s, z, _, _ = encoder_model(purified_image)
    s = s.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([s, z], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        t, xt, ut = flow_matcher(purified_image.repeat(10, 1, 1, 1))
        predicted_ut = unet_model(xt, t.squeeze(), s.repeat(10, 1), z.repeat(10, 1))
        loss = F.mse_loss(predicted_ut, ut)
        loss.backward()
        optimizer.step()
    return s.detach(), z.detach()

# --- Enhanced Causal Factor Inference ---
def infer_causal_factors_enhanced(purified_image, encoder_model, unet_model, flow_matcher, 
                                steps=200, lr=0.005, num_samples=10):
    """
    Stage 2: Enhanced Causal Factor Inference.
    
    Optimizes s and z to maximize likelihood under the flow model,
    starting from encoder estimates and refining them.
    
    Args:
        purified_image: [B, C, H, W] purified image
        encoder_model: trained encoder
        unet_model: trained U-Net flow model
        flow_matcher: conditional flow matcher
        steps: number of optimization steps
        lr: learning rate
        num_samples: number of flow samples per step
    
    Returns:
        s_opt: [B, s_dim] optimized causal factors
        z_opt: [B, z_dim] optimized non-causal factors
    """
    device = purified_image.device
    batch_size = purified_image.shape[0]
    
    # Get initial estimates from encoder
    with torch.no_grad():
        s_init, z_init, _, _ = encoder_model(purified_image)
    
    # Initialize optimization variables
    s = s_init.clone().detach().requires_grad_(True)
    z = z_init.clone().detach().requires_grad_(True)
    
    # Optimizer for s and z
    optimizer = torch.optim.Adam([s, z], lr=lr, betas=(0.9, 0.999))
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Sample multiple timesteps for robust optimization
        total_loss = 0
        
        for _ in range(num_samples):
            # Get flow samples
            t, xt, ut = flow_matcher(purified_image)
            
            # Predict velocity field with current s, z
            predicted_ut = unet_model(xt, t, s, z)
            
            # Loss: minimize prediction error
            loss = F.mse_loss(predicted_ut, ut, reduction='mean')
            total_loss += loss
        
        # Average loss
        avg_loss = total_loss / num_samples
        
        # Add regularization to prevent s, z from deviating too much from encoder
        s_reg = 0.01 * F.mse_loss(s, s_init, reduction='mean')
        z_reg = 0.01 * F.mse_loss(z, z_init, reduction='mean')
        
        total_loss = avg_loss + s_reg + z_reg
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([s, z], max_norm=1.0)
        
        optimizer.step()
        
        # Early stopping
        if step > 50 and total_loss.item() < 1e-4:
            break
    
    return s.detach(), z.detach()

# --- Classification from s ---
def classify_from_s(s_factor, classifier_model):
    logits = classifier_model(s_factor)
    predicted_class = torch.argmax(logits, dim=-1)
    return predicted_class

# --- Main Inference Pipeline (SOTA) ---
def inference_pipeline(adversarial_image, unet, encoder, classifier, flow_matcher, device, 
                      use_likelihood_max=True, use_enhanced_inference=True):
    """
    Complete SOTA inference pipeline for adversarial defense.
    
    Args:
        adversarial_image: [B, C, H, W] potentially adversarial input
        unet, encoder, classifier: trained models
        flow_matcher: conditional flow matcher
        device: computation device
        use_likelihood_max: whether to use likelihood maximization (SOTA) or reverse ODE
        use_enhanced_inference: whether to use enhanced causal factor inference
    
    Returns:
        final_prediction: predicted class labels
        purified_img: purified image (for visualization)
        s_star: optimized causal factors
    """
    # Stage 1: Adversarial Purification
    if use_likelihood_max:
        print("Using SOTA likelihood maximization purification...")
        purified_img = purify_image_likelihood_max(
            adversarial_image, unet, flow_matcher, encoder, 
            steps=100, lr=0.01, lambda_reg=0.1
        )
    else:
        print("Using reverse ODE purification (baseline)...")
        purified_img = purify_image_reverse_ode(adversarial_image, unet, flow_matcher)
    
    # Stage 2: Causal Factor Inference
    if use_enhanced_inference:
        print("Using enhanced causal factor inference...")
        s_star, z_star = infer_causal_factors_enhanced(
            purified_img, encoder, unet, flow_matcher,
            steps=200, lr=0.005
        )
    else:
        print("Using basic causal factor inference...")
        s_star, z_star = infer_causal_factors(purified_img, encoder, unet, flow_matcher)
    
    # Stage 3: Classification
    final_prediction = classify_from_s(s_star, classifier)
    
    return final_prediction, purified_img, s_star

# --- AutoAttack Evaluation (SOTA) ---
def evaluate_autoattack(unet, encoder, classifier, flow_matcher, device, test_loader, 
                       use_likelihood_max=True, use_enhanced_inference=True):
    """
    Evaluate robust accuracy using AutoAttack with SOTA purification.
    
    Args:
        unet, encoder, classifier: trained models
        flow_matcher: conditional flow matcher
        device: computation device
        test_loader: test data loader
        use_likelihood_max: whether to use likelihood maximization
        use_enhanced_inference: whether to use enhanced inference
    
    Returns:
        robust_accuracy: robust accuracy under AutoAttack
    """
    # Define the defended model function
    def defended_model_fn(x):
        """
        Wrapper function for AutoAttack that applies the full defense pipeline.
        """
        # Apply the complete SOTA defense pipeline
        predictions, _, _ = inference_pipeline(
            x, unet, encoder, classifier, flow_matcher, device,
            use_likelihood_max=use_likelihood_max,
            use_enhanced_inference=use_enhanced_inference
        )
        return predictions

    # Wrap for AutoAttack compatibility
    class DefendedModel(torch.nn.Module):
        def __init__(self, model_fn):
            super().__init__()
            self.model_fn = model_fn
            
        def forward(self, x):
            return self.model_fn(x)

    defended_model = DefendedModel(defended_model_fn).to(device)
    
    # Initialize AutoAttack
    adversary = AutoAttack(
        defended_model, 
        norm='Linf', 
        eps=8/255, 
        version='standard',
        device=device
    )
    
    # Collect test data
    print("Collecting test data for AutoAttack evaluation...")
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs, dim=0).to(device)
    ys = torch.cat(ys, dim=0).to(device)
    
    # Run AutoAttack evaluation
    print(f"Running AutoAttack with {'SOTA' if use_likelihood_max else 'baseline'} purification...")
    with torch.no_grad():
        adv_preds = adversary.run_standard_evaluation(xs, ys, bs=64)
    
    # Calculate robust accuracy
    robust_accuracy = (adv_preds == ys.cpu().numpy()).mean()
    print(f'AutoAttack robust accuracy: {robust_accuracy*100:.2f}%')
    
    return robust_accuracy

# --- Main Entrypoint ---
def main():
    """
    Main entrypoint demonstrating both baseline and SOTA adversarial defense.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Model hyperparameters (should match training) ---
    unet_kwargs = dict(
        image_size=32, in_channels=3, model_channels=128, out_channels=3,
        num_res_blocks=2, attention_resolutions=[16,8], s_dim=128, z_dim=128
    )
    encoder_kwargs = dict(
        backbone_arch='WRN', s_dim=128, z_dim=128, wrn_depth=28, wrn_widen_factor=10
    )
    classifier_kwargs = dict(s_dim=128, num_classes=10)
    
    try:
        # --- Load trained models ---
        print("Loading trained models...")
        unet = load_model(UNetModel, 'checkpoints/unet_final.pt', device, **unet_kwargs)
        encoder = load_model(CausalEncoder, 'checkpoints/encoder_final.pt', device, **encoder_kwargs)
        classifier = load_model(LatentClassifier, 'checkpoints/classifier_final.pt', device, **classifier_kwargs)
        flow_matcher = ConditionalFlowMatcher(sigma=0.01)
        
        # --- Load test data ---
        from data.cifar10 import get_cifar10_loaders
        _, test_loader = get_cifar10_loaders(batch_size=128)
        
        # --- Evaluate on clean test set ---
        print('\n' + '='*50)
        print('EVALUATING ON CLEAN TEST SET')
        print('='*50)
        
        correct = 0
        total = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            predictions, _, _ = inference_pipeline(
                x, unet, encoder, classifier, flow_matcher, device,
                use_likelihood_max=True, use_enhanced_inference=True
            )
            correct += (predictions == y).sum().item()
            total += y.size(0)
        clean_accuracy = 100 * correct / total
        print(f'Clean accuracy: {clean_accuracy:.2f}%')
        
        # --- Compare Baseline vs SOTA ---
        print('\n' + '='*50)
        print('COMPARING BASELINE vs SOTA APPROACHES')
        print('='*50)
        
        # Baseline: Reverse ODE + Basic inference
        print("\n1. Baseline Approach (Reverse ODE + Basic Inference)")
        baseline_acc = evaluate_autoattack(
            unet, encoder, classifier, flow_matcher, device, test_loader,
            use_likelihood_max=False, use_enhanced_inference=False
        )
        
        # SOTA: Likelihood Maximization + Enhanced inference
        print("\n2. SOTA Approach (Likelihood Maximization + Enhanced Inference)")
        sota_acc = evaluate_autoattack(
            unet, encoder, classifier, flow_matcher, device, test_loader,
            use_likelihood_max=True, use_enhanced_inference=True
        )
        
        # Summary
        print('\n' + '='*50)
        print('SUMMARY')
        print('='*50)
        print(f'Clean Accuracy: {clean_accuracy:.2f}%')
        print(f'Baseline Robust Accuracy: {baseline_acc*100:.2f}%')
        print(f'SOTA Robust Accuracy: {sota_acc*100:.2f}%')
        print(f'Improvement: {(sota_acc - baseline_acc)*100:.2f} percentage points')
        
        if sota_acc > baseline_acc:
            print("SOTA approach outperforms baseline!")
        else:
            print("Baseline performs better - may need hyperparameter tuning")
            
    except FileNotFoundError as e:
        print(f"Error: Model checkpoint not found. Please train the models first.")
        print(f"Missing file: {e.filename}")
        print("Run: python train_causal_flow.py")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()