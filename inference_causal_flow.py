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

# --- Purification: Likelihood Maximization (placeholder) ---
def purify_image_likelihood_max(adversarial_image, unet_model, flow_matcher, steps=50, lr=0.05, lambda_reg=0.1):
    """
    Purify by maximizing the likelihood under the flow model, with L2 regularization to stay close to the adversarial input.
    """
    device = adversarial_image.device
    x = adversarial_image.clone().detach().to(device)
    x.requires_grad = True
    optimizer = torch.optim.Adam([x], lr=lr)
    s = torch.zeros(x.shape[0], unet_model.s_proj.in_features, device=device)
    z = torch.zeros(x.shape[0], unet_model.z_proj.in_features, device=device)
    for _ in range(steps):
        optimizer.zero_grad()
        t, xt, ut = flow_matcher(x)
        predicted_ut = unet_model(xt, t, s, z)
        nll = F.mse_loss(predicted_ut, ut)
        reg = lambda_reg * ((x - adversarial_image) ** 2).mean()
        loss = nll + reg
        loss.backward()
        optimizer.step()
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

# --- Classification from s ---
def classify_from_s(s_factor, classifier_model):
    logits = classifier_model(s_factor)
    predicted_class = torch.argmax(logits, dim=-1)
    return predicted_class

# --- Main Inference Pipeline ---
def inference_pipeline(adversarial_image, unet, encoder, classifier, flow_matcher, device):
    # 1. Purification (reverse ODE baseline)
    purified_img = purify_image_reverse_ode(adversarial_image, unet, flow_matcher)
    # 2. Causal factor inference
    s_star, z_star = infer_causal_factors(purified_img, encoder, unet, flow_matcher)
    # 3. Classification
    final_prediction = classify_from_s(s_star, classifier)
    return final_prediction

# --- AutoAttack Evaluation ---
def evaluate_autoattack(unet, encoder, classifier, flow_matcher, device, test_loader):
    # Use classifier on purified and inferred s
    def model_fn(x):
        purified = purify_image_reverse_ode(x, unet, flow_matcher)
        s, _ = infer_causal_factors(purified, encoder, unet, flow_matcher)
        logits = classifier(s)
        return logits

    # Wrap model_fn for AutoAttack
    class WrappedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return model_fn(x)

    wrapped_model = WrappedModel().to(device)
    adversary = AutoAttack(wrapped_model, norm='Linf', eps=8/255, version='standard')
    # Collect all test data
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs, dim=0).to(device)
    ys = torch.cat(ys, dim=0).to(device)
    # Run AutoAttack
    with torch.no_grad():
        adv_preds = adversary.run_standard_evaluation(xs, ys, bs=128)
    acc = (adv_preds == ys.cpu().numpy()).mean()
    print(f'AutoAttack robust accuracy: {acc*100:.2f}%')
    return acc

# --- Main Entrypoint ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- Model hyperparameters (should match training) ---
    unet_kwargs = dict(
        image_size=32, in_channels=3, model_channels=128, out_channels=3,
        num_res_blocks=2, attention_resolutions=[16,8], s_dim=128, z_dim=128
    )
    encoder_kwargs = dict(
        backbone_arch='WRN', s_dim=128, z_dim=128, wrn_depth=28, wrn_widen_factor=10
    )
    classifier_kwargs = dict(s_dim=128, num_classes=10)
    # --- Load models ---
    unet = load_model(UNetModel, 'checkpoints/unet_final.pt', device, **unet_kwargs)
    encoder = load_model(CausalEncoder, 'checkpoints/encoder_final.pt', device, **encoder_kwargs)
    classifier = load_model(LatentClassifier, 'checkpoints/classifier_final.pt', device, **classifier_kwargs)
    flow_matcher = ConditionalFlowMatcher(sigma=0.01)
    # --- Load test data ---
    from data.cifar10 import get_cifar10_loaders
    _, test_loader = get_cifar10_loaders(batch_size=128)
    # --- Evaluate on clean and adversarial examples ---
    print('Evaluating on clean test set...')
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        purified = purify_image_reverse_ode(x, unet, flow_matcher)
        s, _ = infer_causal_factors(purified, encoder, unet, flow_matcher)
        logits = classifier(s)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f'Clean accuracy: {100*correct/total:.2f}%')
    # --- AutoAttack evaluation ---
    print('Running AutoAttack evaluation...')
    evaluate_autoattack(unet, encoder, classifier, flow_matcher, device, test_loader)

if __name__ == "__main__":
    main()