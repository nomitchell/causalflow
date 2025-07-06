# modules/cib.py
import torch
import torch.nn.functional as F

class CIBLoss(nn.Module):
    """
    Calculates the full Causal Information Bottleneck (CIB) loss.
    This loss function will be called from the main training script.
    """
    def __init__(self, lambda_kl, gamma_ce, eta_club):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.gamma_ce = gamma_ce
        self.eta_club = eta_club
        # The CLUB loss for estimating mutual information I(S;Z) can be a separate submodule
        self.club_estimator = CLUB()

    def forward(self, predicted_noise, target_noise, logits, target_labels, s, z, mu, logvar):
        # 1. Reconstruction Loss (from CFM)
        reconstruction_loss = F.mse_loss(predicted_noise, target_noise)

        # 2. Label Prediction Loss (Cross-Entropy)
        prediction_loss = F.cross_entropy(logits, target_labels)

        # 3. Disentanglement Loss (Minimizing I(S;Z) via CLUB)
        # This requires a separate model to estimate p(s|z)
        disentanglement_loss = self.club_estimator(s, z)

        # 4. Information Bottleneck (KL Divergence for the encoder)
        # Assuming the encoder q(s,z|x) outputs a Gaussian distribution
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Combine the losses
        total_loss = (reconstruction_loss +
                      self.gamma_ce * prediction_loss +
                      self.eta_club * disentanglement_loss +
                      self.lambda_kl * kl_loss)

        return total_loss, {
            "reconstruction": reconstruction_loss.item(),
            "prediction": prediction_loss.item(),
            "disentanglement": disentanglement_loss.item(),
            "kl": kl_loss.item()
        }

class CLUB(nn.Module):
    # Implementation of the Contrastive Log-Ratio Upper Bound from the CausalDiff paper
    def __init__(self, s_dim, z_dim):
        super().__init__()
        # A simple MLP to approximate p(s|z)
        self.predictor = nn.Sequential(...)

    def forward(self, s, z):
        # ... logic to calculate E[log p(s|z)] - E[log p(s|z)] for random pairs
        # This will require two forward passes.
        return mi_loss