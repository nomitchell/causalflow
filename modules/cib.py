# modules/cib.py
# PURPOSE: Implements the complex CIB loss function from the CausalDiff paper.
# It combines the four key objectives: reconstruction, prediction, disentanglement,
# and the information bottleneck. This is the engine of the "causal" part of CausalFlow.
#
# WHERE TO GET CODE: The logic is from the CausalDiff paper's equations.
# The CLUB implementation below is a standard and faithful adaptation of the
# original CLUB paper, which CausalDiff cites.

import torch
import torch.nn as nn
import torch.nn.functional as F

class CLUB(nn.Module):
    """
    CLUB estimator for mutual information I(S;Z).
    """
    def __init__(self, s_dim, z_dim, hidden_size=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, s_dim)
        )

    def forward(self, s, z):
        predicted_s_for_true_z = self.predictor(z)
        positive_log_likelihood = -F.mse_loss(predicted_s_for_true_z, s, reduction='none').mean(dim=1)
        shuffled_s = s[torch.randperm(s.shape[0])]
        negative_log_likelihood = -F.mse_loss(predicted_s_for_true_z, shuffled_s, reduction='none').mean(dim=1)
        return (positive_log_likelihood - negative_log_likelihood).mean()

class CIBLoss(nn.Module):
    """Causal Information Bottleneck loss."""
    def __init__(self, lambda_kl, gamma_ce, eta_club, s_dim, z_dim, alpha_recon=1.0):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.gamma_ce = gamma_ce
        self.eta_club = eta_club
        self.club_estimator = CLUB(s_dim, z_dim)
        self.alpha_recon = alpha_recon

    def forward(self, predicted_ut, target_ut, logits, target_labels, s, z, s_dist_params, z_dist_params, include_recon=True):
        reconstruction_loss = F.mse_loss(predicted_ut, target_ut)
        prediction_loss = F.cross_entropy(logits, target_labels)
        disentanglement_loss = self.club_estimator(s, z)
        mu_s, logvar_s = s_dist_params
        mu_z, logvar_z = z_dist_params
        kl_s = -0.5 * torch.sum(1 + logvar_s - mu_s.pow(2) - logvar_s.exp(), dim=1).mean()
        kl_z = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1).mean()
        kl_loss = kl_s + kl_z
        total_loss = (self.alpha_recon * reconstruction_loss +
                      self.gamma_ce * prediction_loss +
                      self.eta_club * disentanglement_loss +
                      self.lambda_kl * kl_loss)

        if include_recon:
            total_loss = (self.alpha_recon * reconstruction_loss +
                          self.gamma_ce * prediction_loss +
                          self.eta_club * disentanglement_loss +
                          self.lambda_kl * kl_loss)
        else:
            total_loss = (self.gamma_ce * prediction_loss +
                          self.eta_club * disentanglement_loss +
                          self.lambda_kl * kl_loss)
        
        return total_loss, {
            "reconstruction_loss": reconstruction_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "disentanglement_loss": disentanglement_loss.item(),
            "kl_loss": kl_loss.item()
        }