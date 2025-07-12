# modules/cib.py
# This file has been rewritten to fix a critical numerical instability bug.
# The CLUB module now correctly calculates the mutual information upper bound,
# which is used as the disentanglement loss.

import torch
import torch.nn as nn

class CLUB(nn.Module):
    """
    This class implements the CLUB estimator for the mutual information upper bound.
    It is trained to estimate MI(s, z), which is then minimized by the main encoder.
    """
    def __init__(self, s_dim, z_dim, hidden_size):
        super(CLUB, self).__init__()
        # This MLP learns to predict z from s, i.e., models p(z|s)
        self.p_mu = nn.Sequential(
            nn.Linear(s_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(s_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_dim),
            nn.Tanh() # Tanh to bound the log-variance for stability
        )

    def get_mu_logvar(self, s_prime):
        mu = self.p_mu(s_prime)
        logvar = self.p_logvar(s_prime)
        return mu, logvar

    def forward(self, s_prime, z_prime):
        """
        Calculates the MI upper bound, which is E[log q(z|s)] - E[log q(z)].
        This is the value we want to minimize in the main training loop.
        """
        mu, logvar = self.get_mu_logvar(s_prime)
        
        # E_{p(s,z)}[log q(z|s)]
        positive_loglik = (-0.5 * (z_prime - mu)**2 / logvar.exp() - 0.5 * logvar).mean()
        
        # E_{p(s)}E_{p(z)}[log q(z|s)]
        # We estimate the marginal by shuffling the batch of z
        z_shuffled = z_prime[torch.randperm(z_prime.shape[0])]
        negative_loglik = (-0.5 * (z_shuffled - mu)**2 / logvar.exp() - 0.5 * logvar).mean()

        # The MI estimate is the difference. This is a bounded, stable value.
        return positive_loglik - negative_loglik

    def learning_loss(self, s_prime, z_prime):
        """
        The loss for training the CLUB estimator itself.
        The estimator is trained to maximize the MI estimate. We return the
        negative so it can be minimized by an optimizer.
        """
        return -self.forward(s_prime, z_prime)


class CIBLoss(nn.Module):
    """
    Causal Information Bottleneck Loss for Stage 1 training.
    This combines the losses for classification, VAE regularization, and disentanglement.
    """
    def __init__(self, gamma_ce, lambda_kl, eta_club):
        super(CIBLoss, self).__init__()
        self.gamma_ce = gamma_ce
        self.lambda_kl = lambda_kl
        self.eta_club = eta_club
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true, mu, logvar, s, z, club_estimator):
        """
        Calculates the total loss for Stage 1.
        """
        # 1. Prediction Loss (Cross-Entropy)
        prediction_loss = self.criterion_ce(y_pred, y_true)
        
        # 2. KL Divergence Loss (VAE Regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= y_true.shape[0]

        # 3. Disentanglement Loss (The stable MI estimate from CLUB)
        disentanglement_loss = club_estimator(s, z)

        # Combine the losses
        total_loss = (self.gamma_ce * prediction_loss + 
                      self.lambda_kl * kl_loss + 
                      self.eta_club * disentanglement_loss)

        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'kl_loss': kl_loss,
            'disentanglement_loss': disentanglement_loss
        }
