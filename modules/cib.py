# modules/cib.py
# This file has been updated to support the new two-stage training pipeline.
# The CIBLoss class is now simplified to only calculate the losses needed
# for training the CausalEncoder in Stage 1.

import torch
import torch.nn as nn

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Log-ratio Upper Bound
    """
    This class is from the original CausalDiff implementation and remains unchanged.
    It provides an estimation of the mutual information between s and z, which is
    used to enforce their disentanglement.
    
    Args:
        s_dim (int): Dimension of the semantic factor `s`.
        z_dim (int): Dimension of the non-semantic factor `z`.
        hidden_size (int): Dimension of the hidden layer in the MLP.
    """
    def __init__(self, s_dim, z_dim, hidden_size):
        super(CLUB, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(s_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, z_dim)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(s_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, z_dim),
            nn.Tanh()
        )

    def get_mu_logvar(self, s_prime):
        mu = self.p_mu(s_prime)
        logvar = self.p_logvar(s_prime)
        return mu, logvar

    def forward(self, s_prime, z_prime):
        """
        Calculates the MI upper bound. This is used as the disentanglement loss.
        The goal is to minimize this value, forcing s and z to be independent.
        """
        mu, logvar = self.get_mu_logvar(s_prime)
        
        # log-likelihood of z_prime under the distribution estimated from s_prime
        conditional_loglik = -0.5 * (mu - z_prime) ** 2 / logvar.exp() - 0.5 * logvar
        
        # Calculate the expectation over the batch
        return conditional_loglik.mean()

    def learning_loss(self, s_prime, z_prime):
        """
        The loss for training the CLUB estimator itself.
        It's trained to maximize the log-likelihood of paired (s, z) samples
        while minimizing it for unpaired samples.
        """
        mu, logvar = self.get_mu_logvar(s_prime)
        
        # Positive term: log-likelihood of z_prime given s_prime
        positive = -0.5 * (mu - z_prime) ** 2 / logvar.exp()
        
        # Negative term: log-likelihood of z_prime given a shuffled s_prime
        prediction_from_shuffled_s = self.p_mu(s_prime[torch.randperm(s_prime.shape[0])])
        negative = -0.5 * (prediction_from_shuffled_s - z_prime) ** 2 / logvar.exp()
        
        # The learning loss is the difference, which we want to maximize.
        # So we return its negative to be minimized by the optimizer.
        return -(positive.mean() - negative.mean())


class CIBLoss(nn.Module):
    """
    Causal Information Bottleneck Loss, refactored for Stage 1 training.
    
    This loss function now combines the three components needed to train the CausalEncoder:
    1. Prediction Loss: Ensures the `s` vector is useful for classification.
    2. KL Divergence Loss: Regularizes the encoder's output to be a well-formed distribution.
    3. Disentanglement Loss: Encourages independence between `s` and `z`.
    
    The UNet reconstruction loss has been removed.
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
        
        Args:
            y_pred: Predictions from the LatentClassifier.
            y_true: Ground truth labels.
            mu: Mean from the encoder's VAE output.
            logvar: Log variance from the encoder's VAE output.
            s: The sampled `s` vector.
            z: The sampled `z` vector.
            club_estimator: The trained CLUB network to estimate MI(s;z).
            
        Returns:
            A dictionary containing the total loss and its individual components.
        """
        # 1. Prediction Loss (Cross-Entropy)
        prediction_loss = self.criterion_ce(y_pred, y_true)
        
        # 2. KL Divergence Loss (VAE Regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= y_true.shape[0] # Average over batch size

        # 3. Disentanglement Loss (Mutual Information from CLUB)
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
