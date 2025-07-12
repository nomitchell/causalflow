# models/encoder.py
# This file has been rewritten to fix a critical bug and improve the design.
# 1. Removed the trailing commas that caused the TypeError.
# 2. Created separate linear heads for s and z to allow for better disentanglement.

import torch.nn as nn
import torch

class CausalEncoder(nn.Module):
    """
    Causal Encoder for disentangling an image `x` into a semantic factor `s`
    and a non-semantic factor `z`.
    """
    def __init__(self, s_dim, z_dim, nc=3, ndf=64):
        super(CausalEncoder, self).__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim

        # Shared convolutional body
        self.encoder = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # The flattened feature dimension from the conv body
        feature_dim = ndf * 8 * 1 * 1

        # --- KEY CORRECTION: Separate heads for s and z ---
        # This allows the model to learn to map different parts of the input
        # representation to the semantic and non-semantic factors.
        
        # Head for the semantic factor `s`
        self.s_mu_head = nn.Linear(feature_dim, s_dim)
        self.s_logvar_head = nn.Linear(feature_dim, s_dim)

        # Head for the non-semantic factor `z`
        self.z_mu_head = nn.Linear(feature_dim, z_dim)
        self.z_logvar_head = nn.Linear(feature_dim, z_dim)
        # --- END CORRECTION ---

    def reparameterize(self, mu, logvar):
        """
        The reparameterization trick for VAEs.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the encoder.
        """
        features = self.encoder(x)
        features = features.view(features.size(0), -1) # Flatten

        # Get parameters for `s`
        s_mu = self.s_mu_head(features)
        s_logvar = self.s_logvar_head(features)
        s = self.reparameterize(s_mu, s_logvar)

        # Get parameters for `z`
        z_mu = self.z_mu_head(features)
        z_logvar = self.z_logvar_head(features)
        z = self.reparameterize(z_mu, z_logvar)

        # For the CIB loss, we only need one set of mu/logvar for regularization.
        # We will use the `s` parameters as the primary ones to regularize.
        return s, z, s_mu, s_logvar
