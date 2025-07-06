# models/encoder.py
# PURPOSE: This file defines the CausalEncoder. Its job is to map a high-dimensional
# image `x` into the low-dimensional latent factors `s` and `z`. To enable the
# Causal Information Bottleneck (CIB) loss, we'll structure this as a
# Variational Autoencoder (VAE)-style encoder.
#
# WHERE TO GET CODE: The CausalDiff repo uses a WideResNet (WRN-70-16) as its
# backbone. You can adapt a standard PyTorch implementation of WideResNet here.
# The "head" of the encoder that splits the output into s and z is custom.

import torch
import torch.nn as nn
from models.networks.resnet.wideresnet import WideResNet

# TODO if a vae works well, try a vit 

# --- PLACEHOLDER: A standard WideResNet implementation ---
# You can find PyTorch implementations of WideResNet in many public repos.
# The CausalDiff paper points to a specific one you can adapt.

class CausalEncoder(nn.Module):
    """
    Encodes an image x into latent factors s and z using a VAE-style approach.
    This allows us to calculate the KL-divergence term in the CIB loss.
    """
    def __init__(self, backbone_arch, s_dim, z_dim):
        super().__init__()
        # TO-DO: Instantiate the backbone.
        if backbone_arch == 'WRN':
            self.backbone = WideResNet(depth=28, num_classes=10, widen_factor=10) # Example
            backbone_out_dim = self.backbone.output_dim
        else:
            raise ValueError(f"Unsupported backbone architecture: {backbone_arch}")

        # ### VAE-style Heads ###
        # Instead of directly outputting s and z, we output the parameters of a
        # Gaussian distribution for each. This is what allows us to calculate
        # the KL divergence loss (the "information bottleneck").
        self.fc_mu_s = nn.Linear(backbone_out_dim, s_dim)
        self.fc_logvar_s = nn.Linear(backbone_out_dim, s_dim)
        self.fc_mu_z = nn.Linear(backbone_out_dim, z_dim)
        self.fc_logvar_z = nn.Linear(backbone_out_dim, z_dim)

    def reparameterize(self, mu, logvar):
        """The reparameterization trick to allow backpropagation through a random node."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Get the core features from the backbone
        features = self.backbone(x)

        # Get the distribution parameters for S and Z
        mu_s = self.fc_mu_s(features)
        logvar_s = self.fc_logvar_s(features)
        mu_z = self.fc_mu_z(features)
        logvar_z = self.fc_logvar_z(features)

        # Sample s and z using the reparameterization trick
        s = self.reparameterize(mu_s, logvar_s)
        z = self.reparameterize(mu_z, logvar_z)

        # We need to return the sampled latents AND their distribution parameters
        # for the CIB loss calculation.
        return s, z, (mu_s, logvar_s), (mu_z, logvar_z)