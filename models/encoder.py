# models/encoder.py
# This is the final, rewritten version of the CausalEncoder.
# It now uses a powerful WideResNet backbone, similar to the one described
# in the CausalDiff paper, to provide the necessary representational capacity
# for high-performance disentanglement.
# The forward pass has been corrected to match the layer names of the imported WideResNet.

import torch
import torch.nn as nn
import torch.nn.functional as F

# We will re-use the WideResNet implementation from the victim model's directory.
from .networks.resnet.wideresnet import WideResNet

class CausalEncoder(nn.Module):
    """
    A powerful Causal Encoder that uses a WideResNet as its feature extraction backbone.
    This provides the necessary capacity to learn a rich and disentangled latent space.
    """
    def __init__(self, s_dim, z_dim, wrn_depth=28, wrn_widen_factor=10, dropout_rate=0.3):
        super(CausalEncoder, self).__init__()
        
        # 1. The Backbone
        # We use the WideResNet model as a powerful feature extractor.
        # We will effectively ignore its final linear layer for classification.
        self.backbone = WideResNet(
            depth=wrn_depth, 
            widen_factor=wrn_widen_factor, 
            num_classes=10, # num_classes is needed for init but will be ignored
            dropRate=dropout_rate
        )
        
        # The output feature dimension of this WRN before the final linear layer
        # is `widen_factor * 64`.
        feature_dim = wrn_widen_factor * 64

        # 2. The VAE Heads
        # Separate linear layers to project the rich features into the
        # parameters for the `s` and `z` latent distributions.
        self.s_mu_head = nn.Linear(feature_dim, s_dim)
        self.s_logvar_head = nn.Linear(feature_dim, s_dim)

        self.z_mu_head = nn.Linear(feature_dim, z_dim)
        self.z_logvar_head = nn.Linear(feature_dim, z_dim)

    def reparameterize(self, mu, logvar):
        """Standard reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # --- KEY CORRECTION ---
        # The forward pass now correctly mirrors the structure of the imported
        # `WideResNet` class, using `block1`, `block2`, etc.
        # This intercepts the features just before the final classification layer.
        
        out = self.backbone.conv1(x)
        out = self.backbone.block1(out)
        out = self.backbone.block2(out)
        out = self.backbone.block3(out)
        out = self.backbone.relu(self.backbone.bn1(out))
        out = F.avg_pool2d(out, 8)
        features = out.view(out.size(0), -1)
        # --- END CORRECTION ---

        # Now, use the separate heads to get the VAE parameters
        s_mu = self.s_mu_head(features)
        s_logvar = self.s_logvar_head(features)
        s = self.reparameterize(s_mu, s_logvar)

        z_mu = self.z_mu_head(features)
        z_logvar = self.z_logvar_head(features)
        z = self.reparameterize(z_mu, z_logvar)
        
        # Return the latent vectors and the parameters for the primary (`s`) distribution
        return s, z, s_mu, s_logvar
