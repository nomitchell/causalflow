# models/classifier.py
# PURPOSE: This file defines the LatentClassifier. It is a simple Multi-Layer
# Perceptron (MLP) that learns the mapping from the disentangled causal factor `s`
# to the final class label `y`. Its performance is a direct measure of how
# successfully `s` has captured the essential class information.
#
# WHERE TO GET CODE: This is standard PyTorch. No need to copy from anywhere.

import torch.nn as nn

class LatentClassifier(nn.Module):
    """Classifies a label Y from the latent causal factor s."""
    def __init__(self, s_dim, num_classes, hidden_dim_factor=2):
        super().__init__()
        # Simple two-layer MLP (tunable hyperparameters)
        self.classifier = nn.Sequential(
            nn.Linear(s_dim, s_dim * hidden_dim_factor),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(s_dim * hidden_dim_factor, num_classes)
        )

    def forward(self, s):
        """Takes the latent factor `s` and returns the raw logits for classification."""
        return self.classifier(s)