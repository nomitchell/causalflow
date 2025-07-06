# models/encoder.py
import torch.nn as nn

class CausalEncoder(nn.Module):
    """Encodes an image x into latent factors s and z."""
    def __init__(self, backbone_arch, s_dim, z_dim):
        super().__init__()
        self.backbone = ... # e.g., a WideResNet-70-16 backbone
        self.head_s = nn.Linear(self.backbone.output_dim, s_dim)
        self.head_z = nn.Linear(self.backbone.output_dim, z_dim)

    def forward(self, x):
        features = self.backbone(x)
        s = self.head_s(features)
        z = self.head_z(features)
        return s, z

# models/classifier.py
import torch.nn as nn

class LatentClassifier(nn.Module):
    """Classifies a label Y from the latent factor s."""
    def __init__(self, s_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(s_dim, s_dim // 2),
            nn.ReLU(),
            nn.Linear(s_dim // 2, num_classes)
        )
    def forward(self, s):
        return self.classifier(s)