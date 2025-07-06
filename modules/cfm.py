# modules/cfm.py

import torch
import torch.nn.functional as F

class ConditionalFlowMatcher:
    """
    Implements the Conditional Flow Matching logic, specifically for a Rectified Flow,
    as described in the FlowPure paper.
    
    This class takes a batch of clean images and prepares the inputs needed to train
    the conditional U-Net model.
    """
    def __init__(self, sigma: float = 0.0):
        """
        Initializes the flow matcher.
        
        Args:
            sigma (float): The standard deviation of the noise to add to the
                           interpolated paths. The FlowPure paper uses a very small
                           sigma, effectively creating a deterministic (Dirac delta) path.
        """
        self.sigma = sigma

    def sample_source(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples a source image for the flow path. For simplicity and to match
        common implementations, we will randomly sample another image from the
        same batch to act as the starting point of the flow.
        
        Args:
            x (torch.Tensor): A batch of target images (x_clean).

        Returns:
            torch.Tensor: A batch of source images (x_adv or x_source).
        """
        # Create a random permutation of indices
        batch_size = x.shape[0]
        random_indices = torch.randperm(batch_size, device=x.device)
        
        # Return the shuffled batch
        return x[random_indices]

    def forward(self, x_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares a batch for training the flow model.

        For each clean image in the batch, this method:
        1. Samples a source image (x_source).
        2. Samples a random timestep 't'.
        3. Creates the interpolated image 'x_t' on the path between x_source and x_clean.
        4. Calculates the target velocity 'u_t' for the U-Net to predict.

        Args:
            x_clean (torch.Tensor): A batch of clean target images. Shape: [B, C, H, W]

        Returns:
            A tuple containing:
            - t (torch.Tensor): The sampled timesteps. Shape: [B]
            - x_t (torch.Tensor): The interpolated images at time t. Shape: [B, C, H, W]
            - u_t (torch.Tensor): The target velocity vectors. Shape: [B, C, H, W]
        """
        # Ensure x_clean is a float tensor
        x_clean = x_clean.float()

        # 1. Sample the source images (x_source)
        x_source = self.sample_source(x_clean)

        # 2. Sample timesteps 't' from a uniform distribution [0, 1]
        batch_size = x_clean.shape[0]
        # Reshape t to be [B, 1, 1, 1] for easy broadcasting with image tensors
        t = torch.rand(batch_size, device=x_clean.device).view(-1, 1, 1, 1)

        # 3. Calculate the interpolated image x_t
        # This is the straight-line path (rectified flow) between x_source and x_clean
        # x_t = (1 - t) * x_source + t * x_clean
        x_t = (1 - t) * x_source + t * x_clean
        
        # Add a small amount of Gaussian noise if sigma > 0
        if self.sigma > 0:
            noise = torch.randn_like(x_t) * self.sigma
            x_t = x_t + noise

        # 4. Calculate the target velocity u_t
        # For a rectified flow, the velocity is constant: the difference vector
        u_t = x_clean - x_source

        return t.squeeze(), x_t, u_t