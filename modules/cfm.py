# modules/cfm.py
# This is the corrected and verified version of the Conditional Flow Matcher module.
# It ensures the `sample_location_and_conditional_flow` method is correctly defined
# to be used by the Stage 2 purifier training scripts.

import torch
import torch.nn as nn

class ConditionalFlowMatcher(nn.Module):
    """
    Implements the Conditional Flow Matching (CFM) logic as described in the
    FlowPure paper, using the rectified flow formulation.
    """
    def __init__(self, sigma: float = 0.0):
        """
        Initializes the flow matcher.

        Args:
            sigma (float): The standard deviation of the noise added to the path.
                           A small value is used for rectified flows.
        """
        super().__init__()
        self.sigma = sigma

    def sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor):
        """
        Samples a time `t`, a point `xt` on the path between x0 and x1,
        and the ground-truth velocity `ut` at that point.

        This is the core function for training a CNF with Flow Matching.

        Args:
            x0 (torch.Tensor): The starting point of the flow (e.g., a noisy image).
            x1 (torch.Tensor): The ending point of the flow (e.g., a clean image).

        Returns:
            t (torch.Tensor): A batch of sampled time steps, shape (B,).
            xt (torch.Tensor): A batch of interpolated points on the path, shape (B, C, H, W).
            ut (torch.Tensor): The ground-truth velocity at xt, shape (B, C, H, W).
        """
        # Sample a random time t for each sample in the batch
        t = torch.rand(x0.shape[0], device=x0.device)
        
        # Linearly interpolate between x0 and x1 to get the point xt at time t
        # xt = (1-t)*x0 + t*x1
        xt = t.view(-1, 1, 1, 1) * x1 + (1 - t).view(-1, 1, 1, 1) * x0
        
        # The velocity of the rectified flow is constant along the path
        ut = x1 - x0
        
        # Add a small amount of noise for stability if sigma > 0
        if self.sigma > 0:
            xt = xt + self.sigma * torch.randn_like(xt)
            
        return t, xt, ut

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        """
        The default forward pass simply calls the sampling method.
        """
        return self.sample_location_and_conditional_flow(x0, x1)

