# modules/cfm.py (Conditional Flow Matcher)
# PURPOSE: Implements the Rectified Flow logic from the FlowPure paper. This module
# is responsible for creating the `(t, x_t, u_t)` tuples needed for training the
# U-Net at each step.

import torch
import torch.nn as nn

class ConditionalFlowMatcher(nn.Module):
    def __init__(self, sigma: float = 0.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the training tuple for a flow from a starting point x0 (adversarial)
        to a target point x1 (clean).
        """
        x0 = x0.float()
        x1 = x1.float()

        # Create a 1D tensor for time t
        t = torch.rand(x0.shape[0], device=x0.device)
        
        # Create a broadcastable version of t for interpolation
        t_broadcast = t.view(-1, 1, 1, 1)
        
        # Interpolate between the start (x0) and end (x1) points
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
        
        # Add optional noise, as described in the FlowPure paper
        if self.sigma > 0:
            x_t += torch.randn_like(x_t) * self.sigma
            
        # The velocity field u_t is the direct path from start to end
        u_t = x1 - x0
        
        # Return the 1D time tensor, the interpolated point, and the velocity
        return t, x_t, u_t