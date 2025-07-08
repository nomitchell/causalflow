# modules/cfm.py (Conditional Flow Matcher)
# PURPOSE: Implements the Rectified Flow logic from the FlowPure paper. This module
# is responsible for creating the `(t, x_t, u_t)` tuples needed for training the
# U-Net at each step. It is the engine of the "flow" part of CausalFlow.
#
# WHERE TO GET CODE: The logic is directly from the FlowPure paper's methodology
# and can be adapted from its official implementation.

'''
some improvements could be with source distribution, noise to data
increase sigma
'''

import torch
import torch.nn as nn

class ConditionalFlowMatcher(nn.Module):
    def __init__(self, sigma: float = 0.0):
        super().__init__()
        self.sigma = sigma

    def sample_source(self, x: torch.Tensor) -> torch.Tensor:
        # Sample another image from the batch as the source (could also use noise)
        return x[torch.randperm(x.shape[0], device=x.device)]

    def forward(self, x_clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_clean = x_clean.float()
        x_source = self.sample_source(x_clean)
        
        # Create a 1D tensor for t
        t = torch.rand(x_clean.shape[0], device=x_clean.device)
        
        # Create a broadcastable version of t for interpolation
        t_broadcast = t.view(-1, 1, 1, 1)
        
        x_t = (1 - t_broadcast) * x_source + t_broadcast * x_clean
        if self.sigma > 0:
            x_t += torch.randn_like(x_t) * self.sigma
        u_t = x_clean - x_source
        
        # Return the 1D t tensor
        return t, x_t, u_t