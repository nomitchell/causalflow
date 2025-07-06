# models/unet.py
# (This is a simplified representation of the U-Net from FlowPure)
import torch
import torch.nn as nn

# ... (Keep the original ResBlock, Attention, etc. from FlowPure's unet.py)

class Unet(nn.Module):
    """
    The core U-Net architecture, adapted for S and Z conditioning.
    This will be our v_theta(t, x_t, s, z) model.
    """
    def __init__(self, in_channel, ... , s_dim, z_dim):
        super().__init__()
        # ... (Original U-Net layers from FlowPure)

        # NEW: Projection layers for S and Z from CausalDiff
        self.s_proj = nn.Linear(s_dim, in_channel)
        self.z_proj = nn.Linear(z_dim, in_channel)

    def forward(self, x, time, s=None, z=None):
        # x: input image [batch, channels, height, width]
        # time: timestep embedding
        # s: label-causative factor [batch, s_dim]
        # z: label-non-causative factor [batch, z_dim]

        # --- NEW: Causal Conditioning ---
        s_bias = None
        z_scale = None
        if s is not None and z is not None:
            s_bias = self.s_proj(s).unsqueeze(-1).unsqueeze(-1) # Projects s to act as a bias
            z_scale = self.z_proj(z).unsqueeze(-1).unsqueeze(-1) # Projects z to act as a scale

        # ... (Pass x through the U-Net layers as in FlowPure)
        # At each relevant layer in the U-Net, apply the conditioning:
        # h = ... (some intermediate feature map)
        # if z_scale is not None:
        #     h = h * z_scale
        # if s_bias is not None:
        #     h = h + s_bias
        # ...

        # Return the final predicted noise
        return self.final_conv(h)