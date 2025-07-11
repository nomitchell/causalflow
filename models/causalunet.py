# models/causalunet.py
# This file has been significantly updated to correctly implement the causal guidance mechanism
# as described in the CausalDiff paper and our "Option A" strategy.

import math
import torch
import torch.nn as nn

# --- Helper Modules from the original fpunet/nn.py ---
# These are kept as they are standard building blocks.

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This function is from High-Resolution Image Synthesis with Latent Diffusion Models.
    https://github.com/CompVis/stable-diffusion
    Build sinusoidal embeddings.
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResnetBlock(nn.Module):
    """
    Standard ResNet block with GroupNorm and Swish activation.
    This module remains unchanged from the original implementation.
    """
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        self.swish = Swish()

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        h = h + self.temb_proj(self.swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    """
    Standard Attention block.
    This module remains unchanged from the original implementation.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

# --- NEW Causal ResNet Block ---
# This is the core architectural change. It replaces the standard ResnetBlock
# to correctly inject the causal factors `s` and `z`.

class CausalResnetBlock(nn.Module):
    """
    A modified ResNet block that incorporates causal factors `s` and `z`
    as adaptive scale and shift parameters, following the CausalDiff methodology.
    """
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, s_dim, z_dim):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.swish = Swish()

        # Standard convolutional layers
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Projections for time, s, and z embeddings
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        # --- KEY CHANGE: Projections for s (bias) and z (scale) ---
        # These linear layers will transform the latent vectors s and z into
        # parameters that can modulate the feature maps inside the block.
        self.s_proj = nn.Linear(s_dim, out_channels) # `s` will become a bias/shift
        self.z_proj = nn.Linear(z_dim, out_channels) # `z` will become a scale
        # --- END KEY CHANGE ---

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb, s, z):
        """
        The forward pass now accepts `s` and `z` as additional inputs.
        """
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        # --- KEY CHANGE: Apply causal and time embeddings as scale and shift ---
        # Project time embedding and add it as a bias
        temb_proj = self.temb_proj(self.swish(temb))[:, :, None, None]
        
        # Project `s` embedding and add it as a bias
        s_bias = self.s_proj(self.swish(s))[:, :, None, None]
        
        # Project `z` embedding and use it as a scale factor
        z_scale = self.z_proj(self.swish(z))[:, :, None, None]
        
        # The CausalDiff paper describes the operation as h_out = z_s * h_out + s_b.
        # Here, we combine the biases from time and `s` and apply the scale from `z`.
        h = h * (1 + z_scale) + temb_proj + s_bias
        # --- END KEY CHANGE ---

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


# --- Updated CausalUNet ---
# This UNet now uses the CausalResnetBlock and passes s and z down through the network.

class CausalUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch = config.model_channels, config.out_channels
        ch_mult = tuple(config.channel_mult)
        num_res_blocks = config.num_res_blocks
        attn_resolutions = config.attention_resolutions
        dropout = config.dropout
        in_channels = config.in_channels
        resolution = config.image_size
        resamp_with_conv = config.resamp_with_conv
        
        s_dim = config.s_dim
        z_dim = config.z_dim

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        # Timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                # --- Use CausalResnetBlock instead of the original ResnetBlock ---
                block.append(CausalResnetBlock(in_channels=block_in,
                                               out_channels=block_out,
                                               temb_channels=self.temb_ch,
                                               s_dim=s_dim,
                                               z_dim=z_dim,
                                               dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlock(in_channels=block_in,
                                             out_channels=block_in,
                                             temb_channels=self.temb_ch,
                                             s_dim=s_dim,
                                             z_dim=z_dim,
                                             dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = CausalResnetBlock(in_channels=block_in,
                                             out_channels=block_in,
                                             temb_channels=self.temb_ch,
                                             s_dim=s_dim,
                                             z_dim=z_dim,
                                             dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * in_ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                else:
                    skip_in = block_in
                
                # --- Use CausalResnetBlock ---
                block.append(CausalResnetBlock(in_channels=block_in + skip_in,
                                               out_channels=block_out,
                                               temb_channels=self.temb_ch,
                                               s_dim=s_dim,
                                               z_dim=z_dim,
                                               dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # End
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)
        self.swish = Swish()

    def forward(self, x, t, s, z):
        """
        The forward pass now takes `s` and `z` and threads them through all CausalResnetBlocks.
        """
        assert x.shape[2] == x.shape[3] == self.config.image_size
        
        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = self.swish(temb)
        temb = self.temb.dense[1](temb)

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb, s, z)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb, s, z)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, s, z)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb, s, z)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # End
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h

# --- Upsample and Downsample Modules (Unchanged) ---

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
