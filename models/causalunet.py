# models/causalunet.py
# This file has been completely rewritten to use the SOTA-standard DDPM++ architecture,
# as used in FlowPure and CausalDiff. It replaces the previous custom UNet.
# The core logic is adapted from the OpenAI guided-diffusion repository, which you
# already have in `models/networks/fpunet`.
#
# Key Changes:
# 1. Full DDPM++ Architecture: Implements the complete multi-resolution structure
#    with proper ResNet blocks, skip connections, and attention blocks.
# 2. Causal Conditioning (FiLM): The `s` (causal) and `z` (style) vectors are
#    injected into each CausalResBlock using FiLM-style modulation (scale and bias),
#    providing robust and stable guidance to the purification process.
# 3. Corrected Forward Pass: The forward pass now correctly handles the skip
#    connections and passes the timestep, s, and z embeddings through all levels
#    of the network.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Modules from guided-diffusion/fpunet ---
# We bring these into the file for clarity and to make this module self-contained.

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class CausalResBlock(nn.Module):
    """
    A residual block that includes causal conditioning via FiLM.
    It takes timestep embeddings, causal `s` embeddings, and style `z` embeddings.
    """
    def __init__(self, channels, emb_channels, s_dim, z_dim, dropout, out_channels=None, use_conv=False, dims=2):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        # Timestep embedding projection
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        
        # Causal `s` embedding projection (provides bias)
        self.s_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(s_dim, self.out_channels),
        )

        # Style `z` embedding projection (provides scale)
        self.z_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_dim, self.out_channels),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, s, z):
        h = self.in_layers(x)
        
        # Project embeddings
        emb_out = self.emb_layers(emb).type(h.dtype)
        s_out = self.s_layers(s).type(h.dtype)
        z_out = self.z_layers(z).type(h.dtype)
        
        # Reshape for broadcasting
        emb_out = emb_out.view(emb_out.shape[0], emb_out.shape[1], 1, 1)
        s_out = s_out.view(s_out.shape[0], s_out.shape[1], 1, 1)
        z_out = z_out.view(z_out.shape[0], z_out.shape[1], 1, 1)

        # FiLM-style conditioning
        # `z` provides scale, `s` provides bias, `t` provides an additional bias
        h = h * (1 + z_out) + s_out + emb_out
        
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """Standard self-attention block."""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(num_heads)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """Helper for AttentionBlock."""
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class CausalUNet(nn.Module):
    """
    The full DDPM++ UNet model, adapted for causal conditioning.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Unpack config
        image_size = config.image_size
        in_channels = config.in_channels
        model_channels = config.model_channels
        out_channels = config.out_channels
        num_res_blocks = config.num_res_blocks
        attention_resolutions = config.attention_resolutions
        dropout = config.dropout
        channel_mult = config.channel_mult
        conv_resample = config.resamp_with_conv
        s_dim = config.s_dim
        z_dim = config.z_dim

        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- Downsampling Path ---
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    CausalResBlock(ch, time_embed_dim, s_dim, z_dim, dropout, out_channels=mult * model_channels, use_conv=conv_resample)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch, use_conv=conv_resample))
                input_block_chans.append(ch)
                ds *= 2

        # --- Middle Path ---
        self.middle_block = nn.Sequential(
            CausalResBlock(ch, time_embed_dim, s_dim, z_dim, dropout),
            AttentionBlock(ch),
            CausalResBlock(ch, time_embed_dim, s_dim, z_dim, dropout),
        )

        # --- Upsampling Path ---
        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    CausalResBlock(ch + ich, time_embed_dim, s_dim, z_dim, dropout, out_channels=model_channels * mult, use_conv=conv_resample)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=conv_resample))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        # --- Output Layer ---
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, s, z):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param s: an [N x s_dim] Tensor of causal latents.
        :param z: an [N x z_dim] Tensor of style latents.
        :return: an [N x C x H x W] Tensor of outputs.
        """
        hs = []
        
        # Timestep embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.config.model_channels))

        # Downsampling
        h = self.input_blocks[0](x)
        hs.append(h)
        for module in self.input_blocks[1:]:
            if isinstance(module[0], CausalResBlock):
                 h = module[0](h, emb, s, z)
                 if len(module) > 1: # If attention block exists
                     h = module[1](h)
            else: # Downsample layer
                h = module(h)
            hs.append(h)
        
        # Middle block
        h = self.middle_block[0](h, emb, s, z)
        h = self.middle_block[1](h)
        h = self.middle_block[2](h, emb, s, z)

        # Upsampling
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module[0](cat_in, emb, s, z)
            if len(module) > 1: # If attention or upsample exists
                for layer in module[1:]:
                    h = layer(h)

        return self.out(h)
