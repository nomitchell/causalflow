# models/causalunet.py
# This file has been completely rewritten with a structured, level-by-level UNet architecture
# to definitively fix all skip-connection and dimension mismatch errors.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Modules (No changes) ---

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CausalResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout, temb_channels, s_dim, z_dim):
        super().__init__()
        self.swish = Swish()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.s_proj = nn.Linear(s_dim, out_channels)
        self.z_proj = nn.Linear(z_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb, s, z):
        h = self.swish(self.norm1(x))
        h = self.conv1(h)
        temb_proj = self.temb_proj(self.swish(temb))[:, :, None, None]
        s_bias = self.s_proj(self.swish(s))[:, :, None, None]
        z_scale = self.z_proj(self.swish(z))[:, :, None, None]
        h = h * (1 + z_scale) + temb_proj + s_bias
        h = self.swish(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q, k, v = q.reshape(b, c, h * w), k.reshape(b, c, h * w), v.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        w_ = torch.bmm(q, k) * (c ** -0.5)
        w_ = F.softmax(w_, dim=2)
        h_ = torch.bmm(v, w_.permute(0, 2, 1)).reshape(b, c, h, w)
        return x + self.proj_out(h_)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

# --- FINAL CausalUNet Implementation ---
class CausalUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch = config.model_channels
        ch_mults = tuple(config.channel_mult)
        num_res_blocks = config.num_res_blocks
        attn_resolutions = config.attention_resolutions
        s_dim, z_dim = config.s_dim, config.z_dim
        self.num_resolutions = len(ch_mults)

        # Timestep embedding
        self.temb_ch = ch * 4
        self.temb = nn.Sequential(
            nn.Linear(ch, self.temb_ch),
            Swish(),
            nn.Linear(self.temb_ch, self.temb_ch),
        )

        # --- Downsampling Path (Structured by Resolution Level) ---
        self.conv_in = nn.Conv2d(config.in_channels, ch, kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        current_res = config.image_size
        in_ch = ch
        
        for i_level in range(self.num_resolutions):
            level_blocks = nn.ModuleList()
            out_ch = ch * ch_mults[i_level]
            for _ in range(num_res_blocks):
                level_blocks.append(
                    CausalResnetBlock(in_channels=in_ch, out_channels=out_ch, temb_channels=self.temb_ch, s_dim=s_dim, z_dim=z_dim, dropout=config.dropout)
                )
                in_ch = out_ch
            if current_res in attn_resolutions:
                level_blocks.append(AttnBlock(in_ch))
            
            # Add a downsampler for all levels except the last
            downsampler = Downsample(in_ch) if i_level != self.num_resolutions - 1 else nn.Identity()
            self.down_blocks.append(nn.ModuleDict({'blocks': level_blocks, 'downsampler': downsampler}))
            if i_level != self.num_resolutions - 1:
                current_res //= 2
        
        # --- Middle Path ---
        self.mid_blocks = nn.Sequential(
            CausalResnetBlock(in_channels=in_ch, out_channels=in_ch, temb_channels=self.temb_ch, s_dim=s_dim, z_dim=z_dim, dropout=config.dropout),
            AttnBlock(in_ch),
            CausalResnetBlock(in_channels=in_ch, out_channels=in_ch, temb_channels=self.temb_ch, s_dim=s_dim, z_dim=z_dim, dropout=config.dropout)
        )

        # --- Upsampling Path (Structured by Resolution Level) ---
        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            level_blocks = nn.ModuleList()
            out_ch = ch * ch_mults[i_level]
            # The input to the first ResNet block in an up-level combines the output
            # from the previous up-level and the skip connection.
            skip_in_ch = ch * ch_mults[i_level]
            
            for _ in range(num_res_blocks + 1):
                level_blocks.append(
                    CausalResnetBlock(in_channels=in_ch + skip_in_ch, out_channels=out_ch, temb_channels=self.temb_ch, s_dim=s_dim, z_dim=z_dim, dropout=config.dropout)
                )
                in_ch = out_ch
                # After the first block, the skip connection has been incorporated.
                skip_in_ch = 0 
            
            if current_res in attn_resolutions:
                level_blocks.append(AttnBlock(in_ch))

            # Add an upsampler for all levels except the first
            upsampler = Upsample(in_ch) if i_level != 0 else nn.Identity()
            self.up_blocks.append(nn.ModuleDict({'blocks': level_blocks, 'upsampler': upsampler}))
            if i_level != 0:
                current_res *= 2

        # --- Output Path ---
        self.out = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, config.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, s, z):
        # Timestep embedding
        temb = get_timestep_embedding(t, self.config.model_channels)
        temb = self.temb(temb)

        # --- Downsampling ---
        skips = []
        h = self.conv_in(x)
        for level in self.down_blocks:
            for block in level['blocks']:
                h = block(h, temb, s, z) if isinstance(block, CausalResnetBlock) else block(h)
            skips.append(h)
            h = level['downsampler'](h)

        # --- Middle ---
        h = self.mid_blocks[0](h, temb, s, z)
        h = self.mid_blocks[1](h)
        h = self.mid_blocks[2](h, temb, s, z)

        # --- Upsampling ---
        for level in self.up_blocks:
            # Pop the skip connection. Guaranteed to be the correct size now.
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in level['blocks']:
                 h = block(h, temb, s, z) if isinstance(block, CausalResnetBlock) else block(h)
            h = level['upsampler'](h)
            
        return self.out(h)