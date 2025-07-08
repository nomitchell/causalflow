# models/unet.py
# PURPOSE: This file defines the core generative model of our framework. It's a U-Net
# architecture that learns the "velocity field" of the flow, predicting how an
# interpolated image `x_t` should move to become the clean image.
#
# WHERE TO GET CODE: The entire base U-Net (ResBlock, Attention, etc.) can be
# copied directly from the FlowPure repository. The key modification is adding the
# conditioning mechanism from CausalDiff.

# x: input image [batch, channels, height, width]
# time: timestep embedding
# s: label-causative factor [batch, s_dim] (can be None for unconditional generation)
# z: label-non-causative factor [batch, z_dim] (can be None)

# TO-DO: Implement the forward pass of the U-Net. For each block,
# you need to decide where and how to apply the s_bias and z_scale.
#
# Example for one block:
# h = self.some_res_block(x)
# if s is not None and z is not None:
#     s_bias = self.s_proj(s).unsqueeze(-1).unsqueeze(-1)
#     z_scale = self.z_proj(z).unsqueeze(-1).unsqueeze(-1)
#     h = h * z_scale + s_bias # Apply the causal conditioning

# Placeholder for the full forward pass

import math
import torch
import torch.nn as nn
from models.networks.fpunet.unet import *

# --- Step 1: Copy ALL helper classes from FlowPure's unet.py here ---
# This includes:
# - convert_module_to_f16
# - convert_module_to_f32
# - timestep_embedding
# - TimestepBlock
# - TimestepEmbedSequential
# - Upsample
# - Downsample
# - ResBlock
# - AttentionBlock
# ... and any other helper functions or classes in that file.



# --- (Assuming all the above classes have been copied) ---


# --- Step 2: Modify the Main UNetModel Class ---

class UNetModel(nn.Module):
    """
    UNet model with attention, timestep embedding, and causal conditioning (s, z).
    """
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        s_dim,  # Dimension of causal factor S
        z_dim,  # Dimension of non-causal factor Z
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None, # For class-conditioning (not used)
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

        # ### HARD RESEARCH AREA: Causal Conditioning ###
        # Now, we add the projection layers for S and Z.
        # This is where our causal logic gets injected into the powerful U-Net architecture.
        # We will project S and Z to match the dimension of the time embedding,
        # allowing them to be added in just like the timestep information.

        time_embed_dim = model_channels * 4
        self.s_proj = nn.Linear(s_dim, time_embed_dim)
        self.z_proj = nn.Linear(z_dim, time_embed_dim)

    def convert_to_fp16(self):
        """Convert model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, s=None, z=None):
        """
        Forward pass with optional causal conditioning.
        Args:
            x: [N x C x ...] input
            timesteps: 1-D batch of timesteps
            s: [N x s_dim] causal factors (optional)
            z: [N x z_dim] non-causal factors (optional)
        Returns:
            [N x C x ...] output
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # --- This is the key modification ---
        # If s and z are provided, project them and add them to the embedding.
        # This injects the causal information at the very beginning of the network.
        print("emb here is", emb.shape)
        if s is not None:
            emb = emb + self.s_proj(s)
        if z is not None:
            emb = emb + self.z_proj(z)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)