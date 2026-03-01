"""Flow matching U-Net dynamics model.

Predicts the velocity field v(x_t, t, action, context) for generating the
next game frame. Operates in pixel space — takes 128×128×3 images directly.

Architecture (from docs/build_spec.md §2.3):
  - Input: [B, 15, H, W] = noisy target (3ch) + 4 context frames (12ch)
  - Conditioning: flow time t + action → AdaGN in every ResBlock
  - 4-level encoder-decoder with skip connections
  - Self-attention only at 16×16 (smallest spatial resolution)
  - Output: [B, 3, H, W] predicted velocity field

Reference: docs/build_spec.md §2.3-2.4, docs/foundations_guide.md Part 6-7.
"""

import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import (
    Downsample,
    ResBlock,
    SelfAttention,
    SinusoidalEmbedding,
    Upsample,
)


class UNet(nn.Module):
    """Flow matching U-Net with action conditioning via AdaGN.

    The U-Net processes a concatenation of the noisy interpolated target and
    context frames, conditioned on flow time t and action a. Every ResBlock
    receives the combined conditioning embedding via Adaptive GroupNorm,
    which modulates feature processing based on "what time step are we at"
    and "what action was taken."

    Architecture trace (128×128 input, channels [64, 128, 256, 512]):

        Input conv: 15 → 64
        Down L1: 64→64, 64→64, Downsample → 64×64
        Down L2: 64→128, 128→128, Downsample → 32×32
        Down L3: 128→256, 256→256, Downsample → 16×16
        Down L4: 256→512, 512→512 (no downsample)
        Middle:  512→512, SelfAttention, 512→512
        Up L4:   cat(512+512)=1024→512, 512→256, Upsample → 32×32
        Up L3:   cat(256+256)=512→256, 256→128, Upsample → 64×64
        Up L2:   cat(128+128)=256→128, 128→64, Upsample → 128×128
        Up L1:   cat(64+64)=128→64, 64→64
        Output:  GroupNorm → SiLU → Conv 64→3

    Args:
        in_channels: Input channels (15 = 3 noisy + 4×3 context).
        out_channels: Output channels (3 = velocity field RGB).
        channel_mult: Channel counts at each U-Net level.
        cond_dim: Conditioning embedding dimension (time + action combined).
        num_groups: Groups for GroupNorm in all ResBlocks.
        num_actions: Size of discrete action space (Procgen = 15).
        action_embed_dim: Intermediate action embedding size before projection.
        time_embed_dim: Sinusoidal embedding dimension before MLP.
        cfg_dropout: Probability of zeroing action embedding during training
                     (for classifier-free guidance).
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 3,
        channel_mult: List[int] = [64, 128, 256, 512],
        cond_dim: int = 512,
        num_groups: int = 32,
        num_actions: int = 15,
        action_embed_dim: int = 256,
        time_embed_dim: int = 256,
        cfg_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cfg_dropout = cfg_dropout
        self.cond_dim = cond_dim
        ch = channel_mult  # shorthand: [64, 128, 256, 512]

        # --- Conditioning embeddings ---
        # Time: sinusoidal → MLP(256 → 512 → 512)
        self.time_embed = SinusoidalEmbedding(dim=time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Action: Embedding(15, 256) → MLP(256 → 512 → 512)
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- Input convolution ---
        self.input_conv = nn.Conv2d(in_channels, ch[0], kernel_size=3, padding=1)

        # --- Down path ---
        # Level 1: ResBlock(64→64) × 2, Downsample(64)
        self.down1_res1 = ResBlock(ch[0], ch[0], cond_dim, num_groups)
        self.down1_res2 = ResBlock(ch[0], ch[0], cond_dim, num_groups)
        self.down1_downsample = Downsample(ch[0])

        # Level 2: ResBlock(64→128), ResBlock(128→128), Downsample(128)
        self.down2_res1 = ResBlock(ch[0], ch[1], cond_dim, num_groups)
        self.down2_res2 = ResBlock(ch[1], ch[1], cond_dim, num_groups)
        self.down2_downsample = Downsample(ch[1])

        # Level 3: ResBlock(128→256), ResBlock(256→256), Downsample(256)
        self.down3_res1 = ResBlock(ch[1], ch[2], cond_dim, num_groups)
        self.down3_res2 = ResBlock(ch[2], ch[2], cond_dim, num_groups)
        self.down3_downsample = Downsample(ch[2])

        # Level 4: ResBlock(256→512), ResBlock(512→512) — no downsample
        self.down4_res1 = ResBlock(ch[2], ch[3], cond_dim, num_groups)
        self.down4_res2 = ResBlock(ch[3], ch[3], cond_dim, num_groups)

        # --- Middle ---
        self.mid_res1 = ResBlock(ch[3], ch[3], cond_dim, num_groups)
        self.mid_attn = SelfAttention(ch[3], num_heads=4, num_groups=num_groups)
        self.mid_res2 = ResBlock(ch[3], ch[3], cond_dim, num_groups)

        # --- Up path (with skip connections doubling input channels) ---
        # Level 4: cat(512+512)=1024→512, 512→256, Upsample(256)
        self.up4_res1 = ResBlock(ch[3] + ch[3], ch[3], cond_dim, num_groups)
        self.up4_res2 = ResBlock(ch[3], ch[2], cond_dim, num_groups)
        self.up4_upsample = Upsample(ch[2])

        # Level 3: cat(256+256)=512→256, 256→128, Upsample(128)
        self.up3_res1 = ResBlock(ch[2] + ch[2], ch[2], cond_dim, num_groups)
        self.up3_res2 = ResBlock(ch[2], ch[1], cond_dim, num_groups)
        self.up3_upsample = Upsample(ch[1])

        # Level 2: cat(128+128)=256→128, 128→64, Upsample(64)
        self.up2_res1 = ResBlock(ch[1] + ch[1], ch[1], cond_dim, num_groups)
        self.up2_res2 = ResBlock(ch[1], ch[0], cond_dim, num_groups)
        self.up2_upsample = Upsample(ch[0])

        # Level 1: cat(64+64)=128→64, 64→64 — no upsample
        self.up1_res1 = ResBlock(ch[0] + ch[0], ch[0], cond_dim, num_groups)
        self.up1_res2 = ResBlock(ch[0], ch[0], cond_dim, num_groups)

        # --- Output ---
        self.out_norm = nn.GroupNorm(num_groups, ch[0])
        self.out_conv = nn.Conv2d(ch[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field for flow matching.

        Args:
            x: U-Net input = concat(noisy_target, context_frames),
               shape [B, in_channels, H, W].
            t: Flow time, shape [B]. Values in [0, 1].
            action: Action indices, shape [B]. Integers in [0, num_actions).

        Returns:
            Predicted velocity field, shape [B, out_channels, H, W].
        """
        # --- Build conditioning embedding ---
        # Time: [B] → sinusoidal [B, 256] → MLP [B, 512]
        t_emb = self.time_mlp(self.time_embed(t))  # [B, cond_dim]

        # Action: [B] → Embedding [B, 256] → MLP [B, 512]
        a_emb = self.action_mlp(self.action_embed(action))  # [B, cond_dim]

        # Classifier-free guidance dropout: during training, zero out the
        # action embedding for cfg_dropout fraction of the batch. This teaches
        # the model to predict both conditionally and unconditionally, enabling
        # CFG at inference: v = v_uncond + s * (v_cond - v_uncond).
        if self.training and self.cfg_dropout > 0:
            B = action.shape[0]
            # Per-sample mask: 1 = keep action, 0 = drop action
            keep_mask = (torch.rand(B, device=action.device) >= self.cfg_dropout).float()
            a_emb = a_emb * keep_mask.unsqueeze(1)  # [B, cond_dim]

        # Combined conditioning: additive (both are [B, 512])
        cond = t_emb + a_emb  # [B, cond_dim]

        # --- Input conv ---
        h = self.input_conv(x)  # [B, 64, H, W]

        # --- Down path (save features for skip connections) ---
        # Level 1
        h = self.down1_res1(h, cond)
        h = self.down1_res2(h, cond)
        skip1 = h  # [B, 64, H, W]
        h = self.down1_downsample(h)  # [B, 64, H/2, W/2]

        # Level 2
        h = self.down2_res1(h, cond)
        h = self.down2_res2(h, cond)
        skip2 = h  # [B, 128, H/2, W/2]
        h = self.down2_downsample(h)  # [B, 128, H/4, W/4]

        # Level 3
        h = self.down3_res1(h, cond)
        h = self.down3_res2(h, cond)
        skip3 = h  # [B, 256, H/4, W/4]
        h = self.down3_downsample(h)  # [B, 256, H/8, W/8]

        # Level 4 (no downsample — this is the bottleneck resolution)
        h = self.down4_res1(h, cond)
        h = self.down4_res2(h, cond)
        skip4 = h  # [B, 512, H/8, W/8]

        # --- Middle ---
        h = self.mid_res1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_res2(h, cond)

        # --- Up path (concatenate skip connections) ---
        # Level 4
        h = torch.cat([h, skip4], dim=1)  # [B, 1024, H/8, W/8]
        h = self.up4_res1(h, cond)
        h = self.up4_res2(h, cond)
        h = self.up4_upsample(h)  # [B, 256, H/4, W/4]

        # Level 3
        h = torch.cat([h, skip3], dim=1)  # [B, 512, H/4, W/4]
        h = self.up3_res1(h, cond)
        h = self.up3_res2(h, cond)
        h = self.up3_upsample(h)  # [B, 128, H/2, W/2]

        # Level 2
        h = torch.cat([h, skip2], dim=1)  # [B, 256, H/2, W/2]
        h = self.up2_res1(h, cond)
        h = self.up2_res2(h, cond)
        h = self.up2_upsample(h)  # [B, 64, H, W]

        # Level 1
        h = torch.cat([h, skip1], dim=1)  # [B, 128, H, W]
        h = self.up1_res1(h, cond)
        h = self.up1_res2(h, cond)

        # --- Output ---
        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)  # [B, 3, H, W]

        return h
