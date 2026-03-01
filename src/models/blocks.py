"""Building blocks for the flow matching U-Net.

Contains: SinusoidalEmbedding, ResBlock (with AdaGN), SelfAttention.
These are the primitives that the U-Net and VQ-VAE are composed from.

Reference: docs/build_spec.md §2.3 (U-Net architecture, ResBlock with AdaGN,
conditioning embedding) and docs/foundations_guide.md Part 7 (how AdaGN works).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Map a scalar t ∈ [0, 1] to a fixed-frequency sinusoidal embedding.

    Uses the same encoding as Vaswani et al. (2017) "Attention Is All You Need",
    but applied to a continuous scalar instead of discrete positions.

    Math: for dimension index i in [0, dim):
        embed[2i]   = sin(t * 10000^(-2i/dim))
        embed[2i+1] = cos(t * 10000^(-2i/dim))

    This gives the flow time a rich, high-dimensional representation where
    nearby t values have similar embeddings (smooth) but the model can still
    distinguish fine differences (high-frequency components).

    Args:
        dim: Output embedding dimension. Must be even.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalEmbedding dim must be even, got {dim}")
        self.dim = dim

        # Precompute the frequency denominators: 10000^(2i/dim) for i=0..dim/2-1
        # Stored as a buffer (not a parameter — no gradients needed).
        half_dim = dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32) / half_dim  # [0, 1)
        inv_freq = 1.0 / (10000.0 ** exponents)  # [half_dim]
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed scalar timesteps.

        Args:
            t: Flow time values, shape [B] or [B, 1]. Values in [0, 1].

        Returns:
            Embedding of shape [B, dim].
        """
        # Flatten to [B]
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.view(-1).float()  # [B]

        # Outer product: [B, 1] * [1, half_dim] → [B, half_dim]
        angles = t.unsqueeze(1) * self.inv_freq.unsqueeze(0)  # [B, half_dim]

        # Interleave sin and cos → [B, dim]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, dim]
        return emb


class ResBlock(nn.Module):
    """Residual block with Adaptive Group Normalization (AdaGN).

    Architecture:
        GroupNorm → SiLU → Conv3x3 → GroupNorm (modulated by cond) → SiLU → Conv3x3 → + skip

    The conditioning vector (time + action embedding) modulates the second
    normalization layer via learned scale and shift. This is how the U-Net
    knows *what action was taken* and *what flow time step we're at* — it
    adjusts the internal feature processing based on these signals.

    Why AdaGN and not concatenation or cross-attention?
    - The action doesn't change what's in the image — it changes *how* the image
      should change. AdaGN modulates processing, which is the right inductive bias.
    - Cross-attention is expensive; AdaGN is a single linear layer per block.

    Reference: docs/build_spec.md §2.3 (ResBlock with AdaGN code),
               docs/foundations_guide.md Part 7.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        cond_dim: Conditioning embedding dimension (time + action).
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int = 512,
        num_groups: int = 32,
    ) -> None:
        super().__init__()

        # First conv path: norm → activation → conv
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # Second conv path: norm (modulated by AdaGN) → activation → conv
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # AdaGN: conditioning → scale and shift for norm2
        # Output is 2 * out_ch: first half is scale, second half is shift.
        self.adagn = nn.Linear(cond_dim, out_ch * 2)

        # Skip / residual connection: 1x1 conv if channel count changes, else identity
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map, shape [B, in_ch, H, W].
            cond: Conditioning embedding, shape [B, cond_dim].

        Returns:
            Output feature map, shape [B, out_ch, H, W].
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Adaptive GroupNorm: modulate normalized features with conditioning
        # scale and shift are [B, out_ch], need to broadcast to [B, out_ch, H, W]
        scale, shift = self.adagn(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Standard QKV self-attention applied at 2D spatial positions.

    Used only at 16×16 resolution in the U-Net (compute is O(n²) in spatial
    size — at 128×128 it would be prohibitive: 16384² = 268M attention entries
    vs 256² = 65K at 16×16).

    Architecture: GroupNorm → QKV projection → scaled dot-product attention → output projection

    Args:
        channels: Number of input/output channels.
        num_heads: Number of attention heads. channels must be divisible by num_heads.
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups, channels)
        # Single projection for Q, K, V concatenated
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention over spatial positions.

        Args:
            x: Feature map, shape [B, C, H, W].

        Returns:
            Attended feature map, shape [B, C, H, W] (residual added).
        """
        residual = x
        B, C, H, W = x.shape

        x = self.norm(x)

        # Project to Q, K, V: [B, 3*C, H, W]
        qkv = self.qkv(x)
        # Reshape to [B, 3, num_heads, head_dim, H*W] then permute
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [B, num_heads, head_dim, H*W]

        # Transpose to [B, num_heads, H*W, head_dim] for attention
        q = q.permute(0, 1, 3, 2)  # [B, heads, H*W, head_dim]
        k = k.permute(0, 1, 3, 2)  # [B, heads, H*W, head_dim]
        v = v.permute(0, 1, 3, 2)  # [B, heads, H*W, head_dim]

        # Scaled dot-product attention
        # Using PyTorch's efficient implementation when available
        attn_out = F.scaled_dot_product_attention(q, k, v)  # [B, heads, H*W, head_dim]

        # Reshape back to [B, C, H, W]
        attn_out = attn_out.permute(0, 1, 3, 2)  # [B, heads, head_dim, H*W]
        attn_out = attn_out.reshape(B, C, H, W)

        # Output projection + residual
        return self.out_proj(attn_out) + residual


class Downsample(nn.Module):
    """Spatial downsampling via strided convolution.

    Reduces spatial dimensions by 2× using a 4×4 conv with stride 2.
    Learned downsampling (not just pooling) — the model learns what
    information to preserve vs discard at each resolution.

    Args:
        channels: Number of input/output channels (preserved).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by 2×.

        Args:
            x: Feature map, shape [B, C, H, W].

        Returns:
            Downsampled feature map, shape [B, C, H//2, W//2].
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling via nearest-neighbor interpolation + convolution.

    Increases spatial dimensions by 2×. Nearest-neighbor avoids checkerboard
    artifacts that transposed convolutions can produce. The subsequent conv
    learns to smooth and refine the upsampled features.

    Args:
        channels: Number of input/output channels (preserved).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by 2×.

        Args:
            x: Feature map, shape [B, C, H, W].

        Returns:
            Upsampled feature map, shape [B, C, H*2, W*2].
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
