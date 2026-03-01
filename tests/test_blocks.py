"""Comprehensive tests for src/models/blocks.py.

Tests cover:
  1. Shape correctness (multiple batch sizes, channel sizes, spatial sizes)
  2. Gradient flow to every parameter
  3. Gradient flow through conditioning
  4. Numerical stability (no NaN/Inf with random inputs)

Reference: docs/master_build_plan.md Phase 1, Step 1.5.
"""

import pytest
import torch
import torch.nn as nn

from src.models.blocks import (
    Downsample,
    ResBlock,
    SelfAttention,
    SinusoidalEmbedding,
    Upsample,
)


# ---------------------------------------------------------------------------
# SinusoidalEmbedding
# ---------------------------------------------------------------------------
class TestSinusoidalEmbedding:
    """Tests for SinusoidalEmbedding."""

    def test_output_shape(self) -> None:
        """Output shape is [B, dim] for batched input."""
        emb = SinusoidalEmbedding(dim=256)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert out.shape == (3, 256), f"Expected (3, 256), got {out.shape}"

    def test_output_shape_single(self) -> None:
        """Works with a single scalar (0-dim tensor)."""
        emb = SinusoidalEmbedding(dim=128)
        t = torch.tensor(0.5)
        out = emb(t)
        assert out.shape == (1, 128), f"Expected (1, 128), got {out.shape}"

    def test_different_t_different_embeddings(self) -> None:
        """Different t values produce different embeddings."""
        emb = SinusoidalEmbedding(dim=256)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        # Pairwise L2 distances should all be > 0
        for i in range(3):
            for j in range(i + 1, 3):
                dist = (out[i] - out[j]).norm().item()
                assert dist > 1e-6, f"t={t[i].item()} and t={t[j].item()} produced identical embeddings"

    def test_same_t_same_embedding(self) -> None:
        """Same t value always produces the same embedding (deterministic)."""
        emb = SinusoidalEmbedding(dim=256)
        t = torch.tensor([0.3, 0.3])
        out = emb(t)
        assert torch.allclose(out[0], out[1], atol=1e-7), "Same t produced different embeddings"

    def test_no_nans(self) -> None:
        """No NaN/Inf in output for boundary values."""
        emb = SinusoidalEmbedding(dim=256)
        t = torch.tensor([0.0, 1e-8, 0.5, 1.0 - 1e-8, 1.0])
        out = emb(t)
        assert torch.isfinite(out).all(), "NaN or Inf found in sinusoidal embedding"

    def test_even_dim_required(self) -> None:
        """Odd dim raises ValueError."""
        with pytest.raises(ValueError, match="even"):
            SinusoidalEmbedding(dim=255)

    def test_multiple_dims(self) -> None:
        """Works for various embedding dimensions."""
        for dim in [32, 64, 128, 256, 512]:
            emb = SinusoidalEmbedding(dim=dim)
            out = emb(torch.rand(4))
            assert out.shape == (4, dim)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------
class TestResBlock:
    """Tests for ResBlock with AdaGN conditioning."""

    def test_shape_same_channels(self) -> None:
        """Output shape matches input when in_ch == out_ch."""
        block = ResBlock(in_ch=64, out_ch=64, cond_dim=512)
        x = torch.randn(2, 64, 32, 32)
        cond = torch.randn(2, 512)
        out = block(x, cond)
        assert out.shape == (2, 64, 32, 32), f"Expected (2, 64, 32, 32), got {out.shape}"

    def test_shape_different_channels(self) -> None:
        """Output shape is [B, out_ch, H, W] when in_ch != out_ch."""
        block = ResBlock(in_ch=64, out_ch=128, cond_dim=512)
        x = torch.randn(2, 64, 32, 32)
        cond = torch.randn(2, 512)
        out = block(x, cond)
        assert out.shape == (2, 128, 32, 32), f"Expected (2, 128, 32, 32), got {out.shape}"

    def test_shape_multiple_configs(self) -> None:
        """Test with various channel, spatial, and batch configurations."""
        configs = [
            (1, 32, 64, 16, 16),   # B=1, in=32, out=64, H=W=16
            (4, 128, 256, 8, 8),   # B=4, in=128, out=256, H=W=8
            (2, 256, 512, 16, 16), # B=2, in=256, out=512, H=W=16
            (2, 64, 64, 64, 64),   # B=2, same channels, large spatial
        ]
        for B, in_ch, out_ch, H, W in configs:
            block = ResBlock(in_ch=in_ch, out_ch=out_ch, cond_dim=512)
            x = torch.randn(B, in_ch, H, W)
            cond = torch.randn(B, 512)
            out = block(x, cond)
            assert out.shape == (B, out_ch, H, W), (
                f"Config ({B},{in_ch},{out_ch},{H},{W}): expected {(B, out_ch, H, W)}, got {out.shape}"
            )

    def test_gradient_flow_all_params(self) -> None:
        """Gradient reaches every parameter in the block."""
        block = ResBlock(in_ch=64, out_ch=128, cond_dim=512)
        x = torch.randn(2, 64, 16, 16)
        cond = torch.randn(2, 512)
        out = block(x, cond)
        loss = out.mean()
        loss.backward()

        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_through_conditioning(self) -> None:
        """Gradient flows back to the conditioning input."""
        block = ResBlock(in_ch=64, out_ch=64, cond_dim=512)
        x = torch.randn(2, 64, 16, 16)
        cond = torch.randn(2, 512, requires_grad=True)
        out = block(x, cond)
        loss = out.mean()
        loss.backward()
        assert cond.grad is not None, "No gradient for conditioning"
        assert cond.grad.abs().sum() > 0, "Zero gradient for conditioning"

    def test_conditioning_affects_output(self) -> None:
        """Different conditioning produces different outputs (AdaGN is connected)."""
        block = ResBlock(in_ch=64, out_ch=64, cond_dim=512)
        block.eval()
        x = torch.randn(1, 64, 16, 16)
        cond_a = torch.randn(1, 512)
        cond_b = torch.randn(1, 512)
        out_a = block(x, cond_a)
        out_b = block(x, cond_b)
        assert not torch.allclose(out_a, out_b, atol=1e-5), (
            "Different conditioning produced identical outputs — AdaGN not working"
        )

    def test_skip_connection_identity(self) -> None:
        """When in_ch == out_ch, skip connection is nn.Identity."""
        block = ResBlock(in_ch=64, out_ch=64, cond_dim=512)
        assert isinstance(block.skip, nn.Identity), (
            f"Expected Identity skip, got {type(block.skip)}"
        )

    def test_skip_connection_conv(self) -> None:
        """When in_ch != out_ch, skip connection is a 1x1 Conv."""
        block = ResBlock(in_ch=64, out_ch=128, cond_dim=512)
        assert isinstance(block.skip, nn.Conv2d), (
            f"Expected Conv2d skip, got {type(block.skip)}"
        )

    def test_no_nans(self) -> None:
        """No NaN/Inf in output for random inputs."""
        block = ResBlock(in_ch=128, out_ch=256, cond_dim=512)
        x = torch.randn(4, 128, 16, 16)
        cond = torch.randn(4, 512)
        out = block(x, cond)
        assert torch.isfinite(out).all(), "NaN or Inf found in ResBlock output"


# ---------------------------------------------------------------------------
# SelfAttention
# ---------------------------------------------------------------------------
class TestSelfAttention:
    """Tests for SelfAttention."""

    def test_output_shape(self) -> None:
        """Output shape matches input shape (residual connection)."""
        attn = SelfAttention(channels=256, num_heads=4)
        x = torch.randn(2, 256, 16, 16)
        out = attn(x)
        assert out.shape == (2, 256, 16, 16), f"Expected (2, 256, 16, 16), got {out.shape}"

    def test_different_spatial_sizes(self) -> None:
        """Works at different spatial resolutions."""
        attn = SelfAttention(channels=128, num_heads=4)
        for size in [4, 8, 16, 32]:
            x = torch.randn(2, 128, size, size)
            out = attn(x)
            assert out.shape == (2, 128, size, size), (
                f"Spatial {size}×{size}: expected (2, 128, {size}, {size}), got {out.shape}"
            )

    def test_gradient_flow(self) -> None:
        """Gradient reaches all parameters."""
        attn = SelfAttention(channels=128, num_heads=4)
        x = torch.randn(2, 128, 8, 8)
        out = attn(x)
        loss = out.mean()
        loss.backward()

        for name, param in attn.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_no_nans(self) -> None:
        """No NaN/Inf in output for random inputs."""
        attn = SelfAttention(channels=256, num_heads=8)
        x = torch.randn(2, 256, 16, 16)
        out = attn(x)
        assert torch.isfinite(out).all(), "NaN or Inf found in SelfAttention output"

    def test_channels_divisible_by_heads(self) -> None:
        """Raises ValueError if channels not divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            SelfAttention(channels=100, num_heads=3)

    def test_residual_connection(self) -> None:
        """Output differs from input (attention is doing something)."""
        attn = SelfAttention(channels=64, num_heads=4)
        x = torch.randn(2, 64, 8, 8)
        out = attn(x)
        # The residual connection means output = attn(x) + x,
        # but attn(x) should be nonzero, so out != x in general.
        assert not torch.allclose(out, x, atol=1e-5), (
            "Attention output equals input — projection may be broken"
        )


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------
class TestDownsample:
    """Tests for Downsample (strided convolution)."""

    def test_output_shape(self) -> None:
        """Spatial dimensions halved, channels preserved."""
        down = Downsample(channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = down(x)
        assert out.shape == (2, 64, 16, 16), f"Expected (2, 64, 16, 16), got {out.shape}"

    def test_multiple_resolutions(self) -> None:
        """Works at multiple spatial sizes."""
        down = Downsample(channels=128)
        for size in [16, 32, 64, 128]:
            x = torch.randn(1, 128, size, size)
            out = down(x)
            assert out.shape == (1, 128, size // 2, size // 2)

    def test_gradient_flow(self) -> None:
        """Gradient flows through the strided conv."""
        down = Downsample(channels=64)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        out = down(x)
        out.mean().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Upsample
# ---------------------------------------------------------------------------
class TestUpsample:
    """Tests for Upsample (nearest-neighbor + conv)."""

    def test_output_shape(self) -> None:
        """Spatial dimensions doubled, channels preserved."""
        up = Upsample(channels=64)
        x = torch.randn(2, 64, 16, 16)
        out = up(x)
        assert out.shape == (2, 64, 32, 32), f"Expected (2, 64, 32, 32), got {out.shape}"

    def test_multiple_resolutions(self) -> None:
        """Works at multiple spatial sizes."""
        up = Upsample(channels=128)
        for size in [4, 8, 16, 32, 64]:
            x = torch.randn(1, 128, size, size)
            out = up(x)
            assert out.shape == (1, 128, size * 2, size * 2)

    def test_gradient_flow(self) -> None:
        """Gradient flows through the upsample path."""
        up = Upsample(channels=64)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        out = up(x)
        out.mean().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Down + Up roundtrip
# ---------------------------------------------------------------------------
class TestDownUpRoundtrip:
    """Verify Downsample → Upsample preserves shape."""

    def test_roundtrip_shape(self) -> None:
        """Down then Up returns to original spatial dimensions."""
        down = Downsample(channels=64)
        up = Upsample(channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = up(down(x))
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
