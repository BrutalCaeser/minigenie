"""Tests for evaluation modules: rollout, metrics, and visualize.

Tests run on CPU with small synthetic data — no GPU or real episodes needed.
Validates shapes, value ranges, and output file creation.
"""

import os
import tempfile
from typing import List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.eval.metrics import compute_psnr, compute_ssim
from src.eval.visualize import (
    save_rollout_grid,
    save_rollout_gif,
    save_side_by_side,
    plot_psnr_curve,
    plot_action_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_frames() -> List[torch.Tensor]:
    """Generate a list of random frames [3, 16, 16] in [0, 1]."""
    return [torch.rand(3, 16, 16) for _ in range(10)]


@pytest.fixture
def output_dir() -> str:
    """Create a temporary directory for output files."""
    d = tempfile.mkdtemp(prefix="minigenie_test_eval_")
    yield d
    # Cleanup is optional — tmpdir cleaned up by OS eventually


# ---------------------------------------------------------------------------
# PSNR tests
# ---------------------------------------------------------------------------

class TestPSNR:
    """Tests for compute_psnr."""

    def test_identical_images_high_psnr(self):
        """Identical images should have very high PSNR (>50 dB)."""
        x = torch.rand(4, 3, 32, 32)
        psnr = compute_psnr(x, x)
        assert psnr.shape == (4,)
        # With epsilon 1e-8, identical → ~80 dB
        assert (psnr > 50).all(), f"Identical images PSNR too low: {psnr}"

    def test_different_images_lower_psnr(self):
        """Different images should have lower PSNR."""
        x = torch.rand(4, 3, 32, 32)
        y = torch.rand(4, 3, 32, 32)
        psnr = compute_psnr(x, y)
        assert psnr.shape == (4,)
        # Random images typically give ~8-15 dB PSNR
        assert (psnr < 40).all()
        assert (psnr > 0).all()

    def test_single_image_no_batch(self):
        """Should work with single image [3, H, W] without batch dim."""
        x = torch.rand(3, 32, 32)
        y = torch.rand(3, 32, 32)
        psnr = compute_psnr(x, y)
        assert psnr.ndim == 0  # scalar

    def test_noisy_image_expected_range(self):
        """Adding known noise should give predictable PSNR."""
        x = torch.rand(8, 3, 64, 64)
        # Add noise with std=0.1 → MSE ≈ 0.01 → PSNR ≈ 20 dB
        noise = torch.randn_like(x) * 0.1
        y = (x + noise).clamp(0, 1)
        psnr = compute_psnr(y, x)
        mean_psnr = psnr.mean().item()
        # Should be roughly 18-25 dB for sigma=0.1
        assert 12 < mean_psnr < 30, f"PSNR {mean_psnr} outside expected range"

    def test_psnr_batch_consistency(self):
        """Batched PSNR should equal per-image PSNR."""
        x = torch.rand(4, 3, 16, 16)
        y = torch.rand(4, 3, 16, 16)
        batch_psnr = compute_psnr(x, y)
        for i in range(4):
            single = compute_psnr(x[i], y[i])
            assert abs(batch_psnr[i].item() - single.item()) < 0.01


# ---------------------------------------------------------------------------
# SSIM tests
# ---------------------------------------------------------------------------

class TestSSIM:
    """Tests for compute_ssim."""

    def test_identical_images_ssim_one(self):
        """Identical images should have SSIM ≈ 1.0."""
        x = torch.rand(2, 3, 32, 32)
        ssim = compute_ssim(x, x)
        assert ssim.shape == (2,)
        assert (ssim > 0.99).all(), f"Identical images SSIM: {ssim}"

    def test_different_images_lower_ssim(self):
        """Different random images should have lower SSIM."""
        x = torch.rand(4, 3, 32, 32)
        y = torch.rand(4, 3, 32, 32)
        ssim = compute_ssim(x, y)
        assert ssim.shape == (4,)
        assert (ssim < 0.5).all(), f"Random images SSIM too high: {ssim}"

    def test_single_image(self):
        """Should work with single image [3, H, W]."""
        x = torch.rand(3, 32, 32)
        ssim = compute_ssim(x, x)
        assert ssim.ndim == 0
        assert ssim.item() > 0.99

    def test_ssim_range(self):
        """SSIM should be in [-1, 1] range."""
        x = torch.rand(4, 3, 32, 32)
        y = torch.rand(4, 3, 32, 32)
        ssim = compute_ssim(x, y)
        assert (ssim >= -1.0).all() and (ssim <= 1.0).all()

    def test_ssim_symmetry(self):
        """SSIM(x, y) should equal SSIM(y, x)."""
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        ssim_xy = compute_ssim(x, y)
        ssim_yx = compute_ssim(y, x)
        assert torch.allclose(ssim_xy, ssim_yx, atol=1e-5)


# ---------------------------------------------------------------------------
# Visualization tests (file creation)
# ---------------------------------------------------------------------------

class TestVisualize:
    """Tests for visualization functions — verify files are created and non-empty."""

    def test_save_rollout_grid(self, random_frames, output_dir):
        """Should create a PNG file."""
        path = os.path.join(output_dir, "rollout_grid.png")
        save_rollout_grid(random_frames, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100  # Not empty

    def test_save_rollout_grid_with_gt(self, output_dir):
        """Should create a PNG with GT comparison row."""
        pred = [torch.rand(3, 16, 16) for _ in range(8)]
        gt = [torch.rand(3, 16, 16) for _ in range(8)]
        ctx = [torch.rand(3, 16, 16) for _ in range(4)]
        path = os.path.join(output_dir, "rollout_grid_gt.png")
        save_rollout_grid(pred, path, context_frames=ctx, gt_frames=gt)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100

    def test_save_rollout_gif(self, random_frames, output_dir):
        """Should create a GIF file."""
        path = os.path.join(output_dir, "rollout.gif")
        save_rollout_gif(random_frames, path, fps=4)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100

    def test_plot_psnr_curve(self, output_dir):
        """Should create a PSNR curve plot."""
        psnr = np.linspace(25, 12, 50)  # Degrading PSNR
        std = np.ones(50) * 2
        ssim = np.linspace(0.9, 0.3, 50)
        path = os.path.join(output_dir, "psnr_curve.png")
        plot_psnr_curve(psnr, path, psnr_std_per_step=std, ssim_per_step=ssim)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100

    def test_plot_action_comparison(self, output_dir):
        """Should create an action comparison grid."""
        predictions = {a: torch.rand(3, 16, 16) for a in range(15)}
        context = torch.rand(3, 16, 16)
        path = os.path.join(output_dir, "action_comp.png")
        plot_action_comparison(predictions, path, context_frame=context)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100

    def test_save_side_by_side(self, output_dir):
        """Should create a side-by-side comparison."""
        pred = torch.rand(3, 32, 32)
        gt = torch.rand(3, 32, 32)
        path = os.path.join(output_dir, "side_by_side.png")
        save_side_by_side(pred, gt, path, psnr_val=24.5, title="Test")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100


# ---------------------------------------------------------------------------
# Rollout tests (with a tiny mock model)
# ---------------------------------------------------------------------------

class MockUNet(nn.Module):
    """Tiny mock that returns a slightly shifted input — no real inference."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.cfg_dropout = 0.0

    def forward(self, x, t, action):
        # Return first 3 channels slightly modified
        return x[:, :3] * 0.9 + 0.05


class TestRollout:
    """Tests for rollout generation using a mock model."""

    def test_generate_rollout_shapes(self):
        """Rollout should return correct number of frames with correct shapes."""
        from src.eval.rollout import generate_rollout

        model = MockUNet()
        model.eval()
        context = torch.rand(1, 12, 16, 16)  # 4 context frames
        actions = torch.randint(0, 15, (10,))

        # Monkey-patch generate_next_frame for the mock
        import src.eval.rollout as rollout_mod
        original_fn = rollout_mod.generate_next_frame

        def mock_generate(m, ctx, act, num_steps=15, cfg_scale=2.0):
            B, _, h, w = ctx.shape
            return torch.rand(B, 3, h, w)

        rollout_mod.generate_next_frame = mock_generate
        try:
            frames = generate_rollout(model, context, actions, num_inference_steps=2)
            assert len(frames) == 10
            for f in frames:
                assert f.shape == (3, 16, 16)
                assert f.min() >= 0 and f.max() <= 1
        finally:
            rollout_mod.generate_next_frame = original_fn

    def test_generate_rollout_with_gt_shapes(self):
        """Rollout with GT should return matching prediction and GT lists."""
        from src.eval.rollout import generate_rollout_with_gt

        model = MockUNet()
        model.eval()

        # Create a mock dataset with one episode
        class MockDataset:
            context_length = 4
            frame_skip = 1
            episodes = [
                (
                    np.random.randint(0, 256, (60, 16, 16, 3), dtype=np.uint8),
                    np.random.randint(0, 15, (59,), dtype=np.int32),
                )
            ]

        dataset = MockDataset()

        import src.eval.rollout as rollout_mod
        original_fn = rollout_mod.generate_next_frame

        def mock_generate(m, ctx, act, num_steps=15, cfg_scale=2.0):
            B, _, h, w = ctx.shape
            return torch.rand(B, 3, h, w)

        rollout_mod.generate_next_frame = mock_generate
        try:
            pred, gt, actions = generate_rollout_with_gt(
                model, dataset,
                episode_idx=0, start_t=5, rollout_steps=10,
                device="cpu",
            )
            assert len(pred) == 10
            assert len(gt) == 10
            assert len(actions) == 10
            for p, g in zip(pred, gt):
                assert p.shape == (3, 16, 16)
                assert g.shape == (3, 16, 16)
        finally:
            rollout_mod.generate_next_frame = original_fn


# ---------------------------------------------------------------------------
# Integration: metrics + rollout combined
# ---------------------------------------------------------------------------

class TestMetricsIntegration:
    """Test that metrics work on rollout-shaped outputs."""

    def test_psnr_over_rollout(self):
        """Compute PSNR at each step of a fake rollout."""
        pred_frames = [torch.rand(3, 32, 32) for _ in range(20)]
        gt_frames = [torch.rand(3, 32, 32) for _ in range(20)]

        psnr_per_step = []
        for p, g in zip(pred_frames, gt_frames):
            psnr = compute_psnr(p, g).item()
            psnr_per_step.append(psnr)

        assert len(psnr_per_step) == 20
        # Random images give ~8-15 dB
        assert all(0 < p < 40 for p in psnr_per_step)

    def test_ssim_over_rollout(self):
        """Compute SSIM at each step of a fake rollout."""
        pred_frames = [torch.rand(3, 32, 32) for _ in range(20)]
        gt_frames = [torch.rand(3, 32, 32) for _ in range(20)]

        ssim_per_step = []
        for p, g in zip(pred_frames, gt_frames):
            ssim = compute_ssim(p.unsqueeze(0), g.unsqueeze(0)).item()
            ssim_per_step.append(ssim)

        assert len(ssim_per_step) == 20
        assert all(-1 <= s <= 1 for s in ssim_per_step)
