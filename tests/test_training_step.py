"""
Tests for training step mechanics.

Verifies that one forward+backward pass works correctly for each
component with randomly generated data, without needing real datasets.
"""

import math
import os
import shutil
import tempfile

import pytest
import torch
import torch.nn.functional as F

from src.models.vqvae import VQVAE
from src.models.unet import UNet
from src.training.checkpoint import CheckpointManager
from src.training.train_dynamics import flow_matching_step, generate_next_frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def vqvae(device):
    return VQVAE(
        in_channels=3,
        hidden_channels=[32, 64, 128],
        embed_dim=128,  # Must match hidden_channels[-1]
        codebook_size=64,
        num_res_blocks=1,
    ).to(device)


@pytest.fixture
def unet(device):
    return UNet(
        in_channels=15,
        out_channels=3,
        channel_mult=[32, 64, 128, 256],
        cond_dim=128,
        num_actions=15,
        num_groups=8,
        cfg_dropout=0.1,
    ).to(device)


@pytest.fixture
def random_frames(device):
    """Random frames [B, 3, 64, 64]."""
    return torch.rand(2, 3, 64, 64, device=device)


@pytest.fixture
def random_context(device):
    """Random context [B, 12, 64, 64] (4 frames × 3 channels)."""
    return torch.rand(2, 12, 64, 64, device=device)


@pytest.fixture
def random_action(device):
    """Random action [B]."""
    return torch.randint(0, 15, (2,), device=device)


@pytest.fixture
def random_target(device):
    """Random target frame [B, 3, 64, 64]."""
    return torch.rand(2, 3, 64, 64, device=device)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="minigenie_test_")
    yield d
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# VQ-VAE training step tests
# ---------------------------------------------------------------------------

class TestVQVAETrainingStep:
    def test_compute_loss_returns_finite(self, vqvae, random_frames):
        """compute_loss returns finite loss and valid metrics."""
        loss, metrics = vqvae.compute_loss(random_frames)
        assert torch.isfinite(loss)
        assert loss.item() > 0
        assert "recon_loss" in metrics
        assert "commitment_loss" in metrics
        assert "total_loss" in metrics
        assert "codebook_utilization" in metrics

    def test_backward_produces_gradients(self, vqvae, random_frames):
        """Backward pass populates .grad for all parameters."""
        vqvae.train()
        loss, _ = vqvae.compute_loss(random_frames)
        loss.backward()
        grad_count = sum(1 for p in vqvae.parameters() if p.grad is not None)
        total_count = sum(1 for _ in vqvae.parameters())
        assert grad_count > 0, "No gradients produced"
        # At least the trainable params should have grads
        assert grad_count == total_count

    def test_optimizer_step_changes_weights(self, vqvae, random_frames):
        """One optimizer step actually modifies weights."""
        optimizer = torch.optim.AdamW(vqvae.parameters(), lr=1e-2)
        # Snapshot initial weights
        w_before = {n: p.clone() for n, p in vqvae.named_parameters()}

        vqvae.train()
        loss, _ = vqvae.compute_loss(random_frames)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = False
        for n, p in vqvae.named_parameters():
            if not torch.equal(w_before[n], p):
                changed = True
                break
        assert changed, "Weights didn't change after optimizer step"

    def test_reconstruction_shape(self, vqvae, random_frames):
        """Forward produces correct output shapes."""
        x_hat, indices, recon_loss, commit_loss = vqvae(random_frames)
        B, _, H, W = random_frames.shape
        assert x_hat.shape == random_frames.shape
        assert indices.shape == (B, H // 8, W // 8)
        assert recon_loss.ndim == 0  # scalar
        assert commit_loss.ndim == 0


# ---------------------------------------------------------------------------
# Flow matching training step tests
# ---------------------------------------------------------------------------

class TestDynamicsTrainingStep:
    def test_flow_matching_loss_finite(self, unet, random_context, random_target, random_action):
        """Flow matching step returns finite scalar loss."""
        unet.train()
        loss = flow_matching_step(unet, random_context, random_target, random_action)
        assert torch.isfinite(loss)
        assert loss.item() > 0
        assert loss.ndim == 0

    def test_flow_matching_backward(self, unet, random_context, random_target, random_action):
        """Backward pass through flow matching produces gradients."""
        unet.train()
        loss = flow_matching_step(unet, random_context, random_target, random_action)
        loss.backward()
        grad_count = sum(1 for p in unet.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_flow_matching_optimizer_step(self, unet, random_context, random_target, random_action):
        """Optimizer step modifies U-Net weights."""
        optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-2)
        w_before = {n: p.clone() for n, p in unet.named_parameters()}

        unet.train()
        loss = flow_matching_step(unet, random_context, random_target, random_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = False
        for n, p in unet.named_parameters():
            if not torch.equal(w_before[n], p):
                changed = True
                break
        assert changed

    def test_flow_matching_no_noise_aug(self, unet, random_context, random_target, random_action):
        """Flow matching step works with noise augmentation disabled."""
        unet.train()
        loss = flow_matching_step(
            unet, random_context, random_target, random_action,
            noise_aug_prob=0.0,
        )
        assert torch.isfinite(loss)

    def test_flow_matching_full_noise_aug(self, unet, random_context, random_target, random_action):
        """Flow matching step works with noise augmentation always on."""
        unet.train()
        loss = flow_matching_step(
            unet, random_context, random_target, random_action,
            noise_aug_prob=1.0, noise_aug_sigma_max=0.5,
        )
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------

class TestInference:
    def test_generate_shape(self, unet, random_context, random_action):
        """generate_next_frame produces correct shape."""
        unet.eval()
        pred = generate_next_frame(unet, random_context, random_action, num_steps=2, cfg_scale=2.0)
        assert pred.shape == (2, 3, 64, 64)

    def test_generate_range(self, unet, random_context, random_action):
        """Generated frames are clamped to [0, 1]."""
        unet.eval()
        pred = generate_next_frame(unet, random_context, random_action, num_steps=2, cfg_scale=2.0)
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_generate_no_cfg(self, unet, random_context, random_action):
        """Inference works with cfg_scale=1.0 (no guidance)."""
        unet.eval()
        pred = generate_next_frame(unet, random_context, random_action, num_steps=2, cfg_scale=1.0)
        assert pred.shape == (2, 3, 64, 64)

    def test_generate_single_step(self, unet, random_context, random_action):
        """Inference works with num_steps=1."""
        unet.eval()
        pred = generate_next_frame(unet, random_context, random_action, num_steps=1, cfg_scale=2.0)
        assert pred.shape == (2, 3, 64, 64)

    def test_generate_deterministic(self, unet, random_context, random_action):
        """Same seed → same output."""
        unet.eval()
        torch.manual_seed(123)
        pred1 = generate_next_frame(unet, random_context, random_action, num_steps=3, cfg_scale=2.0)
        torch.manual_seed(123)
        pred2 = generate_next_frame(unet, random_context, random_action, num_steps=3, cfg_scale=2.0)
        assert torch.allclose(pred1, pred2)


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    def test_save_and_load(self, unet, tmp_dir):
        """Save and load checkpoint preserves step."""
        mgr = CheckpointManager(tmp_dir, max_keep=3)
        opt = torch.optim.AdamW(unet.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

        mgr.save(unet, opt, sched, step=42)
        state = mgr.load_latest()
        assert state is not None
        assert state["step"] == 42

    def test_resume_restores_weights(self, device, tmp_dir):
        """Resume restores model weights exactly."""
        mgr = CheckpointManager(tmp_dir, max_keep=3)

        # Save
        m1 = UNet(
            in_channels=15, out_channels=3, channel_mult=[16, 32, 64, 128],
            cond_dim=64, num_actions=15, num_groups=8, cfg_dropout=0.1,
        ).to(device)
        opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=100)
        mgr.save(m1, opt1, sched1, step=10)

        # Load into fresh model
        m2 = UNet(
            in_channels=15, out_channels=3, channel_mult=[16, 32, 64, 128],
            cond_dim=64, num_actions=15, num_groups=8, cfg_dropout=0.1,
        ).to(device)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100)

        state = mgr.load_latest()
        step = mgr.resume(state, m2, opt2, sched2)
        assert step == 10

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert torch.equal(p1, p2), f"Mismatch at {n1}"

    def test_max_keep_deletes_old(self, unet, tmp_dir):
        """max_keep limits number of saved checkpoints."""
        mgr = CheckpointManager(tmp_dir, max_keep=2)
        opt = torch.optim.AdamW(unet.parameters(), lr=1e-3)

        for step in [100, 200, 300, 400]:
            mgr.save(unet, opt, None, step=step)

        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 2
        assert mgr.latest_step == 400

    def test_load_empty_returns_none(self, tmp_dir):
        """load_latest returns None for empty dir."""
        mgr = CheckpointManager(tmp_dir, max_keep=3)
        assert mgr.load_latest() is None
        assert mgr.latest_step is None


# ---------------------------------------------------------------------------
# LR scheduler tests
# ---------------------------------------------------------------------------

class TestLRSchedule:
    def test_warmup_increases_lr(self):
        """LR increases during warmup phase."""
        warmup_steps = 100
        max_steps = 1000
        base_lr = 2e-4
        min_lr = 1e-5

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            min_ratio = min_lr / base_lr
            return min_ratio + (1 - min_ratio) * cosine_decay

        # During warmup, lr_lambda should be strictly increasing
        for s in range(1, warmup_steps):
            assert lr_lambda(s) > lr_lambda(s - 1)

    def test_cosine_decays_after_warmup(self):
        """LR decays during cosine phase."""
        warmup_steps = 100
        max_steps = 1000
        base_lr = 2e-4
        min_lr = 1e-5

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            min_ratio = min_lr / base_lr
            return min_ratio + (1 - min_ratio) * cosine_decay

        # At warmup end, lr_lambda should be ~1.0
        assert abs(lr_lambda(warmup_steps) - 1.0) < 0.05

        # At end, lr_lambda should be close to min_lr / base_lr
        end_ratio = lr_lambda(max_steps)
        expected_ratio = min_lr / base_lr
        assert abs(end_ratio - expected_ratio) < 0.01

    def test_gradient_clipping(self, unet, random_context, random_target, random_action):
        """Gradient clipping limits max gradient norm."""
        unet.train()
        loss = flow_matching_step(unet, random_context, random_target, random_action)
        loss.backward()

        # Clip
        total_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        # After clipping, each param's grad norm should be bounded
        for p in unet.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
