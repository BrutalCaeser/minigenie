#!/usr/bin/env python
"""
Smoke test: run ONE training step of each component with random data.

If this passes, the pipeline is mechanically correct.
Must pass before any Colab session.

Usage:
    python scripts/smoke_test.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vqvae import VQVAE
from src.models.unet import UNet
from src.training.checkpoint import CheckpointManager
from src.training.train_dynamics import flow_matching_step, generate_next_frame


def main() -> None:
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    device = "cpu"
    B, H, W = 2, 64, 64

    # ----- 1. Create tiny random data -----
    print("\n[1/5] Creating random data...")
    frames = torch.rand(B, 3, H, W)
    context = torch.rand(B, 12, H, W)  # 4 context frames × 3 channels
    action = torch.randint(0, 15, (B,))
    target = torch.rand(B, 3, H, W)
    print(f"  frames: {frames.shape}, context: {context.shape}, "
          f"action: {action.shape}, target: {target.shape}")

    # ----- 2. VQ-VAE: 1 training step -----
    print("\n[2/5] VQ-VAE training step...")
    vqvae = VQVAE(
        in_channels=3,
        hidden_channels=[32, 64, 128],  # Smaller for speed
        embed_dim=128,  # Must match hidden_channels[-1]
        codebook_size=64,
        num_res_blocks=1,
    ).to(device)
    vqvae_params = sum(p.numel() for p in vqvae.parameters())
    print(f"  VQ-VAE params: {vqvae_params:,}")

    vqvae.train()
    loss, metrics = vqvae.compute_loss(frames.to(device))
    assert torch.isfinite(loss), f"VQ-VAE loss not finite: {loss.item()}"
    loss.backward()
    print(f"  VQ-VAE loss: {loss.item():.4f} (recon={metrics['recon_loss']:.4f}, "
          f"commit={metrics['commitment_loss']:.4f})")
    print(f"  Codebook utilization: {metrics['codebook_utilization']:.1f}%")

    # ----- 3. Dynamics U-Net: 1 training step -----
    print("\n[3/5] Dynamics U-Net training step...")
    unet = UNet(
        in_channels=15,
        out_channels=3,
        channel_mult=[32, 64, 128, 256],  # Smaller for speed
        cond_dim=128,
        num_actions=15,
        num_groups=8,  # Smaller groups for smaller channels
        cfg_dropout=0.1,
    ).to(device)
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"  U-Net params: {unet_params:,}")

    unet.train()
    dyn_loss = flow_matching_step(
        unet,
        context.to(device),
        target.to(device),
        action.to(device),
    )
    assert torch.isfinite(dyn_loss), f"Dynamics loss not finite: {dyn_loss.item()}"
    dyn_loss.backward()
    print(f"  Dynamics loss: {dyn_loss.item():.4f}")

    # ----- 4. Inference: generate next frame -----
    print("\n[4/5] Inference (generate next frame)...")
    unet.eval()
    pred = generate_next_frame(
        unet,
        context.to(device),
        action.to(device),
        num_steps=3,  # Fewer steps for speed
        cfg_scale=2.0,
    )
    assert pred.shape == (B, 3, H, W), f"Bad prediction shape: {pred.shape}"
    assert pred.min() >= 0.0, f"Prediction has negative values: {pred.min()}"
    assert pred.max() <= 1.0, f"Prediction exceeds 1.0: {pred.max()}"
    print(f"  Prediction shape: {pred.shape}, range [{pred.min():.3f}, {pred.max():.3f}]")

    # ----- 5. Checkpoint save/load round-trip -----
    print("\n[5/5] Checkpoint save/load round-trip...")
    tmpdir = tempfile.mkdtemp(prefix="minigenie_smoke_")
    try:
        ckpt_mgr = CheckpointManager(tmpdir, max_keep=2)

        # Save VQ-VAE checkpoint
        vqvae_opt = torch.optim.AdamW(vqvae.parameters(), lr=1e-3)
        vqvae_sched = torch.optim.lr_scheduler.CosineAnnealingLR(vqvae_opt, T_max=100)
        path = ckpt_mgr.save(vqvae, vqvae_opt, vqvae_sched, step=42)
        print(f"  Saved checkpoint: {os.path.basename(path)}")

        # Load it back
        state = ckpt_mgr.load_latest()
        assert state is not None, "Failed to load checkpoint"
        assert state["step"] == 42, f"Wrong step: {state['step']}"

        # Create fresh model and restore
        vqvae2 = VQVAE(
            in_channels=3,
            hidden_channels=[32, 64, 128],
            embed_dim=128,
            codebook_size=64,
            num_res_blocks=1,
        ).to(device)
        vqvae_opt2 = torch.optim.AdamW(vqvae2.parameters(), lr=1e-3)
        vqvae_sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(vqvae_opt2, T_max=100)
        step = ckpt_mgr.resume(state, vqvae2, vqvae_opt2, vqvae_sched2)
        assert step == 42

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            vqvae.named_parameters(), vqvae2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weights mismatch at {n1}"
        print(f"  Restored step {step}, all weights match ✓")

        # Test max_keep: save 3 more, should keep only 2
        ckpt_mgr.save(vqvae, vqvae_opt, vqvae_sched, step=100)
        ckpt_mgr.save(vqvae, vqvae_opt, vqvae_sched, step=200)
        ckpt_mgr.save(vqvae, vqvae_opt, vqvae_sched, step=300)
        ckpts = ckpt_mgr.list_checkpoints()
        assert len(ckpts) == 2, f"Expected 2 checkpoints, got {len(ckpts)}"
        assert ckpt_mgr.latest_step == 300
        print(f"  Checkpoint cleanup works: {len(ckpts)} kept, latest={ckpt_mgr.latest_step}")
    finally:
        shutil.rmtree(tmpdir)

    # ----- Done -----
    print("\n" + "=" * 60)
    print("🎉 SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
