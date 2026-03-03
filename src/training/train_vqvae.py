"""
VQ-VAE training loop.

Trains the VQ-VAE tokenizer on individual frames from Procgen episodes.
Uses EMA codebook updates, dead code reset, and commitment loss.

Usage (CLI):
    python -m src.training.train_vqvae --config configs/vqvae.yaml --data-dir data/coinrun

Usage (from Python / Colab):
    from src.training.train_vqvae import train
    train(data_dir="data/coinrun", ckpt_dir="checkpoints/vqvae")
"""

import argparse
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.models.vqvae import VQVAE
from src.training.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Single-frame dataset (VQ-VAE doesn't need sequences, just individual frames)
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Dataset that returns individual frames for VQ-VAE training.

    Unlike WorldModelDataset, this doesn't need context or actions — just
    individual frames for reconstruction training.

    Args:
        data_dir: Path to directory with episodes/ subfolder of .npz files.
        resolution: Target resolution (frames are resized if needed).
        max_episodes: Limit number of episodes loaded (for testing).
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int = 64,
        max_episodes: Optional[int] = None,
    ) -> None:
        from glob import glob

        self.resolution = resolution

        # Find episodes
        episodes_dir = os.path.join(data_dir, "episodes")
        if not os.path.isdir(episodes_dir):
            episodes_dir = data_dir

        # Filter ._* Apple Double files
        paths = sorted([
            p for p in glob(os.path.join(episodes_dir, "*.npz"))
            if not os.path.basename(p).startswith("._")
        ])
        if max_episodes is not None:
            paths = paths[:max_episodes]

        if not paths:
            raise FileNotFoundError(f"No .npz files in {episodes_dir}")

        # Collect all frames into a flat list
        self.frames = []
        for p in paths:
            data = np.load(p)
            self.frames.append(data["frames"])  # [T, H, W, 3] uint8

        # Concatenate all frames
        self.frames = np.concatenate(self.frames, axis=0)  # [N, H, W, 3]
        print(f"FrameDataset: {len(self.frames)} frames from {len(paths)} episodes")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]  # [H, W, 3] uint8
        frame = torch.from_numpy(frame.copy()).float().div(255.0)  # [H, W, 3]
        frame = frame.permute(2, 0, 1)  # [3, H, W]

        # Resize if needed
        if frame.shape[1] != self.resolution or frame.shape[2] != self.resolution:
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return frame


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    data_dir: str = "data/coinrun",
    ckpt_dir: str = "checkpoints/vqvae",
    config_path: str = "configs/vqvae.yaml",
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    resume: bool = True,
    seed: int = 42,
    max_episodes: Optional[int] = None,
    device: Optional[str] = None,
) -> None:
    """Train the VQ-VAE model.

    Args:
        data_dir: Path to Procgen episode data.
        ckpt_dir: Directory to save checkpoints.
        config_path: Path to YAML config file.
        max_steps: Override max training steps from config.
        batch_size: Override batch size from config.
        lr: Override learning rate from config.
        resume: If True, resume from latest checkpoint.
        seed: Random seed.
        max_episodes: Limit episodes for testing.
        device: Device to train on (auto-detected if None).
    """
    # Load config
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: config {config_path} not found, using defaults")
        config = {}

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    # Override from arguments
    _max_steps = max_steps or train_cfg.get("max_steps", 50000)
    _batch_size = batch_size or train_cfg.get("batch_size", 64)
    _lr = lr or train_cfg.get("learning_rate", 3e-4)
    _resolution = data_cfg.get("image_resolution", 64)
    _save_every = train_cfg.get("save_every", 2000)
    _log_every = train_cfg.get("log_every", 100)
    _sample_every = train_cfg.get("sample_every", 5000)
    _weight_decay = train_cfg.get("weight_decay", 0.01)
    _seed = config.get("seed", seed)

    seed_everything(_seed)

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset & DataLoader
    dataset = FrameDataset(data_dir, resolution=_resolution, max_episodes=max_episodes)
    loader = DataLoader(
        dataset,
        batch_size=_batch_size,
        shuffle=True,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # Model
    model = VQVAE(
        in_channels=model_cfg.get("in_channels", 3),
        hidden_channels=model_cfg.get("hidden_channels", [64, 128, 256]),
        codebook_size=model_cfg.get("codebook_size", 512),
        embed_dim=model_cfg.get("embed_dim", 256),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        ema_decay=model_cfg.get("ema_decay", 0.99),
        commitment_cost=model_cfg.get("commitment_cost", 0.25),
        dead_code_reset_every=model_cfg.get("dead_code_reset_every", 1000),
        dead_code_threshold=model_cfg.get("dead_code_threshold", 2),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"VQ-VAE parameters: {param_count:,}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_lr,
        weight_decay=_weight_decay,
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_max_steps, eta_min=0.0,
    )

    # Checkpoint manager
    ckpt_manager = CheckpointManager(ckpt_dir, max_keep=train_cfg.get("max_checkpoints_kept", 3))
    start_step = 0

    if resume:
        state = ckpt_manager.load_latest()
        if state is not None:
            start_step = ckpt_manager.resume(state, model, optimizer, scheduler)
            print(f"Resumed from step {start_step}")
        else:
            print("No checkpoint found, starting from scratch")

    # Data integrity check
    sample = dataset[0]
    assert sample.shape == (3, _resolution, _resolution), f"Bad frame shape: {sample.shape}"
    assert 0.0 <= sample.min() and sample.max() <= 1.0, "Frames not in [0, 1]"
    print(f"Data OK: {len(dataset)} frames, shape {sample.shape}")

    # Training loop
    model.train()
    data_iter = iter(loader)
    step = start_step
    t_start = time.time()
    running_loss = 0.0
    running_recon = 0.0
    running_commit = 0.0

    print(f"\nStarting training from step {start_step} to {_max_steps}")
    print(f"Batch size: {_batch_size}, LR: {_lr}, Resolution: {_resolution}×{_resolution}")

    try:
        while step < _max_steps:
            # Get next batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x = batch.to(device)  # [B, 3, H, W]

            # Forward pass: compute_loss returns (total_loss, metrics_dict)
            loss, metrics = model.compute_loss(x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1

            # Accumulate for logging
            running_loss += metrics["total_loss"]
            running_recon += metrics["recon_loss"]
            running_commit += metrics["commitment_loss"]

            # Log
            if step % _log_every == 0:
                avg_loss = running_loss / _log_every
                avg_recon = running_recon / _log_every
                avg_commit = running_commit / _log_every
                elapsed = time.time() - t_start
                steps_per_sec = _log_every / elapsed
                current_lr = scheduler.get_last_lr()[0]

                # Codebook utilization
                with torch.no_grad():
                    z_e = model.encoder(x)
                    _, indices, _ = model.quantizer(z_e)
                    unique_codes = indices.unique().numel()
                    utilization = 100.0 * unique_codes / model.quantizer.codebook.shape[0]

                print(
                    f"Step {step}/{_max_steps} | "
                    f"loss={avg_loss:.4f} (recon={avg_recon:.4f}, commit={avg_commit:.4f}) | "
                    f"codebook={utilization:.1f}% | "
                    f"lr={current_lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                # GPU memory monitoring
                if torch.cuda.is_available() and step % 1000 == 0:
                    alloc = torch.cuda.memory_allocated() / 1e9
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  GPU memory: {alloc:.1f}GB / {peak:.1f}GB peak")

                running_loss = 0.0
                running_recon = 0.0
                running_commit = 0.0
                t_start = time.time()

            # Save checkpoint
            if step % _save_every == 0:
                ckpt_manager.save(model, optimizer, scheduler, step)
                print(f"  Checkpoint saved at step {step}")

            # Save sample reconstructions
            if step % _sample_every == 0:
                _save_samples(model, x[:8], step, ckpt_dir, device)

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nTraining interrupted at step {step}: {e}")
        ckpt_manager.save(model, optimizer, scheduler, step, extra={"interrupted": True})
        print(f"Emergency checkpoint saved at step {step}")
        raise

    # Final save
    ckpt_manager.save(model, optimizer, scheduler, step)
    print(f"\nTraining complete! Final step: {step}")


def _save_samples(
    model: VQVAE,
    batch: torch.Tensor,
    step: int,
    ckpt_dir: str,
    device: str,
) -> None:
    """Save a grid of original vs reconstructed images."""
    try:
        from torchvision.utils import save_image
    except ImportError:
        return  # Skip if torchvision not available

    model.eval()
    with torch.no_grad():
        x_hat, _, _, _ = model(batch.to(device))
        recon = x_hat

    # Interleave original and reconstruction
    images = torch.stack([batch.cpu(), recon.cpu()], dim=1).reshape(-1, *batch.shape[1:])

    samples_dir = os.path.join(os.path.dirname(ckpt_dir), "samples_vqvae")
    os.makedirs(samples_dir, exist_ok=True)
    save_image(images, os.path.join(samples_dir, f"step_{step:07d}.png"), nrow=4)
    print(f"  Samples saved at step {step}")
    model.train()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train VQ-VAE")
    parser.add_argument("--config", type=str, default="configs/vqvae.yaml")
    parser.add_argument("--data-dir", type=str, default="data/coinrun")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/vqvae")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        config_path=args.config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=not args.no_resume,
        seed=args.seed,
        max_episodes=args.max_episodes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
