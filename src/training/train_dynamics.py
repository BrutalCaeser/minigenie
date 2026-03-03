"""
Flow matching dynamics model training loop.

Trains the U-Net to predict velocity fields for frame generation,
using action-conditioned flow matching with noise augmentation and
classifier-free guidance dropout.

Usage (CLI):
    python -m src.training.train_dynamics --config configs/dynamics.yaml --data-dir data/coinrun

Usage (from Python / Colab):
    from src.training.train_dynamics import train
    train(data_dir="data/coinrun", ckpt_dir="checkpoints/dynamics")
"""

import argparse
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.models.unet import UNet
from src.data.dataset import WorldModelDataset
from src.training.checkpoint import CheckpointManager


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
# Flow matching training step
# ---------------------------------------------------------------------------

def flow_matching_step(
    model: UNet,
    context_frames: torch.Tensor,
    target_frame: torch.Tensor,
    action: torch.Tensor,
    noise_aug_prob: float = 0.5,
    noise_aug_sigma_max: float = 0.3,
) -> torch.Tensor:
    """Compute flow matching loss for one batch.

    Implements the core flow matching training logic from build_spec.md §2.4:
    1. Sample flow time t ~ U[0, 1]
    2. Create noisy interpolant: x_t = (1-t)*noise + t*target
    3. Target velocity: v = target - noise (constant along straight path)
    4. Optionally augment context with Gaussian noise (GameNGen technique)
    5. Predict velocity, compute MSE loss

    Args:
        model: U-Net dynamics model.
        context_frames: [B, H*3, h, w] context frames, float [0, 1].
        target_frame: [B, 3, h, w] target next frame, float [0, 1].
        action: [B] action indices.
        noise_aug_prob: Probability of applying noise augmentation to context.
        noise_aug_sigma_max: Maximum sigma for context noise augmentation.

    Returns:
        Scalar MSE loss on velocity prediction.
    """
    B = target_frame.shape[0]
    device = target_frame.device

    # 1. Sample flow time uniformly
    t = torch.rand(B, 1, 1, 1, device=device)  # [B, 1, 1, 1]

    # 2. Sample noise
    noise = torch.randn_like(target_frame)  # [B, 3, h, w]

    # 3. Interpolate along straight path: x_t = (1-t)*noise + t*target
    x_t = (1 - t) * noise + t * target_frame  # [B, 3, h, w]

    # 4. Target velocity = data - noise (constant along straight path)
    target_v = target_frame - noise  # [B, 3, h, w]

    # 5. Noise augmentation on context frames (GameNGen technique)
    # Reduces distribution shift during autoregressive rollout
    if random.random() < noise_aug_prob:
        aug_sigma = random.uniform(0.0, noise_aug_sigma_max)
        context_frames = context_frames + aug_sigma * torch.randn_like(context_frames)
        context_frames = context_frames.clamp(0, 1)

    # 6. Concatenate context with noisy interpolant
    model_input = torch.cat([x_t, context_frames], dim=1)  # [B, 15, h, w]

    # 7. Predict velocity
    # t needs to be [B] for the model's sinusoidal embedding
    t_flat = t.squeeze(-1).squeeze(-1).squeeze(-1)  # [B]
    pred_v = model(model_input, t_flat, action)  # [B, 3, h, w]

    # 8. MSE loss on velocity
    loss = F.mse_loss(pred_v, target_v)

    return loss


# ---------------------------------------------------------------------------
# Inference (next frame generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_next_frame(
    model: UNet,
    context_frames: torch.Tensor,
    action: torch.Tensor,
    num_steps: int = 15,
    cfg_scale: float = 2.0,
) -> torch.Tensor:
    """Generate the next frame using flow matching ODE integration.

    Uses Euler integration along the learned velocity field with
    classifier-free guidance.

    Args:
        model: Trained U-Net dynamics model (in eval mode).
        context_frames: [B, H*3, h, w] context frames.
        action: [B] action indices.
        num_steps: Number of Euler integration steps.
        cfg_scale: Classifier-free guidance scale. 1.0 = no guidance.

    Returns:
        Predicted next frame [B, 3, h, w], clamped to [0, 1].
    """
    B = context_frames.shape[0]
    h, w = context_frames.shape[2], context_frames.shape[3]
    device = context_frames.device

    # Start from pure noise
    x = torch.randn(B, 3, h, w, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        model_input = torch.cat([x, context_frames], dim=1)  # [B, 15, h, w]

        # Conditional velocity (with real action)
        v_cond = model(model_input, t, action)

        if cfg_scale != 1.0:
            # Unconditional velocity (null action — zeros)
            v_uncond = model(model_input, t, torch.zeros_like(action))
            # CFG: steer toward conditional prediction
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        # Euler step
        x = x + dt * v

    return x.clamp(0, 1)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    data_dir: str = "data/coinrun",
    ckpt_dir: str = "checkpoints/dynamics",
    config_path: str = "configs/dynamics.yaml",
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    resume: bool = True,
    seed: int = 42,
    max_episodes: Optional[int] = None,
    device: Optional[str] = None,
) -> None:
    """Train the flow matching dynamics model.

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
    flow_cfg = config.get("flow", {})
    noise_cfg = config.get("noise_augmentation", {})
    data_cfg = config.get("data", {})

    # Override from arguments
    _max_steps = max_steps or train_cfg.get("max_steps", 150000)
    _batch_size = batch_size or train_cfg.get("batch_size", 16)
    _lr = lr or train_cfg.get("learning_rate", 2e-4)
    _min_lr = train_cfg.get("min_learning_rate", 1e-5)
    _resolution = data_cfg.get("image_resolution", 64)
    _save_every = train_cfg.get("save_every", 2000)
    _log_every = train_cfg.get("log_every", 100)
    _sample_every = train_cfg.get("sample_every", 5000)
    _warmup_steps = train_cfg.get("warmup_steps", 1000)
    _weight_decay = train_cfg.get("weight_decay", 0.01)
    _grad_clip = train_cfg.get("gradient_clip_norm", 1.0)
    _mixed_precision = train_cfg.get("mixed_precision", "fp16")
    _grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    _noise_aug_prob = noise_cfg.get("probability", 0.5)
    _noise_aug_sigma_max = noise_cfg.get("sigma_max", 0.3)
    _cfg_scale = flow_cfg.get("cfg_scale", 2.0)
    _num_inference_steps = flow_cfg.get("num_inference_steps", 15)
    _context_length = model_cfg.get("context_frames", 4)
    _seed = config.get("seed", seed)

    seed_everything(_seed)

    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset & DataLoader
    target_res = (_resolution, _resolution) if _resolution != 64 else None
    dataset = WorldModelDataset(
        data_dir,
        context_length=_context_length,
        target_resolution=target_res,
        max_episodes=max_episodes,
    )
    loader = DataLoader(
        dataset,
        batch_size=_batch_size,
        shuffle=True,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} samples from {dataset.num_episodes} episodes")

    # Model
    model = UNet(
        in_channels=model_cfg.get("in_channels", 15),
        out_channels=model_cfg.get("out_channels", 3),
        channel_mult=model_cfg.get("channel_mult", [64, 128, 256, 512]),
        cond_dim=model_cfg.get("cond_dim", 512),
        num_actions=model_cfg.get("num_actions", 15),
        num_groups=model_cfg.get("num_groups", 32),
        cfg_dropout=flow_cfg.get("cfg_dropout", 0.1),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_lr,
        weight_decay=_weight_decay,
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )

    # Cosine scheduler with warmup
    def lr_lambda(step: int) -> float:
        if step < _warmup_steps:
            # Linear warmup from 0 to 1
            return step / max(_warmup_steps, 1)
        # Cosine decay from 1 to min_lr/lr
        progress = (step - _warmup_steps) / max(_max_steps - _warmup_steps, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        min_ratio = _min_lr / _lr
        return min_ratio + (1 - min_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    use_amp = _mixed_precision == "fp16" and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_dtype = torch.float16 if use_amp else torch.float32
    print(f"Mixed precision: {'fp16' if use_amp else 'disabled'}")

    # Checkpoint manager
    ckpt_manager = CheckpointManager(ckpt_dir, max_keep=train_cfg.get("max_checkpoints_kept", 3))
    start_step = 0

    if resume:
        state = ckpt_manager.load_latest()
        if state is not None:
            start_step = ckpt_manager.resume(state, model, optimizer, scheduler)
            if use_amp and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            print(f"Resumed from step {start_step}")
        else:
            print("No checkpoint found, starting from scratch")

    # Data integrity check
    context, action, target = dataset[0]
    expected_ctx_ch = _context_length * 3
    assert context.shape[0] == expected_ctx_ch, f"Context channels: {context.shape[0]} != {expected_ctx_ch}"
    assert target.shape == (3, context.shape[1], context.shape[2])
    assert 0 <= action.item() < 15
    print(f"Data OK: context {context.shape}, action {action.item()}, target {target.shape}")

    # Training loop
    model.train()
    data_iter = iter(loader)
    step = start_step
    t_start = time.time()
    running_loss = 0.0

    print(f"\nStarting training from step {start_step} to {_max_steps}")
    print(f"Batch size: {_batch_size}, LR: {_lr}→{_min_lr}, Resolution: {_resolution}×{_resolution}")
    print(f"Noise aug: p={_noise_aug_prob}, σ_max={_noise_aug_sigma_max}")
    print(f"CFG dropout: {flow_cfg.get('cfg_dropout', 0.1)}, inference scale: {_cfg_scale}")

    try:
        while step < _max_steps:
            # Get next batch
            try:
                context_batch, action_batch, target_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                context_batch, action_batch, target_batch = next(data_iter)

            context_batch = context_batch.to(device)
            action_batch = action_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward + backward with optional mixed precision
            with torch.amp.autocast(device, dtype=autocast_dtype, enabled=use_amp):
                loss = flow_matching_step(
                    model, context_batch, target_batch, action_batch,
                    noise_aug_prob=_noise_aug_prob,
                    noise_aug_sigma_max=_noise_aug_sigma_max,
                )
                loss = loss / _grad_accum

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (step + 1) % _grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scheduler.step()
            step += 1
            running_loss += loss.item() * _grad_accum  # Undo the division for logging

            # Log
            if step % _log_every == 0:
                avg_loss = running_loss / _log_every
                elapsed = time.time() - t_start
                steps_per_sec = _log_every / elapsed
                current_lr = scheduler.get_last_lr()[0]

                print(
                    f"Step {step}/{_max_steps} | "
                    f"loss={avg_loss:.5f} | "
                    f"lr={current_lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                # GPU memory monitoring
                if torch.cuda.is_available() and step % 1000 == 0:
                    alloc = torch.cuda.memory_allocated() / 1e9
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  GPU memory: {alloc:.1f}GB / {peak:.1f}GB peak")

                running_loss = 0.0
                t_start = time.time()

            # Save checkpoint
            if step % _save_every == 0:
                extra = {"scaler": scaler.state_dict()} if use_amp else None
                ckpt_manager.save(model, optimizer, scheduler, step, extra=extra)
                print(f"  Checkpoint saved at step {step}")

            # Generate samples
            if step % _sample_every == 0:
                _save_samples(
                    model, context_batch[:4], action_batch[:4], target_batch[:4],
                    step, ckpt_dir, device, _num_inference_steps, _cfg_scale,
                )

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nTraining interrupted at step {step}: {e}")
        extra = {"scaler": scaler.state_dict(), "interrupted": True} if use_amp else {"interrupted": True}
        ckpt_manager.save(model, optimizer, scheduler, step, extra=extra)
        print(f"Emergency checkpoint saved at step {step}")
        raise

    # Final save
    extra = {"scaler": scaler.state_dict()} if use_amp else None
    ckpt_manager.save(model, optimizer, scheduler, step, extra=extra)
    print(f"\nTraining complete! Final step: {step}")


def _save_samples(
    model: UNet,
    context: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
    step: int,
    ckpt_dir: str,
    device: str,
    num_steps: int = 15,
    cfg_scale: float = 2.0,
) -> None:
    """Save a grid of target vs predicted frames."""
    try:
        from torchvision.utils import save_image
    except ImportError:
        return

    model.eval()
    pred = generate_next_frame(model, context, action, num_steps=num_steps, cfg_scale=cfg_scale)
    model.train()

    # Row 1: targets, Row 2: predictions
    images = torch.cat([target.cpu(), pred.cpu()], dim=0)
    samples_dir = os.path.join(os.path.dirname(ckpt_dir), "samples_dynamics")
    os.makedirs(samples_dir, exist_ok=True)
    save_image(images, os.path.join(samples_dir, f"step_{step:07d}.png"), nrow=4)
    print(f"  Samples saved at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train flow matching dynamics model")
    parser.add_argument("--config", type=str, default="configs/dynamics.yaml")
    parser.add_argument("--data-dir", type=str, default="data/coinrun")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/dynamics")
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
