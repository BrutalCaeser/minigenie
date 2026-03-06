"""Evaluation metrics for the dynamics model.

Implements the quantitative metrics from build_spec.md §7:
  - §7.1  Single-step PSNR and SSIM
  - §7.2  Rollout quality degradation curve (PSNR per step)
  - §7.3  Action differentiation score

All metrics operate on float tensors in [0, 1] range, channels-first format.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet import UNet
from src.training.train_dynamics import generate_next_frame
from src.eval.rollout import generate_rollout, generate_rollout_with_gt


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio per image.

    PSNR = -10 * log10(MSE), where MSE is computed over (C, H, W) for each
    image in the batch. Higher is better; >30 dB is excellent reconstruction,
    >22 dB is the target for single-step dynamics prediction.

    Args:
        pred: Predicted images [B, C, H, W] or [C, H, W], float [0, 1].
        target: Ground truth images, same shape as pred.

    Returns:
        PSNR values in dB, shape [B] or scalar.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # [B]
    psnr = -10.0 * torch.log10(mse + 1e-8)

    return psnr.squeeze(0) if squeeze else psnr


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute Structural Similarity Index per image.

    Implements the standard SSIM formula using a Gaussian window.
    Values range from -1 to 1, with 1 meaning identical images.

    Args:
        pred: Predicted images [B, C, H, W], float [0, 1].
        target: Ground truth images, same shape.
        window_size: Size of Gaussian smoothing window.
        C1: Luminance stability constant.
        C2: Contrast stability constant.

    Returns:
        SSIM values, shape [B].
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    C = pred.shape[1]

    # Create Gaussian window
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords = coords - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)  # [K, 1]
    window_2d = window_1d @ window_1d.t()  # [K, K]
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)  # [C, 1, K, K]

    pad = window_size // 2

    # Compute means
    mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
    mu_target = F.conv2d(target, window, padding=pad, groups=C)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    # Compute variances and covariance
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_cross

    # SSIM formula
    numerator = (2 * mu_cross + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    ssim_map = numerator / denominator  # [B, C, H, W]

    # Average over spatial and channel dimensions
    ssim_per_image = ssim_map.mean(dim=(1, 2, 3))  # [B]

    return ssim_per_image.squeeze(0) if squeeze else ssim_per_image


# ---------------------------------------------------------------------------
# Evaluation routines (build_spec.md §7.1–7.3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_single_step(
    model: UNet,
    dataset,
    num_samples: int = 1000,
    batch_size: int = 16,
    num_inference_steps: int = 15,
    cfg_scale: float = 2.0,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate single-step prediction quality (build_spec.md §7.1).

    For num_samples test sequences, predict one step ahead and measure
    PSNR and SSIM against ground truth.

    Args:
        model: Trained U-Net in eval mode.
        dataset: WorldModelDataset with test episodes.
        num_samples: Number of samples to evaluate.
        batch_size: Batch size for inference.
        num_inference_steps: Euler ODE steps.
        cfg_scale: Classifier-free guidance scale.
        device: Computation device.

    Returns:
        Dict with keys: 'psnr_mean', 'psnr_std', 'psnr_median',
        'psnr_min', 'psnr_max', 'ssim_mean', 'ssim_std', 'num_samples'.
    """
    from torch.utils.data import DataLoader, Subset

    model.eval()

    # Randomly sample indices for representative evaluation
    # (sequential first-N is biased toward the first few episodes)
    rng = np.random.RandomState(42)
    total = len(dataset)
    n = min(num_samples, total)
    indices = rng.choice(total, size=n, replace=False).tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_psnr = []
    all_ssim = []

    for batch_idx, (context, action, target) in enumerate(loader):
        context = context.to(device)
        action = action.to(device)
        target = target.to(device)

        pred = generate_next_frame(
            model, context, action,
            num_steps=num_inference_steps,
            cfg_scale=cfg_scale,
        )

        psnr = compute_psnr(pred, target)
        ssim = compute_ssim(pred, target)

        all_psnr.extend(psnr.cpu().tolist())
        all_ssim.extend(ssim.cpu().tolist())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Single-step eval: {len(all_psnr)}/{num_samples} samples...")

    psnr_arr = np.array(all_psnr)
    ssim_arr = np.array(all_ssim)

    return {
        "psnr_mean": float(psnr_arr.mean()),
        "psnr_std": float(psnr_arr.std()),
        "psnr_median": float(np.median(psnr_arr)),
        "psnr_min": float(psnr_arr.min()),
        "psnr_max": float(psnr_arr.max()),
        "ssim_mean": float(ssim_arr.mean()),
        "ssim_std": float(ssim_arr.std()),
        "num_samples": len(psnr_arr),
    }


@torch.no_grad()
def evaluate_rollout_degradation(
    model: UNet,
    dataset,
    num_rollouts: int = 500,
    max_steps: int = 50,
    num_inference_steps: int = 15,
    cfg_scale: float = 2.0,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Evaluate rollout quality degradation over time (build_spec.md §7.2).

    Generate multi-step rollouts and measure PSNR at each step.
    The PSNR-vs-step curve characterises how quickly errors accumulate.

    Args:
        model: Trained U-Net in eval mode.
        dataset: WorldModelDataset with episodes.
        num_rollouts: Number of independent rollouts.
        max_steps: Steps per rollout.
        num_inference_steps: Euler ODE steps.
        cfg_scale: CFG scale.
        device: Computation device.

    Returns:
        Dict with keys:
            'psnr_per_step': [max_steps] array of mean PSNR at each step
            'psnr_std_per_step': [max_steps] array of std PSNR at each step
            'ssim_per_step': [max_steps] array of mean SSIM at each step
            'num_rollouts_actual': number of rollouts completed
    """
    model.eval()
    H = dataset.context_length
    fs = dataset.frame_skip

    # Collect (episode_idx, start_t) pairs that allow max_steps rollout
    valid_starts = []
    for ep_idx, (frames, actions) in enumerate(dataset.episodes):
        T = len(frames)
        min_t = H * fs
        max_t = min(T - 1, len(actions)) - max_steps
        for t in range(min_t, max(min_t, max_t + 1)):
            valid_starts.append((ep_idx, t))

    if len(valid_starts) == 0:
        raise ValueError(
            f"No episodes long enough for {max_steps}-step rollouts "
            f"with context_length={H}"
        )

    # Subsample if we have more than needed
    rng = np.random.RandomState(42)
    if len(valid_starts) > num_rollouts:
        chosen = rng.choice(len(valid_starts), size=num_rollouts, replace=False)
        valid_starts = [valid_starts[i] for i in chosen]
    else:
        num_rollouts = len(valid_starts)

    # Accumulate per-step metrics
    psnr_all = np.zeros((num_rollouts, max_steps))
    ssim_all = np.zeros((num_rollouts, max_steps))

    for r_idx, (ep_idx, start_t) in enumerate(valid_starts):
        pred_frames, gt_frames, _ = generate_rollout_with_gt(
            model, dataset, ep_idx, start_t, max_steps,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            device=device,
        )

        for step_i in range(max_steps):
            p = pred_frames[step_i].unsqueeze(0)
            g = gt_frames[step_i].unsqueeze(0)
            psnr_all[r_idx, step_i] = compute_psnr(p, g).item()
            ssim_all[r_idx, step_i] = compute_ssim(p, g).item()

        if (r_idx + 1) % 50 == 0:
            print(f"  Rollout eval: {r_idx + 1}/{num_rollouts} rollouts...")

    return {
        "psnr_per_step": psnr_all.mean(axis=0),
        "psnr_std_per_step": psnr_all.std(axis=0),
        "ssim_per_step": ssim_all.mean(axis=0),
        "ssim_std_per_step": ssim_all.std(axis=0),
        "num_rollouts_actual": num_rollouts,
    }


@torch.no_grad()
def evaluate_action_differentiation(
    model: UNet,
    dataset,
    num_start_frames: int = 100,
    num_actions: int = 15,
    num_inference_steps: int = 15,
    cfg_scale: float = 2.0,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate action differentiation (build_spec.md §7.3).

    For each starting context, generate 1-step predictions for all 15 actions
    and measure pairwise L2 distances. If action conditioning works,
    different actions should produce visibly different predictions.

    Args:
        model: Trained U-Net in eval mode.
        dataset: WorldModelDataset.
        num_start_frames: Number of starting contexts to evaluate.
        num_actions: Number of discrete actions (Procgen = 15).
        num_inference_steps: Euler ODE steps.
        cfg_scale: CFG scale.
        device: Computation device.

    Returns:
        Dict with keys: 'mean_l2_distance', 'std_l2_distance',
        'min_l2_distance', 'max_l2_distance',
        'per_action_mean_dist' (list of 15 mean distances),
        'num_start_frames'.
    """
    model.eval()

    # Select random starting contexts
    rng = np.random.RandomState(42)
    num_samples = min(num_start_frames, len(dataset))
    sample_indices = rng.choice(len(dataset), size=num_samples, replace=False)

    all_pairwise_distances = []
    per_action_distances = [[] for _ in range(num_actions)]

    for s_idx, data_idx in enumerate(sample_indices):
        context, _, _ = dataset[data_idx]
        context = context.unsqueeze(0).to(device)  # [1, H*3, h, w]

        # Generate prediction for each action
        predictions = []
        for a in range(num_actions):
            act = torch.tensor([a], dtype=torch.long, device=device)
            pred = generate_next_frame(
                model, context, act,
                num_steps=num_inference_steps,
                cfg_scale=cfg_scale,
            )  # [1, 3, h, w]
            predictions.append(pred[0].cpu())  # [3, h, w]

        # Compute all pairwise L2 distances
        for i in range(num_actions):
            for j in range(i + 1, num_actions):
                dist = (predictions[i] - predictions[j]).pow(2).mean().sqrt().item()
                all_pairwise_distances.append(dist)
                per_action_distances[i].append(dist)
                per_action_distances[j].append(dist)

        if (s_idx + 1) % 20 == 0:
            print(f"  Action diff eval: {s_idx + 1}/{num_samples} contexts...")

    dists = np.array(all_pairwise_distances)
    per_action_means = [
        float(np.mean(d)) if d else 0.0 for d in per_action_distances
    ]

    return {
        "mean_l2_distance": float(dists.mean()),
        "std_l2_distance": float(dists.std()),
        "min_l2_distance": float(dists.min()),
        "max_l2_distance": float(dists.max()),
        "per_action_mean_dist": per_action_means,
        "num_start_frames": num_samples,
    }
