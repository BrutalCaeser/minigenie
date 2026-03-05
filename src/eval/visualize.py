"""Visualization utilities for evaluation outputs.

Creates publication-quality figures for the evaluation report:
  - Rollout grids (context → predicted frames side by side)
  - Animated GIFs of rollouts
  - PSNR degradation curves
  - Action comparison grids (same context, all 15 actions)
  - Side-by-side prediction vs ground truth

Reference: docs/build_spec.md §7.5 (qualitative analysis).
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [3, H, W] or [H, W, 3] float tensor to uint8 numpy HWC."""
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        # CHW → HWC
        tensor = tensor.permute(1, 2, 0)
    img = tensor.clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


def save_rollout_grid(
    frames: List[torch.Tensor],
    path: str,
    context_frames: Optional[List[torch.Tensor]] = None,
    gt_frames: Optional[List[torch.Tensor]] = None,
    cols: int = 10,
    title: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """Save a grid of rollout frames as a PNG image.

    Optionally includes context frames (with border) at the start and/or
    a ground-truth row for comparison.

    Args:
        frames: List of predicted frames, each [3, H, W].
        path: Output file path (.png).
        context_frames: Optional list of context frames to prepend.
        gt_frames: Optional ground truth frames for a comparison row.
        cols: Number of columns in the grid.
        title: Optional figure title.
        dpi: Output resolution.
    """
    import matplotlib.pyplot as plt

    # Build display rows
    rows_data = []
    row_labels = []

    if context_frames is not None and gt_frames is not None:
        # Row 1: context + GT
        row_labels.append("Ground Truth")
        rows_data.append(context_frames + gt_frames)
        # Row 2: context + predicted
        row_labels.append("Predicted")
        rows_data.append(context_frames + frames)
    elif gt_frames is not None:
        row_labels.append("Ground Truth")
        rows_data.append(gt_frames)
        row_labels.append("Predicted")
        rows_data.append(frames)
    else:
        all_frames = (context_frames or []) + frames
        row_labels.append("Rollout")
        rows_data.append(all_frames)

    num_rows = len(rows_data)
    max_cols = max(len(r) for r in rows_data)
    actual_cols = min(cols, max_cols)

    # If too many frames for one row, wrap
    total_rows = 0
    for r in rows_data:
        total_rows += max(1, (len(r) + actual_cols - 1) // actual_cols)

    fig, axes = plt.subplots(
        total_rows, actual_cols,
        figsize=(2 * actual_cols, 2 * total_rows),
        squeeze=False,
    )

    ax_row = 0
    for row_idx, (row_frames, label) in enumerate(zip(rows_data, row_labels)):
        for i, frame in enumerate(row_frames):
            r = ax_row + i // actual_cols
            c = i % actual_cols
            ax = axes[r][c]
            img = _to_numpy_image(frame)
            ax.imshow(img)
            ax.axis("off")

            # Label context frames differently
            if context_frames is not None and i < len(context_frames):
                ax.set_title(f"ctx {i}", fontsize=7, color="gray")
            else:
                pred_idx = i - (len(context_frames) if context_frames else 0)
                ax.set_title(f"t={pred_idx}", fontsize=7)

        # Set row label on first frame
        axes[ax_row][0].set_ylabel(label, fontsize=9, rotation=0, labelpad=40)

        ax_row += max(1, (len(row_frames) + actual_cols - 1) // actual_cols)

    # Hide empty axes
    for r in range(total_rows):
        for c in range(actual_cols):
            if not axes[r][c].has_data():
                axes[r][c].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved rollout grid: {path}")


def save_rollout_gif(
    frames: List[torch.Tensor],
    path: str,
    fps: int = 8,
    context_frames: Optional[List[torch.Tensor]] = None,
) -> None:
    """Save a rollout as an animated GIF.

    Args:
        frames: List of predicted frames, each [3, H, W].
        path: Output file path (.gif).
        fps: Frames per second.
        context_frames: Optional context frames to prepend (shown first).
    """
    import imageio

    all_frames = (context_frames or []) + frames
    images = [_to_numpy_image(f) for f in all_frames]

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    duration = 1000.0 / fps  # milliseconds per frame
    imageio.mimsave(path, images, duration=duration, loop=0)
    print(f"  Saved rollout GIF: {path} ({len(images)} frames, {fps} fps)")


def plot_psnr_curve(
    psnr_per_step: np.ndarray,
    path: str,
    psnr_std_per_step: Optional[np.ndarray] = None,
    ssim_per_step: Optional[np.ndarray] = None,
    target_psnr: float = 22.0,
    title: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """Plot PSNR (and optionally SSIM) degradation over rollout steps.

    This is the key diagnostic for autoregressive quality: shows how
    quickly prediction errors accumulate over time.

    Args:
        psnr_per_step: Mean PSNR at each step, shape [num_steps].
        path: Output file path (.png).
        psnr_std_per_step: Optional std dev for shaded error band.
        ssim_per_step: Optional SSIM values for a secondary axis.
        target_psnr: Target PSNR threshold (dashed line).
        title: Plot title.
        dpi: Output resolution.
    """
    import matplotlib.pyplot as plt

    steps = np.arange(1, len(psnr_per_step) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # PSNR line
    ax1.plot(steps, psnr_per_step, "b-o", markersize=3, label="Mean PSNR")
    if psnr_std_per_step is not None:
        ax1.fill_between(
            steps,
            psnr_per_step - psnr_std_per_step,
            psnr_per_step + psnr_std_per_step,
            alpha=0.2, color="blue",
        )
    ax1.axhline(y=target_psnr, color="red", linestyle="--", alpha=0.7, label=f"Target ({target_psnr} dB)")
    ax1.set_xlabel("Rollout Step")
    ax1.set_ylabel("PSNR (dB)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # Optional SSIM on secondary axis
    if ssim_per_step is not None:
        ax2 = ax1.twinx()
        ax2.plot(steps, ssim_per_step, "g-s", markersize=3, alpha=0.7, label="Mean SSIM")
        ax2.set_ylabel("SSIM", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_ylim(0, 1)

    ax1.legend(loc="upper right")
    plt.title(title or "Rollout Quality Degradation")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PSNR curve: {path}")


def plot_action_comparison(
    predictions: Dict[int, torch.Tensor],
    path: str,
    context_frame: Optional[torch.Tensor] = None,
    action_names: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """Save a grid showing predictions for each action from the same context.

    This visualises action differentiation: do different actions produce
    visibly different next frames?

    Args:
        predictions: Dict mapping action_id → predicted frame [3, H, W].
        path: Output file path (.png).
        context_frame: Optional context frame to show at the start.
        action_names: Optional mapping of action IDs to human-readable names.
        title: Plot title.
        dpi: Output resolution.
    """
    import matplotlib.pyplot as plt

    num_actions = len(predictions)
    cols = min(8, num_actions + (1 if context_frame is not None else 0))
    total = num_actions + (1 if context_frame is not None else 0)
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.4 * rows), squeeze=False)

    idx = 0

    # Optionally show context frame first
    if context_frame is not None:
        ax = axes[0][0]
        ax.imshow(_to_numpy_image(context_frame))
        ax.set_title("Context", fontsize=8, fontweight="bold")
        ax.axis("off")
        idx = 1

    # Show each action's prediction
    for action_id in sorted(predictions.keys()):
        r = (idx) // cols
        c = (idx) % cols
        ax = axes[r][c]
        ax.imshow(_to_numpy_image(predictions[action_id]))

        label = action_names.get(action_id, str(action_id)) if action_names else str(action_id)
        ax.set_title(f"a={label}", fontsize=8)
        ax.axis("off")
        idx += 1

    # Hide unused axes
    for r in range(rows):
        for c in range(cols):
            if not axes[r][c].has_data():
                axes[r][c].axis("off")

    plt.suptitle(title or "Action Comparison (same context, different actions)", fontsize=11)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved action comparison: {path}")


def save_side_by_side(
    pred: torch.Tensor,
    target: torch.Tensor,
    path: str,
    psnr_val: Optional[float] = None,
    title: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """Save a side-by-side comparison of prediction vs ground truth.

    Args:
        pred: Predicted frame [3, H, W].
        target: Ground truth frame [3, H, W].
        path: Output file path (.png).
        psnr_val: Optional PSNR value to annotate.
        title: Optional title.
        dpi: Output resolution.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    ax1.imshow(_to_numpy_image(target))
    ax1.set_title("Ground Truth", fontsize=9)
    ax1.axis("off")

    ax2.imshow(_to_numpy_image(pred))
    subtitle = "Predicted"
    if psnr_val is not None:
        subtitle += f" (PSNR: {psnr_val:.1f} dB)"
    ax2.set_title(subtitle, fontsize=9)
    ax2.axis("off")

    if title:
        fig.suptitle(title, fontsize=11)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
