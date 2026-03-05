"""Autoregressive rollout generation for the dynamics model.

Takes a trained U-Net, starting context frames, and an action sequence,
then generates an N-step rollout by feeding each prediction back as
context for the next step.

This is the core loop that tests whether the model has learned a coherent
world model — errors compound autoregressively, so rollout quality degrades
over time. Noise augmentation during training (GameNGen technique) mitigates
this by teaching the model to handle imperfect inputs.

Reference: docs/build_spec.md §2.4 (inference), §7.2 (rollout evaluation).
"""

from typing import List, Optional, Tuple

import torch

from src.models.unet import UNet
from src.training.train_dynamics import generate_next_frame


@torch.no_grad()
def generate_rollout(
    model: UNet,
    context_frames: torch.Tensor,
    actions: torch.Tensor,
    num_inference_steps: int = 15,
    cfg_scale: float = 2.0,
) -> List[torch.Tensor]:
    """Generate an autoregressive rollout from starting context.

    At each step, the model predicts the next frame given the last H frames
    (context window) and the current action. The prediction is then appended
    to the frame buffer and becomes part of the context for the next step.

    Args:
        model: Trained U-Net dynamics model (should be in eval mode).
        context_frames: Starting context, shape [1, H*3, h, w], float [0, 1].
            H = number of context frames (typically 4).
        actions: Action sequence, shape [num_steps] (int64).
        num_inference_steps: Number of Euler ODE steps per frame generation.
        cfg_scale: Classifier-free guidance scale.

    Returns:
        List of predicted frames, each [3, h, w] on CPU, float [0, 1].
        Length = len(actions). Does NOT include the original context frames.
    """
    model.eval()
    device = context_frames.device
    h, w = context_frames.shape[2], context_frames.shape[3]
    num_context_channels = context_frames.shape[1]
    H = num_context_channels // 3  # number of context frames

    # Extract individual frames from the stacked context tensor
    # context_frames is [1, H*3, h, w] — unstack to list of [3, h, w]
    frame_buffer: List[torch.Tensor] = []
    for i in range(H):
        frame_buffer.append(context_frames[0, i * 3 : (i + 1) * 3])  # [3, h, w]

    predicted_frames: List[torch.Tensor] = []

    for step_idx in range(len(actions)):
        # Build context from the last H frames in the buffer
        ctx = torch.cat(frame_buffer[-H:], dim=0).unsqueeze(0)  # [1, H*3, h, w]
        act = actions[step_idx : step_idx + 1].to(device)  # [1]

        # Generate next frame via flow matching ODE
        pred = generate_next_frame(
            model, ctx, act,
            num_steps=num_inference_steps,
            cfg_scale=cfg_scale,
        )  # [1, 3, h, w]

        pred_frame = pred[0]  # [3, h, w]
        frame_buffer.append(pred_frame)
        predicted_frames.append(pred_frame.cpu())

    return predicted_frames


@torch.no_grad()
def generate_rollout_with_gt(
    model: UNet,
    dataset,
    episode_idx: int,
    start_t: int,
    rollout_steps: int,
    num_inference_steps: int = 15,
    cfg_scale: float = 2.0,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """Generate a rollout alongside ground truth frames for comparison.

    Uses real episode data to provide ground-truth context and actions,
    then generates predictions autoregressively while collecting the
    corresponding ground-truth frames.

    Args:
        model: Trained U-Net dynamics model (should be in eval mode).
        dataset: WorldModelDataset instance with loaded episodes.
        episode_idx: Which episode to use.
        start_t: Starting timestep within the episode (must have enough
            prior frames for context).
        rollout_steps: Number of steps to generate.
        num_inference_steps: Euler ODE steps per frame.
        cfg_scale: CFG scale.
        device: Device for computation.

    Returns:
        Tuple of (predicted_frames, gt_frames, actions_used):
            - predicted_frames: List of [3, h, w] tensors (CPU, float [0, 1])
            - gt_frames: List of [3, h, w] tensors (CPU, float [0, 1])
            - actions_used: List of action ints
    """
    import numpy as np

    model.eval()
    frames_np, actions_np = dataset.episodes[episode_idx]
    H = dataset.context_length
    fs = dataset.frame_skip

    # Validate start_t has enough context and enough future frames
    min_t = H * fs
    assert start_t >= min_t, f"start_t={start_t} too small (need >= {min_t})"
    max_t = min(len(frames_np) - 1, len(actions_np)) - rollout_steps
    assert start_t <= max_t, f"start_t={start_t} too large (max {max_t} for {rollout_steps} steps)"

    # Build initial context: H frames before start_t
    context_indices = [start_t - i * fs for i in range(H, 0, -1)]
    context_np = np.stack([frames_np[i] for i in context_indices])  # [H, h, w, 3]
    context = torch.from_numpy(context_np.copy()).float().div(255.0)  # [H, h, w, 3]
    context = context.permute(0, 3, 1, 2)  # [H, 3, h, w]
    context = context.reshape(1, -1, *context.shape[2:]).to(device)  # [1, H*3, h, w]

    # Get action sequence and ground truth frames
    action_sequence = torch.tensor(
        actions_np[start_t : start_t + rollout_steps], dtype=torch.long
    )
    gt_frames_np = frames_np[start_t + 1 : start_t + rollout_steps + 1]  # [steps, h, w, 3]

    # Convert GT to tensors
    gt_frames = []
    for f in gt_frames_np:
        gt = torch.from_numpy(f.copy()).float().div(255.0).permute(2, 0, 1)  # [3, h, w]
        gt_frames.append(gt)

    # Generate autoregressive rollout
    predicted_frames = generate_rollout(
        model, context, action_sequence,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
    )

    actions_used = action_sequence.tolist()

    return predicted_frames, gt_frames, actions_used
