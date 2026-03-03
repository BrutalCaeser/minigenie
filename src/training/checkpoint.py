"""
Checkpoint management for training runs.

Handles saving, loading, and auto-resuming model checkpoints.
Keeps only the N most recent checkpoints to manage disk space.
Saves/restores full training state: model, optimizer, scheduler, RNG.

Usage:
    manager = CheckpointManager("checkpoints/dynamics", max_keep=3)
    
    # Save
    manager.save(model, optimizer, scheduler, step=5000)
    
    # Resume
    state = manager.load_latest()
    if state:
        start_step = manager.resume(state, model, optimizer, scheduler)
"""

import glob
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages saving and loading training checkpoints.

    Saves checkpoints as `step_{step:07d}.pt` files. Keeps only the most
    recent `max_keep` checkpoints, deleting older ones automatically.

    Stores full training state: model weights, optimizer state, scheduler
    state, RNG states (CPU + CUDA), and any extra metadata.

    Args:
        ckpt_dir: Directory to save checkpoints in. Created if it doesn't exist.
        max_keep: Maximum number of checkpoints to retain. Older ones are deleted.
    """

    def __init__(self, ckpt_dir: str, max_keep: int = 3) -> None:
        self.dir = ckpt_dir
        self.max_keep = max_keep
        os.makedirs(ckpt_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a training checkpoint.

        Args:
            model: The model to save.
            optimizer: The optimizer to save.
            scheduler: The LR scheduler to save (can be None).
            step: Current training step number.
            extra: Optional dict of additional metadata to store.

        Returns:
            Path to the saved checkpoint file.
        """
        state: Dict[str, Any] = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state()
        if extra is not None:
            state.update(extra)

        path = os.path.join(self.dir, f"step_{step:07d}.pt")
        torch.save(state, path)

        # Cleanup old checkpoints beyond max_keep
        ckpts = sorted(glob.glob(os.path.join(self.dir, "step_*.pt")))
        for old in ckpts[: -self.max_keep]:
            os.remove(old)

        return path

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint.

        Returns:
            Checkpoint state dict, or None if no checkpoints exist.
        """
        ckpts = sorted(glob.glob(os.path.join(self.dir, "step_*.pt")))
        if not ckpts:
            return None
        return torch.load(ckpts[-1], map_location="cpu", weights_only=False)

    def load_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint by step number.

        Args:
            step: The training step to load.

        Returns:
            Checkpoint state dict, or None if that step doesn't exist.
        """
        path = os.path.join(self.dir, f"step_{step:07d}.pt")
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    def resume(
        self,
        state: Dict[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ) -> int:
        """Restore training state from a checkpoint.

        Args:
            state: Checkpoint dict from load_latest() or load_step().
            model: Model to restore weights into.
            optimizer: Optimizer to restore state into.
            scheduler: LR scheduler to restore state into (optional).

        Returns:
            The training step the checkpoint was saved at.
        """
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        torch.random.set_rng_state(state["rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state(state["cuda_rng"])
        return state["step"]

    def list_checkpoints(self) -> list:
        """List all available checkpoint paths, sorted by step."""
        return sorted(glob.glob(os.path.join(self.dir, "step_*.pt")))

    @property
    def latest_step(self) -> Optional[int]:
        """Get the step number of the latest checkpoint, or None."""
        ckpts = self.list_checkpoints()
        if not ckpts:
            return None
        # Parse step from filename: step_0005000.pt → 5000
        basename = os.path.basename(ckpts[-1])
        return int(basename.replace("step_", "").replace(".pt", ""))
