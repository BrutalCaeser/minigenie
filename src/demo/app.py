"""
MiniGenie — Interactive Gradio demo for the flow matching world model.

Lets users step through a CoinRun game frame-by-frame by choosing actions.
The dynamics model predicts the next frame given 4 context frames + action.

Usage (Colab — recommended, needs GPU):
    python -m src.demo.app --ckpt-dir checkpoints/dynamics --data-dir data/coinrun/episodes

Usage (local CPU — slow but works):
    python -m src.demo.app --ckpt-dir checkpoints/dynamics --data-dir data/coinrun/episodes --device cpu

The demo focuses on single-step prediction where the model excels (27 dB PSNR).
Multi-step rollouts are shown as a filmstrip but degrade after ~5 steps
(known limitation — documented in docs/EVALUATION.md).
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Lazy import — Gradio may not be installed in test environments
gradio = None


# ---------------------------------------------------------------------------
# CoinRun action mapping
# ---------------------------------------------------------------------------
# Procgen uses a shared 15-action space. In CoinRun, only ~6 are distinct:
#   Directions: LEFT, RIGHT, UP (jump), plus diagonals
#   Actions 9–14 are button keys (no-op in CoinRun)
#   DOWN is ignored (platformer with gravity)

COINRUN_ACTIONS: Dict[str, int] = {
    "⬅️ Left": 1,
    "➡️ Right": 7,
    "⬆️ Jump": 5,
    "↗️ Jump Right": 8,
    "↖️ Jump Left": 2,
    "⏸️ No-op": 4,
}

ACTION_NAMES: Dict[int, str] = {v: k for k, v in COINRUN_ACTIONS.items()}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    ckpt_dir: str,
    config_path: str = "configs/dynamics.yaml",
    device: str = "cuda",
) -> Tuple["torch.nn.Module", int, dict]:
    """Load the trained dynamics model from checkpoint.

    Args:
        ckpt_dir: Directory containing step_XXXXXXX.pt checkpoint files.
        config_path: Path to dynamics YAML config.
        device: Device to load model onto.

    Returns:
        Tuple of (model, training_step, config_dict).
    """
    from src.models.unet import UNet
    from src.training.checkpoint import CheckpointManager

    # Load config
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    mcfg = config.get("model", {})

    model = UNet(
        in_channels=mcfg.get("in_channels", 15),
        out_channels=mcfg.get("out_channels", 3),
        channel_mult=mcfg.get("channel_mult", [64, 128, 256, 512]),
        cond_dim=mcfg.get("cond_dim", 512),
        num_actions=mcfg.get("num_actions", 15),
        num_groups=mcfg.get("num_groups", 32),
        cfg_dropout=0.0,  # No dropout at inference
    ).to(device)

    ckpt_mgr = CheckpointManager(ckpt_dir)
    state = ckpt_mgr.load_latest()
    if state is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    model.load_state_dict(state["model"])
    model.eval()
    step = state["step"]

    return model, step, config


def load_seed_frames(
    data_dir: str,
    num_seeds: int = 20,
    context_length: int = 4,
) -> List[np.ndarray]:
    """Load seed frame sequences from real episodes for the Reset button.

    Each seed is a sequence of `context_length` consecutive frames that
    can serve as starting context for the model.

    Args:
        data_dir: Path to directory with .npz episode files.
        num_seeds: Number of seed sequences to pre-load.
        context_length: Number of frames per seed.

    Returns:
        List of arrays, each [context_length, 64, 64, 3] uint8.
    """
    from glob import glob

    npz_paths = sorted([
        p for p in glob(os.path.join(data_dir, "*.npz"))
        if not os.path.basename(p).startswith("._")
    ])

    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    rng = random.Random(42)
    seeds = []

    # Sample from different episodes for variety
    sampled_paths = rng.sample(npz_paths, min(num_seeds, len(npz_paths)))

    for path in sampled_paths:
        data = np.load(path)
        frames = data["frames"]  # [T, H, W, 3] uint8
        T = len(frames)

        if T < context_length + 1:
            continue

        # Pick a random starting point (not too early, not too late)
        min_start = 0
        max_start = T - context_length - 1
        if max_start <= min_start:
            continue

        start = rng.randint(min_start, max_start)
        seed_frames = frames[start : start + context_length].copy()
        seeds.append(seed_frames)

        if len(seeds) >= num_seeds:
            break

    if not seeds:
        raise ValueError("Could not extract any seed frames from episodes")

    return seeds


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_next_frame(
    model: "torch.nn.Module",
    context_frames: List[np.ndarray],
    action: int,
    num_steps: int = 15,
    cfg_scale: float = 2.0,
    device: str = "cuda",
) -> np.ndarray:
    """Generate the next frame given context frames and an action.

    Args:
        model: Trained U-Net in eval mode.
        context_frames: List of 4 frames, each [64, 64, 3] uint8 or float [0,1].
        action: Action index (0–14).
        num_steps: Euler ODE integration steps.
        cfg_scale: Classifier-free guidance scale.
        device: Computation device.

    Returns:
        Predicted frame as [64, 64, 3] uint8 numpy array.
    """
    from src.training.train_dynamics import generate_next_frame

    # Convert frames to tensor: [1, H*3, h, w] float [0, 1]
    tensors = []
    for f in context_frames:
        if f.dtype == np.uint8:
            t = torch.from_numpy(f.copy()).float().div(255.0)
        else:
            t = torch.from_numpy(f.copy()).float()
        t = t.permute(2, 0, 1)  # [3, h, w]
        tensors.append(t)

    context = torch.cat(tensors, dim=0).unsqueeze(0).to(device)  # [1, 12, 64, 64]
    act = torch.tensor([action], dtype=torch.long, device=device)

    pred = generate_next_frame(
        model, context, act,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
    )  # [1, 3, 64, 64]

    # Convert back to uint8 numpy
    frame = pred[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()  # [64, 64, 3]
    frame = (frame * 255).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_demo(
    model: "torch.nn.Module",
    seed_frames: List[np.ndarray],
    model_step: int,
    num_steps: int = 15,
    cfg_scale: float = 2.0,
    device: str = "cuda",
) -> "gradio.Blocks":
    """Build the Gradio interface.

    Args:
        model: Loaded dynamics model.
        seed_frames: Pre-loaded seed frame sequences.
        model_step: Training step of the loaded checkpoint.
        num_steps: Euler ODE steps for generation.
        cfg_scale: CFG scale for generation.
        device: Computation device.

    Returns:
        Gradio Blocks app ready to .launch().
    """
    import gradio as gr

    CONTEXT_LENGTH = 4
    MAX_FILMSTRIP = 20  # Show last 20 frames in filmstrip

    def reset_state():
        """Pick a random seed and reset the frame buffer."""
        seed = random.choice(seed_frames)  # [4, 64, 64, 3] uint8
        frame_buffer = [seed[i] for i in range(CONTEXT_LENGTH)]

        current_frame = frame_buffer[-1]
        filmstrip = _make_filmstrip(frame_buffer)
        status = "🔄 Reset — pick an action to start generating"
        step_count = 0

        return (
            current_frame,   # current_frame_display
            filmstrip,        # filmstrip_display
            frame_buffer,     # state: frame_buffer
            step_count,       # state: step_count
            status,           # status_text
        )

    def take_action(action_name, frame_buffer, step_count):
        """Generate next frame for the chosen action."""
        if frame_buffer is None or len(frame_buffer) < CONTEXT_LENGTH:
            # Not initialised — reset first
            return reset_state()

        action_idx = COINRUN_ACTIONS[action_name]

        # Use last 4 frames as context
        context = frame_buffer[-CONTEXT_LENGTH:]
        next_frame = predict_next_frame(
            model, context, action_idx,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            device=device,
        )

        frame_buffer = frame_buffer + [next_frame]
        step_count += 1

        current_frame = next_frame
        filmstrip = _make_filmstrip(frame_buffer[-MAX_FILMSTRIP:])
        status = (
            f"Step {step_count} — action: {action_name} "
            f"(idx {action_idx})"
        )

        return (
            current_frame,
            filmstrip,
            frame_buffer,
            step_count,
            status,
        )

    def _make_filmstrip(frames: List[np.ndarray]) -> np.ndarray:
        """Stitch frames into a horizontal strip with borders.

        Args:
            frames: List of [64, 64, 3] uint8 arrays.

        Returns:
            Single [66, N*66, 3] uint8 array (1px border around each frame).
        """
        if not frames:
            return np.zeros((66, 66, 3), dtype=np.uint8)

        bordered = []
        for f in frames:
            # Add 1px white border
            h, w = f.shape[:2]
            b = np.ones((h + 2, w + 2, 3), dtype=np.uint8) * 200
            b[1:-1, 1:-1] = f
            bordered.append(b)

        strip = np.concatenate(bordered, axis=1)
        return strip

    # --- Build UI ---
    with gr.Blocks(
        title="MiniGenie — World Model Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(f"""
# 🧞 MiniGenie — Interactive World Model

**Play CoinRun one frame at a time.** Choose an action and the model predicts
what happens next. The model was trained for {model_step:,} steps on 5,000
CoinRun episodes using flow matching with classifier-free guidance.

- **Single-step PSNR:** 27.0 dB | **SSIM:** 0.84
- **Architecture:** 42M-param U-Net, 15-step Euler ODE, CFG scale {cfg_scale}
- ⚠️ Quality degrades after ~5 generated steps (autoregressive error accumulation)

*Built entirely from scratch in PyTorch — no pretrained models or diffusion libraries.*
        """)

        # State (not displayed)
        frame_buffer_state = gr.State(value=None)
        step_count_state = gr.State(value=0)

        with gr.Row():
            with gr.Column(scale=2):
                current_frame_display = gr.Image(
                    label="Current Frame (64×64)",
                    height=320,
                    width=320,
                    show_label=True,
                    interactive=False,
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Press 'Reset / New Seed' to start",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Actions")
                action_buttons = {}
                for name in COINRUN_ACTIONS:
                    btn = gr.Button(name, variant="secondary", size="lg")
                    action_buttons[name] = btn

                gr.Markdown("---")
                reset_btn = gr.Button(
                    "🔄 Reset / New Seed",
                    variant="primary",
                    size="lg",
                )

        gr.Markdown("### Frame History (most recent on the right)")
        filmstrip_display = gr.Image(
            label="Filmstrip",
            height=100,
            show_label=False,
            interactive=False,
        )

        gr.Markdown(f"""
---
### About this demo
- Each click generates **one frame** using 15 Euler ODE steps with CFG scale {cfg_scale}
- The model sees the **last 4 frames** as context (context window = 4)
- First 4 frames are real CoinRun frames; all subsequent frames are model-generated
- Action conditioning is weak at 80K training steps — different actions may look similar
- **Best experience:** try 3–5 actions from a reset, then reset again

*Model checkpoint: step {model_step:,} | Game: CoinRun | [GitHub](https://github.com/BrutalCaeser/minigenie)*
        """)

        # --- Wire events ---
        all_outputs = [
            current_frame_display,
            filmstrip_display,
            frame_buffer_state,
            step_count_state,
            status_text,
        ]

        reset_btn.click(
            fn=reset_state,
            inputs=[],
            outputs=all_outputs,
        )

        for name, btn in action_buttons.items():
            btn.click(
                fn=take_action,
                inputs=[
                    gr.State(value=name),
                    frame_buffer_state,
                    step_count_state,
                ],
                outputs=all_outputs,
            )

        # Auto-reset on load
        demo.load(fn=reset_state, inputs=[], outputs=all_outputs)

    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniGenie interactive world model demo"
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default="checkpoints/dynamics",
        help="Path to dynamics model checkpoint directory",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/coinrun/episodes",
        help="Path to CoinRun episode data (for seed frames)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/dynamics.yaml",
        help="Path to dynamics YAML config",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detected if not specified)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=15,
        help="Euler ODE integration steps",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=2.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for the Gradio server",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Loading model from {args.ckpt_dir}...")
    model, step, config = load_model(args.ckpt_dir, args.config, device)
    print(f"Model loaded: step {step:,}, {sum(p.numel() for p in model.parameters()):,} params")

    print(f"Loading seed frames from {args.data_dir}...")
    seeds = load_seed_frames(
        args.data_dir,
        num_seeds=20,
        context_length=config.get("model", {}).get("context_frames", 4),
    )
    print(f"Loaded {len(seeds)} seed frame sequences")

    print("Building Gradio interface...")
    demo = create_demo(
        model, seeds, step,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        device=device,
    )

    print(f"Launching on port {args.port} (share={args.share})...")
    demo.launch(
        server_port=args.port,
        share=args.share,
        inline=False,
    )


if __name__ == "__main__":
    main()
