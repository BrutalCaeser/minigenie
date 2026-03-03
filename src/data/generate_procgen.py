"""
Procgen data generation script.

Generates episodes from Procgen environments and saves them as compressed .npz files.
Each episode contains:
  - frames: uint8 array [T, H, W, 3] (T = number of frames including initial obs)
  - actions: int32 array [T-1] (one action per transition)

Designed to run on Colab (x86 Linux) where procgen is installable.
Includes a synthetic fallback for local testing on Apple Silicon.

Usage (Colab):
    python -m src.data.generate_procgen --game coinrun --episodes 5000 --save-dir data/coinrun/episodes

Usage (local testing):
    python -m src.data.generate_procgen --synthetic --episodes 10 --save-dir data/synthetic/episodes
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


def generate_procgen_episodes(
    game: str,
    save_dir: str,
    num_episodes: int = 5000,
    max_steps: int = 500,
    start_level: int = 0,
    num_levels: int = 0,
    distribution_mode: str = "hard",
) -> None:
    """Generate episodes from a Procgen environment.

    Args:
        game: Procgen game name (e.g., 'coinrun', 'starpilot').
        save_dir: Directory to save episode .npz files.
        num_episodes: Number of episodes to generate.
        max_steps: Maximum steps per episode before forced termination.
        start_level: Starting level seed for reproducibility.
        num_levels: Number of unique levels (0 = unlimited).
        distribution_mode: Difficulty mode ('easy' or 'hard').
    """
    try:
        import gym
    except ImportError:
        raise ImportError(
            "gym is required for procgen data generation. "
            "Install with: pip install gym==0.26.2 procgen"
        )

    try:
        import procgen  # noqa: F401 — registers procgen envs with gym
    except ImportError:
        raise ImportError(
            "procgen is not available on this platform. "
            "Run this script on Colab (x86 Linux) or use --synthetic for testing."
        )

    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(
        f"procgen:procgen-{game}-v0",
        start_level=start_level,
        num_levels=num_levels,
        distribution_mode=distribution_mode,
        render_mode="rgb_array",
    )

    print(f"Generating {num_episodes} episodes of {game} (max {max_steps} steps each)")
    print(f"Saving to: {save_dir}")

    episode_lengths = []
    for ep in tqdm(range(num_episodes), desc=f"Generating {game}"):
        obs = env.reset()
        frames = [obs]  # uint8 [64, 64, 3] — Procgen native resolution
        actions = []

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            actions.append(action)
            frames.append(obs)
            if done:
                break

        frames_arr = np.stack(frames).astype(np.uint8)  # [T, 64, 64, 3]
        actions_arr = np.array(actions, dtype=np.int32)  # [T-1]

        np.savez_compressed(
            os.path.join(save_dir, f"ep_{ep:05d}.npz"),
            frames=frames_arr,
            actions=actions_arr,
        )
        episode_lengths.append(len(frames))

    env.close()

    # Print summary statistics
    lengths = np.array(episode_lengths)
    print(f"\nGeneration complete!")
    print(f"  Episodes: {num_episodes}")
    print(f"  Episode length — mean: {lengths.mean():.1f}, min: {lengths.min()}, max: {lengths.max()}")
    print(f"  Total frames: {lengths.sum():,}")
    print(f"  Saved to: {save_dir}")


def generate_synthetic_episodes(
    save_dir: str,
    num_episodes: int = 10,
    num_frames: int = 50,
    height: int = 64,
    width: int = 64,
    num_actions: int = 15,
    seed: Optional[int] = 42,
) -> None:
    """Generate synthetic episodes for testing when procgen is not available.

    Creates episodes with random pixel data and random actions in the same
    .npz format as real procgen episodes. Useful for:
    - Local testing on Apple Silicon (where procgen can't install)
    - Unit tests for the dataset class
    - Smoke tests for the training pipeline

    Args:
        save_dir: Directory to save episode .npz files.
        num_episodes: Number of synthetic episodes to generate.
        num_frames: Number of frames per episode (all same length).
        height: Frame height in pixels.
        width: Frame width in pixels.
        num_actions: Number of discrete actions (Procgen has 15).
        seed: Random seed for reproducibility. None for random.
    """
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating {num_episodes} synthetic episodes ({num_frames} frames, {height}x{width})")
    print(f"Saving to: {save_dir}")

    for ep in range(num_episodes):
        # Random pixel data — uint8 [T, H, W, 3]
        frames = np.random.randint(0, 256, size=(num_frames, height, width, 3), dtype=np.uint8)
        # Random actions — int32 [T-1]
        actions = np.random.randint(0, num_actions, size=(num_frames - 1,), dtype=np.int32)

        np.savez_compressed(
            os.path.join(save_dir, f"ep_{ep:05d}.npz"),
            frames=frames,
            actions=actions,
        )

    print(f"Synthetic generation complete! {num_episodes} episodes saved.")


def main() -> None:
    """CLI entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate Procgen episodes for world model training"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="coinrun",
        choices=["coinrun", "starpilot", "climber", "ninja", "bigfish"],
        help="Procgen game to generate data from (default: coinrun)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Number of episodes to generate (default: 5000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save episodes (default: data/<game>/episodes)",
    )
    parser.add_argument(
        "--start-level",
        type=int,
        default=0,
        help="Starting level seed (default: 0)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=0,
        help="Number of unique levels, 0=unlimited (default: 0)",
    )
    parser.add_argument(
        "--distribution-mode",
        type=str,
        default="hard",
        choices=["easy", "hard"],
        help="Difficulty mode (default: hard)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data for testing (no procgen needed)",
    )
    parser.add_argument(
        "--synthetic-frames",
        type=int,
        default=50,
        help="Frames per synthetic episode (default: 50)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Frame height for synthetic data (default: 64)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Frame width for synthetic data (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.synthetic:
        save_dir = args.save_dir or "data/synthetic/episodes"
        generate_synthetic_episodes(
            save_dir=save_dir,
            num_episodes=args.episodes,
            num_frames=args.synthetic_frames,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )
    else:
        save_dir = args.save_dir or f"data/{args.game}/episodes"
        generate_procgen_episodes(
            game=args.game,
            save_dir=save_dir,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            start_level=args.start_level,
            num_levels=args.num_levels,
            distribution_mode=args.distribution_mode,
        )


if __name__ == "__main__":
    main()
