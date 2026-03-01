# MiniGenie 🧞

> Action-conditioned video world model for Procgen game environments, built entirely from scratch in PyTorch.

![Status: Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Built from Scratch](https://img.shields.io/badge/built-from%20scratch-blue)

## What is this?

MiniGenie is a flow matching video world model that predicts the next frame of a game given past frames and a player action. Given a sequence of 4 context frames and an action (e.g., "move right"), it generates what the next frame should look like — learning the physics, dynamics, and visual style of the game purely from observation.

**Built entirely from scratch.** No pretrained models, no diffusion libraries, no external frameworks beyond PyTorch. Every component — VQ-VAE tokenizer, flow matching U-Net, training loops, evaluation — is implemented from first principles.

## Architecture

```
Context frames (4×3 channels) ──┐
                                 ├─ concat ─→ U-Net ─→ predicted velocity ─→ ODE integrate ─→ next frame
Noisy target (3 channels) ──────┘              ↑
                                          action + time
                                        (AdaGN conditioning)
```

- **Dynamics model:** Pixel-space flow matching U-Net (~30-35M params)
- **Conditioning:** Adaptive Group Normalization (AdaGN) injects action + flow time into every ResBlock
- **Generation:** 15-step Euler ODE integration with classifier-free guidance
- **VQ-VAE:** Standalone tokenizer with 512-entry codebook, EMA updates, L2 normalization (for analysis/evaluation)

*Full architecture details: [docs/build_spec.md](docs/build_spec.md)*
*Mathematical foundations: [docs/foundations_guide.md](docs/foundations_guide.md)*

## Games

Trained on 5 Procgen environments:
- 🪙 CoinRun — platformer with coin collection
- 🚀 StarPilot — scrolling shooter
- 🧗 Climber — vertical platformer
- 🥷 Ninja — stealth platformer
- 🐟 BigFish — eat-or-be-eaten

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/minigenie.git
cd minigenie

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/ -v
```

## Training

*Training instructions will be added after training infrastructure is built.*

## Results

*Results will be added after training and evaluation are complete.*

## Project Structure

```
src/
├── models/
│   ├── blocks.py        # ResBlock, AdaGN, SelfAttention, SinusoidalEmbed
│   ├── unet.py          # Flow matching U-Net (dynamics model)
│   └── vqvae.py         # VQ-VAE tokenizer
├── training/
│   ├── train_vqvae.py   # VQ-VAE training loop
│   ├── train_dynamics.py # Flow matching training loop
│   └── checkpoint.py    # CheckpointManager (save/load/resume)
├── data/
│   ├── generate_procgen.py # Data collection
│   └── dataset.py       # PyTorch Dataset
├── eval/
│   ├── rollout.py       # Autoregressive generation
│   ├── metrics.py       # PSNR, SSIM, action differentiation
│   └── visualize.py     # Rollout grids, plots, GIFs
└── demo/
    └── app.py           # Gradio interactive demo
```

## References

Draws ideas from:
- VQ-VAE (van den Oord et al., 2017)
- Flow Matching (Lipman et al., 2023)
- DIAMOND (Alonso et al., 2024) — conditioning design
- GameNGen (Valevski et al., 2024) — noise augmentation

The architecture and analysis are original. Not a reimplementation of any specific paper.

## License

MIT
