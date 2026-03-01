# Build Specification: Flow Matching Video World Model

**Read foundations_guide.md first.** This document assumes you understand VQ-VAE, flow matching, and action conditioning. If any term below is unclear, check that document.

---

## 1. Project summary

Build an action-conditioned video world model from scratch in PyTorch. Train it on Procgen game environments (which provide real action labels). The model predicts the next frame given past frames and an action, using flow matching for generation. Ship an interactive Gradio demo and thorough analysis.

**Not a reimplementation of any specific paper.** Draws ideas from VQ-VAE (van den Oord 2017), flow matching (Lipman 2023), DIAMOND's conditioning design, and GameNGen's noise augmentation, but the architecture and analysis are original.

**Hardware:** MacBook Air M4 24GB (local development) + Google Colab Pro (GPU training).

---

## 2. Architecture

### 2.1 Decision: pixel-space dynamics, not latent-space

The dynamics model operates in **pixel space** — it directly predicts the next 128×128×3 frame. The VQ-VAE is trained separately and used only for evaluation/analysis (measuring reconstruction quality, understanding what the model represents), not as part of the dynamics pipeline.

**Why pixel-space:** Latent-space dynamics (predicting next tokens) requires careful integration between the tokenizer and dynamics model — errors in tokenization propagate and compound. DIAMOND showed that pixel-space diffusion outperforms latent-token approaches for world modeling. For a from-scratch build, pixel-space is simpler and produces better visual results. The VQ-VAE is still a valuable standalone component for your portfolio and understanding.

**Implication:** The dynamics model U-Net takes 128×128×3 images directly. No tokenization in the generation loop. The VQ-VAE is a separate training phase that you can use for frame compression, analysis, or as a baseline comparison.

### 2.2 VQ-VAE tokenizer (standalone component)

**Input:** Frame $x \in \mathbb{R}^{3 \times 128 \times 128}$, normalized to $[0, 1]$.

**Encoder:**
```
Conv2d(3→64, k=4, s=2, p=1) + ReLU           → [B, 64, 64, 64]
Conv2d(64→128, k=4, s=2, p=1) + ReLU         → [B, 128, 32, 32]
Conv2d(128→256, k=4, s=2, p=1) + ReLU        → [B, 256, 16, 16]
ResBlock(256) × 2                              → [B, 256, 16, 16]
Conv2d(256→256, k=1, s=1) + L2Normalize       → [B, 256, 16, 16]
```

**Vector quantization:**
- Codebook: $K = 512$ entries, each $D = 256$ dimensional, L2-normalized
- Assignment: cosine similarity (both encoder output and codebook are on unit sphere)
- Gradient: straight-through estimator
- Codebook update: EMA with decay $\gamma = 0.99$
- Dead code reset: every 1000 steps, replace entries used $< 2$ times since last reset with random encoder outputs $+ \mathcal{N}(0, 0.01)$

**Decoder:** Mirror of encoder with ConvTranspose2d, final Sigmoid activation.

**Loss:** $\mathcal{L} = \|x - \hat{x}\|^2 + 0.25 \cdot \|z_e - \text{sg}(z_q)\|^2$

**Parameters:** ~8M total. **Training:** 50K steps, batch 64, lr $3 \times 10^{-4}$ AdamW with cosine decay.

### 2.3 Flow matching dynamics model (core component)

**Input specification:**
- $x_{t_{flow}}$: noisy interpolation between noise and target next frame, shape $[B, 3, 128, 128]$
- Context: last $H = 4$ frames channel-stacked, shape $[B, 12, 128, 128]$
- Combined U-Net input: $[B, 15, 128, 128]$ (concatenate along channel dimension)
- Flow time: $t_{flow} \in [0, 1]$, scalar per batch element
- Action: integer $\in \{0, 1, ..., 14\}$ (Procgen has 15 discrete actions)

**U-Net architecture:**

```
Input conv: Conv2d(15→64, k=3, p=1)

Down path:
  Level 1 (128×128): ResBlock(64→64, AdaGN) → ResBlock(64→64, AdaGN) → Downsample(64)
  Level 2 (64×64):   ResBlock(64→128, AdaGN) → ResBlock(128→128, AdaGN) → Downsample(128)
  Level 3 (32×32):   ResBlock(128→256, AdaGN) → ResBlock(256→256, AdaGN) → Downsample(256)
  Level 4 (16×16):   ResBlock(256→512, AdaGN) → ResBlock(512→512, AdaGN)

Middle (16×16):
  ResBlock(512, AdaGN) → SelfAttention(512) → ResBlock(512, AdaGN)

Up path (with skip connections from down path):
  Level 4 (16×16):  ResBlock(1024→512, AdaGN) → ResBlock(512→256, AdaGN) → Upsample(256)
  Level 3 (32×32):  ResBlock(512→256, AdaGN) → ResBlock(256→128, AdaGN) → Upsample(128)
  Level 2 (64×64):  ResBlock(256→128, AdaGN) → ResBlock(128→64, AdaGN) → Upsample(64)
  Level 1 (128×128): ResBlock(128→64, AdaGN) → ResBlock(64→64, AdaGN)

Output: GroupNorm(64) → SiLU → Conv2d(64→3, k=3, p=1)
```

**Downsample:** Conv2d(C→C, k=4, s=2, p=1). **Upsample:** nearest-neighbor ×2 then Conv2d(C→C, k=3, p=1).

**Self-attention only at 16×16** resolution (compute is $O(n^2)$ in spatial size — at 128×128 it would be prohibitive).

**ResBlock with AdaGN:**
```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=512):
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adagn = nn.Linear(cond_dim, out_ch * 2)  # scale + shift
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # Adaptive GroupNorm
        scale, shift = self.adagn(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)
```

**Conditioning embedding:**
```python
# Time embedding
t_emb = sinusoidal_embedding(t_flow, dim=256)  # [B, 256]
t_emb = MLP(256 → 512 → 512)(t_emb)           # [B, 512]

# Action embedding
a_emb = nn.Embedding(15, 256)(action)           # [B, 256]
a_emb = MLP(256 → 512 → 512)(a_emb)            # [B, 512]

# Combined conditioning
cond = t_emb + a_emb                             # [B, 512]
```

**Total parameters:** ~30-35M.

### 2.4 Training the dynamics model

**Flow matching training loop (the complete core logic):**
```python
def train_step(model, context_frames, target_frame, action, optimizer):
    B = target_frame.shape[0]
    
    # 1. Sample flow time uniformly
    t = torch.rand(B, 1, 1, 1, device=target_frame.device)  # [B,1,1,1]
    
    # 2. Sample noise
    noise = torch.randn_like(target_frame)                    # [B,3,128,128]
    
    # 3. Interpolate along straight path
    x_t = (1 - t) * noise + t * target_frame                 # [B,3,128,128]
    
    # 4. Target velocity = data - noise (constant along straight path)
    target_v = target_frame - noise                           # [B,3,128,128]
    
    # 5. Noise augmentation on context (GameNGen technique)
    if random.random() < 0.5:
        aug_sigma = random.uniform(0.0, 0.3)
        context_frames = context_frames + aug_sigma * torch.randn_like(context_frames)
        context_frames = context_frames.clamp(0, 1)
    
    # 6. Concatenate context with noisy sample
    model_input = torch.cat([x_t, context_frames], dim=1)    # [B,15,128,128]
    
    # 7. Predict velocity
    pred_v = model(model_input, t.squeeze(), action)          # [B,3,128,128]
    
    # 8. Loss = MSE on velocity
    loss = F.mse_loss(pred_v, target_v)
    
    # 9. Classifier-free guidance prep: 10% of the time, zero out action
    # (Already handled inside model forward with random dropout)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()
```

**Inference (generating the next frame):**
```python
@torch.no_grad()
def generate_next_frame(model, context_frames, action, num_steps=15, cfg_scale=2.0):
    x = torch.randn(1, 3, 128, 128, device=context_frames.device)
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.tensor([i * dt], device=x.device)
        model_input = torch.cat([x, context_frames], dim=1)
        
        # Conditional velocity
        v_cond = model(model_input, t, action)
        # Unconditional velocity (action = zeros)
        v_uncond = model(model_input, t, torch.zeros_like(action))
        # Classifier-free guidance
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        x = x + dt * v
    
    return x.clamp(0, 1)
```

---

## 3. Data pipeline

### 3.1 Collection (run on M4)

```python
# src/data/generate_procgen.py
import gym
import numpy as np
import os

GAMES = ["coinrun", "starpilot", "climber", "ninja", "bigfish"]
EPISODES_PER_GAME = 5000
MAX_STEPS = 500

for game in GAMES:
    env = gym.make(f"procgen:procgen-{game}-v0",
                   start_level=0, num_levels=0,
                   distribution_mode="hard",
                   render_mode="rgb_array")
    
    save_dir = f"data/{game}/episodes"
    os.makedirs(save_dir, exist_ok=True)
    
    for ep in range(EPISODES_PER_GAME):
        obs = env.reset()
        frames = [obs]     # uint8 [64, 64, 3] — Procgen native
        actions = []
        
        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            actions.append(action)
            frames.append(obs)
            if done:
                break
        
        np.savez_compressed(f"{save_dir}/ep_{ep:05d}.npz",
                           frames=np.stack(frames).astype(np.uint8),
                           actions=np.array(actions, dtype=np.int32))

    env.close()
```

**Note on resolution:** Procgen outputs 64×64 natively. For 128×128, either set `render_mode` to higher resolution or bilinear-upscale during preprocessing. Start with **64×64** to validate the pipeline faster, then move to 128×128 once everything works. The architecture handles both — just change input dimensions.

**Storage:** 5 games × 5000 episodes × ~200 frames × 64×64×3 bytes ≈ 20GB compressed.

### 3.2 Dataset class

```python
# src/data/dataset.py
class WorldModelDataset(torch.utils.data.Dataset):
    """Returns (context_frames, action, target_frame) tuples."""
    
    def __init__(self, data_dir, context_length=4, frame_skip=1):
        self.context_length = context_length
        self.frame_skip = frame_skip
        self.episodes = []  # list of (frames_array, actions_array)
        
        for path in sorted(glob(f"{data_dir}/episodes/*.npz")):
            # Filter Mac ._ metadata files (created on external SSD)
            if os.path.basename(path).startswith('._'):
                continue
            data = np.load(path)
            self.episodes.append((data['frames'], data['actions']))
        
        # Build index: (episode_idx, frame_idx) for all valid starting points
        self.index = []
        for ep_idx, (frames, actions) in enumerate(self.episodes):
            T = len(frames)
            start = context_length * frame_skip
            for t in range(start, T - 1):
                self.index.append((ep_idx, t))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        frames, actions = self.episodes[ep_idx]
        
        # Context: H frames before time t, with frame_skip spacing
        context_indices = [t - i * self.frame_skip for i in range(self.context_length, 0, -1)]
        context = np.stack([frames[i] for i in context_indices])  # [H, H, W, 3]
        
        target = frames[t + 1]                # [H, W, 3]
        action = actions[t]                    # scalar int
        
        # Normalize to [0, 1] float32, rearrange to [C, H, W]
        context = torch.from_numpy(context).float().div(255).permute(0, 3, 1, 2)  # [H, 3, h, w]
        context = context.reshape(-1, *context.shape[2:])  # [H*3, h, w] — channel stack
        target = torch.from_numpy(target).float().div(255).permute(2, 0, 1)       # [3, h, w]
        action = torch.tensor(action, dtype=torch.long)
        
        return context, action, target
```

---

## 4. Training schedule

### Phase 0: VQ-VAE (standalone, 1 Colab session)
- **Duration:** 4-6 hours on A100
- **Data:** All collected frames from all games (load as individual images, no sequences needed)
- **Steps:** 50K, batch 64, lr 3e-4 AdamW cosine decay
- **Checkpoint:** Every 2000 steps to Google Drive
- **Validation:** Reconstruction PSNR > 28 dB, codebook utilization > 80%
- **This phase is optional for the dynamics pipeline** but builds your understanding and is a portfolio piece

### Phase 1: Dynamics model (main training, 3-4 Colab sessions)
- **Duration:** 8-12 hours per session, 3-4 sessions total
- **Data:** Sequences from all 5 games
- **Steps:** 100K-150K total, batch 16 (sequences are memory-heavy), lr 2e-4 AdamW cosine decay
- **Checkpoint:** Every 2000 steps to Google Drive
- **Auto-resume:** Every training script checks for latest checkpoint and resumes

**Session plan:**
| Session | Steps | Focus |
|---------|-------|-------|
| 1 | 0-40K | Initial training on CoinRun only (validate pipeline) |
| 2 | 40K-80K | Add all 5 games, continue training |
| 3 | 80K-120K | Continue. Start generating evaluation rollouts |
| 4 | 120K-150K | Final training. Full evaluation suite. Ablations if time |

**Monitoring per session:**
- Flow matching loss (should decrease steadily, expect ~0.01-0.05 range)
- Sample generations every 5000 steps (save to Drive for inspection)
- PSNR of 1-step predictions vs ground truth (should be > 20 dB)

### Phase 2: Evaluation and demo (1-2 sessions + local work)
- Generate 1000 rollouts (20 steps each) per game
- Compute metrics (PSNR per step, action differentiation)
- Build Gradio demo
- Much of this can run on T4 or even M4 for analysis

---

## 5. VSCode + Colab workflow

### Local (VSCode on M4)
All code lives here. Write, test, commit.

```
minigenie/
├── src/
│   ├── models/
│   │   ├── vqvae.py
│   │   ├── unet.py          # flow matching U-Net
│   │   └── blocks.py        # ResBlock, AdaGN, SelfAttention, sinusoidal embed
│   ├── training/
│   │   ├── train_vqvae.py
│   │   ├── train_dynamics.py
│   │   └── checkpoint.py    # CheckpointManager (save/load/resume)
│   ├── data/
│   │   ├── generate_procgen.py
│   │   └── dataset.py
│   ├── eval/
│   │   ├── rollout.py       # autoregressive generation
│   │   ├── metrics.py       # PSNR, action differentiation score
│   │   └── visualize.py     # rollout grids, comparison plots
│   └── demo/
│       └── app.py           # Gradio interactive demo
├── notebooks/
│   ├── 01_train_vqvae.ipynb
│   ├── 02_train_dynamics.ipynb
│   └── 03_evaluate.ipynb
├── configs/
│   └── default.yaml
├── tests/
│   ├── test_shapes.py       # verify all tensor shapes match spec
│   └── test_gradient_flow.py # verify gradients reach all parameters
└── requirements.txt
```

### Testing locally before Colab (critical step)

Before spending any Colab hours, verify everything on M4:
```python
# tests/test_shapes.py
def test_unet_shapes():
    model = UNet(in_channels=15, out_channels=3, cond_dim=512)
    x = torch.randn(2, 15, 64, 64)  # use 64×64 locally for speed
    t = torch.rand(2)
    action = torch.randint(0, 15, (2,))
    out = model(x, t, action)
    assert out.shape == (2, 3, 64, 64), f"Expected (2,3,64,64), got {out.shape}"

def test_gradient_flow():
    model = UNet(in_channels=15, out_channels=3, cond_dim=512)
    x = torch.randn(2, 15, 64, 64)
    t = torch.rand(2)
    action = torch.randint(0, 15, (2,))
    out = model(x, t, action)
    loss = out.mean()
    loss.backward()
    # Check that gradients exist for action embedding
    for name, param in model.named_parameters():
        if 'action' in name and param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
```

Run these on M4 with small inputs (64×64, batch 2). Fix all bugs before touching Colab.

### Colab notebook template

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')
!pip install -q procgen wandb einops

# Sync code from Drive (or git pull from GitHub)
!cp -r /content/drive/MyDrive/minigenie/src /content/src
import sys; sys.path.insert(0, '/content')

# Cell 2: Train
from src.training.train_dynamics import train
train(
    data_dir='/content/drive/MyDrive/minigenie/data/coinrun',
    ckpt_dir='/content/drive/MyDrive/minigenie/checkpoints/dynamics',
    max_steps=40000,
    batch_size=16,
    lr=2e-4,
    save_every=2000,
    resume=True,    # auto-loads latest checkpoint from ckpt_dir
)
```

### Checkpoint manager

```python
# src/training/checkpoint.py
import torch, os, glob

class CheckpointManager:
    def __init__(self, ckpt_dir, max_keep=3):
        self.dir = ckpt_dir
        self.max_keep = max_keep
        os.makedirs(ckpt_dir, exist_ok=True)
    
    def save(self, model, optimizer, scheduler, step, extra=None):
        state = {
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'rng': torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['cuda_rng'] = torch.cuda.get_rng_state()
        if extra:
            state.update(extra)
        path = os.path.join(self.dir, f'step_{step:07d}.pt')
        torch.save(state, path)
        # Cleanup old
        ckpts = sorted(glob.glob(os.path.join(self.dir, 'step_*.pt')))
        for old in ckpts[:-self.max_keep]:
            os.remove(old)
    
    def load_latest(self):
        ckpts = sorted(glob.glob(os.path.join(self.dir, 'step_*.pt')))
        if not ckpts:
            return None
        return torch.load(ckpts[-1], map_location='cpu', weights_only=False)
    
    def resume(self, state, model, optimizer, scheduler=None):
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler and state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])
        torch.random.set_rng_state(state['rng'])
        if torch.cuda.is_available() and 'cuda_rng' in state:
            torch.cuda.set_rng_state(state['cuda_rng'])
        return state['step']
```

---

## 6. Hyperparameter reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image resolution | 64×64 (Phase 1), 128×128 (stretch goal) | Start small, scale up |
| Context frames $H$ | 4 | Matches DIAMOND |
| Frame skip | 1 | Procgen has visible changes every step |
| Actions | 15 (Procgen discrete) | Embedded to 256-dim then projected to 512 |
| U-Net channels | [64, 128, 256, 512] | ~30-35M parameters at 128×128 |
| Self-attention | 16×16 resolution only | Compute constraint |
| Flow matching steps (inference) | 15 | Euler integration |
| CFG scale | 2.0 | Tune: try 1.5, 2.0, 3.0 |
| CFG dropout | 10% (drop action) | For classifier-free guidance training |
| Noise augmentation | $\sigma \in [0, 0.3]$, prob 0.5 | On context frames during training |
| Optimizer | AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.999$ | Weight decay 0.01 |
| Learning rate | 2e-4 with cosine decay to 1e-5 | Warmup: 1000 steps linear from 0 |
| Batch size | 16 (A100 40GB) | Accumulate gradients ×2 if needed |
| Total training steps | 100K-150K | ~30-40 A100-hours |
| Mixed precision | fp16 via torch.amp | Cuts memory ~40% |
| Gradient clipping | max_norm 1.0 | Prevents instabilities |
| Checkpoint interval | 2000 steps | ~15-20 min between saves |

---

## 7. Evaluation specification

### 7.1 Single-step prediction quality
For 1000 test sequences (from unseen Procgen levels), predict one step and measure PSNR and SSIM vs ground truth. Target: PSNR > 22 dB.

### 7.2 Rollout quality degradation
Generate 50-step rollouts. Plot mean PSNR vs step number (averaged over 500 rollouts). The curve should start high and decay — characterize the decay rate. Compare: with vs without noise augmentation during training.

### 7.3 Action differentiation
For 100 starting frames, generate 1-step predictions for all 15 actions. Measure mean L2 distance between predictions with different actions. If action conditioning works, this should be significantly > 0. Compute per-action: some actions (move left vs move right) should produce large differences; others (no-op variations) may produce small differences.

### 7.4 Cross-game generalization
Train on 4 games, test on the 5th (held-out). How much does prediction quality drop? Repeat for each game as the held-out one.

### 7.5 Qualitative analysis
Save rollout videos (GIF/MP4) for cherry-picked good cases AND failure cases. Document: what does the model get right (character movement, background scrolling, enemy behavior)? What does it get wrong (physics violations, object permanence failures, texture degradation)?

---

## 8. Failure modes and fixes

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Loss doesn't decrease | Architecture bug (shapes wrong, conditioning not connected) | Run test_gradient_flow.py, verify action embedding gradients are nonzero |
| All predictions look the same regardless of action | Action conditioning not working | Check AdaGN implementation. Increase CFG dropout to 20%, increase CFG scale to 3.0 |
| Predictions are blurry | Not enough training or steps | Increase training from 100K to 200K steps. Increase inference steps from 15 to 25 |
| Rollouts degrade after 5 steps | Distribution shift | Enable noise augmentation. Increase aug probability from 0.5 to 0.8, increase σ_max from 0.3 to 0.5 |
| Out of memory on A100 | Batch too large or model too big | Reduce batch to 8 with gradient accumulation ×2. Use mixed precision. Reduce channels |
| Colab disconnects | Expected behavior | Checkpoint manager handles this. Resume from latest. Use keep-alive JS |

---

## 9. Week-by-week plan

**Week 1:** Write all code locally. Test on M4 with tiny data. Generate Procgen datasets. Train VQ-VAE on Colab (1 session). Verify reconstructions look sharp.

**Week 2:** Train dynamics model on CoinRun only (2 sessions). Generate first rollouts. Debug action conditioning. Verify the model produces different outputs for different actions.

**Week 3:** Add multi-game data. Continue training (1-2 sessions). Run evaluation metrics. Start analysis. Implement noise augmentation if rollouts degrade fast.

**Week 4:** Run full evaluation suite. Ablation: with/without noise augmentation, different CFG scales, different context lengths. Build Gradio demo. Start blog post.

**Week 5:** Polish demo. Finish blog post with training curves, rollout examples, and failure analysis. Clean up code. Upload checkpoints to HuggingFace. Record demo video.

---

## 10. Deliverables

- [ ] GitHub repo: clean PyTorch code, all components from scratch, clear README
- [ ] HuggingFace Hub: pretrained checkpoints (VQ-VAE + dynamics model)
- [ ] Gradio demo: interactive frame-by-frame generation with action buttons
- [ ] Blog post: architecture, math, training process, results, failure analysis
- [ ] Evaluation report: metrics, plots, ablations
- [ ] 2-min demo video for social media

---

## 11. How to hand this to a Claude agent

Open a new conversation and say:

> "I'm building a flow matching video world model for Procgen game environments. I have two reference documents: a foundations guide (foundations_guide.md) explaining VQ-VAE and flow matching from first principles, and a build spec (build_spec.md) with exact architecture, hyperparameters, and training procedures. I work in VSCode on MacBook Air M4 and train on Google Colab Pro with A100. I want to start with [component]. Help me implement it following the spec, with proper shape tests, and prepare the Colab notebook."

**Start with:** `src/models/blocks.py` (ResBlock, AdaGN, SelfAttention, sinusoidal embedding). These are the building blocks everything else depends on. Test shapes locally. Then build the U-Net from these blocks. Then the training loop. Then data loading. Then Colab notebook.
