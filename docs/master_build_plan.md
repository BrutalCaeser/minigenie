# Master Build Plan: Flow Matching Video World Model

## Who you are

You are a senior ML engineer building a flow matching video world model from scratch. You work methodically, test everything before moving forward, and never assume code works without verification. You document as you go. You commit frequently with precise messages. When something fails, you diagnose before fixing — you don't randomly change things hoping they work.

## Project context

Read these two files before doing ANYTHING:
- `docs/foundations_guide.md` — explains VQ-VAE, flow matching, and world models from first principles
- `docs/build_spec.md` — exact architecture, hyperparameters, training procedures, and evaluation plan

These files are the source of truth. If this build plan contradicts them, the spec files win.

## Hardware and environment

- **Local machine:** MacBook Air M4, 24GB unified memory
- **Storage:** External 1TB SSD mounted at `/Volumes/[SSD_NAME]/`. The entire project lives here — code, data, virtual environment, everything. Do NOT store project files on the internal 256GB drive.
- **Mac metadata on external SSD:** macOS creates `._` prefixed files (e.g., `._blocks.py`, `._episode_00001.npz`) on non-APFS/non-HFS+ external drives. These are Apple Double files storing extended attributes. They will pollute every directory. You MUST handle this in `.gitignore` and in any file-globbing code (data loaders, scripts).
- **GPU training:** Google Colab Pro (T4 or A100, sessions up to 24 hours, will disconnect)
- **Editor:** VS Code with terminal access. Open the project folder from the SSD directly.
- **No local GPU training.** All GPU work happens on Colab. Local machine is for writing code, running CPU tests, generating data, and analysis.

---

## Critical rules you must follow at all times

### Rule 1: Test before you build the next thing
Never move to the next phase or file until the current one passes all tests. "It looks right" is not a test. Run the test. See green. Then move on.

### Rule 2: Small commits with precise messages
Commit after every meaningful unit of work. Not at the end of a day. Not after building three files. After each file, each test, each fix. The commit message format is:
```
<type>(<scope>): <description>

Types: feat, fix, test, docs, refactor, data, config
Examples:
  feat(blocks): implement ResBlock with AdaGN conditioning
  test(blocks): add shape and gradient flow tests for ResBlock
  fix(unet): correct skip connection channel mismatch at level 3
  docs(readme): add architecture diagram and training instructions
  data(procgen): add CoinRun data generation script
```

### Rule 3: Update logs after every work session
The file `logs/BUILD_LOG.md` is your running diary. After every significant action (completing a module, fixing a bug, running a training session, discovering something unexpected), append an entry with the date, what you did, what you learned, and what's next. This log is invaluable for debugging later and for the final blog post.

### Rule 4: Never write code you can't explain
If you're implementing something and you're not sure why it works, stop. Go back to the foundations guide. Understand the math first. Then implement. Comments in the code should explain WHY, not WHAT.

### Rule 5: Project lives entirely on the SSD
All code, data, virtual environment, and outputs live on the external SSD. The internal drive is not used for this project. With 1TB available, disk space is not a constraint — but keep things organized. The directory structure on the SSD should be:
```
/Volumes/[SSD_NAME]/
└── minigenie/                  # git repo root
    ├── .venv/                  # virtual environment (on SSD, not internal)
    ├── data/                   # all datasets (excluded from git)
    ├── outputs/                # generated images, rollouts (excluded from git)
    └── [all project files]
```

### Rule 6: Filter out `._` files in all code that globs files
Any code that lists directory contents (data loaders, scripts) must filter out Apple Double metadata files:
```python
# WRONG — will pick up ._episode_00001.npz alongside episode_00001.npz
paths = sorted(glob(f"{data_dir}/*.npz"))

# CORRECT — filter out Mac metadata files
paths = sorted([p for p in glob(f"{data_dir}/*.npz") if not os.path.basename(p).startswith('._')])
```
This applies to `generate_procgen.py`, `dataset.py`, and any script that reads files from directories. Forgetting this will cause silent data corruption (loading metadata files as if they were real data, getting decode errors or garbage tensors).

---

## Repository structure (create this first)

```
minigenie/
├── .gitignore
├── README.md                          # Updated continuously
├── requirements.txt                   # Pinned versions
├── setup.py                          # For local editable install
├── docs/
│   ├── foundations_guide.md           # Theory reference (provided)
│   ├── build_spec.md                 # Architecture spec (provided)
│   ├── ARCHITECTURE.md               # Diagram + component descriptions (you create)
│   └── EVALUATION.md                 # Results, plots, analysis (you create during eval)
├── logs/
│   ├── BUILD_LOG.md                  # Running development diary
│   └── TRAINING_LOG.md               # Per-session Colab training records
├── configs/
│   ├── vqvae.yaml                    # VQ-VAE hyperparameters
│   ├── dynamics.yaml                 # Flow matching model hyperparameters
│   └── eval.yaml                     # Evaluation settings
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py                 # ResBlock, AdaGN, SelfAttention, SinusoidalEmbed
│   │   ├── unet.py                   # Flow matching U-Net (dynamics model)
│   │   └── vqvae.py                  # VQ-VAE tokenizer
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_vqvae.py            # VQ-VAE training loop
│   │   ├── train_dynamics.py         # Flow matching training loop
│   │   └── checkpoint.py             # CheckpointManager
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate_procgen.py       # Data collection script
│   │   └── dataset.py                # PyTorch Dataset class
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── rollout.py                # Autoregressive generation
│   │   ├── metrics.py                # PSNR, SSIM, action differentiation
│   │   └── visualize.py              # Rollout grids, plots, GIFs
│   └── demo/
│       └── app.py                    # Gradio interactive demo
├── notebooks/
│   ├── 01_train_vqvae.ipynb
│   ├── 02_train_dynamics.ipynb
│   ├── 03_evaluate.ipynb
│   └── colab_template.ipynb          # Base template with setup cells
├── tests/
│   ├── __init__.py
│   ├── test_blocks.py
│   ├── test_unet.py
│   ├── test_vqvae.py
│   ├── test_dataset.py
│   └── test_training_step.py         # Verify one training step runs without error
└── scripts/
    └── smoke_test.py                 # Quick end-to-end test with random data
```

---

## Phase 0: Project initialization (Day 1)

### Step 0.1: Create repository and initial files

Create the directory structure above. Initialize git. Create these files:

**.gitignore:**
```
# Mac metadata (critical — SSD creates these everywhere)
._*
.DS_Store
.Spotlight-V100
.Trashes
.fseventsd

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
.venv/

# Model artifacts
*.pt
*.pth
*.ckpt

# Data (stored on SSD, not in git)
data/
outputs/

# Experiment tracking
wandb/

# Notebooks
notebooks/.ipynb_checkpoints/

# Misc
*.npz
*.tar.gz
```

**requirements.txt:**
```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
einops>=0.7.0
pyyaml>=6.0
wandb>=0.16.0
gradio>=4.0.0
matplotlib>=3.7.0
Pillow>=10.0.0
tqdm>=4.65.0
imageio>=2.31.0
scipy>=1.11.0
# procgen — install separately, may need special handling on Apple Silicon
# See Phase 4 Step 4.1 for instructions
```

**README.md** — create an initial version with:
- Project title and one-sentence description
- "Work in progress" badge
- Architecture overview (placeholder — fill in as you build)
- Setup instructions (virtual env, pip install)
- Training instructions (placeholder)
- Results (placeholder)
- "Built from scratch" statement — no pretrained models, no external frameworks beyond PyTorch

**logs/BUILD_LOG.md:**
```markdown
# Build Log

## Day 1 — Project initialization
- Created repository structure
- Set up virtual environment
- Next: implement blocks.py (ResBlock, AdaGN, SelfAttention)
```

**setup.py:**
```python
from setuptools import setup, find_packages
setup(
    name="minigenie",
    version="0.1.0",
    packages=find_packages(),
)
```

### Step 0.2: Create virtual environment on the SSD

The virtual environment lives inside the project directory on the SSD — not on the internal drive, not in a global location.

```bash
# Navigate to project root on SSD
cd /Volumes/[SSD_NAME]/minigenie

# Create virtual environment
python3 -m venv .venv

# Activate it (you must do this every time you open a new terminal)
source .venv/bin/activate

# Verify you're in the venv
which python   # should show /Volumes/[SSD_NAME]/minigenie/.venv/bin/python
which pip      # should show /Volumes/[SSD_NAME]/minigenie/.venv/bin/pip

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install project in editable mode (so 'from src.models import ...' works)
pip install -e .

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import einops; print('einops OK')"
```

**VS Code integration:** Open VS Code, press Cmd+Shift+P → "Python: Select Interpreter" → choose the one at `/Volumes/[SSD_NAME]/minigenie/.venv/bin/python`. This ensures VS Code terminals auto-activate the venv and linting/IntelliSense use the correct packages.

**Important:** If the SSD is ejected/disconnected while the venv is active, your terminal session will break. Always eject safely. If this happens, just open a new terminal and re-activate.

**Colab does NOT use this venv.** Colab has its own Python environment. Dependencies are installed via `!pip install` in notebook cells.

### Step 0.3: Initial commit

```bash
cd /Volumes/[SSD_NAME]/minigenie
git init
git add .
git commit -m "feat(init): project structure, requirements, README, build log"
```

**Push to GitHub** (create private repo first):
```bash
git remote add origin https://github.com/YOUR_USERNAME/minigenie.git
git push -u origin main
```

**Verify `.gitignore` is working:** Run `git status` — you should NOT see `.venv/`, `data/`, `._*` files, or `.DS_Store`. If you do, your `.gitignore` is wrong. Fix it before any further commits.

---

## Phase 1: Building blocks (Days 2-3)

This phase produces `src/models/blocks.py` and `tests/test_blocks.py`. Everything else depends on these.

### Step 1.1: Implement SinusoidalEmbedding

The time embedding used throughout the U-Net. Maps a scalar $t \in [0, 1]$ to a vector of dimension `dim`.

```python
# Math: embed(t, i) = sin(t / 10000^(2i/dim)) for even i, cos(...) for odd i
```

**Test:** `SinusoidalEmbedding(dim=256)(torch.tensor([0.0, 0.5, 1.0]))` should produce shape `[3, 256]`. Different `t` values should produce different embeddings. Same `t` should produce same embedding.

### Step 1.2: Implement AdaGN (Adaptive Group Normalization)

Takes features `[B, C, H, W]` and conditioning `[B, cond_dim]`. Outputs modulated features.

**Test:** Output shape matches input shape. When conditioning is all zeros, output equals standard GroupNorm output. When conditioning varies, outputs differ.

### Step 1.3: Implement ResBlock with AdaGN

Residual block with two conv layers, AdaGN after the second, and a skip connection.

**Test:** Output shape is `[B, out_ch, H, W]` regardless of whether `in_ch == out_ch`. Gradient flows through both the skip path and the main path. Gradient w.r.t. conditioning is nonzero.

### Step 1.4: Implement SelfAttention

Standard QKV self-attention applied at 2D spatial positions.

**Test:** Output shape matches input. Works at different spatial sizes. Gradient exists.

### Step 1.5: Write comprehensive tests

`tests/test_blocks.py` should verify:
- All shape correctness (multiple batch sizes, channel sizes, spatial sizes)
- Gradient flow to every parameter
- Gradient flow through conditioning
- Numerical stability (no NaN/Inf with random inputs)

Run: `python -m pytest tests/test_blocks.py -v`

ALL TESTS MUST PASS before moving to Phase 2.

### Step 1.6: Commit

```bash
git add src/models/blocks.py tests/test_blocks.py
git commit -m "feat(blocks): implement ResBlock, AdaGN, SelfAttention, SinusoidalEmbed"
git commit -m "test(blocks): add shape, gradient, and stability tests"
```

Update `logs/BUILD_LOG.md` with what you built and any issues encountered.

---

## Phase 2: U-Net dynamics model (Days 3-5)

### Step 2.1: Implement the U-Net

Build `src/models/unet.py` following the architecture in `build_spec.md` Section 2.3 exactly. Use the blocks from Phase 1.

**Implementation order within the file:**
1. Downsample module (strided conv)
2. Upsample module (nearest neighbor + conv)
3. UNet class with encoder path, middle block, decoder path with skip connections
4. Time + action embedding inside the UNet
5. Forward method

**Critical details to get right:**
- Skip connections: the decoder at each level concatenates features from the corresponding encoder level. This doubles the channel count, which the first ResBlock at each decoder level must handle.
- Conditioning: `cond_emb` (time + action) is passed to EVERY ResBlock in the network.
- Input channels: 15 = 3 (noisy target) + 12 (4 context frames × 3 channels each).
- Output channels: 3 (predicted velocity field).

### Step 2.2: Test the U-Net

`tests/test_unet.py`:
- Shape test: input `[2, 15, 64, 64]` → output `[2, 3, 64, 64]`
- Different resolutions: 64×64 and 128×128 both work
- Gradient flow: loss on output → backward → check gradients exist for time embedding, action embedding, all conv layers, all normalization layers
- Parameter count: print total params, verify ~30-35M for 128×128 config
- Memory: estimate peak memory for batch=16 at 64×64 (should fit in A100's 40GB)

### Step 2.3: Implement classifier-free guidance dropout

Inside the UNet forward method, during training, randomly zero out the action embedding with probability 10%. Store a flag `self.training` to control this.

### Step 2.4: Commit and log

```bash
git add src/models/unet.py tests/test_unet.py
git commit -m "feat(unet): implement flow matching U-Net with AdaGN action conditioning"
git commit -m "test(unet): add shape, gradient, memory, and CFG dropout tests"
```

---

## Phase 3: VQ-VAE tokenizer (Days 5-7)

### Step 3.1: Implement VQ-VAE

Build `src/models/vqvae.py` following `build_spec.md` Section 2.2.

**Key components:**
1. Encoder (conv stack with residual blocks)
2. VectorQuantizer (codebook, nearest neighbor lookup, straight-through, EMA update, dead code reset)
3. Decoder (transposed conv stack)
4. Full VQVAE module combining all three

**The VectorQuantizer is the trickiest part.** Implement carefully:
- L2 normalize both encoder outputs and codebook entries
- Use `torch.cdist` or manual distance computation for nearest neighbor
- Straight-through: `z_q = z_e + (z_q_actual - z_e).detach()`
- EMA update: maintain running count and running sum per codebook entry
- Dead code reset: track usage, replace dead entries every N steps

### Step 3.2: Test VQ-VAE

`tests/test_vqvae.py`:
- Reconstruction: pass a batch through encode → quantize → decode, verify output shape matches input
- Codebook indices: verify they're integers in range `[0, K-1]`
- Gradient flow: verify encoder gets gradients through straight-through
- Codebook utilization: after 100 random batches, how many codes are used?
- EMA update: verify codebook vectors change after a training step
- Dead code reset: set all entries to zeros except one, run reset, verify they get reinitialized

### Step 3.3: Commit

```bash
git add src/models/vqvae.py tests/test_vqvae.py
git commit -m "feat(vqvae): implement VQ-VAE with EMA codebook, L2 norm, dead code reset"
git commit -m "test(vqvae): add reconstruction, gradient, utilization, and reset tests"
```

---

## Phase 4: Data pipeline (Days 7-9)

### Step 4.1: Implement data generation

Build `src/data/generate_procgen.py` following `build_spec.md` Section 3.1. Start with CoinRun only.

**Procgen on Apple Silicon:** `pip install procgen` may fail on M4. If it does, try:
```bash
# Option 1: Install from conda-forge
conda install -c conda-forge procgen
# Option 2: Build from source
pip install procgen --no-binary procgen
# Option 3: Use gymnasium's procgen wrapper if available
pip install procgen-mirror  # community Apple Silicon fork
```
If procgen absolutely won't install on M4, generate data on a Colab CPU session instead and download to the SSD.

**Data output path:** All generated data goes to `/Volumes/[SSD_NAME]/minigenie/data/coinrun/episodes/`. With 1TB SSD, storage is not a concern.

**Run locally on M4:** Generate 5000 episodes for CoinRun (~2 hours).

**Verify the data:**
- Load 10 random episodes, display first and last frames (matplotlib)
- Check shapes: frames should be `[T, 64, 64, 3]` uint8, actions should be `[T-1]` int32
- Check action distribution: should be roughly uniform across 15 values
- Check episode lengths: should vary (some end early due to death/completion)

### Step 4.2: Implement Dataset class

Build `src/data/dataset.py` following `build_spec.md` Section 3.2.

**Test:** `tests/test_dataset.py`:
- Loads data correctly
- Returns correct shapes: context `[12, 64, 64]`, action `scalar`, target `[3, 64, 64]`
- Values are in `[0, 1]` range
- Different indices return different samples
- DataLoader works with multiple workers: `DataLoader(ds, batch_size=8, num_workers=2)`

### Step 4.3: Upload data to Google Drive for Colab access

The data lives on the SSD locally, but Colab needs it on Google Drive:
```bash
# From the SSD project directory
cd /Volumes/[SSD_NAME]/minigenie
tar czf coinrun_data.tar.gz data/coinrun/
# Upload coinrun_data.tar.gz to Google Drive (manually or via rclone)
# On Colab: !tar xzf /content/drive/MyDrive/minigenie/coinrun_data.tar.gz
```

**Note:** When tarring, the `._` metadata files will be included. On Colab (Linux), they're harmless but the dataset loader must still filter them out.

### Step 4.4: Commit

```bash
git add src/data/ tests/test_dataset.py
git commit -m "feat(data): add Procgen data generation and PyTorch Dataset"
git commit -m "data(coinrun): generate 5000 episodes for CoinRun"
```

---

## Phase 5: Training infrastructure (Days 9-11)

### Step 5.1: Implement CheckpointManager

Build `src/training/checkpoint.py` following `build_spec.md` Section 5.

### Step 5.2: Implement VQ-VAE training loop

Build `src/training/train_vqvae.py`:
- Loads config from YAML
- Creates model, optimizer, scheduler
- Resumes from checkpoint if available
- Training loop with: loss computation, backward, optimizer step, logging
- Logs every 100 steps: loss, codebook utilization, learning rate
- Saves checkpoint every 2000 steps
- Saves sample reconstructions every 5000 steps (as PNG grid)
- Uses wandb for experiment tracking (optional but recommended)

### Step 5.3: Implement dynamics training loop

Build `src/training/train_dynamics.py`:
- Same structure as VQ-VAE training
- Implements the flow matching training step from `build_spec.md` Section 2.4
- Includes noise augmentation on context frames
- Includes classifier-free guidance dropout
- Logs: loss, sample 1-step predictions (as PNG), learning rate
- Mixed precision training via `torch.amp.autocast` and `GradScaler`

### Step 5.4: Smoke test (critical checkpoint)

Build `scripts/smoke_test.py`:
```python
"""
Run ONE training step of each component with random data.
If this passes, the pipeline is mechanically correct.
Must pass before any Colab session.
"""
# 1. Create tiny random dataset (10 frames)
# 2. Create VQ-VAE, run 1 training step, verify loss is finite
# 3. Create dynamics U-Net, run 1 training step, verify loss is finite
# 4. Run 1 inference step (generate next frame), verify output shape
# 5. Save and load checkpoint, verify state matches
# 6. Print "SMOKE TEST PASSED"
```

Run: `python scripts/smoke_test.py` — this must print "SMOKE TEST PASSED" before you touch Colab.

### Step 5.5: Create Colab notebooks

Build `notebooks/colab_template.ipynb` with standard setup cells:
1. Mount Google Drive
2. Clone/sync code
3. Install dependencies
4. Verify GPU available

Build `notebooks/01_train_vqvae.ipynb`:
1. Template cells
2. Call `train_vqvae.py` with config pointing to Drive paths
3. Cell to display sample reconstructions after training

Build `notebooks/02_train_dynamics.ipynb`:
1. Template cells
2. Call `train_dynamics.py` with config pointing to Drive paths
3. Cell to generate and display sample rollouts after training

### Step 5.6: Commit

```bash
git add src/training/ scripts/ notebooks/
git commit -m "feat(training): implement VQ-VAE and dynamics training loops with checkpointing"
git commit -m "feat(infra): add smoke test and Colab notebooks"
```

---

## Phase 6: VQ-VAE training on Colab (Day 12-13)

### Before the session
- Verify smoke test passes locally
- Push latest code to GitHub
- Upload CoinRun data to Google Drive if not done
- Plan: this session should take 4-6 hours

### During the session
1. Open `01_train_vqvae.ipynb` in Colab
2. Run setup cells
3. Start training: 50K steps, batch 64
4. Monitor: loss should decrease steadily, codebook utilization should be >80% by step 10K
5. If codebook utilization is low (<50%) by step 5K: stop, increase codebook reset frequency, restart

### After the session
- Download sample reconstruction images from Drive
- Verify visual quality: reconstructions should be sharp, recognizable
- Update `logs/TRAINING_LOG.md`:
  ```markdown
  ## VQ-VAE Training Session 1
  - Date: [date]
  - GPU: A100/T4
  - Steps: 0 → 50K
  - Final loss: [value]
  - Codebook utilization: [value]%
  - PSNR: [value] dB
  - Visual quality: [sharp/blurry/notes]
  - Issues: [none/describe]
  ```
- Commit any code changes made during the session

---

## Phase 7: Dynamics model training on Colab (Days 14-22)

### Session 1: CoinRun only (8-12 hours)
- Train dynamics model on CoinRun data for 40K steps
- **Milestone check at step 5K:** loss should be decreasing. If flat or NaN, something is wrong — stop and debug.
- **Milestone check at step 20K:** generate a 1-step prediction. It should look vaguely like a CoinRun frame (not random noise). If it's pure noise, action conditioning or context concatenation is likely broken.
- Save checkpoints every 2000 steps

### Session 2: Continue + multi-game (8-12 hours)
- Resume from last checkpoint
- If CoinRun is working: add data from other Procgen games
- Train for another 40K steps (total: 80K)
- Generate 5-step rollouts and inspect visually

### Session 3: Final training + noise augmentation tuning (8-12 hours)
- Resume from last checkpoint
- If rollouts degrade fast: increase noise augmentation
- Train for another 40K-60K steps (total: 120K-140K)
- Generate 20-step rollouts for preliminary evaluation

### After each session
Update `logs/TRAINING_LOG.md` with all metrics, observations, and sample images.

---

## Phase 8: Evaluation (Days 23-27)

### Step 8.1: Implement evaluation tools

Build `src/eval/rollout.py`:
- Function that takes model + starting frames + action sequence → generates N-step rollout
- Returns list of predicted frames

Build `src/eval/metrics.py`:
- `compute_psnr(pred, target)` — per-frame PSNR in dB
- `compute_ssim(pred, target)` — structural similarity
- `action_differentiation_score(model, start_frame, context)` — generate with all actions, measure pairwise distances
- `rollout_quality_curve(model, test_episodes, steps=50)` — PSNR at each step, averaged

Build `src/eval/visualize.py`:
- `save_rollout_grid(frames, path)` — save a grid of rollout frames as PNG
- `save_rollout_gif(frames, path)` — save rollout as animated GIF
- `plot_psnr_curve(psnr_per_step, path)` — save PSNR vs step plot
- `plot_action_comparison(per_action_frames, path)` — grid showing predictions for each action

### Step 8.2: Run evaluation (Colab session or local)

1. Generate 1000 single-step predictions → compute mean PSNR
2. Generate 500 rollouts of 50 steps → plot PSNR degradation curve
3. For 100 start frames, generate with all 15 actions → compute action differentiation
4. Generate cherry-picked rollout GIFs (best and worst cases)
5. If multi-game: compute per-game metrics

### Step 8.3: Create evaluation report

Write `docs/EVALUATION.md` with:
- All quantitative metrics in tables
- PSNR degradation plots
- Action comparison grids
- Example rollout GIFs (embedded or linked)
- Failure mode documentation with specific examples
- Comparison: with vs without noise augmentation

### Step 8.4: Commit

```bash
git add src/eval/ docs/EVALUATION.md
git commit -m "feat(eval): implement metrics, rollout generation, and visualization"
git commit -m "docs(eval): add evaluation report with metrics, plots, and analysis"
```

---

## Phase 9: Demo and polish (Days 28-32)

### Step 9.1: Build Gradio demo

Build `src/demo/app.py`:
- Load pretrained dynamics model
- Display current frame
- 15 action buttons (or a subset of meaningful ones)
- On button click: generate next frame, display it, shift context
- Show last 10 frames as a filmstrip
- "Reset" button to start from a random real frame
- Display which game / level it's from

### Step 9.2: Update README

Full README with:
- Project description (what it is, why it matters)
- Architecture diagram (create with draw.io, Excalidraw, or Mermaid)
- Key results (PSNR, sample images)
- Setup instructions
- Training instructions (with Colab links)
- Demo screenshot
- Links to blog post, HuggingFace, demo
- Citation/references to papers that inspired the work

### Step 9.3: Write blog post

Create `docs/BLOG_POST.md` (or publish directly to a blog):
- Motivation: what is a world model, why build one from scratch
- Architecture walkthrough with diagrams
- Key implementation details (noise augmentation, CFG, AdaGN)
- Training process and what went well / wrong
- Results with honest failure analysis
- What you'd do differently / next steps
- Use images from your evaluation: rollout grids, PSNR plots, action comparisons

### Step 9.4: Upload to HuggingFace

- Create model cards for VQ-VAE and dynamics model
- Upload pretrained checkpoints
- If possible, deploy Gradio demo on HuggingFace Spaces

### Step 9.5: Final commits and cleanup

```bash
# Clean up any debugging code, print statements, hardcoded paths
# Verify all tests still pass
python -m pytest tests/ -v
# Final commit
git add .
git commit -m "docs(readme): complete README with architecture, results, and instructions"
git commit -m "feat(demo): add interactive Gradio demo"
git tag v1.0.0
git push origin main --tags
```

---

## Phase 10: Record and share (Day 33-35)

- Record 2-min screen capture of the demo in action
- Post on Twitter/X tagging relevant researchers/companies
- Share on Reddit (r/MachineLearning)
- Add to your resume/portfolio site

---

## What to do when things go wrong

### "Loss is NaN"
1. Check learning rate (too high?)
2. Check for division by zero in normalization layers
3. Reduce batch size to 1, run single step, print intermediate values
4. Disable mixed precision temporarily
5. Check data: are there any NaN/Inf values in the dataset?

### "Loss plateaus and won't decrease"
1. Check that the model is actually receiving gradients: `for p in model.parameters(): print(p.grad.abs().mean())`
2. Verify the target velocity computation is correct
3. Try reducing learning rate by 10x
4. Verify the data loader is shuffling properly (not feeding the same batch repeatedly)

### "Generated frames are all identical regardless of input"
1. The model has mode-collapsed. Reduce learning rate.
2. Check that context frames are actually different across batches
3. Verify the conditioning (time + action) is connected to the network (not accidentally detached)

### "Colab runs out of memory"
1. Reduce batch size
2. Enable gradient checkpointing: `torch.utils.checkpoint.checkpoint()`
3. Reduce model channels (e.g., [32, 64, 128, 256] instead of [64, 128, 256, 512])
4. Use 64×64 resolution instead of 128×128

### "I'm behind schedule"
1. Cut scope, not quality. Train on 1 game instead of 5. Generate 10-step rollouts instead of 50. Skip the VQ-VAE phase if needed — the dynamics model is the core deliverable.
2. Ship what works. A working demo with honest analysis of a simpler model beats a broken demo of a complex model.

---

## Daily checklist

Before ending each work session:
- [ ] All new code has tests
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] Changes committed with descriptive messages
- [ ] Pushed to GitHub
- [ ] `logs/BUILD_LOG.md` updated
- [ ] If training was done: `logs/TRAINING_LOG.md` updated
- [ ] README updated if any visible feature was added

---

## Success criteria

The project is DONE when:
- [ ] VQ-VAE trains and produces sharp reconstructions (PSNR > 28 dB)
- [ ] Dynamics model produces coherent 1-step predictions (PSNR > 20 dB)
- [ ] Autoregressive rollouts are visually coherent for at least 10 steps
- [ ] Different actions produce visibly different predictions
- [ ] Gradio demo is functional and interactive
- [ ] README is complete with architecture diagram and results
- [ ] Evaluation report exists with metrics and honest failure analysis
- [ ] All code is clean, tested, and committed
- [ ] Checkpoints uploaded to HuggingFace
- [ ] Blog post written

---

## Addendum: Things the user hasn't asked for but you should do anyway

### Type hints everywhere
Every function should have type hints. Not optional. This catches bugs early and makes the code self-documenting:
```python
def compute_loss(model: UNet, target: torch.Tensor, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
```

### Docstrings on every public function
Google-style docstrings with Args, Returns, and a one-line description:
```python
def generate_next_frame(model, context, action, num_steps=15):
    """Generate the next frame using flow matching ODE integration.
    
    Args:
        model: Trained UNet dynamics model
        context: Last H frames, shape [1, H*3, h, w]
        action: Action index, shape [1]
        num_steps: Number of Euler integration steps
        
    Returns:
        Predicted next frame, shape [1, 3, h, w], values in [0, 1]
    """
```

### Config validation
When loading a YAML config, validate that all required keys exist and values are in expected ranges. Don't silently use defaults — fail loudly if the config is incomplete.

### Reproducibility
Set random seeds at the start of every training run:
```python
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

### Memory monitoring on Colab
Print GPU memory usage every 1000 steps:
```python
if step % 1000 == 0:
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
```

### Graceful handling of Colab disconnections
Wrap the training loop in a try/except that saves an emergency checkpoint:
```python
try:
    for step in range(start_step, max_steps):
        # training step
except (KeyboardInterrupt, Exception) as e:
    print(f"Training interrupted: {e}")
    ckpt_manager.save(model, optimizer, scheduler, step, extra={'interrupted': True})
    raise
```

### Data integrity checks
Before training, verify:
- Dataset is not empty
- Sample shapes match expected dimensions
- Value ranges are correct (0-255 for uint8 raw, 0-1 for normalized)
- Actions are in valid range [0, num_actions-1]
- No corrupted episodes (NaN values, zero-length sequences)
