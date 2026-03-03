# Build Log

## 2026-02-28 — Project initialization (Phase 0)

### What was done
- Created full directory tree: `src/models/`, `src/training/`, `src/data/`, `src/eval/`, `src/demo/`, `tests/`, `configs/`, `logs/`, `notebooks/`, `scripts/`, `docs/`, `data/`, `outputs/`
- Added all `__init__.py` files for Python packages
- Moved `build_spec.md`, `foundations_guide.md`, `master_build_plan.md` into `docs/`
- Created `.gitignore` with `._*` filter (critical for external SSD), `data/`, `outputs/`, `.venv/` exclusions
- Created `requirements.txt` with pinned deps (procgen deferred to Phase 4)
- Created `setup.py` for editable install
- Created `README.md` with WIP skeleton, architecture overview, and "built from scratch" statement
- Created `BUILD_LOG.md` and `TRAINING_LOG.md`
- Created full hyperparameter YAML configs: `configs/vqvae.yaml`, `configs/dynamics.yaml`, `configs/eval.yaml`
- Set up virtual environment on SSD, installed all deps, verified PyTorch + einops
- Initialized git repo, verified `.gitignore` is clean, made initial commit

### Decisions
- **Public GitHub repo** from the start (not private)
- **Full YAML configs** populated now (not stubs) — every hyperparameter from the spec's §6 table
- **Procgen deferred** to Phase 4 — will handle Apple Silicon compatibility when data generation starts

### Next
- Phase 1: Implement `src/models/blocks.py` (SinusoidalEmbedding, AdaGN, ResBlock, SelfAttention)
- Write `tests/test_blocks.py` with shape, gradient, and stability tests
- All tests must pass before moving to Phase 2 (U-Net)

---

## 2026-02-28 — Phase 1: Building blocks

### What was done
- Implemented `src/models/blocks.py` with 5 components:
  - `SinusoidalEmbedding` — maps scalar t ∈ [0,1] to high-dim vector via sin/cos frequencies (precomputed inv_freq buffer)
  - `ResBlock` — residual block with AdaGN: GroupNorm→SiLU→Conv→AdaGN(cond)→SiLU→Conv + skip. The AdaGN linear projects cond_dim→2*out_ch (scale + shift), modulating norm2 output.
  - `SelfAttention` — QKV self-attention at 2D spatial positions with multi-head support. Uses `F.scaled_dot_product_attention` for efficiency. Residual connection.
  - `Downsample` — strided 4×4 conv (stride=2, pad=1) for learned 2× spatial downsampling
  - `Upsample` — nearest-neighbor 2× interpolation + 3×3 conv for artifact-free upsampling
- Wrote `tests/test_blocks.py` with 29 tests covering:
  - Shape correctness across multiple batch/channel/spatial configs
  - Gradient flow to every parameter (named_parameters loop)
  - Gradient flow through conditioning input
  - AdaGN actually affects output (different cond → different output)
  - Skip connection type correctness (Identity vs Conv2d)
  - Numerical stability (no NaN/Inf)
  - Edge cases (even dim requirement, divisibility check)
  - Downsample/Upsample roundtrip shape preservation

### Issues encountered
- **Critical SSD issue:** macOS creates `._*` Apple Double files inside `.venv/lib/site-packages/`. A `._distutils-precedence.pth` file was parsed by Python as a real `.pth` file, injecting garbage bytes → `UnicodeDecodeError` on every pip/python command. Fix: `find .venv -name '._*' -delete` after every pip install. This will recur — need to clean after every install.
- **`.gitignore` pattern `data/`** matched `src/data/` too. Fixed by using `/data/` (anchored to repo root).

### Test results
- 29/29 passed in 1.57s. All green.

### Next
- Phase 2: Implement `src/models/unet.py` — the flow matching U-Net dynamics model
- Use the blocks from Phase 1 to build encoder, middle, decoder paths with skip connections
- Test shapes, gradient flow, parameter count (~30-35M), CFG dropout

---

## 2026-03-01 — Phase 2: U-Net dynamics model

### What was done
- Implemented `src/models/unet.py` — the full flow matching U-Net:
  - 4-level encoder (Down L1-L4) with ResBlock pairs + Downsample at each level
  - Middle block: ResBlock → SelfAttention → ResBlock at 16×16 (bottleneck)
  - 4-level decoder (Up L4-L1) with skip-connection concatenation + ResBlock pairs + Upsample
  - Conditioning: SinusoidalEmbedding(256) → MLP(256→512→512) for time, Embedding(15,256) → MLP(256→512→512) for action, summed → cond [B, 512]
  - CFG dropout: during training, zero out action embedding with p=cfg_dropout (default 10%)
  - Output: GroupNorm(64) → SiLU → Conv(64→3)
- Wrote `tests/test_unet.py` with 17 tests covering:
  - Shape correctness at 64×64 and 128×128, batch 1/2/8
  - Gradient flow to every parameter (all convs, norms, embeddings)
  - Gradient flow specifically to action_embed + action_mlp (critical)
  - Gradient flow specifically to time_mlp
  - Parameter count: 41,826,883 (~42M) — spec estimated ~30-35M but the skip-connection channel doubling in the up path adds ~7M extra
  - CFG dropout: with dropout=1.0, different actions produce identical output; with dropout=0.0 or in eval mode, different actions produce different output
  - Conditioning effect: different times and actions produce different outputs
  - Numerical stability: no NaN/Inf at 64×64, 128×128, and boundary t values

### Notes
- Param count is 42M vs spec's ~30-35M estimate. The architecture matches the spec exactly — the estimate was rough. 42M is fine for A100 40GB with batch 16 at 64×64.
- Used explicit named attributes (self.down1_res1, self.up4_res2, etc.) instead of ModuleLists for clarity — makes the forward pass read like the spec diagram.

### Test results
- 46/46 passed (29 blocks + 17 unet) in 6.57s. All green.

### Next
- Phase 3: Implement `src/models/vqvae.py` — VQ-VAE tokenizer with EMA codebook, L2 normalization, dead code reset
- Write `tests/test_vqvae.py`
---

## 2026-03-01 — Phase 3: VQ-VAE tokenizer

### What was done
- Implemented `src/models/vqvae.py` — standalone VQ-VAE tokenizer (not part of dynamics pipeline):
  - `VQVAEResBlock` — simple conv-GroupNorm-ReLU-conv + skip residual block (no conditioning, unlike the U-Net ResBlock)
  - `Encoder` — 3 strided Conv2d (3→64→128→256) with GroupNorm+ReLU, 2 VQVAEResBlocks, 1×1 conv to codebook_dim, L2-normalize output → 8× spatial downsample
  - `Decoder` — mirror: 1×1 conv from codebook_dim, 2 VQVAEResBlocks, 3 ConvTranspose2d (256→128→64→3) with GroupNorm+ReLU, Sigmoid output
  - `VectorQuantizer` — 512 entries, 256-dim, all core features:
    - L2-normalized codebook stored as buffer (not parameter — updated via EMA, not gradient)
    - Cosine similarity assignment via argmax of dot product (both vectors on unit sphere)
    - Straight-through estimator: `z_q = z_e + (z_q - z_e).detach()`
    - EMA update with decay γ=0.99 and Laplace smoothing on counts
    - Dead code reset every 1000 steps: codes used < 2 times get replaced with random encoder outputs + N(0, 0.01)
  - `VQVAE` — full pipeline wrapping encoder → quantizer → decoder, with `encode()`, `decode()`, `forward()`, `compute_loss()` methods
- Wrote `tests/test_vqvae.py` with 24 tests covering:
  - Encoder: output shape at 128×128 and 64×64, L2 normalization of output
  - Decoder: output shape at 128×128 and 64×64, output range [0,1] via Sigmoid
  - VectorQuantizer: output shapes, indices in [0, K), straight-through gradient flow, codebook L2-normalization, EMA actually updates codebook in training, no update in eval, dead code reset replaces unused entries, codebook utilization (≥50% unique codes used)
  - VQVAE end-to-end: reconstruction shape, output range, losses finite, gradient flow to encoder and decoder, encode→decode roundtrip shape, compute_loss API, param count 4M-12M, no NaN/Inf

### Notes
- VQ-VAE param count: 6,103,939 (~6.1M) — well within expected range. Codebook: [512, 256].
- Used a separate `VQVAEResBlock` (simpler, no conditioning) instead of reusing blocks.py `ResBlock` — VQ-VAE is a standalone autoencoder that doesn't need AdaGN.
- Codebook is a `register_buffer`, not a `nn.Parameter`. This is correct because it's updated via EMA, not backprop. Avoids optimizer applying weight decay or momentum to it.

### Test results
- 70/70 passed (29 blocks + 17 unet + 24 vqvae) in 7.86s. All green.

### Next
- Phase 4: Data pipeline — `src/data/generate_procgen.py` and `src/data/dataset.py`
- Handle procgen Apple Silicon compatibility
- Generate CoinRun episodes, write dataset loader with ._* filtering

---

## 2026-03-02 — Phase 4: Data pipeline

### What was done
- Implemented `src/data/generate_procgen.py`:
  - Full CLI with argparse: `--game`, `--episodes`, `--max-steps`, `--save-dir`, `--synthetic`
  - Procgen generation via `gym.make("procgen:procgen-{game}-v0")` with random policy
  - Synthetic fallback mode (`--synthetic`) for local testing on Apple Silicon
  - Saves episodes as compressed `.npz` files: `frames` [T, H, W, 3] uint8, `actions` [T-1] int32
- Implemented `src/data/dataset.py` — `WorldModelDataset`:
  - Loads all `.npz` episodes into memory, builds flat index over valid (episode, timestep) pairs
  - Returns `(context, action, target)` tuples: context [H*3, h, w] float32 [0,1], action scalar int64, target [3, h, w] float32 [0,1]
  - `._*` Apple Double file filtering (Rule 6)
  - Optional `target_resolution` for bilinear upscaling (64→128)
  - `frame_skip` support, `max_episodes` cap, `validate` flag with integrity checks
  - Works with `DataLoader` multi-worker
- Generated 5,000 CoinRun episodes on Google Colab using `procgen2` package (Python 3.11 Linux wheel)
  - Original `procgen` package is dead — no wheels for Python ≥ 3.9. `procgen2` is a maintained fork on PyPI.
  - Tar'd on Drive, downloaded to Mac, extracted to SSD at `data/coinrun/episodes/`
- Wrote `tests/test_dataset.py` with 33 tests covering:
  - Synthetic generation: file creation, naming, contents, shapes, dtypes, reproducibility
  - Dataset: loading, shapes (context [12,64,64], target [3,64,64]), value ranges, dtypes, alignment
  - Temporal alignment: verified target = frames[t+1], context order is chronological, action = actions[t]
  - Apple Double filtering: ._*.npz files are correctly ignored
  - Resizing: target_resolution=(128,128) produces correct shapes
  - Frame skip: frame_skip=2 selects correct indices
  - Short episodes: barely-enough and too-short episodes handled gracefully
  - DataLoader: basic, multi-worker, full-epoch iteration with NaN/Inf checks
  - Validation: empty dir raises FileNotFoundError, bad action range raises ValueError

### Issues encountered
- **procgen won't install on Apple Silicon M4** — `pip install procgen` returns "no matching distribution". Not even `procgen-mirror` or `gymnasium[procgen]` work. Procgen only published wheels for Python ≤ 3.8 on x86. Solution: generate data on Colab using `procgen2` (has cp311 manylinux wheel), tar to Drive, download to SSD.
- **Terminal cwd corruption** — after SSD disconnect/reconnect or long idle, zsh terminals get a stale cwd (`getcwd: cannot access parent directories`). Fix: `cd /Volumes/Crucial_X9/Projects/minigenie` in a fresh terminal.

### Data verification
- 5,000 episodes at `data/coinrun/episodes/ep_00000.npz` through `ep_04999.npz`
- Frames: `[T, 64, 64, 3]` uint8 — episode lengths vary (35–501, mean ~407)
- Actions: `[T-1]` int32, range [0, 14] — roughly uniform across all 15 actions
- No NaN, no corrupted files

### Test results
- 103/103 passed (29 blocks + 17 unet + 24 vqvae + 33 dataset) in 10.46s. All green.

### Next
- Phase 5: Training infrastructure — CheckpointManager, train_vqvae.py, train_dynamics.py, smoke_test.py

---

## 2026-03-02 — Phase 5: Training infrastructure

### What was done
- Implemented `src/training/checkpoint.py` — `CheckpointManager`:
  - Saves `step_{step:07d}.pt` files with model, optimizer, scheduler, RNG states + optional extras
  - Auto-deletes old checkpoints beyond `max_keep` (default 3)
  - `load_latest()`, `load_step()`, `resume()`, `list_checkpoints()`, `latest_step` property
  - Stores CPU + CUDA RNG states for perfect reproducibility across resume

- Implemented `src/training/train_vqvae.py` — VQ-VAE training loop:
  - `FrameDataset` — loads individual frames from .npz episodes (VQ-VAE doesn't need sequences)
  - Loads config from YAML, creates model/optimizer/scheduler, auto-resumes from checkpoint
  - CosineAnnealingLR scheduler, AdamW optimizer
  - Training loop with logging (loss, codebook utilization, lr, steps/s) every 100 steps
  - Checkpoint save every 2000 steps, sample reconstruction grids every 5000 steps
  - GPU memory monitoring every 1000 steps
  - Emergency checkpoint on KeyboardInterrupt/Exception
  - Full CLI via argparse

- Implemented `src/training/train_dynamics.py` — flow matching dynamics training loop:
  - `flow_matching_step()` — core flow matching train step from build_spec §2.4:
    - Sample t ~ U[0,1], interpolate x_t = (1-t)*noise + t*target, target velocity v = target - noise
    - Noise augmentation on context frames (GameNGen technique): p=0.5, σ∈[0,0.3]
    - Concat [x_t, context] → [B,15,H,W], predict velocity, MSE loss
  - `generate_next_frame()` — Euler ODE integration with classifier-free guidance:
    - Start from noise, N steps of v_uncond + cfg_scale*(v_cond - v_uncond), clamp [0,1]
  - Mixed precision via `torch.amp.autocast` + `GradScaler` (fp16 on CUDA)
  - Gradient clipping max_norm=1.0
  - AdamW + LambdaLR with linear warmup (1000 steps) → cosine decay (lr 2e-4 → 1e-5)
  - Gradient accumulation support
  - Same logging/checkpointing/sample/emergency structure as VQ-VAE trainer
  - Full CLI via argparse

- Built `scripts/smoke_test.py`:
  - Creates random data (no real dataset needed)
  - Runs 1 VQ-VAE training step → verifies finite loss + backward
  - Runs 1 dynamics training step → verifies finite loss + backward
  - Runs 1 inference step → verifies output shape + range [0,1]
  - Checkpoint save/load round-trip → verifies weights match exactly
  - Tests max_keep cleanup
  - Prints "🎉 SMOKE TEST PASSED"

- Wrote `tests/test_training_step.py` with 21 tests:
  - VQ-VAE: compute_loss finite, backward produces gradients, optimizer step changes weights, reconstruction shapes
  - Dynamics: flow matching loss finite, backward, optimizer step, no-noise-aug mode, full-noise-aug mode
  - Inference: shape, range [0,1], no-CFG mode, single-step, deterministic with same seed
  - Checkpoint: save/load, resume restores weights exactly, max_keep deletes old, empty dir returns None
  - LR schedule: warmup increases lr, cosine decays after warmup, gradient clipping works

### Bug fixes
- `train_vqvae.py`: Fixed `compute_loss` call — was passing `(x, output)` but API takes just `(x)` and returns `(loss, metrics_dict)`. Also fixed `_save_samples` which accessed `output["x_recon"]` but forward returns a tuple.
- `train_dynamics.py`: Fixed UNet constructor kwarg `channels=` → `channel_mult=` to match actual signature.
- Smoke test VQ-VAE: embed_dim must equal `hidden_channels[-1]` (decoder's first ConvTranspose2d expects this).

### Test results
- Smoke test: PASSED ✅
- 124/124 passed (29 blocks + 17 unet + 24 vqvae + 33 dataset + 21 training) in 12.22s. All green.

### Next
- Phase 5.5: Create Colab notebooks (colab_template, 01_train_vqvae, 02_train_dynamics)
- Phase 6: Push to GitHub, upload data to Drive, start VQ-VAE training on Colab

---

## 2026-03-03 — Phase 5.5: Colab notebooks

### What was done
- Built `notebooks/colab_template.ipynb` (15 cells: 8 md, 7 code):
  - Mount Google Drive, create project dirs on Drive
  - Clone/sync repo from GitHub
  - Install dependencies (einops, pyyaml, wandb, etc.) + editable install
  - Verify GPU (assert CUDA, print GPU name/VRAM)
  - Extract data from tar.gz on Drive, filter `._*` files
  - Verify data integrity (spot-check episodes, assert ≥1000)
  - Optional smoke test cell

- Built `notebooks/01_train_vqvae.ipynb` (19 cells: 9 md, 10 code):
  - Full setup cells (Drive, clone, deps, GPU)
  - Data extraction with `._*` filtering
  - Symlink `checkpoints/vqvae/` and `samples_vqvae/` to Drive for persistence
  - Smoke test before training
  - Train cell: calls `from src.training.train_vqvae import train` with Drive paths
  - Post-training: display reconstruction samples from Drive
  - PSNR computation: loads checkpoint, runs 1024 frames, reports mean/median/min/max vs 28 dB target
  - Codebook utilization check vs 80% target
  - Drive contents summary

- Built `notebooks/02_train_dynamics.ipynb` (21 cells: 10 md, 11 code):
  - Full setup cells with GPU-aware batch size recommendation (A100 vs T4)
  - Data extraction with sequence structure verification (frames = actions + 1)
  - Symlink `checkpoints/dynamics/` and `samples_dynamics/` to Drive
  - Smoke test before training
  - Train cell: calls `from src.training.train_dynamics import train` with Drive paths
  - Post-training: display sample prediction images (target vs predicted grids)
  - 20-step autoregressive rollout generation with visualization grid
  - Single-step PSNR computation over 320 samples vs 22 dB target
  - Drive contents summary

### Design decisions
- **Symlinks to Drive** rather than writing directly — checkpoints survive Colab disconnects automatically
- **`._*` filtering** in every data glob (Rule 6)
- **Auto-resume** — notebooks always pass `resume=True`; just re-run after disconnect
- **GPU batch size guidance** in dynamics notebook (A100=16, T4=8, small=4+accum)
- **Emergency checkpoint** already handled by training scripts' try/except

### Bug found and fixed
- `scripts/write_notebooks.py` had wrong path resolution — used `os.path.dirname(__file__)` which pointed to `scripts/notebooks/` instead of `notebooks/`. Fixed to `os.path.dirname(os.path.dirname(__file__))`.
- ExFAT SSD: `create_file` tool writes content to `._*` Apple Double metadata files instead of the actual files (0-byte `.ipynb` files). Workaround: write notebooks via Python script with explicit `f.flush()` + `os.fsync(f.fileno())`.

### Validation
- All 3 notebooks valid JSON, correct nbformat 4.0
- All notebook `train()` calls match actual function signatures in train_vqvae.py and train_dynamics.py
- 124/124 tests still passing

### Phase 5 is now COMPLETE ✅

### Next
- Phase 6: Push to GitHub, upload CoinRun data to Drive, start VQ-VAE training on Colab