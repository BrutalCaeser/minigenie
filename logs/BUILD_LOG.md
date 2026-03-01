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