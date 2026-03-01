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
