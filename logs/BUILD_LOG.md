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
