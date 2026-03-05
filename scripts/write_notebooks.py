"""Write all three Colab notebooks as valid .ipynb JSON files.
Run once, then delete this script.
"""
import json
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

COLAB_META = {
    "accelerator": "GPU",
    "colab": {"gpuType": "A100", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}


def md(source):
    """Markdown cell."""
    if isinstance(source, str):
        source = source.split("\n")
        source = [line + "\n" for line in source[:-1]] + [source[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    """Code cell."""
    if isinstance(source, str):
        source = source.split("\n")
        source = [line + "\n" for line in source[:-1]] + [source[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def write_nb(path, cells):
    nb = {"cells": cells, "metadata": COLAB_META, "nbformat": 4, "nbformat_minor": 0}
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    size = os.path.getsize(path)
    print(f"  {os.path.basename(path)}: {size:,} bytes, {len(cells)} cells")


# ============================================================
# colab_template.ipynb
# ============================================================
write_nb(os.path.join(NOTEBOOKS_DIR, "colab_template.ipynb"), [
    md("# MiniGenie \u2014 Colab Setup Template\n\nBase template for all MiniGenie Colab training sessions.  \nRun these cells at the start of **every** session to mount Drive, sync code, install deps, and verify GPU.\n\n**Do not run training in this notebook** \u2014 use the dedicated training notebooks:\n- `01_train_vqvae.ipynb` \u2014 VQ-VAE tokenizer training\n- `02_train_dynamics.ipynb` \u2014 Flow matching dynamics model training"),
    md("## 1. Mount Google Drive"),
    code("""from google.colab import drive
drive.mount('/content/drive')

# Project root on Drive (checkpoints + data persist here across sessions)
DRIVE_PROJECT = '/content/drive/MyDrive/minigenie'

import os
os.makedirs(DRIVE_PROJECT, exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT}/checkpoints/vqvae', exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT}/checkpoints/dynamics', exist_ok=True)
print(f'Drive project root: {DRIVE_PROJECT}')
print('Contents:', os.listdir(DRIVE_PROJECT))"""),
    md("## 2. Clone / Sync Code from GitHub"),
    code("""REPO_URL = 'https://github.com/BrutalCaeser/minigenie.git'  # <-- UPDATE THIS
LOCAL_CODE = '/content/minigenie'

if os.path.exists(LOCAL_CODE):
    !cd {LOCAL_CODE} && git pull --ff-only
else:
    !git clone {REPO_URL} {LOCAL_CODE}

os.chdir(LOCAL_CODE)
!git log --oneline -5
print(f'\\nWorking directory: {os.getcwd()}')"""),
    md("## 3. Install Dependencies"),
    code("""# Install project dependencies (Colab already has PyTorch + CUDA)
!pip install -q einops pyyaml wandb tqdm imageio scipy pillow matplotlib torchvision

# Install project in editable mode so 'from src.models import ...' works
!pip install -q -e .

# Verify imports
import torch
import einops
import yaml
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print('einops OK, pyyaml OK')"""),
    md("## 4. Verify GPU"),
    code("""assert torch.cuda.is_available(), 'No GPU detected! Go to Runtime > Change runtime type > GPU'

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'GPU: {gpu_name}')
print(f'VRAM: {gpu_mem:.1f} GB')

# Quick sanity check
x = torch.randn(2, 3, 64, 64, device='cuda')
print(f'Tensor on GPU: {x.shape}, dtype={x.dtype}, device={x.device}')
del x
torch.cuda.empty_cache()
print('\\nGPU verified and ready!')"""),
    md("## 5. Extract Data (if needed)\n\nUpload `coinrun_data.tar.gz` to Google Drive at `MyDrive/minigenie/` before running this."),
    code("""DATA_TAR = f'{DRIVE_PROJECT}/coinrun_data.tar.gz'
LOCAL_DATA = '/content/minigenie/data/coinrun/episodes'

if os.path.exists(LOCAL_DATA) and len(os.listdir(LOCAL_DATA)) > 100:
    n_eps = len([f for f in os.listdir(LOCAL_DATA) if f.endswith('.npz') and not f.startswith('._')])
    print(f'Data already extracted: {n_eps} episodes in {LOCAL_DATA}')
elif os.path.exists(DATA_TAR):
    print(f'Extracting {DATA_TAR} ...')
    !tar xzf {DATA_TAR} -C /content/minigenie/
    n_eps = len([f for f in os.listdir(LOCAL_DATA) if f.endswith('.npz') and not f.startswith('._')])
    print(f'Extracted {n_eps} episodes')
else:
    print(f'Data archive not found at {DATA_TAR}')
    print('Upload coinrun_data.tar.gz to Google Drive at MyDrive/minigenie/')"""),
    md("## 6. Verify Data Integrity"),
    code("""import numpy as np
import glob

data_dir = '/content/minigenie/data/coinrun/episodes'
paths = sorted([p for p in glob.glob(f'{data_dir}/*.npz') if not os.path.basename(p).startswith('._')])
print(f'Found {len(paths)} episodes')

# Spot-check a few episodes
for i in [0, len(paths)//2, -1]:
    ep = np.load(paths[i])
    frames = ep['frames']
    actions = ep['actions']
    print(f'  {os.path.basename(paths[i])}: frames {frames.shape} {frames.dtype}, '
          f'actions {actions.shape} range [{actions.min()}, {actions.max()}]')

assert len(paths) >= 1000, f'Expected >=1000 episodes, got {len(paths)}'
print('\\nData integrity check passed!')"""),
    md("---\n\n## Setup Complete\n\nEverything is ready. Switch to the appropriate training notebook:\n- **VQ-VAE training:** `01_train_vqvae.ipynb`\n- **Dynamics training:** `02_train_dynamics.ipynb`\n\nOr continue below to run the smoke test as a final verification."),
    code("# Optional: run smoke test to verify everything works end-to-end\n!cd /content/minigenie && python scripts/smoke_test.py"),
])

# ============================================================
# 01_train_vqvae.ipynb
# ============================================================
write_nb(os.path.join(NOTEBOOKS_DIR, "01_train_vqvae.ipynb"), [
    md("# MiniGenie \u2014 VQ-VAE Training\n\nTrain the VQ-VAE tokenizer on CoinRun episode frames.  \n**Spec:** `docs/build_spec.md` \u00a72.2 \u2014 512\u00d7256 codebook, EMA updates, dead code reset.  \n**Config:** `configs/vqvae.yaml` \u2014 50K steps, batch 64, lr 3e-4, cosine schedule.\n\n### Targets\n- Reconstruction PSNR > 28 dB\n- Codebook utilization > 80%\n\n### Expected runtime\n- T4: ~4\u20136 hours for 50K steps\n- A100: ~1\u20132 hours for 50K steps"),
    md("---\n## 1. Setup (mount Drive, clone code, install deps, verify GPU)"),
    code("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_PROJECT = '/content/drive/MyDrive/minigenie'
os.makedirs(f'{DRIVE_PROJECT}/checkpoints/vqvae', exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT}/samples_vqvae', exist_ok=True)
print(f'Drive project root: {DRIVE_PROJECT}')"""),
    code("""REPO_URL = 'https://github.com/BrutalCaeser/minigenie.git'  # <-- UPDATE THIS
LOCAL_CODE = '/content/minigenie'

if os.path.exists(LOCAL_CODE):
    !cd {LOCAL_CODE} && git pull --ff-only
else:
    !git clone {REPO_URL} {LOCAL_CODE}

os.chdir(LOCAL_CODE)
!git log --oneline -3"""),
    code("""!pip install -q einops pyyaml wandb tqdm imageio scipy pillow matplotlib torchvision
!pip install -q -e .

import torch
assert torch.cuda.is_available(), 'No GPU! Go to Runtime > Change runtime type > GPU'
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')"""),
    md("---\n## 2. Extract & Verify Data"),
    code("""import glob
import numpy as np

DATA_TAR = f'{DRIVE_PROJECT}/coinrun_data.tar.gz'
LOCAL_DATA = '/content/minigenie/data/coinrun/episodes'

if os.path.exists(LOCAL_DATA) and len(os.listdir(LOCAL_DATA)) > 100:
    print('Data already extracted.')
elif os.path.exists(DATA_TAR):
    print('Extracting data...')
    !tar xzf {DATA_TAR} -C /content/minigenie/
else:
    raise FileNotFoundError(
        f'Data archive not found at {DATA_TAR}\\n'
        'Upload coinrun_data.tar.gz to Google Drive at MyDrive/minigenie/'
    )

# Filter out Apple Double ._* files
paths = sorted([p for p in glob.glob(f'{LOCAL_DATA}/*.npz')
                if not os.path.basename(p).startswith('._')])
print(f'Episodes: {len(paths)}')

# Spot check
ep = np.load(paths[0])
print(f'Sample: frames {ep["frames"].shape} {ep["frames"].dtype}, '
      f'actions {ep["actions"].shape} range [{ep["actions"].min()}, {ep["actions"].max()}]')
assert len(paths) >= 1000, f'Expected >=1000 episodes, got {len(paths)}'
print('Data OK')"""),
    md("---\n## 3. Symlink Checkpoints to Google Drive\n\nCheckpoints are saved to Drive so they persist across Colab disconnections.  \nThe training script writes to `checkpoints/vqvae/` \u2014 we symlink this to Drive."),
    code("""import shutil

# Symlink local checkpoint dir -> Drive (persists across disconnects)
LOCAL_CKPT = '/content/minigenie/checkpoints/vqvae'
DRIVE_CKPT = f'{DRIVE_PROJECT}/checkpoints/vqvae'

os.makedirs(os.path.dirname(LOCAL_CKPT), exist_ok=True)
if os.path.islink(LOCAL_CKPT):
    os.remove(LOCAL_CKPT)
elif os.path.isdir(LOCAL_CKPT):
    shutil.rmtree(LOCAL_CKPT)
os.symlink(DRIVE_CKPT, LOCAL_CKPT)

# Also symlink samples dir
LOCAL_SAMPLES = '/content/minigenie/samples_vqvae'
DRIVE_SAMPLES = f'{DRIVE_PROJECT}/samples_vqvae'
os.makedirs(DRIVE_SAMPLES, exist_ok=True)
if os.path.islink(LOCAL_SAMPLES):
    os.remove(LOCAL_SAMPLES)
elif os.path.isdir(LOCAL_SAMPLES):
    shutil.rmtree(LOCAL_SAMPLES)
os.symlink(DRIVE_SAMPLES, LOCAL_SAMPLES)

existing = sorted(glob.glob(f'{DRIVE_CKPT}/step_*.pt'))
if existing:
    print(f'Found {len(existing)} existing checkpoint(s): {[os.path.basename(p) for p in existing]}')
    print('Training will auto-resume from the latest.')
else:
    print('No existing checkpoints. Training will start from scratch.')
print(f'\\nCheckpoints -> {DRIVE_CKPT}')"""),
    md("---\n## 4. Run Smoke Test\n\nQuick end-to-end verification with random data before committing to a long training run."),
    code("!cd /content/minigenie && python scripts/smoke_test.py"),
    md("---\n## 5. Train VQ-VAE\n\nThis calls the full training loop from `src/training/train_vqvae.py`.  \nTraining resumes automatically from the latest checkpoint on Drive.\n\n**Monitor:**\n- Loss should decrease steadily\n- Codebook utilization should reach >80% by step 10K\n- If utilization <50% at step 5K: stop, increase reset frequency, restart\n\n**Override defaults** by passing keyword arguments to `train()` below."),
    code("""from src.training.train_vqvae import train

train(
    data_dir='/content/minigenie/data/coinrun/episodes',
    ckpt_dir='/content/minigenie/checkpoints/vqvae',
    config_path='/content/minigenie/configs/vqvae.yaml',
    # --- Override any config values here ---
    # max_steps=50000,
    # batch_size=64,
    # lr=3e-4,
    resume=True,
    device='cuda',
)"""),
    md("---\n## 6. Inspect Results\n\nAfter training completes (or after interruption), inspect reconstructions and metrics."),
    code("""import matplotlib.pyplot as plt
from PIL import Image

# Display saved reconstruction samples
samples_dir = f'{DRIVE_PROJECT}/samples_vqvae'
sample_files = sorted(glob.glob(f'{samples_dir}/step_*.png'))

if sample_files:
    print(f'Found {len(sample_files)} sample images')
    # Show the last 3
    for path in sample_files[-3:]:
        img = Image.open(path)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(img)
        ax.set_title(os.path.basename(path))
        ax.axis('off')
        plt.tight_layout()
        plt.show()
else:
    print('No sample images found yet. Train for at least sample_every steps.')"""),
    md("### Compute PSNR on a Batch"),
    code("""import torch
import numpy as np
from src.models.vqvae import VQVAE
from src.training.checkpoint import CheckpointManager
from src.training.train_vqvae import FrameDataset
from torch.utils.data import DataLoader

# Load latest checkpoint
ckpt_mgr = CheckpointManager('/content/minigenie/checkpoints/vqvae')
state = ckpt_mgr.load_latest()

if state is not None:
    import yaml
    with open('/content/minigenie/configs/vqvae.yaml') as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg['model']

    model = VQVAE(
        in_channels=mcfg.get('in_channels', 3),
        hidden_channels=mcfg.get('hidden_channels', [64, 128, 256]),
        codebook_size=mcfg.get('codebook_size', 512),
        embed_dim=mcfg.get('embed_dim', 256),
        num_res_blocks=mcfg.get('num_res_blocks', 2),
        ema_decay=mcfg.get('ema_decay', 0.99),
        commitment_cost=mcfg.get('commitment_cost', 0.25),
    ).cuda()
    model.load_state_dict(state['model'])
    model.eval()
    step = state['step']
    print(f'Loaded checkpoint at step {step}')

    # Compute PSNR over ~1000 frames
    dataset = FrameDataset('/content/minigenie/data/coinrun/episodes', resolution=64)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    psnr_values = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 16:  # 16 x 64 = 1024 frames
                break
            x = batch.cuda()
            x_hat, _, _, _ = model(x)
            # Per-image MSE -> PSNR
            mse = ((x - x_hat) ** 2).mean(dim=(1, 2, 3))
            psnr = -10 * torch.log10(mse + 1e-8)
            psnr_values.extend(psnr.cpu().tolist())

    psnr_arr = np.array(psnr_values)
    print(f'\\nPSNR over {len(psnr_arr)} frames:')
    print(f'   Mean:   {psnr_arr.mean():.2f} dB')
    print(f'   Median: {np.median(psnr_arr):.2f} dB')
    print(f'   Min:    {psnr_arr.min():.2f} dB')
    print(f'   Max:    {psnr_arr.max():.2f} dB')
    target_psnr = cfg.get('targets', {}).get('psnr_db', 28.0)
    status = 'PASS' if psnr_arr.mean() >= target_psnr else 'BELOW TARGET'
    print(f'   Target: {target_psnr} dB [{status}]')

    # Codebook utilization
    all_indices = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 16:
                break
            z_e = model.encoder(batch.cuda())
            _, indices, _ = model.quantizer(z_e)
            all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices).flatten()
    unique = all_indices.unique().numel()
    total = model.quantizer.codebook.shape[0]
    util = 100.0 * unique / total
    target_util = cfg.get('targets', {}).get('codebook_utilization_pct', 80.0)
    status = 'PASS' if util >= target_util else 'BELOW TARGET'
    print(f'\\nCodebook utilization: {unique}/{total} = {util:.1f}% (target: {target_util}%) [{status}]')
else:
    print('No checkpoint found. Train the model first.')"""),
    md("---\n## 7. Generate Reconstructions from Checkpoint\n\nLoad the trained VQ-VAE and display side-by-side original vs reconstructed frames with per-image PSNR.  \nThis works even if sample images were not saved during training."),
    code("""import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.vqvae import VQVAE
from src.training.checkpoint import CheckpointManager
from src.training.train_vqvae import FrameDataset
from torch.utils.data import DataLoader
import yaml

# Load config
with open('/content/minigenie/configs/vqvae.yaml') as f:
    cfg = yaml.safe_load(f)
mcfg = cfg['model']

# Build model
model = VQVAE(
    in_channels=mcfg.get('in_channels', 3),
    hidden_channels=mcfg.get('hidden_channels', [64, 128, 256]),
    codebook_size=mcfg.get('codebook_size', 512),
    embed_dim=mcfg.get('embed_dim', 256),
    num_res_blocks=mcfg.get('num_res_blocks', 2),
    ema_decay=mcfg.get('ema_decay', 0.99),
    commitment_cost=mcfg.get('commitment_cost', 0.25),
).cuda()

# Load checkpoint
ckpt_mgr = CheckpointManager('/content/minigenie/checkpoints/vqvae')
state = ckpt_mgr.load_latest()
assert state is not None, 'No checkpoint found — train VQ-VAE first!'
model.load_state_dict(state['model'])
model.eval()
print(f"Loaded checkpoint at step {state['step']}")

# Grab a random batch of 8 frames
dataset = FrameDataset('/content/minigenie/data/coinrun/episodes', resolution=64)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
originals = next(iter(loader)).cuda()

with torch.no_grad():
    recons, _, _, _ = model(originals)

# Compute per-image PSNR
mse_per = ((originals - recons) ** 2).mean(dim=(1, 2, 3))
psnr_per = -10 * torch.log10(mse_per + 1e-8)

# Plot side-by-side
fig, axes = plt.subplots(2, 8, figsize=(20, 5))
for i in range(8):
    orig_np = originals[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    rec_np = recons[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[0, i].imshow(orig_np)
    axes[0, i].set_title('Original', fontsize=9)
    axes[0, i].axis('off')
    axes[1, i].imshow(rec_np)
    axes[1, i].set_title(f'PSNR {psnr_per[i]:.1f} dB', fontsize=9)
    axes[1, i].axis('off')

fig.suptitle(f"VQ-VAE Reconstructions  (step {state['step']}, mean PSNR {psnr_per.mean():.1f} dB)", fontsize=13)
plt.tight_layout()
plt.show()
print(f"\\nMean PSNR: {psnr_per.mean():.2f} dB | Min: {psnr_per.min():.2f} dB | Max: {psnr_per.max():.2f} dB")"""),
    md("---\n## 8. Post-Session Checklist\n\nAfter training completes or before Colab disconnects:\n\n1. Checkpoints are already on Drive (symlinked)\n2. Sample reconstructions are already on Drive (symlinked)\n3. Update `logs/TRAINING_LOG.md` with session results\n4. Record: final loss, PSNR, codebook utilization, GPU type, steps completed\n5. Push any code changes to GitHub"),
    code("""# Summary of what's on Drive
print('=== Drive Contents ===')
for subdir in ['checkpoints/vqvae', 'samples_vqvae']:
    full = f'{DRIVE_PROJECT}/{subdir}'
    if os.path.exists(full):
        files = os.listdir(full)
        print(f'  {subdir}/: {len(files)} files')
        for f in sorted(files)[-5:]:
            size_mb = os.path.getsize(os.path.join(full, f)) / 1e6
            print(f'    {f} ({size_mb:.1f} MB)')
    else:
        print(f'  {subdir}/: (not found)')"""),
])

# ============================================================
# 02_train_dynamics.ipynb
# ============================================================
write_nb(os.path.join(NOTEBOOKS_DIR, "02_train_dynamics.ipynb"), [
    md("# MiniGenie \u2014 Flow Matching Dynamics Training\n\nTrain the flow matching U-Net dynamics model on CoinRun episodes.  \n**Spec:** `docs/build_spec.md` \u00a72.3\u20132.4 \u2014 U-Net with AdaGN, flow matching ODE, CFG.  \n**Config:** `configs/dynamics.yaml` \u2014 150K steps, batch 16, lr 2e-4\u21921e-5, fp16, noise aug.\n\n### Prerequisites\n- VQ-VAE must be trained first (for future latent-space training \u2014 currently training in pixel space)\n- CoinRun data extracted on Drive\n\n### Targets\n- Single-step PSNR > 22 dB\n- Flow matching loss in [0.01, 0.05] range\n\n### Expected runtime\n- T4: ~18\u201324 hours for 150K steps (multiple sessions)\n- A100: ~6\u201310 hours for 150K steps"),
    md("---\n## 1. Setup"),
    code("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_PROJECT = '/content/drive/MyDrive/minigenie'
os.makedirs(f'{DRIVE_PROJECT}/checkpoints/dynamics', exist_ok=True)
os.makedirs(f'{DRIVE_PROJECT}/samples_dynamics', exist_ok=True)
print(f'Drive project root: {DRIVE_PROJECT}')"""),
    code("""REPO_URL = 'https://github.com/BrutalCaeser/minigenie.git'  # <-- UPDATE THIS
LOCAL_CODE = '/content/minigenie'

if os.path.exists(LOCAL_CODE):
    !cd {LOCAL_CODE} && git pull --ff-only
else:
    !git clone {REPO_URL} {LOCAL_CODE}

os.chdir(LOCAL_CODE)
!git log --oneline -3"""),
    code("""!pip install -q einops pyyaml wandb tqdm imageio scipy pillow matplotlib torchvision
!pip install -q -e .

import torch
assert torch.cuda.is_available(), 'No GPU! Go to Runtime > Change runtime type > GPU'
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')

# Recommend batch size based on GPU
if gpu_mem >= 35:
    print('A100 detected -- default batch_size=16 should be fine.')
elif gpu_mem >= 14:
    print('T4 detected -- may need batch_size=8 or gradient_accumulation_steps=2.')
else:
    print(f'Small GPU ({gpu_mem:.0f} GB). Use batch_size=4 + gradient accumulation.')"""),
    md("---\n## 2. Extract & Verify Data"),
    code("""import glob
import numpy as np

DATA_TAR = f'{DRIVE_PROJECT}/coinrun_data.tar.gz'
LOCAL_DATA = '/content/minigenie/data/coinrun/episodes'

if os.path.exists(LOCAL_DATA) and len(os.listdir(LOCAL_DATA)) > 100:
    print('Data already extracted.')
elif os.path.exists(DATA_TAR):
    print('Extracting data...')
    !tar xzf {DATA_TAR} -C /content/minigenie/
else:
    raise FileNotFoundError(
        f'Data archive not found at {DATA_TAR}\\n'
        'Upload coinrun_data.tar.gz to Google Drive at MyDrive/minigenie/'
    )

paths = sorted([p for p in glob.glob(f'{LOCAL_DATA}/*.npz')
                if not os.path.basename(p).startswith('._')])
print(f'Episodes: {len(paths)}')

# Spot check -- verify sequence structure for dynamics training
ep = np.load(paths[0])
frames, actions = ep['frames'], ep['actions']
print(f'Sample episode: {frames.shape[0]} frames, {actions.shape[0]} actions')
print(f'  frames: {frames.shape} {frames.dtype}')
print(f'  actions: {actions.shape} range [{actions.min()}, {actions.max()}]')
assert frames.shape[0] == actions.shape[0] + 1, 'Frame/action count mismatch'
assert len(paths) >= 1000
print('Data OK')"""),
    md("---\n## 3. Symlink Checkpoints & Samples to Drive"),
    code("""import shutil

symlinks = {
    '/content/minigenie/checkpoints/dynamics': f'{DRIVE_PROJECT}/checkpoints/dynamics',
    '/content/minigenie/samples_dynamics': f'{DRIVE_PROJECT}/samples_dynamics',
}

for local, remote in symlinks.items():
    os.makedirs(os.path.dirname(local), exist_ok=True)
    os.makedirs(remote, exist_ok=True)
    if os.path.islink(local):
        os.remove(local)
    elif os.path.isdir(local):
        shutil.rmtree(local)
    os.symlink(remote, local)
    print(f'  {local} -> {remote}')

# Check for existing checkpoints (auto-resume)
DRIVE_CKPT = f'{DRIVE_PROJECT}/checkpoints/dynamics'
existing = sorted(glob.glob(f'{DRIVE_CKPT}/step_*.pt'))
if existing:
    print(f'\\nFound {len(existing)} checkpoint(s): {[os.path.basename(p) for p in existing]}')
    print('Training will auto-resume from the latest.')
else:
    print('\\nNo existing checkpoints. Training will start from scratch.')
print('\\nSymlinks ready')"""),
    md("---\n## 4. Smoke Test"),
    code("!cd /content/minigenie && python scripts/smoke_test.py"),
    md("---\n## 5. Train Dynamics Model\n\nCalls the full training loop from `src/training/train_dynamics.py`.  \nResumes automatically from the latest checkpoint on Drive.\n\n**Monitor:**\n- Loss should decrease steadily \u2192 expect [0.01, 0.05] range by convergence\n- **Step 5K checkpoint:** loss should be clearly decreasing. If flat or NaN \u2192 stop & debug.\n- **Step 20K checkpoint:** 1-step predictions should look vaguely like CoinRun frames (not noise).\n- Sample predictions saved every 5K steps to Drive.\n\n**If session disconnects:** Just re-run this notebook \u2014 training resumes from the last checkpoint.\n\n**Override defaults** by passing keyword arguments below."),
    code("""from src.training.train_dynamics import train

train(
    data_dir='/content/minigenie/data/coinrun/episodes',
    ckpt_dir='/content/minigenie/checkpoints/dynamics',
    config_path='/content/minigenie/configs/dynamics.yaml',
    # --- Override any config values here ---
    # max_steps=150000,
    # batch_size=16,      # Reduce to 8 on T4, use grad accum x2
    # lr=2e-4,
    resume=True,
    device='cuda',
)"""),
    md("---\n## 6. Inspect Sample Predictions\n\nEach sample image shows:\n- **Top row:** Ground truth target frames\n- **Bottom row:** Model's 1-step predictions"),
    code("""import matplotlib.pyplot as plt
from PIL import Image

samples_dir = f'{DRIVE_PROJECT}/samples_dynamics'
sample_files = sorted(glob.glob(f'{samples_dir}/step_*.png'))

if sample_files:
    print(f'Found {len(sample_files)} sample images')
    for path in sample_files[-3:]:
        img = Image.open(path)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(img)
        ax.set_title(os.path.basename(path))
        ax.axis('off')
        plt.tight_layout()
        plt.show()
else:
    print('No sample images yet. Train for at least sample_every steps.')"""),
    md("---\n## 7. Generate Multi-Step Rollout\n\nAfter significant training, generate a multi-step autoregressive rollout to see temporal coherence."),
    code("""import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.models.unet import UNet
from src.training.checkpoint import CheckpointManager
from src.training.train_dynamics import generate_next_frame
from src.data.dataset import WorldModelDataset

# Load config
with open('/content/minigenie/configs/dynamics.yaml') as f:
    cfg = yaml.safe_load(f)
mcfg = cfg['model']
fcfg = cfg['flow']

# Load model
ckpt_mgr = CheckpointManager('/content/minigenie/checkpoints/dynamics')
state = ckpt_mgr.load_latest()

if state is not None:
    model = UNet(
        in_channels=mcfg.get('in_channels', 15),
        out_channels=mcfg.get('out_channels', 3),
        channel_mult=mcfg.get('channel_mult', [64, 128, 256, 512]),
        cond_dim=mcfg.get('cond_dim', 512),
        num_actions=mcfg.get('num_actions', 15),
        num_groups=mcfg.get('num_groups', 32),
        cfg_dropout=0.0,  # No dropout at inference
    ).cuda()
    model.load_state_dict(state['model'])
    model.eval()
    step = state['step']
    print(f'Loaded checkpoint at step {step}')

    # Load a real episode for starting context
    dataset = WorldModelDataset(
        '/content/minigenie/data/coinrun/episodes',
        context_length=mcfg.get('context_frames', 4),
    )
    context, action, target = dataset[0]
    context = context.unsqueeze(0).cuda()  # [1, 12, H, W]

    # Autoregressive rollout
    ROLLOUT_STEPS = 20
    NUM_INFERENCE_STEPS = fcfg.get('num_inference_steps', 15)
    CFG_SCALE = fcfg.get('cfg_scale', 2.0)
    H = mcfg.get('context_frames', 4)

    # Random actions for the rollout
    actions = torch.randint(0, mcfg.get('num_actions', 15), (ROLLOUT_STEPS,)).cuda()

    # Collect context frames (first H frames from the context tensor)
    frames = []
    for i in range(H):
        frames.append(context[0, i*3:(i+1)*3].cpu())  # [3, h, w]

    # Generate frames autoregressively
    print(f'Generating {ROLLOUT_STEPS}-step rollout...')
    with torch.no_grad():
        for t_step in range(ROLLOUT_STEPS):
            # Build context from last H frames
            ctx = torch.cat(frames[-H:], dim=0).unsqueeze(0).cuda()  # [1, H*3, h, w]
            act = actions[t_step:t_step+1]  # [1]
            pred = generate_next_frame(
                model, ctx, act,
                num_steps=NUM_INFERENCE_STEPS,
                cfg_scale=CFG_SCALE,
            )
            frames.append(pred[0].cpu())  # [3, h, w]

    # Display as grid
    total = len(frames)
    cols = min(10, total)
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    if rows == 1:
        axes = [axes]
    for i, frame in enumerate(frames):
        r, c = i // cols, i % cols
        ax = axes[r][c] if rows > 1 else axes[c]
        img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        label = f'ctx {i}' if i < H else f'pred {i-H}'
        ax.set_title(label, fontsize=8)
        ax.axis('off')
    # Hide empty axes
    for i in range(total, rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r][c] if rows > 1 else axes[c]
        ax.axis('off')
    plt.suptitle(f'Autoregressive Rollout (step {step}, {ROLLOUT_STEPS} predicted frames)', fontsize=12)
    plt.tight_layout()
    plt.show()
else:
    print('No checkpoint found. Train the model first.')"""),
    md("---\n## 8. Compute Single-Step PSNR"),
    code("""# Re-use model from above (or reload if needed)
if state is not None:
    from torch.utils.data import DataLoader

    dataset = WorldModelDataset(
        '/content/minigenie/data/coinrun/episodes',
        context_length=mcfg.get('context_frames', 4),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    NUM_INFERENCE_STEPS = fcfg.get('num_inference_steps', 15)
    CFG_SCALE = fcfg.get('cfg_scale', 2.0)

    psnr_values = []
    with torch.no_grad():
        for i, (ctx, act, tgt) in enumerate(loader):
            if i >= 20:  # 20 x 16 = 320 samples
                break
            ctx, act, tgt = ctx.cuda(), act.cuda(), tgt.cuda()
            pred = generate_next_frame(model, ctx, act, NUM_INFERENCE_STEPS, CFG_SCALE)
            mse = ((tgt - pred) ** 2).mean(dim=(1, 2, 3))
            psnr = -10 * torch.log10(mse + 1e-8)
            psnr_values.extend(psnr.cpu().tolist())
            if (i + 1) % 5 == 0:
                print(f'  Batch {i+1}/20 done...')

    psnr_arr = np.array(psnr_values)
    print(f'\\nSingle-step PSNR over {len(psnr_arr)} predictions:')
    print(f'   Mean:   {psnr_arr.mean():.2f} dB')
    print(f'   Median: {np.median(psnr_arr):.2f} dB')
    print(f'   Min:    {psnr_arr.min():.2f} dB')
    print(f'   Max:    {psnr_arr.max():.2f} dB')
    target_psnr = cfg.get('targets', {}).get('single_step_psnr_db', 22.0)
    status = 'PASS' if psnr_arr.mean() >= target_psnr else 'BELOW TARGET'
    print(f'   Target: {target_psnr} dB [{status}]')
else:
    print('No model loaded.')"""),
    md("---\n## 9. Post-Session Checklist\n\nAfter training completes or before Colab disconnects:\n\n1. Checkpoints are on Drive (symlinked)\n2. Sample predictions are on Drive (symlinked)\n3. Update `logs/TRAINING_LOG.md` with:\n   - Date, GPU type, steps completed (from -> to)\n   - Final loss value\n   - Single-step PSNR\n   - Rollout quality (sharp/blurry/collapsed)\n   - Any issues or observations\n4. Push code changes to GitHub"),
    code("""# Summary of what's on Drive
print('=== Drive Contents ===')
for subdir in ['checkpoints/dynamics', 'samples_dynamics']:
    full = f'{DRIVE_PROJECT}/{subdir}'
    if os.path.exists(full):
        files = os.listdir(full)
        print(f'  {subdir}/: {len(files)} files')
        for f in sorted(files)[-5:]:
            size_mb = os.path.getsize(os.path.join(full, f)) / 1e6
            print(f'    {f} ({size_mb:.1f} MB)')
    else:
        print(f'  {subdir}/: (not found)')"""),
])

# ============================================================
# 03_evaluate.ipynb
# ============================================================
write_nb(os.path.join(NOTEBOOKS_DIR, "03_evaluate.ipynb"), [
    md("# MiniGenie \u2014 Full Evaluation Suite\n\nRun the complete evaluation pipeline on the trained dynamics model (CoinRun).\n\n**Spec:** `docs/build_spec.md` \u00a77 \u2014 Single-step PSNR, rollout degradation, action differentiation, qualitative analysis.\n\n**Config:** `configs/eval.yaml`\n\n### What this notebook produces\n1. **Single-step PSNR & SSIM** \u2014 1000 samples, target >22 dB\n2. **Rollout degradation curve** \u2014 PSNR vs step over 200 rollouts of 50 steps\n3. **Action differentiation score** \u2014 do different actions produce different frames?\n4. **Cherry-picked rollout GIFs** \u2014 best and worst cases\n5. **Action comparison grids** \u2014 same context, all 15 actions\n\nAll outputs are saved to Google Drive under `outputs/eval/`."),
    md("---\n## 1. Setup"),
    code("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_PROJECT = '/content/drive/MyDrive/minigenie'
OUTPUT_DIR = f'{DRIVE_PROJECT}/outputs/eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Drive project: {DRIVE_PROJECT}')
print(f'Eval outputs: {OUTPUT_DIR}')"""),
    code("""REPO_URL = 'https://github.com/BrutalCaeser/minigenie.git'
LOCAL_CODE = '/content/minigenie'

if os.path.exists(LOCAL_CODE):
    !cd {LOCAL_CODE} && git pull --ff-only
else:
    !git clone {REPO_URL} {LOCAL_CODE}

os.chdir(LOCAL_CODE)
!git log --oneline -3"""),
    code("""!pip install -q einops pyyaml tqdm imageio scipy pillow matplotlib torchvision
!pip install -q -e .

import torch
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"""),
    md("---\n## 2. Extract Data & Load Model"),
    code("""import glob
import numpy as np
import yaml

# Extract data if needed
DATA_TAR = f'{DRIVE_PROJECT}/coinrun_data.tar.gz'
LOCAL_DATA = '/content/minigenie/data/coinrun/episodes'

if os.path.exists(LOCAL_DATA) and len(os.listdir(LOCAL_DATA)) > 100:
    print('Data already extracted.')
elif os.path.exists(DATA_TAR):
    print('Extracting data...')
    !tar xzf {DATA_TAR} -C /content/minigenie/
else:
    raise FileNotFoundError(f'Data archive not found at {DATA_TAR}')

paths = sorted([p for p in glob.glob(f'{LOCAL_DATA}/*.npz')
                if not os.path.basename(p).startswith('._')])
print(f'Episodes: {len(paths)}')"""),
    code("""# Symlink checkpoints from Drive
import shutil

ckpt_local = '/content/minigenie/checkpoints/dynamics'
ckpt_drive = f'{DRIVE_PROJECT}/checkpoints/dynamics'

os.makedirs(os.path.dirname(ckpt_local), exist_ok=True)
if os.path.islink(ckpt_local):
    os.remove(ckpt_local)
elif os.path.isdir(ckpt_local):
    shutil.rmtree(ckpt_local)
os.symlink(ckpt_drive, ckpt_local)
print(f'Checkpoints: {ckpt_local} -> {ckpt_drive}')
print(f'Available: {sorted(os.listdir(ckpt_drive))}')"""),
    code("""from src.models.unet import UNet
from src.training.checkpoint import CheckpointManager
from src.data.dataset import WorldModelDataset

# Load config
with open('/content/minigenie/configs/dynamics.yaml') as f:
    dyn_cfg = yaml.safe_load(f)
with open('/content/minigenie/configs/eval.yaml') as f:
    eval_cfg = yaml.safe_load(f)

mcfg = dyn_cfg['model']
fcfg = dyn_cfg['flow']
NUM_STEPS = eval_cfg['inference']['num_steps']
CFG_SCALE = eval_cfg['inference']['cfg_scale']

# Build & load model
model = UNet(
    in_channels=mcfg.get('in_channels', 15),
    out_channels=mcfg.get('out_channels', 3),
    channel_mult=mcfg.get('channel_mult', [64, 128, 256, 512]),
    cond_dim=mcfg.get('cond_dim', 512),
    num_actions=mcfg.get('num_actions', 15),
    num_groups=mcfg.get('num_groups', 32),
    cfg_dropout=0.0,  # No dropout at inference
).cuda()

ckpt_mgr = CheckpointManager('/content/minigenie/checkpoints/dynamics')
state = ckpt_mgr.load_latest()
assert state is not None, 'No checkpoint found!'
model.load_state_dict(state['model'])
model.eval()
step = state['step']
print(f'Loaded dynamics model at step {step}')
print(f'Inference: {NUM_STEPS} Euler steps, CFG scale {CFG_SCALE}')

# Load dataset
dataset = WorldModelDataset(
    '/content/minigenie/data/coinrun/episodes',
    context_length=mcfg.get('context_frames', 4),
)
print(f'Dataset: {len(dataset)} samples from {dataset.num_episodes} episodes')"""),
    md("---\n## 3. Single-Step Prediction Quality (\u00a77.1)\n\nGenerate 1-step predictions for 1000 samples. Target: PSNR > 22 dB."),
    code("""from src.eval.metrics import evaluate_single_step

single_step_results = evaluate_single_step(
    model, dataset,
    num_samples=eval_cfg['single_step']['num_test_sequences'],
    batch_size=eval_cfg['single_step'].get('batch_size', 16),
    num_inference_steps=NUM_STEPS,
    cfg_scale=CFG_SCALE,
    device='cuda',
)

target_psnr = eval_cfg['single_step']['target_psnr_db']
status = 'PASS' if single_step_results['psnr_mean'] >= target_psnr else 'BELOW TARGET'

print(f\"\"\"
=== Single-Step Prediction Quality ===
  PSNR Mean:   {single_step_results['psnr_mean']:.2f} dB  (target: {target_psnr} dB) [{status}]
  PSNR Median: {single_step_results['psnr_median']:.2f} dB
  PSNR Std:    {single_step_results['psnr_std']:.2f} dB
  PSNR Range:  [{single_step_results['psnr_min']:.2f}, {single_step_results['psnr_max']:.2f}] dB
  SSIM Mean:   {single_step_results['ssim_mean']:.4f}
  SSIM Std:    {single_step_results['ssim_std']:.4f}
  Samples:     {single_step_results['num_samples']}
\"\"\")"""),
    md("---\n## 4. Rollout Quality Degradation (\u00a77.2)\n\nGenerate multi-step rollouts and plot PSNR at each step.\nThis shows how fast errors accumulate autoregressively."),
    code("""from src.eval.metrics import evaluate_rollout_degradation
from src.eval.visualize import plot_psnr_curve

rollout_results = evaluate_rollout_degradation(
    model, dataset,
    num_rollouts=eval_cfg['rollout']['num_rollouts'],
    max_steps=eval_cfg['rollout']['max_steps'],
    num_inference_steps=NUM_STEPS,
    cfg_scale=CFG_SCALE,
    device='cuda',
)

print(f\"\"\"
=== Rollout Quality Degradation ===
  Rollouts: {rollout_results['num_rollouts_actual']}
  Steps:    {eval_cfg['rollout']['max_steps']}
  Step  1 PSNR: {rollout_results['psnr_per_step'][0]:.2f} dB
  Step 10 PSNR: {rollout_results['psnr_per_step'][9]:.2f} dB
  Step 25 PSNR: {rollout_results['psnr_per_step'][24]:.2f} dB
  Step 50 PSNR: {rollout_results['psnr_per_step'][49]:.2f} dB
\"\"\")

# Save PSNR degradation curve
plot_psnr_curve(
    rollout_results['psnr_per_step'],
    f'{OUTPUT_DIR}/psnr_degradation_curve.png',
    psnr_std_per_step=rollout_results['psnr_std_per_step'],
    ssim_per_step=rollout_results['ssim_per_step'],
    target_psnr=target_psnr,
    title=f'Rollout Quality Degradation (step {step}, {rollout_results[\"num_rollouts_actual\"]} rollouts)',
)"""),
    code("""# Also display the curve inline
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open(f'{OUTPUT_DIR}/psnr_degradation_curve.png')
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.imshow(img)
ax.axis('off')
plt.tight_layout()
plt.show()"""),
    md("---\n## 5. Action Differentiation (\u00a77.3)\n\nFor 100 starting contexts, generate 1-step predictions with all 15 actions.\nMeasure pairwise L2 distance \u2014 if conditioning works, different actions should produce different outputs."),
    code("""from src.eval.metrics import evaluate_action_differentiation

action_results = evaluate_action_differentiation(
    model, dataset,
    num_start_frames=eval_cfg['action_differentiation']['num_start_frames'],
    num_actions=eval_cfg['action_differentiation']['num_actions'],
    num_inference_steps=NUM_STEPS,
    cfg_scale=CFG_SCALE,
    device='cuda',
)

print(f\"\"\"
=== Action Differentiation ===
  Mean L2 distance:  {action_results['mean_l2_distance']:.4f}
  Std L2 distance:   {action_results['std_l2_distance']:.4f}
  Range:             [{action_results['min_l2_distance']:.4f}, {action_results['max_l2_distance']:.4f}]
  Start frames:      {action_results['num_start_frames']}
\"\"\")

# Per-action mean distances
print('Per-action mean L2 distance from other actions:')
for a, d in enumerate(action_results['per_action_mean_dist']):
    bar = '#' * int(d * 200)
    print(f'  Action {a:2d}: {d:.4f} {bar}')"""),
    md("---\n## 6. Action Comparison Grids (\u00a77.3 visual)\n\nVisualize predictions for all 15 actions from the same starting context."),
    code("""from src.eval.visualize import plot_action_comparison
from src.training.train_dynamics import generate_next_frame

# Pick 3 different starting contexts
import random
random.seed(42)
sample_indices = random.sample(range(len(dataset)), min(3, len(dataset)))

for s_idx, data_idx in enumerate(sample_indices):
    context, _, _ = dataset[data_idx]
    context_gpu = context.unsqueeze(0).cuda()

    # Generate for all 15 actions
    predictions = {}
    with torch.no_grad():
        for a in range(15):
            act = torch.tensor([a], dtype=torch.long, device='cuda')
            pred = generate_next_frame(model, context_gpu, act, NUM_STEPS, CFG_SCALE)
            predictions[a] = pred[0].cpu()

    # Last context frame for display
    H = mcfg.get('context_frames', 4)
    last_ctx = context[(H-1)*3:H*3]  # [3, h, w]

    path = f'{OUTPUT_DIR}/action_comparison_{s_idx}.png'
    plot_action_comparison(
        predictions, path,
        context_frame=last_ctx,
        title=f'Action Comparison (sample {s_idx})',
    )

    # Display inline
    img = Image.open(path)
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()"""),
    md("---\n## 7. Cherry-Picked Rollouts (\u00a77.5)\n\nGenerate rollouts, rank by mean PSNR, save the best and worst as GIFs and grids."),
    code("""from src.eval.rollout import generate_rollout_with_gt
from src.eval.visualize import save_rollout_grid, save_rollout_gif
from src.eval.metrics import compute_psnr

ROLLOUT_STEPS = eval_cfg['qualitative'].get('rollout_steps', 20)
NUM_CHERRY = eval_cfg['qualitative']['num_cherry_picked']
NUM_FAIL = eval_cfg['qualitative']['num_failure_cases']
H = dataset.context_length

# Collect valid start points
valid_starts = []
for ep_idx, (frames, actions) in enumerate(dataset.episodes):
    T = len(frames)
    min_t = H * dataset.frame_skip
    max_t = min(T - 1, len(actions)) - ROLLOUT_STEPS
    for t in range(min_t, max(min_t, max_t + 1)):
        valid_starts.append((ep_idx, t))

# Sample a manageable number to rank
rng = np.random.RandomState(42)
num_candidates = min(100, len(valid_starts))
chosen = rng.choice(len(valid_starts), size=num_candidates, replace=False)

rollout_scores = []
rollout_data = []

print(f'Generating {num_candidates} rollouts of {ROLLOUT_STEPS} steps to rank...')
for i, idx in enumerate(chosen):
    ep_idx, start_t = valid_starts[idx]
    pred, gt, acts = generate_rollout_with_gt(
        model, dataset, ep_idx, start_t, ROLLOUT_STEPS,
        num_inference_steps=NUM_STEPS, cfg_scale=CFG_SCALE, device='cuda',
    )
    # Mean PSNR across all steps
    psnr_vals = [compute_psnr(p, g).item() for p, g in zip(pred, gt)]
    mean_psnr = np.mean(psnr_vals)
    rollout_scores.append(mean_psnr)
    rollout_data.append((pred, gt, acts, ep_idx, start_t))
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{num_candidates} done (last mean PSNR: {mean_psnr:.1f} dB)')

# Rank
ranked = np.argsort(rollout_scores)
best_indices = ranked[-NUM_CHERRY:][::-1]
worst_indices = ranked[:NUM_FAIL]

print(f'\\nBest {NUM_CHERRY} rollouts (mean PSNR):')
for i in best_indices:
    print(f'  ep={rollout_data[i][3]} t={rollout_data[i][4]}: {rollout_scores[i]:.2f} dB')

print(f'\\nWorst {NUM_FAIL} rollouts (mean PSNR):')
for i in worst_indices:
    print(f'  ep={rollout_data[i][3]} t={rollout_data[i][4]}: {rollout_scores[i]:.2f} dB')"""),
    code("""# Save best rollouts as GIFs and grids
os.makedirs(f'{OUTPUT_DIR}/best_rollouts', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/worst_rollouts', exist_ok=True)

def save_rollout_outputs(indices, label, subdir):
    for rank, i in enumerate(indices):
        pred, gt, acts, ep_idx, start_t = rollout_data[i]
        score = rollout_scores[i]

        # Build context frames for display
        frames_np, actions_np = dataset.episodes[ep_idx]
        ctx_list = []
        for ci in range(H):
            t_idx = start_t - (H - ci) * dataset.frame_skip
            f = torch.from_numpy(frames_np[t_idx].copy()).float().div(255).permute(2, 0, 1)
            ctx_list.append(f)

        # Grid: GT row + Predicted row
        save_rollout_grid(
            pred, f'{OUTPUT_DIR}/{subdir}/rollout_{rank:02d}_grid.png',
            context_frames=ctx_list, gt_frames=gt,
            title=f'{label} #{rank} (PSNR {score:.1f} dB, ep={ep_idx} t={start_t})',
        )

        # GIF of predictions
        save_rollout_gif(
            pred, f'{OUTPUT_DIR}/{subdir}/rollout_{rank:02d}.gif',
            fps=8, context_frames=ctx_list,
        )

save_rollout_outputs(best_indices, 'Best', 'best_rollouts')
save_rollout_outputs(worst_indices, 'Worst', 'worst_rollouts')

print('\\nAll rollout outputs saved to Drive.')"""),
    code("""# Display a few inline
for rank, i in enumerate(best_indices[:3]):
    pred, gt, _, ep_idx, start_t = rollout_data[i]
    score = rollout_scores[i]

    fig, axes = plt.subplots(2, min(10, ROLLOUT_STEPS), figsize=(20, 4))
    cols = min(10, ROLLOUT_STEPS)
    for j in range(cols):
        axes[0, j].imshow(gt[j].permute(1,2,0).clamp(0,1).numpy())
        axes[0, j].set_title(f't={j}', fontsize=7)
        axes[0, j].axis('off')
        axes[1, j].imshow(pred[j].permute(1,2,0).clamp(0,1).numpy())
        axes[1, j].axis('off')
    axes[0, 0].set_ylabel('GT', fontsize=9)
    axes[1, 0].set_ylabel('Pred', fontsize=9)
    fig.suptitle(f'Best #{rank} — PSNR {score:.1f} dB (ep={ep_idx}, t={start_t})', fontsize=11)
    plt.tight_layout()
    plt.show()

for rank, i in enumerate(worst_indices[:3]):
    pred, gt, _, ep_idx, start_t = rollout_data[i]
    score = rollout_scores[i]

    fig, axes = plt.subplots(2, min(10, ROLLOUT_STEPS), figsize=(20, 4))
    cols = min(10, ROLLOUT_STEPS)
    for j in range(cols):
        axes[0, j].imshow(gt[j].permute(1,2,0).clamp(0,1).numpy())
        axes[0, j].set_title(f't={j}', fontsize=7)
        axes[0, j].axis('off')
        axes[1, j].imshow(pred[j].permute(1,2,0).clamp(0,1).numpy())
        axes[1, j].axis('off')
    axes[0, 0].set_ylabel('GT', fontsize=9)
    axes[1, 0].set_ylabel('Pred', fontsize=9)
    fig.suptitle(f'Worst #{rank} — PSNR {score:.1f} dB (ep={ep_idx}, t={start_t})', fontsize=11)
    plt.tight_layout()
    plt.show()"""),
    md("---\n## 8. Summary & Export\n\nCollect all metrics into a summary for the evaluation report."),
    code("""import json

summary = {
    'model_step': step,
    'game': 'coinrun',
    'inference_steps': NUM_STEPS,
    'cfg_scale': CFG_SCALE,
    'single_step': single_step_results,
    'rollout': {
        'num_rollouts': rollout_results['num_rollouts_actual'],
        'max_steps': eval_cfg['rollout']['max_steps'],
        'psnr_step_1': float(rollout_results['psnr_per_step'][0]),
        'psnr_step_10': float(rollout_results['psnr_per_step'][9]),
        'psnr_step_25': float(rollout_results['psnr_per_step'][24]),
        'psnr_step_50': float(rollout_results['psnr_per_step'][49]),
    },
    'action_differentiation': {
        'mean_l2': action_results['mean_l2_distance'],
        'std_l2': action_results['std_l2_distance'],
    },
}

# Save to Drive
json_path = f'{OUTPUT_DIR}/eval_results.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Results saved to {json_path}')

# Pretty print
print('\\n' + '='*50)
print('  EVALUATION SUMMARY')
print('='*50)
print(f\"\"\"
  Model:           dynamics @ step {step}
  Game:            CoinRun

  Single-step PSNR:  {single_step_results['psnr_mean']:.2f} dB (target: {target_psnr} dB) [{'PASS' if single_step_results['psnr_mean'] >= target_psnr else 'BELOW'}]
  Single-step SSIM:  {single_step_results['ssim_mean']:.4f}

  Rollout PSNR:
    Step  1: {rollout_results['psnr_per_step'][0]:.2f} dB
    Step 10: {rollout_results['psnr_per_step'][9]:.2f} dB
    Step 25: {rollout_results['psnr_per_step'][24]:.2f} dB
    Step 50: {rollout_results['psnr_per_step'][49]:.2f} dB

  Action differentiation:
    Mean L2 distance: {action_results['mean_l2_distance']:.4f}

  Outputs saved to: {OUTPUT_DIR}/
\"\"\")"""),
    md("---\n## 9. Post-Evaluation Checklist\n\n1. \u2705 All outputs saved to `outputs/eval/` on Drive\n2. Copy `eval_results.json` into `docs/EVALUATION.md`\n3. Update `logs/BUILD_LOG.md` with evaluation results\n4. Push code to GitHub\n5. Proceed to Phase 9 (demo)"),
    code("""# List all output files
print('=== Evaluation Outputs on Drive ===')
for root, dirs, files in os.walk(OUTPUT_DIR):
    level = root.replace(OUTPUT_DIR, '').count(os.sep)
    indent = '  ' * level
    subdir = os.path.basename(root)
    print(f'{indent}{subdir}/')
    sub_indent = '  ' * (level + 1)
    for f in sorted(files):
        size_kb = os.path.getsize(os.path.join(root, f)) / 1024
        print(f'{sub_indent}{f} ({size_kb:.1f} KB)')"""),
])

print("\nAll notebooks written successfully.")
