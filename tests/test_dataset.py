"""
Tests for the data pipeline: generate_procgen.py and dataset.py.

Uses synthetic data generated in a temporary directory to test:
- Synthetic data generation creates valid .npz files
- Dataset loads data correctly and returns proper shapes
- Values are in expected ranges
- Different indices return different samples
- Context frames and targets are properly aligned
- ._* file filtering works
- DataLoader with multiple workers functions correctly
- Target resolution resizing works
- Edge cases: short episodes, frame_skip > 1
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import WorldModelDataset
from src.data.generate_procgen import generate_synthetic_episodes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data_dir():
    """Create a temporary directory with synthetic episodes for testing.
    
    Generates 20 episodes with 50 frames each at 64×64 resolution.
    Cleaned up after all tests in this module complete.
    """
    tmpdir = tempfile.mkdtemp(prefix="minigenie_test_")
    episodes_dir = os.path.join(tmpdir, "episodes")
    generate_synthetic_episodes(
        save_dir=episodes_dir,
        num_episodes=20,
        num_frames=50,
        height=64,
        width=64,
        num_actions=15,
        seed=42,
    )
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="module")
def short_episode_dir():
    """Create episodes that are very short (barely enough for context)."""
    tmpdir = tempfile.mkdtemp(prefix="minigenie_test_short_")
    episodes_dir = os.path.join(tmpdir, "episodes")
    generate_synthetic_episodes(
        save_dir=episodes_dir,
        num_episodes=5,
        num_frames=8,  # Just barely enough for context_length=4 + 1 target
        height=64,
        width=64,
        num_actions=15,
        seed=123,
    )
    yield tmpdir
    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# TestSyntheticGeneration
# ---------------------------------------------------------------------------

class TestSyntheticGeneration:
    """Tests for generate_synthetic_episodes."""

    def test_creates_npz_files(self, synthetic_data_dir):
        episodes_dir = os.path.join(synthetic_data_dir, "episodes")
        npz_files = sorted([f for f in os.listdir(episodes_dir) if f.endswith(".npz")])
        assert len(npz_files) == 20

    def test_file_naming(self, synthetic_data_dir):
        episodes_dir = os.path.join(synthetic_data_dir, "episodes")
        for i in range(20):
            expected = f"ep_{i:05d}.npz"
            assert os.path.exists(os.path.join(episodes_dir, expected)), \
                f"Missing expected file: {expected}"

    def test_npz_contents(self, synthetic_data_dir):
        episodes_dir = os.path.join(synthetic_data_dir, "episodes")
        data = np.load(os.path.join(episodes_dir, "ep_00000.npz"))
        assert "frames" in data, "Missing 'frames' key"
        assert "actions" in data, "Missing 'actions' key"

    def test_frames_shape_and_dtype(self, synthetic_data_dir):
        episodes_dir = os.path.join(synthetic_data_dir, "episodes")
        data = np.load(os.path.join(episodes_dir, "ep_00000.npz"))
        assert data["frames"].shape == (50, 64, 64, 3)
        assert data["frames"].dtype == np.uint8

    def test_actions_shape_and_dtype(self, synthetic_data_dir):
        episodes_dir = os.path.join(synthetic_data_dir, "episodes")
        data = np.load(os.path.join(episodes_dir, "ep_00000.npz"))
        assert data["actions"].shape == (49,)  # T-1 actions for T frames
        assert data["actions"].dtype == np.int32

    def test_actions_in_range(self, synthetic_data_dir):
        episodes_dir = os.path.join(synthetic_data_dir, "episodes")
        data = np.load(os.path.join(episodes_dir, "ep_00000.npz"))
        assert np.all(data["actions"] >= 0)
        assert np.all(data["actions"] < 15)

    def test_reproducible_with_seed(self):
        """Same seed produces identical data."""
        tmpdir1 = tempfile.mkdtemp()
        tmpdir2 = tempfile.mkdtemp()
        try:
            generate_synthetic_episodes(save_dir=tmpdir1, num_episodes=3, seed=99)
            generate_synthetic_episodes(save_dir=tmpdir2, num_episodes=3, seed=99)
            for i in range(3):
                d1 = np.load(os.path.join(tmpdir1, f"ep_{i:05d}.npz"))
                d2 = np.load(os.path.join(tmpdir2, f"ep_{i:05d}.npz"))
                np.testing.assert_array_equal(d1["frames"], d2["frames"])
                np.testing.assert_array_equal(d1["actions"], d2["actions"])
        finally:
            shutil.rmtree(tmpdir1)
            shutil.rmtree(tmpdir2)


# ---------------------------------------------------------------------------
# TestWorldModelDataset — basic functionality
# ---------------------------------------------------------------------------

class TestWorldModelDataset:
    """Core tests for the WorldModelDataset class."""

    def test_loads_all_episodes(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir)
        assert ds.num_episodes == 20

    def test_max_episodes(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir, max_episodes=5)
        assert ds.num_episodes == 5

    def test_len_positive(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir)
        assert len(ds) > 0

    def test_sample_shapes_64(self, synthetic_data_dir):
        """Context [12, 64, 64], action scalar, target [3, 64, 64]."""
        ds = WorldModelDataset(synthetic_data_dir, context_length=4)
        context, action, target = ds[0]
        assert context.shape == (12, 64, 64), f"Got {context.shape}"
        assert action.shape == (), f"Got action shape {action.shape}"
        assert target.shape == (3, 64, 64), f"Got {target.shape}"

    def test_context_channels(self, synthetic_data_dir):
        """Context should have H*3 channels for H context frames."""
        for H in [1, 2, 4, 8]:
            ds = WorldModelDataset(synthetic_data_dir, context_length=H)
            if len(ds) > 0:
                context, _, _ = ds[0]
                assert context.shape[0] == H * 3

    def test_value_range_context(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir)
        context, _, _ = ds[0]
        assert context.min() >= 0.0
        assert context.max() <= 1.0

    def test_value_range_target(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir)
        _, _, target = ds[0]
        assert target.min() >= 0.0
        assert target.max() <= 1.0

    def test_action_dtype(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir)
        _, action, _ = ds[0]
        assert action.dtype == torch.long

    def test_action_range(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir)
        for i in range(min(100, len(ds))):
            _, action, _ = ds[i]
            assert 0 <= action.item() < 15

    def test_different_indices_different_samples(self, synthetic_data_dir):
        """Different indices should generally produce different samples."""
        ds = WorldModelDataset(synthetic_data_dir)
        c1, a1, t1 = ds[0]
        c2, a2, t2 = ds[len(ds) // 2]
        # At least one of the tensors should differ
        different = (
            not torch.equal(c1, c2)
            or not torch.equal(t1, t2)
            or a1.item() != a2.item()
        )
        assert different, "Samples at different indices are identical"

    def test_tensor_dtypes(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir)
        context, action, target = ds[0]
        assert context.dtype == torch.float32
        assert action.dtype == torch.int64
        assert target.dtype == torch.float32


# ---------------------------------------------------------------------------
# TestDatasetAlignment
# ---------------------------------------------------------------------------

class TestDatasetAlignment:
    """Tests that context, action, and target are correctly aligned in time."""

    def test_target_is_next_frame(self, synthetic_data_dir):
        """Verify target = frames[t+1] by loading raw data and comparing."""
        ds = WorldModelDataset(synthetic_data_dir, context_length=4)
        # Get first sample's episode and timestep from the index
        ep_idx, t = ds.index[0]
        frames, actions = ds.episodes[ep_idx]

        _, _, target = ds[0]

        # Manually compute expected target
        expected = torch.from_numpy(frames[t + 1].copy()).float().div(255.0).permute(2, 0, 1)
        torch.testing.assert_close(target, expected)

    def test_context_frames_order(self, synthetic_data_dir):
        """Verify context frames are in chronological order."""
        ds = WorldModelDataset(synthetic_data_dir, context_length=4, frame_skip=1)
        ep_idx, t = ds.index[0]
        frames, _ = ds.episodes[ep_idx]

        context, _, _ = ds[0]

        # Context should be frames[t-4], frames[t-3], frames[t-2], frames[t-1]
        for h in range(4):
            frame_idx = t - (4 - h)
            expected = torch.from_numpy(frames[frame_idx].copy()).float().div(255.0).permute(2, 0, 1)
            actual = context[h * 3 : (h + 1) * 3]
            torch.testing.assert_close(actual, expected)

    def test_action_matches_timestep(self, synthetic_data_dir):
        """Verify returned action is actions[t]."""
        ds = WorldModelDataset(synthetic_data_dir, context_length=4)
        ep_idx, t = ds.index[0]
        _, actions = ds.episodes[ep_idx]

        _, action, _ = ds[0]
        assert action.item() == actions[t]


# ---------------------------------------------------------------------------
# TestAppleDoubleFiltering
# ---------------------------------------------------------------------------

class TestAppleDoubleFiltering:
    """Tests that macOS ._* Apple Double metadata files are filtered out."""

    def test_filters_dotunderscore_files(self):
        """Create ._*.npz files alongside real ones and verify they're ignored."""
        tmpdir = tempfile.mkdtemp(prefix="minigenie_test_apple_")
        episodes_dir = os.path.join(tmpdir, "episodes")
        os.makedirs(episodes_dir)
        try:
            # Create 3 real episodes
            for i in range(3):
                frames = np.random.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8)
                actions = np.random.randint(0, 15, (19,), dtype=np.int32)
                np.savez_compressed(
                    os.path.join(episodes_dir, f"ep_{i:05d}.npz"),
                    frames=frames, actions=actions,
                )

            # Create 3 ._* Apple Double metadata files (garbage data)
            for i in range(3):
                # Write some garbage bytes to simulate Apple Double files
                with open(os.path.join(episodes_dir, f"._ep_{i:05d}.npz"), "wb") as f:
                    f.write(b"\x00\x05\x16\x07" + os.urandom(100))

            ds = WorldModelDataset(tmpdir)
            assert ds.num_episodes == 3, f"Should load 3 real episodes, got {ds.num_episodes}"
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# TestResizing
# ---------------------------------------------------------------------------

class TestResizing:
    """Tests for target_resolution resizing."""

    def test_resize_to_128(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir, target_resolution=(128, 128))
        context, _, target = ds[0]
        assert context.shape == (12, 128, 128)
        assert target.shape == (3, 128, 128)

    def test_resize_preserves_range(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir, target_resolution=(128, 128))
        context, _, target = ds[0]
        # Bilinear interpolation can produce values slightly outside [0,1]
        # but should be very close
        assert context.min() >= -0.01
        assert context.max() <= 1.01
        assert target.min() >= -0.01
        assert target.max() <= 1.01


# ---------------------------------------------------------------------------
# TestFrameSkip
# ---------------------------------------------------------------------------

class TestFrameSkip:
    """Tests for frame_skip > 1."""

    def test_frame_skip_2(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir, context_length=4, frame_skip=2)
        # With frame_skip=2, context_length=4: need at least 4*2 = 8 prior frames
        assert len(ds) > 0

        ep_idx, t = ds.index[0]
        frames, _ = ds.episodes[ep_idx]
        context, _, _ = ds[0]

        # Context should be frames[t-8], frames[t-6], frames[t-4], frames[t-2]
        for h in range(4):
            frame_idx = t - (4 - h) * 2
            expected = torch.from_numpy(frames[frame_idx].copy()).float().div(255.0).permute(2, 0, 1)
            actual = context[h * 3 : (h + 1) * 3]
            torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# TestShortEpisodes
# ---------------------------------------------------------------------------

class TestShortEpisodes:
    """Tests with very short episodes."""

    def test_short_episodes_have_samples(self, short_episode_dir):
        ds = WorldModelDataset(short_episode_dir, context_length=4, frame_skip=1)
        # 8 frames, context_length=4, frame_skip=1 → min_t=4, max_t=6
        # Valid t: 4, 5, 6 → 3 samples per episode × 5 episodes = 15
        assert len(ds) == 15

    def test_too_short_episodes_no_crash(self):
        """Episodes shorter than context window should produce 0 samples, not crash."""
        tmpdir = tempfile.mkdtemp(prefix="minigenie_test_tiny_")
        episodes_dir = os.path.join(tmpdir, "episodes")
        os.makedirs(episodes_dir)
        try:
            # Create episodes with only 3 frames (not enough for context_length=4)
            for i in range(3):
                frames = np.random.randint(0, 256, (3, 64, 64, 3), dtype=np.uint8)
                actions = np.random.randint(0, 15, (2,), dtype=np.int32)
                np.savez_compressed(
                    os.path.join(episodes_dir, f"ep_{i:05d}.npz"),
                    frames=frames, actions=actions,
                )
            ds = WorldModelDataset(tmpdir, context_length=4, frame_skip=1)
            assert len(ds) == 0
        finally:
            shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# TestDataLoader
# ---------------------------------------------------------------------------

class TestDataLoader:
    """Tests that the dataset works correctly with PyTorch DataLoader."""

    def test_dataloader_basic(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir, max_episodes=5)
        dl = DataLoader(ds, batch_size=8, shuffle=True)
        batch = next(iter(dl))
        context, action, target = batch
        assert context.shape == (8, 12, 64, 64)
        assert action.shape == (8,)
        assert target.shape == (8, 3, 64, 64)

    def test_dataloader_multiworker(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir, max_episodes=5)
        dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
        batch = next(iter(dl))
        context, action, target = batch
        assert context.shape[0] == 4
        assert action.shape[0] == 4
        assert target.shape[0] == 4

    def test_dataloader_full_epoch(self, synthetic_data_dir):
        """Iterate through the full dataset without errors."""
        ds = WorldModelDataset(synthetic_data_dir, max_episodes=3)
        dl = DataLoader(ds, batch_size=16, shuffle=False)
        total_samples = 0
        for context, action, target in dl:
            total_samples += context.shape[0]
            # Verify no NaN/Inf
            assert torch.isfinite(context).all()
            assert torch.isfinite(target).all()
        assert total_samples == len(ds)


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------

class TestRepr:
    """Test string representation."""

    def test_repr_contains_info(self, synthetic_data_dir):
        ds = WorldModelDataset(synthetic_data_dir)
        r = repr(ds)
        assert "episodes=20" in r
        assert "context_length=4" in r


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for data validation."""

    def test_empty_dir_raises(self):
        tmpdir = tempfile.mkdtemp(prefix="minigenie_test_empty_")
        os.makedirs(os.path.join(tmpdir, "episodes"))
        try:
            with pytest.raises(FileNotFoundError):
                WorldModelDataset(tmpdir)
        finally:
            shutil.rmtree(tmpdir)

    def test_bad_action_range_raises(self):
        """Actions outside [0, 14] should raise ValueError during validation."""
        tmpdir = tempfile.mkdtemp(prefix="minigenie_test_badact_")
        episodes_dir = os.path.join(tmpdir, "episodes")
        os.makedirs(episodes_dir)
        try:
            frames = np.random.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8)
            actions = np.array([99] * 19, dtype=np.int32)  # Invalid action
            np.savez_compressed(
                os.path.join(episodes_dir, "ep_00000.npz"),
                frames=frames, actions=actions,
            )
            with pytest.raises(ValueError, match="actions outside valid range"):
                WorldModelDataset(tmpdir)
        finally:
            shutil.rmtree(tmpdir)
