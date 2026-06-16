"""Tests for mapsnap.oim_truth."""

import numpy as np
from PIL import Image

from mapsnap.oim_truth import locate_oim_splits, locate_split_in_unsplit


def _save(path, arr):
    Image.fromarray(arr).save(path, quality=95)


def _smooth_page(height: int, width: int, seed: int) -> np.ndarray:
    """A smooth, non-periodic grayscale page that survives JPEG and localizes uniquely.

    Built by bilinearly upscaling coarse random noise so it has low spatial frequency
    (JPEG-robust) yet a single sharp template-match peak.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, size=(height // 10, width // 10), dtype=np.uint8)
    resized = Image.fromarray(coarse).resize((width, height), Image.Resampling.BILINEAR)
    return np.array(resized)


def test_locate_split_in_unsplit_finds_offset(tmp_path):
    unsplit = _smooth_page(200, 160, seed=0)
    # The split is a sub-window of the unsplit page at a known offset.
    split = unsplit[40:140, 30:110].copy()
    _save(tmp_path / "u.jpg", unsplit)
    _save(tmp_path / "s.jpg", split)

    offset_x, offset_y = locate_split_in_unsplit(tmp_path / "s.jpg", tmp_path / "u.jpg")
    assert abs(offset_x - 30) <= 1
    assert abs(offset_y - 40) <= 1


def test_locate_oim_splits_writes_canvas_rings(tmp_path):
    raw_dir = tmp_path / "raw"
    oim_dir = tmp_path / "oim"
    raw_dir.mkdir()
    oim_dir.mkdir()

    unsplit = _smooth_page(200, 160, seed=1)
    _save(raw_dir / "p4.jpg", unsplit)
    _save(oim_dir / "p4__1.jpg", unsplit[0:120, 0:160].copy())
    _save(oim_dir / "p4__2.jpg", unsplit[120:200, 0:160].copy())

    panels = locate_oim_splits("p4", [1, 2], raw_dir, oim_dir)

    assert (panels["width"], panels["height"]) == (160, 200)
    assert len(panels["panels"]) == 2
    # Second region starts at y≈120 (top-left corner of the ring).
    second_ring = panels["panels"][1]
    assert abs(second_ring[0][1] - 120) <= 1
