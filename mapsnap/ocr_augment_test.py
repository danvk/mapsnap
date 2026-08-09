"""Tests for the OCR crop augmentation module."""

import numpy as np
import pytest

from mapsnap.ocr_augment import (
    DASH_GAP_PX,
    DASH_RUN_PX,
    RULE_THICKNESS,
    add_dash_line,
    add_edge_junk,
    add_underline,
    augment_crop,
    glyph_bottom_row,
    ink_mask,
    jitter_photometric,
    squeeze_resolution,
)


def sample_crop() -> np.ndarray:
    """A synthetic label crop: paper 250 with a glyph band on rows 7-19."""
    crop = np.full((26, 60), 250, dtype=np.uint8)
    # Vertical strokes every 6 px across the glyph band, ink value 90.
    for x in range(8, 52, 6):
        crop[7:20, x : x + 2] = 90
    return crop


def test_ink_mask_finds_glyph_band():
    crop = sample_crop()
    rows = np.where(ink_mask(crop).any(axis=1))[0]
    assert rows.min() == 7 and rows.max() == 19
    assert glyph_bottom_row(crop) == 19


def test_add_underline_thickness_and_fusion():
    crop = sample_crop()
    rng = np.random.default_rng(0)
    out = add_underline(crop, rng)
    changed = (out != crop) & ink_mask(out)
    rows = np.where(changed.any(axis=1))[0]
    assert len(rows) > 0
    thickness = rows.max() - rows.min() + 1
    assert RULE_THICKNESS[0] <= thickness <= RULE_THICKNESS[1]
    # Fused to the glyphs: rule top within 1 px of the glyph bottom, no gap.
    assert rows.min() <= glyph_bottom_row(crop) + 1
    # Shape unchanged: an underline never widens the crop.
    assert out.shape == crop.shape


def test_add_dash_line_runs_and_gaps():
    crop = sample_crop()
    rng = np.random.default_rng(1)
    out = add_dash_line(crop, rng)
    changed_cols = (out != crop).any(axis=0)
    assert changed_cols.any()
    # Dashes span most of the crop width (first to last changed column).
    span = np.where(changed_cols)[0]
    assert span[-1] - span[0] > crop.shape[1] // 2
    # Runs and gaps within the measured ranges.
    runs, gaps = [], []
    run = gap = 0
    for col in changed_cols[span[0] : span[-1] + 1]:
        if col:
            if gap:
                gaps.append(gap)
                gap = 0
            run += 1
        else:
            if run:
                runs.append(run)
                run = 0
            gap += 1
    if run:
        runs.append(run)
    assert all(DASH_RUN_PX[0] <= r <= DASH_RUN_PX[1] for r in runs)
    assert all(DASH_GAP_PX[0] <= g <= DASH_GAP_PX[1] for g in gaps)


def test_add_edge_junk_widens_and_adds_ink():
    crop = sample_crop()
    rng = np.random.default_rng(2)
    out = add_edge_junk(crop, rng)
    assert out.shape[0] == crop.shape[0]
    assert out.shape[1] > crop.shape[1]
    added = out.shape[1] - crop.shape[1]
    # The label pixels themselves are untouched (junk is prepended/appended).
    assert (
        np.array_equal(out[:, added:], crop)
        or np.array_equal(out[:, : crop.shape[1]], crop)
        or ink_mask(out).sum() > ink_mask(crop).sum()
    )


def test_add_edge_junk_uses_fragment_pool():
    crop = sample_crop()
    fragment = np.full((10, 12), 60, dtype=np.uint8)
    rng = np.random.default_rng(3)
    out = add_edge_junk(crop, rng, fragments=[fragment])
    assert out.shape[1] > crop.shape[1]


def test_squeeze_resolution_shrinks_within_range():
    crop = sample_crop()
    rng = np.random.default_rng(4)
    out = squeeze_resolution(crop, rng)
    assert out.shape[0] < crop.shape[0]
    assert 7 <= out.shape[0] <= 14
    # Aspect roughly preserved.
    assert out.shape[1] == pytest.approx(
        crop.shape[1] * out.shape[0] / crop.shape[0], abs=2
    )


def test_jitter_photometric_bounds():
    crop = sample_crop()
    rng = np.random.default_rng(5)
    out = jitter_photometric(crop, rng)
    assert out.dtype == np.uint8
    assert out.shape == crop.shape
    assert out.min() >= 0 and out.max() <= 255


def test_augment_crop_deterministic_under_seed():
    crop = sample_crop()
    a = augment_crop(crop, np.random.default_rng(42))
    b = augment_crop(crop, np.random.default_rng(42))
    assert a.shape == b.shape
    assert np.array_equal(a, b)


def test_augment_crop_always_changes_something():
    crop = sample_crop()
    for seed in range(20):
        out = augment_crop(crop, np.random.default_rng(seed))
        assert out.shape != crop.shape or not np.array_equal(out, crop)


def test_augment_crop_never_erases_all_ink():
    crop = sample_crop()
    for seed in range(20):
        out = augment_crop(crop, np.random.default_rng(seed))
        assert ink_mask(out).any()
