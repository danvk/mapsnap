"""Tests for split_pages.py."""

from pathlib import Path

import numpy as np
import pytest

from mapsnap.split_pages import find_thick_bands, partition_image, split_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"

CHAMPAIGN_UNSPLIT = DATA_DIR / "champaign_ill_1915" / "p4.unsplit.jpg"
CHAMPAIGN_P2 = DATA_DIR / "champaign_ill_1915" / "p2.unsplit.jpg"
CHAMPAIGN_P20 = DATA_DIR / "champaign_ill_1915" / "p20.unsplit.jpg"
DETROIT_NO_SPLIT = DATA_DIR / "detroit_mich_1929_vol_11" / "p11.raw.jpg"
NEW_ORLEANS_DIAGONAL = DATA_DIR / "new_orleans_la_1896_vol_2" / "p101.unsplit.jpg"


def _make_dark_stripe_h(
    height: int, width: int, band_y: int, band_h: int
) -> np.ndarray:
    """Grayscale image (0=black, 255=white) with a horizontal dark stripe."""
    img = np.full((height, width), 200, dtype=np.uint8)
    img[band_y : band_y + band_h, :] = 0
    return img


def _make_dark_stripe_v(
    height: int, width: int, band_x: int, band_w: int
) -> np.ndarray:
    """Grayscale image with a vertical dark stripe."""
    img = np.full((height, width), 200, dtype=np.uint8)
    img[:, band_x : band_x + band_w] = 0
    return img


# ---------------------------------------------------------------------------
# find_thick_bands
# ---------------------------------------------------------------------------


def test_find_thick_bands_simple():
    # Band from index 30 to 39 (10 wide), value 80 > min_run=50.
    profile = np.zeros(100, dtype=np.int32)
    profile[30:40] = 80
    bands = find_thick_bands(profile, min_run=50, min_thickness=5)
    assert bands == [(30, 39)]


def test_find_thick_bands_none():
    profile = np.zeros(100, dtype=np.int32)
    profile[30:40] = 80
    # min_run higher than all profile values → no bands.
    bands = find_thick_bands(profile, min_run=100, min_thickness=5)
    assert bands == []


def test_find_thick_bands_thin_filtered():
    # Band only 2 wide, below min_thickness=5.
    profile = np.zeros(100, dtype=np.int32)
    profile[30:32] = 80
    bands = find_thick_bands(profile, min_run=50, min_thickness=5)
    assert bands == []


def test_find_thick_bands_multiple():
    profile = np.zeros(200, dtype=np.int32)
    profile[20:30] = 80  # band 1
    profile[100:115] = 80  # band 2
    bands = find_thick_bands(profile, min_run=50, min_thickness=5)
    assert len(bands) == 2
    assert bands[0] == (20, 29)
    assert bands[1] == (100, 114)


def test_find_thick_bands_at_end():
    # Band that runs to the very end of the array.
    profile = np.zeros(100, dtype=np.int32)
    profile[90:] = 80
    bands = find_thick_bands(profile, min_run=50, min_thickness=5)
    assert bands == [(90, 99)]


# ---------------------------------------------------------------------------
# partition_image — synthetic images
# ---------------------------------------------------------------------------


def test_partition_no_split():
    # Uniform white image → single section covering the whole image.
    img = np.full((100, 100), 200, dtype=np.uint8)
    sections = partition_image(img, min_run_fraction=0.3, min_thickness=3)
    assert len(sections) == 1
    assert sections[0].bounds == (0.0, 0.0, 100.0, 100.0)


def test_partition_vertical_split():
    # Vertical dark stripe at x=45..54 → 2 sections.
    img = _make_dark_stripe_v(height=100, width=100, band_x=45, band_w=10)
    sections = partition_image(img, min_run_fraction=0.3, min_thickness=3)
    assert len(sections) == 2
    left, right = sections
    assert left.bounds[0] == 0  # x0
    assert right.bounds[2] == 100  # x1
    assert left.bounds[2] <= right.bounds[0] + 2


def test_partition_horizontal_split():
    # Horizontal dark stripe at y=45..54 → 2 sections.
    img = _make_dark_stripe_h(height=100, width=100, band_y=45, band_h=10)
    sections = partition_image(img, min_run_fraction=0.3, min_thickness=3)
    assert len(sections) == 2
    top, bottom = sections
    assert top.bounds[1] == 0  # y0
    assert bottom.bounds[3] == 100  # y1
    assert top.bounds[3] <= bottom.bounds[1] + 2


def test_partition_t_shape():
    # T-shaped layout: one vertical band + two horizontal bands only in the right column.
    height, width = 100, 200
    img = np.full((height, width), 200, dtype=np.uint8)

    # Vertical split at x=90..99 — spans full height.
    img[:, 90:100] = 0

    # Horizontal splits in the right portion (x=100..200) at y=30..39 and y=60..69.
    img[30:40, 100:] = 0
    img[60:70, 100:] = 0

    sections = partition_image(img, min_run_fraction=0.3, min_thickness=3)
    # Expected: left column (1 section) + right column with 2 horizontal splits (3 sections) = 4.
    assert len(sections) == 4

    # First section is the full-height left column.
    left = sections[0]
    assert left.bounds[0] == 0  # x0
    assert left.bounds[1] == 0  # y0
    assert left.bounds[3] == height  # y1

    # Remaining 3 sections are stacked in the right column.
    right_sections = sections[1:]
    assert all(s.bounds[0] > 80 for s in right_sections)
    assert right_sections[0].bounds[1] == 0
    assert right_sections[-1].bounds[3] == height


# ---------------------------------------------------------------------------
# Integration tests — real images (skipped when data files are absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CHAMPAIGN_UNSPLIT.exists(), reason="test data not present")
def test_split_image_champaign(tmp_path):
    # p4.unsplit.jpg has 4 sections: 1 full-height left + 3 stacked on the right.
    import shutil

    src = shutil.copy(CHAMPAIGN_UNSPLIT, tmp_path / "p4.unsplit.jpg")
    paths = split_image(Path(src))
    assert len(paths) == 4
    names = {p.name for p in paths}
    assert names == {"p4__1.raw.jpg", "p4__2.raw.jpg", "p4__3.raw.jpg", "p4__4.raw.jpg"}


@pytest.mark.skipif(not CHAMPAIGN_P2.exists(), reason="test data not present")
def test_split_image_champaign_p2(tmp_path):
    # p2.unsplit.jpg has a corner inset (top-left): expect 2 sections.
    import shutil

    src = shutil.copy(CHAMPAIGN_P2, tmp_path / "p2.unsplit.jpg")
    paths = split_image(Path(src))
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert names == {"p2__1.raw.jpg", "p2__2.raw.jpg"}


@pytest.mark.skipif(not CHAMPAIGN_P20.exists(), reason="test data not present")
def test_split_image_champaign_p20(tmp_path):
    # p20.unsplit.jpg has a corner inset (bottom-right): expect 2 sections.
    import shutil

    src = shutil.copy(CHAMPAIGN_P20, tmp_path / "p20.unsplit.jpg")
    paths = split_image(Path(src))
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert names == {"p20__1.raw.jpg", "p20__2.raw.jpg"}


@pytest.mark.skipif(not DETROIT_NO_SPLIT.exists(), reason="test data not present")
def test_split_image_detroit_no_split(tmp_path):
    # p11.raw.jpg has no splits; split_image should return an empty list.
    import shutil

    src = shutil.copy(DETROIT_NO_SPLIT, tmp_path / "p11.raw.jpg")
    paths = split_image(Path(src))
    assert paths == []
