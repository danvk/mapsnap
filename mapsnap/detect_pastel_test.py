import numpy as np

from mapsnap.detect_pastel import (
    auto_threshold,
    chroma_distance,
    clean_mask,
    estimate_paper_color,
    paint_pastels,
    pastel_mask,
)

WHITE_PAPER = (238, 232, 210)
SEPIA_PAPER = (184, 168, 136)

PINK = (240, 170, 190)
BLUE = (160, 195, 225)
GREEN = (175, 210, 175)
YELLOW = (235, 220, 120)
INK = (25, 25, 25)


def scene(paper: tuple[int, int, int], patch: tuple[int, int, int]) -> np.ndarray:
    """A 100x100 image of mostly `paper` with a 20x20 `patch` in one corner.

    Paper stays the dominant color so it is estimated correctly; the patch sits at
    rows/cols 0:20 and the paper corner at the opposite side.
    """
    img = np.empty((100, 100, 3), dtype=np.uint8)
    img[:] = paper
    img[0:20, 0:20] = patch
    return img


def patch_is_detected(paper: tuple[int, int, int], patch: tuple[int, int, int]) -> bool:
    """Whether the patch is flagged and the paper corner is not."""
    mask = pastel_mask(scene(paper, patch))
    return bool(mask[5:15, 5:15].all()) and not bool(mask[80:, 80:].any())


def test_estimate_paper_color_finds_dominant_sepia():
    paper = estimate_paper_color(scene(SEPIA_PAPER, PINK))
    # Within one quantization bin of the true sepia paper.
    assert np.abs(paper.astype(int) - SEPIA_PAPER).max() <= 16


def test_pastels_detected_on_white_paper():
    for patch in (PINK, BLUE, GREEN, YELLOW):
        assert patch_is_detected(WHITE_PAPER, patch), patch


def test_pastels_detected_on_sepia_paper():
    # The whole point of the per-image calibration: sepia paper must not swamp it.
    for patch in (PINK, BLUE, GREEN, YELLOW):
        assert patch_is_detected(SEPIA_PAPER, patch), patch


def test_ink_is_not_pastel():
    mask = pastel_mask(scene(WHITE_PAPER, INK))
    assert not mask[5:15, 5:15].any()


def test_chroma_distance_ignores_lightness():
    # A darker shade of the paper color (same hue, lower lightness) stays near zero.
    paper = np.array(SEPIA_PAPER, dtype=np.uint8)
    darker = (paper.astype(np.float32) * 0.6).astype(np.uint8)
    img = darker.reshape(1, 1, 3)
    assert chroma_distance(img, paper)[0, 0] < 10


def test_auto_threshold_floors_when_no_pastels():
    # All-zero distance (uniform page) must not yield a tiny threshold.
    assert auto_threshold(np.zeros((50, 50), dtype=np.float32)) >= 10.0


def test_paint_pastels_marks_only_masked_pixels_red():
    img = scene(WHITE_PAPER, PINK)
    painted = paint_pastels(img, pastel_mask(img))
    assert (painted[0:20, 0:20] == (255, 0, 0)).all()  # pink patch
    assert (painted[80:, 80:] == img[80:, 80:]).all()  # paper corner untouched


def test_clean_mask_removes_isolated_speckle():
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 20] = True  # a single stray pixel
    assert not clean_mask(mask).any()


def test_clean_mask_bridges_a_thin_gap():
    # Two blocks separated by a 3px gap (like a lot line) should reconnect.
    mask = np.ones((40, 40), dtype=bool)
    mask[:, 18:21] = False
    cleaned = clean_mask(mask, open_size=3, close_size=11)
    assert cleaned[:, 18:21].all()


def test_clean_mask_keeps_wide_gap_separate():
    # A wide gap (like a street) must survive closing so regions stay distinct.
    mask = np.ones((60, 60), dtype=bool)
    mask[:, 25:35] = False
    cleaned = clean_mask(mask, open_size=3, close_size=7)
    assert not cleaned[:, 28:32].any()
