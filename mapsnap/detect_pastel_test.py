import cv2
import numpy as np

from mapsnap.detect_pastel import (
    INK,
    INK_DISPLAY_COLOR,
    PAPER,
    PAPER_DISPLAY_COLOR,
    PASTEL,
    assign_palette,
    classify_centroids,
    classify_pastel_centroids,
    clean_mask,
    cluster_display_colors,
    cluster_segment_image,
    color_palette,
    detect_pastels,
    paint_pastels,
    palette_segmentation,
    pastel_mask,
    segment_image,
    segment_page,
)

WHITE_PAPER = (238, 232, 210)
SEPIA_PAPER = (184, 168, 136)

PINK = (240, 170, 190)
BLUE = (160, 195, 225)
GREEN = (175, 210, 175)
YELLOW = (235, 220, 120)
INK_COLOR = (25, 25, 25)


def page(paper: tuple[int, int, int]) -> np.ndarray:
    """A 120x120 page: mostly `paper`, with one patch each of ink and four pastels.

    Paper stays the dominant color so it is the most populous cluster. Each patch is
    20x20; the rest is paper.
    """
    img = np.empty((120, 120, 3), dtype=np.uint8)
    img[:] = paper
    img[0:20, 0:20] = INK_COLOR
    img[0:20, 40:60] = PINK
    img[0:20, 80:100] = BLUE
    img[40:60, 0:20] = GREEN
    img[40:60, 40:60] = YELLOW
    return img


def test_color_palette_returns_k_centroids():
    centroids = color_palette(page(WHITE_PAPER), 7)
    assert centroids.shape == (7, 3)


def test_color_palette_recovers_planted_colors():
    # Each planted color should have a centroid close to it (in RGB, after LAB->RGB).
    img = page(SEPIA_PAPER)
    centroids = color_palette(img, 7)
    centroid_rgb = cv2.cvtColor(
        np.clip(centroids, 0, 255).astype(np.uint8).reshape(1, -1, 3),
        cv2.COLOR_LAB2RGB,
    ).reshape(-1, 3)
    for color in (SEPIA_PAPER, INK_COLOR, PINK, BLUE, GREEN, YELLOW):
        nearest = np.abs(centroid_rgb.astype(int) - color).sum(axis=1).min()
        assert nearest < 40, (color, nearest)


def test_assign_palette_picks_nearest_centroid():
    centroids = np.array([[0, 128, 128], [255, 128, 128]], dtype=np.float32)
    lab = np.array([[[10, 128, 128], [250, 128, 128]]], dtype=np.float32)
    labels = assign_palette(lab, centroids)
    assert labels.tolist() == [[0, 1]]


def test_classify_flags_pastels_not_paper_or_ink():
    img = page(SEPIA_PAPER)
    centroids = color_palette(img, 7)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    labels = assign_palette(lab, centroids)
    is_pastel = classify_pastel_centroids(centroids, labels)
    # Four pastel patches -> at least four pastel centroids; paper and ink excluded.
    assert is_pastel.sum() >= 4
    # The darkest centroid (ink) is never pastel.
    assert not is_pastel[centroids[:, 0].argmin()]
    # The most populous centroid (paper) is never pastel.
    counts = np.bincount(labels.reshape(-1), minlength=len(centroids))
    assert not is_pastel[counts.argmax()]


def test_classify_separates_cool_and_off_axis_from_warm_paper_shade():
    # On yellow paper, a darker/yellower shade stays on the warm axis (paper), while blue
    # (opposite the axis), green and pink (off the axis) are pastel. This is the whole
    # point of the directional test: it must not lump cool/off-axis washes with paper.
    centroids = np.array(
        [
            [180, 128, 145],  # paper (yellow)
            [140, 128, 152],  # darker, yellower paper shade — along the warm axis
            [170, 126, 128],  # blue — opposite the warm axis
            [170, 120, 150],  # green — off the axis
            [175, 145, 145],  # pink — off the axis
            [30, 128, 128],  # ink
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [[0, 0, 0, 1], [2, 3, 4, 5]], dtype=np.int32
    )  # paper most populous
    categories = classify_centroids(centroids, labels)
    assert categories[0] == PAPER
    assert categories[1] == PAPER  # warm shade is not mistaken for a pastel
    assert categories[2] == PASTEL  # blue
    assert categories[3] == PASTEL  # green
    assert categories[4] == PASTEL  # pink
    assert categories[5] == INK


def test_pastels_detected_on_white_and_sepia_paper():
    for paper in (WHITE_PAPER, SEPIA_PAPER):
        mask = pastel_mask(page(paper))
        for r0, c0 in [
            (0, 40),
            (0, 80),
            (40, 0),
            (40, 40),
        ]:  # pink, blue, green, yellow
            assert mask[r0 + 5 : r0 + 15, c0 + 5 : c0 + 15].all(), (paper, r0, c0)
        assert not mask[5:15, 5:15].any()  # ink patch
        assert not mask[100:115, 100:115].any()  # paper corner


def test_segment_image_recolors_to_centroid():
    centroids = np.array([[200, 120, 150], [40, 128, 128]], dtype=np.float32)
    labels = np.array([[0, 1], [1, 0]], dtype=np.int32)
    out = segment_image(centroids, labels)
    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    assert (out[0, 0] == out[1, 1]).all()  # same label -> same color
    assert (out[0, 0] != out[0, 1]).any()  # different labels -> different colors


def test_segment_page_preserves_size():
    img = page(WHITE_PAPER)
    out = segment_page(img)
    assert out.shape == img.shape


def test_cluster_segment_paints_exactly_one_white_one_black_rest_colored():
    img = page(SEPIA_PAPER)
    centroids, labels = palette_segmentation(img, margin_fraction=0.0)
    colors = cluster_display_colors(centroids, labels)
    # Exactly one cluster white (background) and exactly one black (darkest).
    assert np.all(colors == PAPER_DISPLAY_COLOR, axis=1).sum() == 1
    assert np.all(colors == INK_DISPLAY_COLOR, axis=1).sum() == 1

    out = cluster_segment_image(centroids, labels)
    assert out.shape == img.shape
    # Paper corner -> white, ink patch -> black.
    assert (out[100:115, 100:115] == PAPER_DISPLAY_COLOR).all()
    assert (out[5:15, 5:15] == INK_DISPLAY_COLOR).all()
    # Each pastel patch -> a single solid color that is neither white nor black.
    for r0, c0 in [(0, 40), (0, 80), (40, 0), (40, 40)]:
        patch = out[r0 + 5 : r0 + 15, c0 + 5 : c0 + 15]
        assert (patch == patch[0, 0]).all()
        assert tuple(patch[0, 0]) not in (PAPER_DISPLAY_COLOR, INK_DISPLAY_COLOR)


def test_detect_pastels_preserves_image_size():
    mask = detect_pastels(page(WHITE_PAPER))
    assert mask.shape == page(WHITE_PAPER).shape[:2]


def test_detect_pastels_ignores_pastels_in_the_margin():
    # 200x200 with a 4% (8px) margin: a patch in the corner margin must be ignored,
    # while an interior patch is still detected.
    img = np.empty((200, 200, 3), dtype=np.uint8)
    img[:] = WHITE_PAPER
    img[0:6, 0:6] = PINK  # inside the 8px margin
    img[90:130, 90:130] = PINK  # interior
    mask = detect_pastels(img, raw=True)
    assert not mask[0:6, 0:6].any()
    assert mask[95:125, 95:125].all()


def test_paint_pastels_marks_only_masked_pixels_red():
    img = page(WHITE_PAPER)
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[40:60, 40:60] = True
    painted = paint_pastels(img, mask)
    assert (painted[40:60, 40:60] == (255, 0, 0)).all()
    assert (painted[100:115, 100:115] == img[100:115, 100:115]).all()


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
