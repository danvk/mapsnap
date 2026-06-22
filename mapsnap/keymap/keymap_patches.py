"""Build a patch dataset for the page-number localizer from point labels.

The labeler tool writes ``<stem>.labels.json`` (``{width,height,labels:[{x,y,text}]}``)
marking the center of every bold page number. This module turns those points into a
binary patch dataset for a tiny CNN: positive patches centered on a label, negative
patches sampled away from every label. Work happens at a fixed downscale (SCALE) so a
page number is a roughly constant size regardless of the scan's resolution.

The helpers here are pure (array in, array out) so they can be unit-tested without a
model; image loading and augmentation live in the trainer.
"""

import json
from pathlib import Path

import numpy as np

# Fraction of a FULL-resolution scan we work at, so a page number is ~40px and a
# PATCH_SIZE window captures the number plus its block context.
SCALE = 0.25

# A scan whose larger side is below this is assumed to already be at SCALE (25%); a
# larger one is full resolution and must be downscaled by SCALE to reach working res.
FULL_SCALE_THRESHOLD = 4000


def working_scale(
    width: int, height: int, *, full_threshold: int = FULL_SCALE_THRESHOLD
) -> float:
    """Factor to bring an image (and its label coords) to working resolution.

    Returns SCALE for a full-resolution scan (larger side >= full_threshold) and 1.0 for
    one already downscaled to ~25%, so both reach the same ~40px-per-number working size.
    Label coordinates live in the image's own pixel space, so the same factor applies.
    """
    return SCALE if max(width, height) >= full_threshold else 1.0


# Side length (px, at SCALE) of the square patches fed to the CNN.
PATCH_SIZE = 128

# Minimum distance (px, at SCALE) a negative patch center must be from every label, so
# negatives never accidentally contain a centered page number.
MIN_NEG_DIST = 40

# Negative patches sampled per positive label.
NEG_PER_POS = 3

# Largest a page number gets in a FULL-resolution scan (px); at working scale (SCALE) a
# number is therefore ~this * SCALE, roughly constant across scans. Used to decide whether
# a candidate negative crop is safe — i.e. contains no part of a real page number.
NUMBER_MAX_W_FULL = 170
NUMBER_MAX_H_FULL = 100
NUMBER_HALF_W_WORKING = NUMBER_MAX_W_FULL * SCALE / 2
NUMBER_HALF_H_WORKING = NUMBER_MAX_H_FULL * SCALE / 2


def crop_excludes_numbers(
    cx: float,
    cy: float,
    label_centers: list[tuple[float, float]],
    *,
    crop_half_w: float,
    crop_half_h: float,
    number_half_w: float = NUMBER_HALF_W_WORKING,
    number_half_h: float = NUMBER_HALF_H_WORKING,
) -> bool:
    """Whether a crop at (cx, cy) overlaps no page-number box (so it is a safe negative).

    All coordinates and half-sizes are in working units. A number box (centered on a label,
    half-size number_half_*) and the crop box (half-size crop_half_*) overlap when they are
    close in BOTH axes; the crop is a safe negative only if every label clears it in at
    least one axis. This lets negatives sit in the colorful gaps between numbers (e.g.
    vertically between rows) without clipping a real number into the crop.
    """
    margin_x = crop_half_w + number_half_w
    margin_y = crop_half_h + number_half_h
    return all(
        abs(cx - lx) >= margin_x or abs(cy - ly) >= margin_y for lx, ly in label_centers
    )


def load_label_points(path: str) -> tuple[int, int, list[tuple[float, float, str]]]:
    """Read a .labels.json file; return (width, height, [(x, y, text), ...])."""
    doc = json.load(open(path))
    labels = doc["labels"] if isinstance(doc, dict) else doc
    points = [
        (float(label["x"]), float(label["y"]), str(label["text"])) for label in labels
    ]
    width = int(doc["width"]) if isinstance(doc, dict) else 0
    height = int(doc["height"]) if isinstance(doc, dict) else 0
    return width, height, points


def scale_points(
    points: list[tuple[float, float, str]], scale: float
) -> list[tuple[float, float, str]]:
    """Multiply each point's x/y by ``scale`` (text unchanged)."""
    return [(x * scale, y * scale, text) for x, y, text in points]


def crop_patch(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Square ``size``x``size`` RGB patch centered on (cx, cy), white-padded at edges."""
    half = size // 2
    x0, y0 = round(cx) - half, round(cy) - half
    x1, y1 = x0 + size, y0 + size
    height, width = image.shape[:2]
    patch = np.full((size, size, 3), 255, dtype=np.uint8)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x1), min(height, y1)
    if sx1 > sx0 and sy1 > sy0:
        patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return patch


def is_far_from_all(
    cx: float, cy: float, points: list[tuple[float, float, str]], min_dist: float
) -> bool:
    """Whether (cx, cy) is at least ``min_dist`` from every point (ignoring text)."""
    min_sq = min_dist * min_dist
    return all((cx - px) ** 2 + (cy - py) ** 2 >= min_sq for px, py, _ in points)


def sample_negative_centers(
    width: int,
    height: int,
    positives: list[tuple[float, float, str]],
    count: int,
    *,
    min_dist: float = MIN_NEG_DIST,
    rng: np.random.Generator,
    max_attempts_per: int = 50,
) -> list[tuple[float, float]]:
    """Sample ``count`` random centers in [0,width)x[0,height) far from every positive.

    Rejection-samples points at least ``min_dist`` from all positives; gives up on an
    individual point after ``max_attempts_per`` tries, so the returned list may be
    slightly shorter than ``count`` on a tiny/crowded image.
    """
    centers: list[tuple[float, float]] = []
    for _ in range(count):
        for _attempt in range(max_attempts_per):
            cx = float(rng.integers(0, width))
            cy = float(rng.integers(0, height))
            if is_far_from_all(cx, cy, positives, min_dist):
                centers.append((cx, cy))
                break
    return centers


def build_image_patches(
    image: np.ndarray,
    scaled_points: list[tuple[float, float, str]],
    *,
    size: int = PATCH_SIZE,
    neg_per_pos: int = NEG_PER_POS,
    min_neg_dist: float = MIN_NEG_DIST,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int]]:
    """Positive (label 1) and negative (label 0) patches for one downscaled image.

    ``image`` and ``scaled_points`` must already be at the working scale. Returns
    (patches, labels) where patches are ``size``x``size``x3 uint8 arrays.
    """
    height, width = image.shape[:2]
    patches: list[np.ndarray] = []
    labels: list[int] = []

    for px, py, _ in scaled_points:
        patches.append(crop_patch(image, px, py, size))
        labels.append(1)

    negatives = sample_negative_centers(
        width,
        height,
        scaled_points,
        count=neg_per_pos * len(scaled_points),
        min_dist=min_neg_dist,
        rng=rng,
    )
    for nx, ny in negatives:
        patches.append(crop_patch(image, nx, ny, size))
        labels.append(0)

    return patches, labels


def labels_path_for(image_path: str) -> Path:
    """Path of the .labels.json sidecar for an image."""
    p = Path(image_path)
    return p.parent / (p.name.split(".")[0] + ".labels.json")
