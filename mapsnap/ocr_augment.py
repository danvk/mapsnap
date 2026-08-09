"""Synthetic artifact augmentation for street-label OCR crops (#265 Phase 2).

Fine-tuning the recognizer on clean inlier crops alone would teach it nothing
about the artifacts that actually break reads. This module corrupts clean
training crops with the three measured artifact classes, using geometry taken
from the real cases in ``testdata/erase_underlines/`` (see the ranges below),
plus the junk that surrounds labels on the sheets:

- **underline rules**: 2-3 px thick, fused to the glyph bottoms (the measured
  gap is zero - the rule shares rows with descender ink), drawn full-width,
  partial, or as the Fargo-style double dash (two 8-20 px runs);
- **dashed lines through the text** (water pipes, dashed street edges): 1-9 px
  dashes with 3-10 px gaps at glyph height, spanning the crop;
- **edge junk**: pipe-size annotations (``6"``-style tick pairs), leader dots,
  and ink fragments from neighboring boxes, prepended/appended to the label;
- **low-resolution squeeze**: the squished-ordinal class - the crop is
  downsampled to a few pixels of text height, as tiny sheet ordinals are;
- **photometric jitter**: brightness/contrast and paper tint.

All functions are deterministic given the caller's ``numpy`` Generator and
operate on grayscale uint8 arrays (the recognizer sees grayscale).
"""

import numpy as np

INK_THRESHOLD = 175  # matches detect_text's underline machinery
RULE_THICKNESS = (2, 3)  # measured: every fixture rule is 2-3 px
RULE_RUN_PX = (8, 20)  # measured partial/double-dash run lengths
DASH_RUN_PX = (1, 9)  # measured dash lengths (nashville pipe, champaign S NEIL)
DASH_GAP_PX = (3, 10)  # measured gaps between dashes
SQUEEZE_TEXT_HEIGHT = (7, 14)  # squished ordinals: a few px of glyph height


def ink_mask(crop: np.ndarray) -> np.ndarray:
    """Boolean mask of glyph/rule ink (dark pixels) in a grayscale crop."""
    return crop < INK_THRESHOLD


def ink_and_paper_values(crop: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """(ink value, paper value) sampled from the crop's own pixel population.

    Synthetic strokes must match the label's own ink (median ~80-100 on the
    fixtures, never pure black) and repairs its own paper (235-255), or the
    corruption is trivially separable from the print.
    """
    mask = ink_mask(crop)
    ink = int(np.median(crop[mask])) if mask.any() else 90
    paper = int(np.median(crop[~mask])) if (~mask).any() else 250
    ink = int(np.clip(ink + rng.integers(-15, 16), 30, INK_THRESHOLD - 20))
    return ink, paper


def glyph_bottom_row(crop: np.ndarray) -> int:
    """The last row containing glyph ink (where an underline rule fuses on)."""
    rows = np.where(ink_mask(crop).any(axis=1))[0]
    return int(rows[-1]) if len(rows) else crop.shape[0] - 1


def add_underline(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw an underline rule fused to the glyph bottoms.

    Styles match the fixtures: full-width, partial (one 8-20 px run), or the
    Fargo double dash (two runs with a gap). The rule's top row overlaps the
    last 0-1 glyph ink rows - the measured fixtures have no white gap, which
    is exactly why row-threshold erasure clipped digits (#250).
    """
    out = crop.copy()
    h, w = out.shape
    ink, _ = ink_and_paper_values(out, rng)
    thickness = int(rng.integers(RULE_THICKNESS[0], RULE_THICKNESS[1] + 1))
    top = min(glyph_bottom_row(out) - int(rng.integers(0, 2)) + 1, h - thickness)
    top = max(top, 0)
    style = rng.choice(["full", "partial", "double"])
    if style == "full":
        runs = [(0, w)]
    else:
        n_runs = 2 if style == "double" else 1
        runs = []
        x = int(rng.integers(0, max(1, w // 3)))
        for _ in range(n_runs):
            run = int(rng.integers(RULE_RUN_PX[0], RULE_RUN_PX[1] + 1))
            runs.append((x, min(x + run, w)))
            x = runs[-1][1] + int(rng.integers(3, 9))
            if x >= w:
                break
    for x0, x1 in runs:
        out[top : top + thickness, x0:x1] = np.minimum(
            out[top : top + thickness, x0:x1], ink
        )
    return out


def add_dash_line(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a dashed line through the text at glyph height (water-pipe class).

    Dashes run the full crop width at a row inside the glyph band, so erasure
    cannot remove the line without eating the label - the S NEIL failure mode.
    """
    out = crop.copy()
    h, w = out.shape
    ink, _ = ink_and_paper_values(out, rng)
    rows = np.where(ink_mask(out).any(axis=1))[0]
    if len(rows):
        y = int(rng.integers(rows[0], rows[-1] + 1))
    else:
        y = h // 2
    thickness = int(rng.integers(1, 3))
    y = min(y, h - thickness)
    x = int(rng.integers(0, DASH_GAP_PX[1]))
    while x < w:
        run = int(rng.integers(DASH_RUN_PX[0], DASH_RUN_PX[1] + 1))
        out[y : y + thickness, x : min(x + run, w)] = np.minimum(
            out[y : y + thickness, x : min(x + run, w)], ink
        )
        x += run + int(rng.integers(DASH_GAP_PX[0], DASH_GAP_PX[1] + 1))
    return out


def make_tick_junk(
    height: int, ink: int, paper: int, rng: np.random.Generator
) -> np.ndarray:
    """A small junk patch: pipe-size ticks (6"-style) or leader dots."""
    w = int(rng.integers(6, 14))
    patch = np.full((height, w), paper, dtype=np.uint8)
    if rng.random() < 0.5:
        # Two short vertical ticks (the inch marks after a pipe size).
        top = max(1, height // 4)
        for x in (w // 3, 2 * w // 3):
            patch[top : top + max(2, height // 4), x] = ink
    else:
        # Leader dots along the text midline.
        y = height // 2
        for x in range(1, w - 1, 3):
            patch[y : y + 2, x : x + 2] = ink
    return patch


def add_edge_junk(
    crop: np.ndarray,
    rng: np.random.Generator,
    fragments: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Attach junk before and/or after the text, widening the crop.

    Junk is a real ink fragment from a neighboring box when a pool is given
    (resized to this crop's height), else a synthetic tick/dot patch. This
    teaches the recognizer that a crop may carry non-label ink at its ends -
    the CRAFT box often includes it.
    """
    out = crop
    h, label_w = out.shape
    ink, paper = ink_and_paper_values(crop, rng)
    sides = ["left", "right"] if rng.random() < 0.3 else [rng.choice(["left", "right"])]
    for side in sides:
        if fragments and rng.random() < 0.6:
            frag = fragments[int(rng.integers(0, len(fragments)))]
            # Junk stays subordinate to the label: at most ~60% of its height
            # and ~35% of its width, and never upscaled (a fragment blown up
            # to label size reads as a competing label, not as junk).
            frag_h = min(frag.shape[0], max(4, int(h * rng.uniform(0.35, 0.6))))
            scale = frag_h / frag.shape[0]
            frag_w = max(3, min(int(frag.shape[1] * scale), int(label_w * 0.35)))
            import cv2

            small = cv2.resize(frag, (frag_w, frag_h), interpolation=cv2.INTER_AREA)
            patch = np.full((h, frag_w), paper, dtype=np.uint8)
            top = int(rng.integers(0, h - frag_h + 1))
            patch[top : top + frag_h] = small
        else:
            patch = make_tick_junk(h, ink, paper, rng)
        gap = np.full((h, int(rng.integers(1, 5))), paper, dtype=np.uint8)
        parts = [patch, gap, out] if side == "left" else [out, gap, patch]
        out = np.concatenate(parts, axis=1)
    return out


def squeeze_resolution(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Downsample to squished-ordinal resolution (a few px of text height).

    The recognizer's own prep will upscale it back to model height, which is
    exactly what happens to a tiny "2 ND" box in production - detail is
    already gone by the time the model sees it.
    """
    import cv2

    h, w = crop.shape
    target_h = int(rng.integers(SQUEEZE_TEXT_HEIGHT[0], SQUEEZE_TEXT_HEIGHT[1] + 1))
    if target_h >= h:
        return crop.copy()
    target_w = max(3, round(w * target_h / h))
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)


def jitter_photometric(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness/contrast jitter and paper-tint shift."""
    arr = crop.astype(np.float32)
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-20, 20))
    arr = (arr - 128.0) * gain + 128.0 + bias
    return np.clip(arr, 0, 255).astype(np.uint8)


def augment_crop(
    crop: np.ndarray,
    rng: np.random.Generator,
    fragments: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Apply a random combination of artifact corruptions (at least one).

    Underline and dash-line are mutually exclusive (they occupy the same
    visual role); junk, squeeze, and photometric jitter compose freely.
    """
    out = crop
    line = rng.random()
    applied = False
    if line < 0.45:
        out = add_underline(out, rng)
        applied = True
    elif line < 0.75:
        out = add_dash_line(out, rng)
        applied = True
    if rng.random() < 0.35:
        out = add_edge_junk(out, rng, fragments)
        applied = True
    if rng.random() < 0.25:
        out = squeeze_resolution(out, rng)
        applied = True
    if rng.random() < 0.5 or not applied:
        out = jitter_photometric(out, rng)
    return out
