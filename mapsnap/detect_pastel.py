"""Identify the pastel-colored regions on a Sanborn key map.

A key map page is mostly three things: the dominant color of aged paper (anywhere
from near-white to a deep sepia); black ink (text, lines, hatching); and
pastel-colored regions (pink, yellow, green, blue) that show where each detailed
sheet sits. This module picks out just the pastel pixels.

The paper color varies enormously from scan to scan, so a fixed color threshold
does not generalize: a threshold that isolates pastels on near-white paper flags
the whole page when the paper has yellowed to sepia. Instead we calibrate per image,
after first cropping a margin off every side (pastel regions sit in the page
interior, so the margin only contributes scan edges, binding shadows, and edge
stains that would skew the calibration):

1. Estimate the paper color as the modal color (the largest uniform area on the
   page is always the paper).
2. Measure each pixel's chroma distance from the paper in CIELAB's a*/b* plane.
   This ignores lightness, so paper shadows and the paper's own tint sit near zero,
   while colored washes stand out regardless of how dark the paper is.
3. Threshold that distance with the triangle method, which adapts to each scan: the
   distances form a tall paper peak near zero with a tail of pastel pixels, and the
   triangle method places the cut in the valley after the peak.
4. Drop dark pixels by a lightness floor. Black ink is chroma-neutral, so on tinted
   paper it sits a full paper-chroma away from the paper and would otherwise be
   flagged; pastels are always light washes, so a lightness floor separates them.

The raw per-pixel mask is then cleaned with a morphological opening (to drop
isolated speckle on the paper) followed by a closing (to bridge the black lot
lines, street names, and bold sheet numbers that otherwise carve a single colored
region into many disconnected fragments). The closing kernel is sized to fill those
gaps while staying smaller than the white streets between regions, so neighboring
regions are not merged. Kernel sizes are tuned for full-resolution scans (~5000-6000
px wide); scale them down for smaller images.

Localized paper damage (foxing, water stains) drifts the paper color toward the
warm/yellow part of the a*/b* plane, the same direction as yellow pastels, so badly
stained areas can be flagged as pastel. Heavy stains are a known limitation.

Run as a script to write a sidecar "<stem>.pastel.png" next to each input image: a
copy of the original with every pastel pixel painted bright red.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mapsnap.utils import image_stem

# Bin width (per RGB channel) used when finding the modal paper color. Coarse enough
# to pool scanning noise into one bin, fine enough not to merge paper with pastels.
PAPER_QUANT = 16

# Chroma distances are clipped to this before the triangle threshold, so a handful of
# very saturated pixels can't stretch the histogram and skew the computed cut.
DISTANCE_CAP = 64

# Floor on the auto-computed chroma threshold. On very clean scans the paper peak is
# so tall and narrow that the triangle method cuts right beside it, flagging the JPEG
# color fringe around dense text; real pastel washes always sit at least this far from
# the paper, so a wash perceptibly different from paper never drops below this.
MIN_CHROMA_THRESHOLD = 10.0

# Minimum CIELAB lightness (OpenCV 0-255 scale) for a pixel to count as pastel.
# Excludes black ink, which is chroma-neutral and otherwise reads as far from tinted
# paper. Ink lands near L=30; the palest pastel washes stay well above L=80.
LIGHTNESS_FLOOR = 60

# Fraction of each side cropped off before analysis. Pastel regions sit in the page
# interior, while the margin holds scan edges, binding shadows, and edge stains that
# would skew the paper-color estimate and the auto threshold.
MARGIN_FRACTION = 0.04

# Morphological cleanup kernel diameters (pixels), tuned for full-res scans.
# Opening removes speckle; closing bridges lot lines, text, and sheet numbers.
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 25

# Color used to paint pastel pixels in the sidecar image.
RED = (255, 0, 0)


def estimate_paper_color(rgb: np.ndarray) -> np.ndarray:
    """Modal (most common) color of an RGB image, as a length-3 uint8 array.

    Colors are quantized into PAPER_QUANT-wide bins before counting, so scanning
    noise within the paper pools into a single bin. The paper is the largest uniform
    area on a key map, so its color wins the vote.
    """
    flat = rgb.reshape(-1, 3)
    quantized = flat // PAPER_QUANT
    codes = (
        (quantized[:, 0].astype(np.int32) << 16)
        + (quantized[:, 1].astype(np.int32) << 8)
        + quantized[:, 2].astype(np.int32)
    )
    values, counts = np.unique(codes, return_counts=True)
    code = int(values[counts.argmax()])
    half = PAPER_QUANT // 2
    return np.array(
        [
            ((code >> 16) & 0xFF) * PAPER_QUANT + half,
            ((code >> 8) & 0xFF) * PAPER_QUANT + half,
            (code & 0xFF) * PAPER_QUANT + half,
        ],
        dtype=np.uint8,
    )


def lab_distance_ab(lab: np.ndarray, paper_lab: np.ndarray) -> np.ndarray:
    """a*/b* (chroma) distance of a float32 LAB image from a single LAB color."""
    da = lab[..., 1] - paper_lab[1]
    db = lab[..., 2] - paper_lab[2]
    return np.sqrt(da * da + db * db)


def chroma_distance(rgb: np.ndarray, paper_rgb: np.ndarray) -> np.ndarray:
    """Per-pixel CIELAB a*/b* distance from the paper color (float32 HxW).

    Distance is measured only in the a*/b* (chroma) plane, ignoring lightness, so it
    responds to colored washes but not to how light or dark the paper is.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    paper_lab = cv2.cvtColor(paper_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[
        0, 0
    ].astype(np.float32)
    return lab_distance_ab(lab, paper_lab)


def auto_threshold(distance: np.ndarray) -> float:
    """Triangle-method chroma threshold for a distance map, clamped to a floor.

    The distances form one tall peak (paper) with a tail (pastels); the triangle
    method puts the cut after the peak. Returns at least MIN_CHROMA_THRESHOLD.
    """
    capped = np.clip(distance, 0, DISTANCE_CAP).astype(np.uint8)
    threshold, _ = cv2.threshold(
        capped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )
    return max(threshold, MIN_CHROMA_THRESHOLD)


def pastel_mask(rgb: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    """Boolean mask of the pastel-colored pixels in an RGB image.

    Takes an HxWx3 uint8 RGB array and returns an HxW boolean array that is True for
    pixels belonging to a pastel region (and False for paper and ink). The paper
    color is estimated per image; the chroma-distance ``threshold`` is chosen
    automatically per image unless one is supplied.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    paper_lab = cv2.cvtColor(
        estimate_paper_color(rgb).reshape(1, 1, 3), cv2.COLOR_RGB2LAB
    )[0, 0].astype(np.float32)
    distance = lab_distance_ab(lab, paper_lab)
    if threshold is None:
        threshold = auto_threshold(distance)
    return (distance > threshold) & (lab[..., 0] >= LIGHTNESS_FLOOR)


def clean_mask(
    mask: np.ndarray,
    *,
    open_size: int = OPEN_KERNEL_SIZE,
    close_size: int = CLOSE_KERNEL_SIZE,
) -> np.ndarray:
    """Speckle-free, gap-bridged version of a boolean pastel mask.

    Applies a morphological opening of diameter ``open_size`` to remove isolated
    speckle, then a closing of diameter ``close_size`` to fill the lot lines, text,
    and sheet numbers that fragment a region. Returns a boolean HxW array.
    """
    binary = mask.astype(np.uint8)
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary.astype(bool)


def detect_pastels(
    rgb: np.ndarray,
    *,
    margin_fraction: float = MARGIN_FRACTION,
    threshold: float | None = None,
    raw: bool = False,
) -> np.ndarray:
    """Full-size boolean pastel mask, ignoring a margin around the page edge.

    Crops ``margin_fraction`` off every side before estimating the paper color and
    thresholding, so scan edges, binding shadows, and edge stains do not skew the
    per-image calibration. Detection and (unless ``raw``) the morphological cleanup
    run on the interior; the result is pasted back into a full-size mask so its pixels
    line up one-for-one with ``rgb``, with the margin left all False.
    """
    height, width = rgb.shape[:2]
    margin_y = round(height * margin_fraction)
    margin_x = round(width * margin_fraction)
    interior = rgb[margin_y : height - margin_y, margin_x : width - margin_x]

    interior_mask = pastel_mask(interior, threshold=threshold)
    if not raw:
        interior_mask = clean_mask(interior_mask)

    mask = np.zeros((height, width), dtype=bool)
    mask[margin_y : height - margin_y, margin_x : width - margin_x] = interior_mask
    return mask


def paint_pastels(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of an RGB image with every pixel in ``mask`` painted bright red."""
    out = rgb.copy()
    out[mask] = RED
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect pastel-colored regions on key map images and write a sidecar "
            "<stem>.pastel.png with those pixels painted bright red."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE", help="Input image files.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip morphological cleanup and paint the raw per-pixel mask.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Chroma-distance threshold to override the per-image auto value.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=MARGIN_FRACTION,
        help="Fraction of each side to ignore during analysis (default %(default)s).",
    )
    args = parser.parse_args()

    for image in args.images:
        image_path = Path(image)
        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"))
        mask = detect_pastels(
            rgb, margin_fraction=args.margin, threshold=args.threshold, raw=args.raw
        )
        painted = paint_pastels(rgb, mask)
        output_path = image_path.parent / (image_stem(image) + ".pastel.png")
        Image.fromarray(painted).save(output_path)
        print(
            f"{image_path} → {output_path}  ({mask.mean():.1%} pastel)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
