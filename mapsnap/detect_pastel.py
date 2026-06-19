"""Identify the pastel-colored regions on a Sanborn key map.

A key map page is mostly three things: the dominant color of aged paper (anywhere
from near-white to a deep sepia); black ink (text, lines, hatching); and
pastel-colored regions (pink, yellow, green, blue) that show where each detailed
sheet sits. This module picks out just the pastel pixels.

We take that "mostly three things" thesis literally: if the page really is paper plus
ink plus a few pastels, then a small fixed palette should reconstruct it. We quantize
each scan to a handful of colors (default 7) with k-means in CIELAB, then assign every
pixel to its nearest palette color. Seven colors rather than the bare six (paper, ink,
and four pastels) leave a spare cluster or two to absorb background and shading
variation instead of smearing it across the real colors. Because the palette is fit
per image, it adapts to paper that has yellowed all the way to sepia.

Lightness is down-weighted during clustering (see LIGHTNESS_WEIGHT): aged paper spans
a wide lightness range at nearly constant hue, so at full weight k-means spends every
cluster splitting paper into light/dark shades and the faint pastels collapse into it.
Down-weighting lightness lets chroma — what actually distinguishes the pastels — drive
the clustering, while dark ink still separates out as its own cluster.

Analysis runs on the page interior only: a margin is cropped off every side first
(pastel regions sit in the interior, so the margin only contributes scan edges,
binding shadows, and edge stains that would pull a palette cluster).

Each palette color is then classified:
- ink: the centroid is dark (lightness below LIGHTNESS_FLOOR).
- paper: the most populous non-ink centroid, plus any non-ink centroid whose a*/b*
  chroma is within CENTROID_CHROMA_THRESHOLD of it (shading and stain variants).
- pastel: every remaining non-ink centroid.

The pastel pixels are then cleaned with a morphological opening (to drop isolated
speckle on the paper) followed by a closing (to bridge the black lot lines, street
names, and bold sheet numbers that otherwise carve a single colored region into many
disconnected fragments). The closing kernel is sized to fill those gaps while staying
smaller than the white streets between regions, so neighboring regions are not merged.
Kernel sizes are tuned for full-resolution scans (~5000-6000 px wide).

Localized paper damage (foxing, water stains) can drift far enough from the paper to
form its own palette cluster classified as pastel, so badly stained areas can be
flagged. Heavy stains are a known limitation.

Run as a script to write a sidecar "<stem>.segments.png" next to each input image: the
page with each cluster shown in a distinct color (one cluster white, one black, the rest
a color palette), so the raw partition can be inspected directly. Pass --natural to
recolor each cluster to its own posterized color instead, or --overlay to also write
"<stem>.pastel.png" with the detected pastels painted red.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mapsnap.utils import image_stem

# Default number of palette colors: paper, ink, four pastels, plus a spare cluster or
# two for background/shading variation.
DEFAULT_NUM_COLORS = 7

# Pixels sampled to fit the palette. Fitting on every pixel of a full-res scan (~50M)
# is needlessly slow; a random subsample gives the same centroids.
KMEANS_SAMPLE_SIZE = 100_000

# Fixed seed so the (randomized) k-means produces identical output across runs.
KMEANS_SEED = 0

# Weight applied to CIELAB lightness when clustering. Aged paper spans a wide range of
# lightness at nearly constant hue, so at full weight k-means spends its whole budget
# splitting paper into light/dark shades and the pastels collapse into the paper
# clusters. Down-weighting lightness lets chroma (which is what actually separates the
# pastels) drive the clustering, while still keeping dark ink its own cluster.
LIGHTNESS_WEIGHT = 0.5

# Minimum CIELAB lightness (OpenCV 0-255 scale) for a centroid to be a color rather
# than ink. Ink lands near L=30; the palest pastel washes stay well above L=80.
LIGHTNESS_FLOOR = 60

# A non-ink centroid is pastel if its a*/b* offset from the paper centroid is far enough
# from the paper's warm/yellow tint axis. Paper shading and staining only slide color
# along that axis, so deviations are judged three ways:
#  - PASTEL_PERP_THRESHOLD: sideways off the axis (pink, green, and blends).
#  - PASTEL_WARM_THRESHOLD: far up the axis = a distinct yellow wash, not just yellower
#    paper (set high because yellow pastel and yellowed paper share the warm direction).
#  - PASTEL_COOL_THRESHOLD: far down the axis = blue, the opposite of yellow; paper never
#    goes cool, so a strongly anti-warm centroid is blue even with little sideways offset.
PASTEL_PERP_THRESHOLD = 4.0
PASTEL_WARM_THRESHOLD = 15.0
PASTEL_COOL_THRESHOLD = 15.0

# Fraction of each side cropped off before analysis. Pastel regions sit in the page
# interior, while the margin holds scan edges, binding shadows, and edge stains that
# would pull a palette cluster.
MARGIN_FRACTION = 0.04

# Morphological cleanup kernel diameters (pixels), tuned for full-res scans.
# Opening removes speckle; closing bridges lot lines, text, and sheet numbers.
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 25

# Color used to paint pastel pixels in the optional overlay image.
RED = (255, 0, 0)

# Centroid categories.
PAPER, INK, PASTEL = 0, 1, 2

# Display colors for the cluster segmentation image: the background cluster washes out to
# white, the darkest to black, and every other cluster gets a distinct vivid color, so the
# raw k-means partition is obvious at a glance.
PAPER_DISPLAY_COLOR = (255, 255, 255)
INK_DISPLAY_COLOR = (0, 0, 0)
# Each pastel cluster is shown in whichever of these the cluster's own hue is closest to,
# so a blue wash renders blue, a yellow wash yellow, etc. Spans the hue circle so every
# pastel finds a near match.
PASTEL_DISPLAY_COLORS = [
    (230, 25, 75),  # red
    (245, 130, 48),  # orange
    (255, 225, 25),  # yellow
    (60, 180, 75),  # green
    (70, 240, 240),  # cyan
    (0, 130, 200),  # blue
    (145, 30, 180),  # purple
    (240, 50, 230),  # magenta
    (170, 110, 40),  # brown
]

# Names for the display colors, for the printed legend.
DISPLAY_COLOR_NAMES = {
    PAPER_DISPLAY_COLOR: "white",
    INK_DISPLAY_COLOR: "black",
    (230, 25, 75): "red",
    (245, 130, 48): "orange",
    (255, 225, 25): "yellow",
    (60, 180, 75): "green",
    (70, 240, 240): "cyan",
    (0, 130, 200): "blue",
    (145, 30, 180): "purple",
    (240, 50, 230): "magenta",
    (170, 110, 40): "brown",
}


def color_palette(
    rgb: np.ndarray,
    k: int = DEFAULT_NUM_COLORS,
    *,
    lightness_weight: float = LIGHTNESS_WEIGHT,
    sample_size: int = KMEANS_SAMPLE_SIZE,
    seed: int = KMEANS_SEED,
) -> np.ndarray:
    """Fit a ``k``-color palette to an image, as a Kx3 float32 array of LAB centroids.

    Runs k-means in CIELAB on a random subsample of ``sample_size`` pixels (the full
    image is unnecessary and slow), with lightness scaled by ``lightness_weight`` so
    chroma drives the clustering. Returned centroids are in true (unweighted) LAB.
    Seeded for reproducible output.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    if lab.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        lab = lab[rng.choice(lab.shape[0], sample_size, replace=False)]

    weighted = lab.copy()
    weighted[:, 0] *= lightness_weight
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    labels = np.empty((weighted.shape[0], 1), dtype=np.int32)
    _, _, centers = cv2.kmeans(
        np.ascontiguousarray(weighted), k, labels, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers[:, 0] /= lightness_weight
    return centers


def assign_palette(
    lab: np.ndarray,
    centroids: np.ndarray,
    *,
    lightness_weight: float = LIGHTNESS_WEIGHT,
) -> np.ndarray:
    """Index of the nearest centroid for every pixel of a float LAB image (HxW int).

    Distances use the same ``lightness_weight`` the palette was fit with, so assignment
    is consistent with the clustering. Loops over the (few) centroids tracking a running
    minimum, which avoids materializing an HxWxK distance array.
    """
    scale = np.array([lightness_weight, 1.0, 1.0], dtype=np.float32)
    flat = lab.reshape(-1, 3) * scale
    scaled_centroids = centroids * scale
    best_distance = np.full(flat.shape[0], np.inf, dtype=np.float32)
    best_label = np.zeros(flat.shape[0], dtype=np.int32)
    for index, centroid in enumerate(scaled_centroids):
        distance = ((flat - centroid) ** 2).sum(axis=1)
        closer = distance < best_distance
        best_distance[closer] = distance[closer]
        best_label[closer] = index
    return best_label.reshape(lab.shape[:2])


def classify_centroids(centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Category (PAPER / INK / PASTEL) of each palette centroid, as a length-K array.

    A centroid is ink if it is dark. Otherwise it is judged by its a*/b* offset from the
    paper centroid (the most populous non-ink centroid), split into a component along the
    paper's warm/yellow tint axis and one perpendicular to it. Paper shading and staining
    slide the color along that axis, so a centroid is pastel if it deviates off the axis
    (PASTEL_PERP_THRESHOLD, for pink/green), runs far up the axis (PASTEL_WARM_THRESHOLD,
    a yellow wash), or far down it (PASTEL_COOL_THRESHOLD, toward blue).
    """
    counts = np.bincount(labels.reshape(-1), minlength=len(centroids))
    is_ink = centroids[:, 0] < LIGHTNESS_FLOOR
    categories = np.full(len(centroids), PAPER, dtype=np.int8)
    categories[is_ink] = INK
    if is_ink.all():
        return categories

    paper_index = np.where(is_ink, -1, counts).argmax()
    paper_ab = centroids[paper_index, 1:3]
    tint = paper_ab - 128.0  # paper's offset from neutral gray = its warm/yellow axis
    norm = float(np.hypot(*tint))
    axis = tint / norm if norm > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)

    offset = centroids[:, 1:3] - paper_ab
    parallel = offset @ axis
    perpendicular = np.hypot(*(offset - np.outer(parallel, axis)).T)
    is_pastel = ~is_ink & (
        (perpendicular > PASTEL_PERP_THRESHOLD)
        | (parallel > PASTEL_WARM_THRESHOLD)
        | (parallel < -PASTEL_COOL_THRESHOLD)
    )
    categories[is_pastel] = PASTEL
    return categories


def classify_pastel_centroids(centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Boolean mask (length K) of which palette centroids are pastel."""
    return classify_centroids(centroids, labels) == PASTEL


def segment_image(centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Recolor a label map so every pixel takes the RGB color of its palette centroid.

    ``labels`` is an HxW array of centroid indices; the result is an HxWx3 uint8 RGB
    image. This reproduces the page in its own (posterized) colors.
    """
    centroid_lab = np.clip(centroids, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    centroid_rgb = cv2.cvtColor(centroid_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    return centroid_rgb[labels]


def palette_display_hues() -> np.ndarray:
    """Hue angle (radians, in CIELAB a*/b*) of each PASTEL_DISPLAY_COLORS entry."""
    palette = np.array(PASTEL_DISPLAY_COLORS, dtype=np.uint8).reshape(1, -1, 3)
    lab = cv2.cvtColor(palette, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    return np.arctan2(lab[:, 2] - 128, lab[:, 1] - 128)


def cluster_display_colors(centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Display color (Kx3 uint8) for each centroid, one per cluster, to show the partition.

    Exactly one cluster is painted white (the background: the most populous cluster) and
    exactly one black (the darkest cluster); every other cluster gets a distinct palette
    color, chosen as the PASTEL_DISPLAY_COLORS entry nearest its own hue (most saturated
    first, no reuse). Unlike a category-based coloring this makes the raw k-means split
    visible — paper or ink spread across several clusters shows up as several colors.
    """
    counts = np.bincount(labels.reshape(-1), minlength=len(centroids))
    black_index = int(centroids[:, 0].argmin())
    counts_without_black = counts.copy()
    counts_without_black[black_index] = -1
    white_index = int(counts_without_black.argmax())

    colors = np.zeros((len(centroids), 3), dtype=np.uint8)
    colors[black_index] = INK_DISPLAY_COLOR
    colors[white_index] = PAPER_DISPLAY_COLOR

    cluster_hue = np.arctan2(centroids[:, 2] - 128, centroids[:, 1] - 128)
    chroma = np.hypot(centroids[:, 1] - 128, centroids[:, 2] - 128)
    palette_hue = palette_display_hues()
    used: set[int] = set()
    remaining = [
        i for i in range(len(centroids)) if i not in (black_index, white_index)
    ]
    for index in sorted(remaining, key=lambda i: -chroma[i]):
        # Circular hue gap to each display color, wrapped to [0, pi].
        gap = np.abs((cluster_hue[index] - palette_hue + np.pi) % (2 * np.pi) - np.pi)
        ranked = [int(slot) for slot in np.argsort(gap)]
        choice = next((slot for slot in ranked if slot not in used), ranked[0])
        used.add(choice)
        colors[index] = PASTEL_DISPLAY_COLORS[choice]
    return colors


def cluster_segment_image(centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Recolor a label map so each cluster shows in a distinct color (one white, one black)."""
    return cluster_display_colors(centroids, labels)[labels]


def pastel_mask(rgb: np.ndarray, *, k: int = DEFAULT_NUM_COLORS) -> np.ndarray:
    """Boolean mask of the pastel-colored pixels in an RGB image.

    Fits a ``k``-color palette, assigns each pixel to a centroid, and returns the union
    of the pixels whose centroid was classified as pastel.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    centroids = color_palette(rgb, k)
    labels = assign_palette(lab, centroids)
    is_pastel = classify_pastel_centroids(centroids, labels)
    return is_pastel[labels]


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


def margin_pixels(shape: tuple[int, int], margin_fraction: float) -> tuple[int, int]:
    """(margin_y, margin_x) in pixels for an image of the given (height, width)."""
    height, width = shape
    return round(height * margin_fraction), round(width * margin_fraction)


def palette_segmentation(
    rgb: np.ndarray,
    *,
    k: int = DEFAULT_NUM_COLORS,
    margin_fraction: float = MARGIN_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a palette on the page interior and assign every pixel; return centroids, labels.

    The palette is fit on the margin-cropped interior so edge artifacts do not claim a
    cluster, but every pixel (margin included) is assigned for a complete picture.
    ``centroids`` is Kx3 LAB; ``labels`` is an HxW map of centroid indices.
    """
    height, width = rgb.shape[:2]
    margin_y, margin_x = margin_pixels((height, width), margin_fraction)
    interior = rgb[margin_y : height - margin_y, margin_x : width - margin_x]

    centroids = color_palette(interior, k)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    labels = assign_palette(lab, centroids)
    return centroids, labels


def segment_page(
    rgb: np.ndarray,
    *,
    k: int = DEFAULT_NUM_COLORS,
    margin_fraction: float = MARGIN_FRACTION,
) -> np.ndarray:
    """Recolor a page to its own (posterized) palette, ignoring the edge margin."""
    centroids, labels = palette_segmentation(rgb, k=k, margin_fraction=margin_fraction)
    return segment_image(centroids, labels)


def segmentation_legend(centroids: np.ndarray, labels: np.ndarray) -> list[str]:
    """Human-readable lines describing each cluster (coverage, true color, display color).

    Sorted by coverage, descending. Pairs each centroid's true color with the display
    color it is painted in the cluster segmentation image.
    """
    display = cluster_display_colors(centroids, labels)
    centroid_rgb = cv2.cvtColor(
        np.clip(centroids, 0, 255).astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_LAB2RGB
    ).reshape(-1, 3)
    counts = np.bincount(labels.reshape(-1), minlength=len(centroids))
    fractions = counts / counts.sum()

    lines = []
    for index in np.argsort(-counts):
        true_r, true_g, true_b = centroid_rgb[index]
        display_name = DISPLAY_COLOR_NAMES.get(
            tuple(int(c) for c in display[index]), "?"
        )
        lines.append(
            f"  {fractions[index]:5.1%}  "
            f"true rgb({true_r:3d},{true_g:3d},{true_b:3d}) → {display_name}"
        )
    return lines


def detect_pastels(
    rgb: np.ndarray,
    *,
    k: int = DEFAULT_NUM_COLORS,
    margin_fraction: float = MARGIN_FRACTION,
    raw: bool = False,
) -> np.ndarray:
    """Full-size boolean pastel mask, ignoring a margin around the page edge.

    Crops ``margin_fraction`` off every side before fitting the palette, so scan edges,
    binding shadows, and edge stains do not claim a cluster. Detection and (unless
    ``raw``) the morphological cleanup run on the interior; the result is pasted back
    into a full-size mask so its pixels line up one-for-one with ``rgb``, with the
    margin left all False.
    """
    height, width = rgb.shape[:2]
    margin_y, margin_x = margin_pixels((height, width), margin_fraction)
    interior = rgb[margin_y : height - margin_y, margin_x : width - margin_x]

    interior_mask = pastel_mask(interior, k=k)
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
            "Segment key map images into a small color palette and write a sidecar "
            "<stem>.segments.png recolored to that palette."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE", help="Input image files.")
    parser.add_argument(
        "-k",
        "--colors",
        type=int,
        default=DEFAULT_NUM_COLORS,
        help="Number of palette colors (default %(default)s).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=MARGIN_FRACTION,
        help="Fraction of each side to ignore during analysis (default %(default)s).",
    )
    parser.add_argument(
        "--natural",
        action="store_true",
        help="Recolor to each cluster's own (posterized) color instead of the "
        "one-white/one-black/distinct-palette cluster view.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Also write <stem>.pastel.png with detected pastels painted red.",
    )
    args = parser.parse_args()

    for image in args.images:
        image_path = Path(image)
        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"))

        centroids, labels = palette_segmentation(
            rgb, k=args.colors, margin_fraction=args.margin
        )
        if args.natural:
            segments = segment_image(centroids, labels)
        else:
            segments = cluster_segment_image(centroids, labels)
        segments_path = image_path.parent / (image_stem(image) + ".segments.png")
        Image.fromarray(segments).save(segments_path)
        print(f"{image_path} → {segments_path}", file=sys.stderr)
        for line in segmentation_legend(centroids, labels):
            print(line, file=sys.stderr)

        if args.overlay:
            mask = detect_pastels(rgb, k=args.colors, margin_fraction=args.margin)
            overlay_path = image_path.parent / (image_stem(image) + ".pastel.png")
            Image.fromarray(paint_pastels(rgb, mask)).save(overlay_path)
            print(
                f"{image_path} → {overlay_path}  ({mask.mean():.1%} pastel)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
