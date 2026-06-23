"""Find the colored area drawn around each page number on a Sanborn key map.

On a key map every page is a saturated colored block (yellow, green, blue, pink, tan) with
its page number printed in black at the centre. Blocks are separated either by a black
boundary line or by a direct change of colour; thinner black street-grid lines run *inside* a
block. The white/tan paper between blocks is unsaturated.

Recovering one polygon per page is a marker-controlled watershed: each detected page number is
a seed, and the floods from neighbouring seeds meet along the ridges between pages (black
boundaries and colour changes). An internal grid line is crossed freely because no competing
seed sits inside the block, so the whole block becomes one basin. Two equally-coloured blocks
that abut are still separated, because each owns a seed and they meet at the ridge between them.

The elevation surface is the CIELAB gradient, weighting colour changes above black lines, so a
basin stops at a colour boundary but flows across an internal grid line. The unsaturated paper is
given its own marker so floods stop at the block edge (no distance cap, so a tall block fills its
full height). Each page-number seed is its digit's bounding box grown by a margin, so the marker
clears the digit's strokes (e.g. the hole in a "0") instead of getting trapped inside them.

    uv run python -m mapsnap.keymap.page_regions data/chicago_il_1950_vol_1/p0b.keymap.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.color import lab2rgb, rgb2lab
from skimage.filters import sobel
from skimage.segmentation import watershed

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 (full-res pixels)


def polygon_bounds(polygon: list[list[float]]) -> Box:
    """Axis-aligned bounding box (x0, y0, x1, y1) of a polygon's vertices."""
    xs = [vertex[0] for vertex in polygon]
    ys = [vertex[1] for vertex in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def box_center(box: Box) -> Point:
    """Centre point of a bounding box."""
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def load_seeds(keymap_path: Path) -> tuple[list[Box], list[str]]:
    """Page-number bounding boxes and their texts from a ``<stem>.keymap.json``."""
    streets = json.load(open(keymap_path)).get("streets", [])
    boxes: list[Box] = []
    texts: list[str] = []
    for street in streets:
        boxes.append(polygon_bounds(street["polygon"]))
        texts.append(str(street.get("text", "")))
    return boxes, texts


@dataclass
class RegionParams:
    """Tunables for key-map page-region segmentation.

    The image is first downscaled so its longer side is ``target_long_side`` (a key map may be
    anywhere from ~2k to ~7k px). Sizes that should track the map's drawing scale — the line-smooth
    window and the arm-opening radius — are then derived from the median seed spacing rather than
    set in absolute pixels, so the same params work across maps of different resolution and block
    density (Chicago vs New Orleans).

    target_long_side: the segmentation runs on a copy scaled so max(width, height) is this many
        pixels (never upscaled); polygons are returned in full-resolution coordinates.
    bg_lightness / bg_chroma: a pixel is unsaturated "paper" if its CIELAB lightness exceeds
        bg_lightness and its chroma is below bg_chroma.
    n_clusters / cluster_seed: the elevation comes from a k-means colour quantization of the
        (line-smoothed) image; ~8 clusters separate the pastel block colours from the several
        background/paper tints. See elevation_map.
    line_smooth_frac: median-blur window as a fraction of the median seed spacing; it erases black
        lines thinner than the window (so basins flow across grid lines) while preserving block
        colour boundaries.
    smooth_sigma: Gaussian blur (in scaled pixels) applied to the gradient elevation.
    seed_pad_frac: each page-number marker is its digit's bounding box grown by this fraction of
        the box's longer side, so the seed clears the digit's strokes (e.g. the hole in a "0" or
        "8") and starts from the surrounding block colour rather than getting trapped inside it.
    surround_reach_factor: a coloured pixel farther than this many median seed spacings from the
        nearest seed is treated as the map's surround (margin/water) and barriered off, so an edge
        block cannot flood it. Distance is to the nearest *any* seed, so the densely-seeded block
        grid stays under the threshold and tall blocks are not clipped.
    arm_open_frac: morphological-opening disk radius, as a fraction of the median seed spacing,
        applied to each region to delete thin "arms" that flowed down a same-coloured street/margin
        corridor; bigger than a street half-width, far smaller than a block. See open_thin_arms.
    simplify_tolerance: Douglas-Peucker tolerance (in scaled pixels) for the output polygon.
    """

    target_long_side: int = 3000
    bg_lightness: float = 78.0
    bg_chroma: float = 12.0
    n_clusters: int = 8
    cluster_seed: int = 0
    line_smooth_frac: float = 0.04
    smooth_sigma: float = 1.0
    seed_pad_frac: float = 0.3
    surround_reach_factor: float = 1.5
    arm_open_frac: float = 0.12
    simplify_tolerance: float = 2.0


def nearest_neighbor_distance(points: list[Point]) -> float:
    """Median distance from each point to its closest other point.

    A robust estimate of the local block spacing. Returns 0.0 for fewer than two points.
    """
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    nearest: list[float] = []
    for i in range(len(pts)):
        deltas = pts - pts[i]
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        distances[i] = np.inf
        nearest.append(float(distances.min()))
    return float(np.median(nearest))


def background_mask(
    lab: np.ndarray, bg_lightness: float, bg_chroma: float
) -> np.ndarray:
    """Boolean mask of unsaturated "paper" pixels (light and near-grey) in a CIELAB image."""
    lightness = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return (lightness > bg_lightness) & (chroma < bg_chroma)


def quantize_colors(lab: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Replace every pixel with its nearest of ``n_clusters`` k-means colour centres (CIELAB).

    k-means is fit on a random subsample for speed, then every pixel is assigned to its nearest
    centre. Returns a piecewise-constant image the same shape as ``lab``.
    """
    flat = lab.reshape(-1, 3).astype(np.float32)
    generator = np.random.default_rng(seed)
    if len(flat) > 40000:
        sample = flat[generator.choice(len(flat), size=40000, replace=False)]
    else:
        sample = flat
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(
        sample, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = np.zeros(len(flat), dtype=np.int32)
    best = np.full(len(flat), np.inf, dtype=np.float32)
    for index, center in enumerate(centers):
        distance = ((flat - center) ** 2).sum(axis=1)
        closer = distance < best
        best[closer] = distance[closer]
        labels[closer] = index
    return centers[labels].reshape(lab.shape)


def quantized_lab(
    rgb_u8: np.ndarray, n_clusters: int, line_smooth_size: int, seed: int
) -> np.ndarray:
    """Median-blur the image (erasing thin lines) then k-means colour-quantize it, in CIELAB."""
    smoothed = cv2.medianBlur(rgb_u8, line_smooth_size)
    return quantize_colors(rgb2lab(smoothed / 255.0), n_clusters, seed)


def elevation_map(
    rgb_u8: np.ndarray,
    n_clusters: int,
    line_smooth_size: int,
    smooth_sigma: float,
    seed: int,
) -> np.ndarray:
    """Watershed elevation from the boundaries of a colour quantization of the key map.

    A page block's interior carries thin black street-grid lines and the page number; a block
    *boundary* is a colour change or a colour-to-paper edge. A plain gradient can't tell them
    apart — a black line is a strong lightness edge that would dam a basin inside its own block,
    while the boundary between a *pale* block and a similar-coloured street is a weak ridge a basin
    floods across as an "arm".

    Two steps fix this. (1) Median-blur the image (``line_smooth_size``): a line thinner than the
    window is the local minority and is replaced by the surrounding block colour, so internal grid
    lines vanish. (2) k-means quantize the blurred colours into ``n_clusters`` and take the gradient
    of the quantized image — piecewise-constant, so it is flat inside a cluster (grid lines, now the
    block colour, are crossable) and rises at every cluster boundary by the CIELAB distance between
    the two centres. That distance is a clean step even for two pale-but-distinct colours, so the
    pale-block/street boundary becomes a real ridge instead of a weak gradient.
    """
    quant = quantized_lab(rgb_u8, n_clusters, line_smooth_size, seed)
    gradient = (
        sobel(quant[..., 0] / 100.0)
        + sobel(quant[..., 1] / 128.0)
        + sobel(quant[..., 2] / 128.0)
    )
    return ndi.gaussian_filter(gradient, smooth_sigma)


def mask_to_polygon(mask: np.ndarray, simplify_tolerance: float) -> list[Point]:
    """Simplified outer-contour polygon of the largest blob in a boolean mask.

    Returns the Douglas-Peucker-simplified vertices (pixel coords) of the biggest connected
    component, or an empty list if the mask is empty.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, simplify_tolerance, closed=True)
    return [(float(x), float(y)) for [[x, y]] in approx]


def open_thin_arms(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening that deletes protrusions narrower than ``2 * radius`` pixels.

    A basin sometimes flows down a same-coloured street/margin corridor as a thin "arm" (the
    corridor is walled by colour ridges on its long sides, so no neighbour contests it). Opening
    with a disk erodes such an arm away (it is thinner than the disk) while a compact block, far
    wider than the disk, is preserved bar some corner rounding. Returns ``mask`` unchanged if the
    opening would erase it entirely (a genuinely thin block), so a region is never lost.
    """
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened.astype(bool) if opened.any() else mask


def keep_seed_component(basin: np.ndarray) -> np.ndarray:
    """Largest connected component of a basin mask, with interior holes filled."""
    filled: np.ndarray = ndi.binary_fill_holes(basin)  # type: ignore[assignment]
    labels, count = ndi.label(filled)  # type: ignore[misc]
    if count <= 1:
        return filled
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background
    return labels == int(np.argmax(sizes))


def stamp_seed_markers(
    shape: tuple[int, int], boxes: list[Box], scale: float, pad_frac: float
) -> np.ndarray:
    """Marker image: label 1 is unset, each seed box (grown by ``pad_frac``) gets label index+2.

    Two passes so dense digits cannot erase each other: first the padded boxes (overlaps go to
    whichever is stamped last), then the tight boxes, guaranteeing every digit keeps its own core.
    """
    markers = np.zeros(shape, dtype=np.int32)
    height, width = shape

    def stamp(box: Box, label: int, pad: float) -> None:
        x0, y0, x1, y1 = box
        col0 = max(0, int(round((x0 - pad) * scale)))
        row0 = max(0, int(round((y0 - pad) * scale)))
        col1 = min(width, int(round((x1 + pad) * scale)) + 1)
        row1 = min(height, int(round((y1 + pad) * scale)) + 1)
        markers[row0:row1, col0:col1] = label

    for index, box in enumerate(boxes, start=2):
        stamp(box, index, pad_frac * max(box[2] - box[0], box[3] - box[1]))
    for index, box in enumerate(boxes, start=2):
        stamp(box, index, 0.0)
    return markers


def working_scale(image_shape: tuple[int, ...], target_long_side: int) -> float:
    """Downscale factor so the image's longer side is ``target_long_side`` (never upscales)."""
    height, width = image_shape[:2]
    return min(1.0, target_long_side / max(height, width))


def segment_page_regions(
    rgb: np.ndarray, seeds: list[Box], params: RegionParams
) -> dict[int, list[Point]]:
    """Polygon (full-res pixels) of the colored block around each seeded page number.

    rgb is a float image in [0, 1] (H x W x 3). ``seeds`` are page-number bounding boxes in
    full-resolution coordinates. Each block grows by watershed from its (padded) digit box until
    it meets a neighbouring page or the unsaturated paper between blocks; there is no distance
    cap, so a tall block fills its full height. Returns a map from seed index to its block
    polygon; a seed whose block could not be recovered is omitted.
    """
    scale = working_scale(rgb.shape, params.target_long_side)
    height, width = rgb.shape[:2]
    scaled = cv2.resize(
        rgb, (max(1, round(width * scale)), max(1, round(height * scale)))
    )
    scaled_u8 = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    paper = background_mask(rgb2lab(scaled), params.bg_lightness, params.bg_chroma)

    # Sizes that track the map's drawing scale are derived from the median seed spacing, so the
    # same params transfer across maps of different resolution and block density.
    spacing = nearest_neighbor_distance([box_center(box) for box in seeds]) * scale
    line_smooth_size = max(3, int(round(params.line_smooth_frac * spacing)) | 1)
    arm_open_radius = round(params.arm_open_frac * spacing)
    elevation = elevation_map(
        scaled_u8,
        params.n_clusters,
        line_smooth_size,
        params.smooth_sigma,
        params.cluster_seed,
    )

    markers = stamp_seed_markers(scaled.shape[:2], seeds, scale, params.seed_pad_frac)
    markers[paper & (markers == 0)] = 1  # unsaturated paper is its own (non-page) basin
    markers[0, :] = markers[-1, :] = markers[:, 0] = markers[:, -1] = (
        1  # frame is paper too
    )

    # Barrier off the map's coloured surround (margin/water): coloured pixels too far from any
    # seed to belong to a page block. Distance is to the nearest seed, so the densely-seeded grid
    # stays below the threshold (a block's far end is near its neighbour's seed) and isn't clipped.
    if spacing > 0:
        distance_to_seed = ndi.distance_transform_edt(markers < 2)
        surround = (distance_to_seed > params.surround_reach_factor * spacing) & ~paper
        markers[surround & (markers == 0)] = 1

    basins = watershed(elevation, markers)

    inverse_scale = 1.0 / scale
    polygons: dict[int, list[Point]] = {}
    for index in range(2, len(seeds) + 2):
        basin = (basins == index) & ~paper
        if not basin.any():
            continue
        basin = open_thin_arms(basin, arm_open_radius)
        component = keep_seed_component(basin)
        polygon = mask_to_polygon(component, params.simplify_tolerance)
        if len(polygon) >= 3:
            polygons[index - 2] = [
                (x * inverse_scale, y * inverse_scale) for x, y in polygon
            ]
    return polygons


def keymap_image_path(keymap_path: Path) -> Path:
    """Sibling JPEG of a ``<stem>.keymap.json`` (e.g. ``p0b.keymap.json`` -> ``p0b.jpg``)."""
    name = keymap_path.name
    suffix = ".keymap.json"
    stem = name[: -len(suffix)] if name.endswith(suffix) else keymap_path.stem
    return keymap_path.with_name(stem + ".jpg")


def render_overlay(
    image_path: Path, polygons: dict[int, list[Point]], texts: list[str], output: Path
) -> None:
    """Draw each segmented region (and its page number) on a downscaled copy of the key map."""
    image = Image.open(image_path).convert("RGB")
    view_scale = 1200 / max(image.size)
    image = image.resize(
        (round(image.width * view_scale), round(image.height * view_scale))
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rng = np.random.default_rng(0)
    for index, polygon in polygons.items():
        color = tuple(int(v) for v in rng.integers(40, 230, size=3))
        points = [(x * view_scale, y * view_scale) for x, y in polygon]
        draw.polygon(points, fill=color + (90,), outline=color + (255,))
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        draw.text((cx, cy), texts[index], fill=(0, 0, 0, 255))
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output)


def regions_panels_doc(
    image_name: str,
    size: tuple[int, int],
    polygons: dict[int, list[Point]],
    texts: list[str],
) -> dict:
    """A panels.json sidecar for the detected regions, for the debugger app to display.

    Matches the existing panels.json schema (image / width / height / panels-as-polygon-rings)
    and adds a parallel ``labels`` array giving each panel's page number, so the app can show the
    page number instead of a positional index. ``size`` is (width, height) in full-res pixels.
    """
    width, height = size
    panels: list[list[list[float]]] = []
    labels: list[str] = []
    for index, polygon in polygons.items():
        panels.append([[round(x, 1), round(y, 1)] for x, y in polygon])
        labels.append(texts[index])
    return {
        "image": image_name,
        "width": width,
        "height": height,
        "panels": panels,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "keymap", type=Path, help="<stem>.keymap.json with page-number detections."
    )
    parser.add_argument(
        "--image", type=Path, help="Key-map JPEG (default: sibling of the JSON)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Panels JSON for the debugger (default: <stem>.regions.panels.json).",
    )
    parser.add_argument(
        "--overlay", type=Path, help="Also write a labelled overlay PNG to this path."
    )
    parser.add_argument(
        "--blur-debug",
        type=Path,
        help="Also write the median-blurred image the elevation is built from to this path.",
    )
    parser.add_argument(
        "--cluster-debug",
        type=Path,
        help="Also write the k-means colour-quantized image (the elevation's basis) to this path.",
    )
    args = parser.parse_args()

    image_path = args.image or keymap_image_path(args.keymap)
    output = args.output or args.keymap.with_name(
        image_path.stem + ".regions.panels.json"
    )

    params = RegionParams()
    seeds, texts = load_seeds(args.keymap)
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    polygons = segment_page_regions(rgb.astype(np.float64) / 255.0, seeds, params)
    doc = regions_panels_doc(image_path.name, image.size, polygons, texts)
    output.write_text(json.dumps(doc))
    print(f"Wrote {output}: {len(polygons)}/{len(seeds)} regions found.")
    if args.overlay:
        render_overlay(image_path, polygons, texts, args.overlay)
        print(f"Wrote {args.overlay}")
    if args.blur_debug or args.cluster_debug:
        scale = working_scale(rgb.shape, params.target_long_side)
        scaled = cv2.resize(
            rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
        )
        spacing = nearest_neighbor_distance([box_center(b) for b in seeds]) * scale
        line_smooth_size = max(3, int(round(params.line_smooth_frac * spacing)) | 1)
        if args.blur_debug:
            Image.fromarray(cv2.medianBlur(scaled, line_smooth_size)).save(
                args.blur_debug
            )
            print(f"Wrote {args.blur_debug}")
        if args.cluster_debug:
            quant = quantized_lab(
                scaled, params.n_clusters, line_smooth_size, params.cluster_seed
            )
            quant_rgb = (np.clip(lab2rgb(quant), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(quant_rgb).save(args.cluster_debug)
            print(f"Wrote {args.cluster_debug}")


if __name__ == "__main__":
    main()
