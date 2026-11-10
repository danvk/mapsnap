"""Train the key-map region UNet (mapsnap.keymap.region_unet) on OIM-derived labels.

Every OldInsuranceMaps-georeferenced page of a volume has a footprint polygon; the
key map's own georeference (``<key>.georef.json``) maps it into key-map pixels. The
labels are the *drawn* blocks, the colored fills a tracer would outline, and the
projected footprints only say which block is which page, with corrections for what
they are not:

* A key map is a stylized drawing, so footprints land 0.05-0.3 spacings off the blocks
  and not uniformly across the sheet (a better georef does not help: Chicago's right
  edge is 50 px off under two independent fits). Each footprint is first moved by the
  local median offset between footprints and page_regions' same-numbered polygons.
* A footprint reaches the street centerline; the drawn block stops at the curb. Each
  footprint is shrunk to a *core* and a marker watershed on an edge-strength map (ink
  darkness and color change) grows it back to the drawn line. Paper connected to the
  open sheet is a background marker even between two blocks, so a colored block's flood
  stops at its color edge and the street between two blocks stays paper. (Hand traces
  showed an earlier version without that marker labeling streets as block interior,
  and the model learned it: regions ran to the street centerline.)
* Where page_regions' polygon sits inside its footprint and covers most of it
  (SHARP_PRECISION, SHARP_COVERAGE), it follows the ink exactly and replaces the
  footprint as a sharp core.
* Blocks nobody labeled (pages OIM never georeferenced or did not clip, the volumes
  inset, and detected numbers with no footprint) are *ignored*, not background: an
  unlabeled block taught as paper would teach the model to miss blocks.

Hand-traced blocks (the regions labeler, ``data/<vol>/raw/truth/<key>.regions.panels.json``)
are the evaluation that matters: ``eval`` and the per-epoch validation score against
them wherever they exist, beside the footprint-based score.

    uv run python -m mapsnap.keymap.train_region_unet prep            # labels -> training dir
    uv run python -m mapsnap.keymap.train_region_unet relabel         # new label recipe, same examples
    uv run python -m mapsnap.keymap.train_region_unet train           # -> models/keymap_region_unet.pt
    uv run python -m mapsnap.keymap.train_region_unet eval            # held-out IoU vs k-means and hand traces
"""

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import KDTree
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid
from skimage.segmentation import watershed

from mapsnap.compare_iiif_georef import (
    label_split_index,
    parse_svg_polygon,
    truth_page_number,
    truth_polygon_world,
)
from mapsnap.fix_truth_splits import gcp_containment
from mapsnap.keymap.page_regions import (
    Point,
    nearest_neighbor_distance,
    regions_panels_doc,
)
from mapsnap.keymap.region_unet import (
    BACKGROUND,
    BOUNDARY,
    BOUNDARY_WIDTH_PX,
    IGNORE,
    INTERIOR,
    MIN_SEEDS_FOR_SPACING,
    MODEL_PATH,
    TARGET_SPACING_PX,
    TILE_SIZE,
    UNSEEDED_LABEL,
    DecodeParams,
    KeymapSources,
    build_model,
    decode_regions,
    discover_keymap_sources,
    load_region_unet,
    model_input,
    predict_class_maps,
    read_rgb,
    region_polygons,
    resize_image,
    scale_points,
    seed_points,
    select_device,
    working_scale_for_seeds,
)
from mapsnap.keymap.score_regions import (
    RegionScore,
    panel_polygons,
    scale_to,
    score_regions,
)
from mapsnap.keymap.truth_regions import MIN_INSIDE_FRACTION, world_to_pixel_affine
from mapsnap.road_model import UNet

TRAINING_DIR = Path.home() / "Documents" / "mapsnap" / "keymap-region-training"
# Held out of training. The hand-traced volumes are the real benchmark, so none of
# them may be trained on.
DEFAULT_VAL_VOLUMES = (
    "hudson_co_nj_1950_vol_9",  # pale fills, where k-means loses blocks to paper
    "chicago_il_1950_vol_1",  # dense, colorful
    "miami_fl_1950_vol_1",  # faint, foxed; hand-traced
    "san-antonio_texas_1952_sanborn08740_020",  # corpus-style low-resolution scan
    "kansas_city_mo_1951_vol_4",  # hand-traced
    "grand_rapids_mi_1953_vol7",  # hand-traced
    "brooklyn_ny_1939_vol_1",  # hand-traced
)
# Validation picks the best epoch by the hand-traced score when at least this many
# validation maps have traces, else by the footprint score.
MIN_HAND_VAL_MAPS = 3

# Label construction, in working pixels unless noted.
BAND_FRAC = 0.15  # truth-edge uncertainty band as a fraction of the seed spacing
MIN_BAND_PX = 4
# A color polygon is the exact block when it lies inside its footprint and covers most
# of it: a footprint reaches the street centerline, so IoU alone would reject correct
# blocks on wide-street maps.
SHARP_PRECISION = 0.9
SHARP_COVERAGE = 0.55
SHARP_OTHER_OVERLAP = (
    0.1  # ... unless it covers more than this of another page's footprint
)
# Footprint shift: a local median over the nearest same-numbered color polygons.
SHIFT_NEIGHBORS = 6
SHIFT_MIN_NEAR = 3
SHIFT_MIN_SAMPLES = 4
SHIFT_RADIUS_FRAC = 3.0  # of the spacing: how far a sample may be to count as local
SHIFT_CANDIDATE_AREA = (
    0.3,
    1.6,
)  # color polygon area / footprint area, else not a sample
SHIFT_MAX_FRAC = 0.6  # of the spacing: a larger shift is noise, clipped
# A block counts as a filled block when this share of its core is not paper-like; only
# then may paper inside its band be a background marker (an outline-only block is
# paper inside). Miami's fills are so pale that half of them fail the colored test
# against the margin paper, but almost none of their pixels pass as paper.
COLORED_BLOCK_FRACTION = 0.5
PAPER_MIN_LIGHTNESS = 60.0
PAPER_CHROMA_PERCENTILE = 90  # of the paper's own chroma spread, the paper-like cutoff
PAPER_CHROMA_FLOOR = 4.0
# Paper that is cut off from the open sheet (by ink lines, stains or bleed) still marks
# background when the pocket is at least this share of spacing²; smaller enclosed pockets
# are taken for speckles in a pale fill. None keeps only connected paper.
OPEN_PAPER_POCKET_FRAC: float | None = 0.005
# Drop paper-like pixels from a filled block's core: a footprint that overshoots its
# drawn block onto the street must not mark the street as block.
CLIP_CORES_TO_FILL = True
SHARP_BAND_PX = 3
WEAK_EDGE = (
    0.15  # edge strength below which a snapped boundary is unsupported (ignored)
)
UNCLIPPED_FRACTION = (
    0.95  # selector area / sheet area at or above this: volunteers didn't clip
)
MAX_UNCLIPPED_SHARE = 0.3  # drop a key map whose truth is mostly unclipped
MIN_TRUTH_FOOTPRINTS = 8
MAX_FOOTPRINT_AREA_RATIO = 20.0  # vs the median footprint: larger is a broken truth fit
# A pixel whose Lab chroma differs from the sheet's paper by more than this, and is not
# ink-dark, is a colored fill; outside every footprint that means an unlabeled block.
UNLABELED_CHROMA_DISTANCE = 12.0
COLORED_MIN_LIGHTNESS = 40.0
UNLABELED_SEED_RADIUS_FRAC = (
    0.5  # ignore disk around a detected number with no footprint
)

# Training.
# Labeled pixels are 82% background, 17% interior and 1.4% boundary. Weighted
# cross-entropy alone left the model conservative about interior on pale fills
# (precision 0.9, recall 0.3-0.6 on held-out maps), so interior is weighted up and
# gets a Dice term of its own, which does not care about class balance.
CLASS_WEIGHTS = (1.0, 2.5, 3.0)
DICE_CLASSES = (INTERIOR, BOUNDARY)
DICE_WEIGHT = 1.0
SCALE_JITTER = (0.85, 1.15)
# This share of tiles is centered on a page number (jittered), the rest placed at
# random: blocks are a small part of a pale corpus sheet, and uniform placement
# shows the model mostly paper there.
SEED_CENTERED_TILE_SHARE = 0.6
TTA_FLIPS = (
    True  # flip-averaged prediction in validation and eval (--no-flips turns it off)
)
SEED_CENTER_JITTER = 0.3  # of the window, each axis


@dataclass
class TruthFootprint:
    """A labeled page block in key-map pixels."""

    label: str
    ring: list[Point]


@dataclass
class ProjectedTruth:
    """The footprints a volume's truth projects onto one key map, with what was skipped."""

    footprints: list[TruthFootprint] = field(default_factory=list)
    unclipped: int = 0
    total: int = 0


def selector_fraction(item: dict) -> float | None:
    """Selector area over the annotation's whole sheet, or None without a usable selector."""
    selector = (item.get("target") or {}).get("selector") or {}
    points = parse_svg_polygon(selector.get("value", ""))
    source = item["target"]["source"]
    width, height = float(source.get("width") or 0), float(source.get("height") or 0)
    if len(points) < 3 or width <= 0 or height <= 0:
        return None
    if label_split_index(item) is not None and gcp_containment(item, points) < 0.5:
        return None  # a split selector written in the crop's frame (OIM#402)
    return make_valid(Polygon(points)).area / (width * height)


def project_truth(truth_path: Path, georef: dict) -> ProjectedTruth:
    """Truth footprints in the key map's full-resolution pixels, skipping unclipped ones."""
    affine = world_to_pixel_affine(georef)
    width, height = georef["width"], georef["height"]
    image_box = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    projected = ProjectedTruth()
    for item in json.loads(truth_path.read_text()).get("items", []):
        number = truth_page_number(item)
        world = truth_polygon_world(item)
        fraction = selector_fraction(item)
        if number is None or world is None or fraction is None:
            continue
        projected.total += 1
        if fraction >= UNCLIPPED_FRACTION:
            projected.unclipped += 1
            continue
        points = np.hstack(
            [np.asarray(world, dtype=np.float64), np.ones((len(world), 1))]
        )
        pixels = points @ affine.T
        polygon = make_valid(Polygon((float(x), float(y)) for x, y in pixels))
        if polygon.area <= 0:
            continue
        if polygon.intersection(image_box).area < MIN_INSIDE_FRACTION * polygon.area:
            continue
        ring = [(float(x), float(y)) for x, y in pixels]
        projected.footprints.append(TruthFootprint(str(number), ring))
    areas = [polygon_area(footprint.ring) for footprint in projected.footprints]
    if areas:
        median = float(np.median(areas))
        projected.footprints = [
            footprint
            for footprint, area in zip(projected.footprints, areas)
            if area <= MAX_FOOTPRINT_AREA_RATIO * median
        ]
    return projected


def polygon_area(ring: list[Point]) -> float:
    """Area of a ring (shoelace), as a float."""
    return float(abs(make_valid(Polygon(ring)).area))


def panels_doc_rings(doc: dict, scale: float) -> list[tuple[str, list[Point]]]:
    """(label, ring) pairs of a panels.json document, rescaled by ``scale``."""
    labels = doc.get("labels") or [""] * len(doc["panels"])
    return [
        (str(label), [(x * scale, y * scale) for x, y in ring])
        for ring, label in zip(doc["panels"], labels)
        if len(ring) >= 3
    ]


def rasterize(rings: list[list[Point]], shape: tuple[int, int]) -> np.ndarray:
    """Boolean HxW mask of the union of the rings."""
    mask = np.zeros(shape, np.uint8)
    polygons = [
        np.round(np.asarray(ring, np.float64)).astype(np.int32)
        for ring in rings
        if len(ring) >= 3
    ]
    if polygons:
        cv2.fillPoly(mask, polygons, 1)
    return mask.astype(bool)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """Float32 CIE Lab (L 0-100, a/b about -128..127) of a uint8 RGB image, via OpenCV.

    scikit-image's conversion segfaults in its XYZ matmul on 10-megapixel sheets (seen
    on four maps in the corpus pull); OpenCV agrees with it to within half a unit.
    """
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def edge_strength(lab: np.ndarray) -> np.ndarray:
    """Per-pixel evidence of a block boundary in [0, 1], from the image in float32 Lab.

    A heavy black line is dark; a boundary drawn as a change of fill has a strong Lab
    gradient; the thin street grid inside a block is, after downscaling, neither.
    """
    lightness = lab[..., 0]
    darkness = np.clip((55.0 - lightness) / 45.0, 0.0, 1.0)
    gradient = np.zeros_like(lightness)
    for channel, divisor in (
        (lab[..., 0], 100.0),
        (lab[..., 1], 128.0),
        (lab[..., 2], 128.0),
    ):
        scaled = channel / divisor
        gx = cv2.Sobel(scaled, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(scaled, cv2.CV_32F, 0, 1, ksize=3)
        gradient += gx * gx + gy * gy
    gradient = np.sqrt(gradient)
    high = float(np.percentile(gradient, 99)) or 1.0
    gradient = np.clip(gradient / high, 0.0, 1.0)
    strength = np.maximum(darkness, gradient)
    return cv2.GaussianBlur(strength, (0, 0), 1.0)


def paper_color(lab: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Median Lab of the light pixels in ``background``: the sheet's paper tint."""
    light = lab[..., 0] > 60
    sample = lab[background & light]
    if len(sample) == 0:
        sample = lab[light]
    if len(sample) == 0:
        return np.array([95.0, 0.0, 0.0], np.float32)
    return np.median(sample, axis=0)


def colored_mask(lab: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Pixels filled with a color other than the paper's: chroma far from the paper's, not ink.

    Aged paper is itself yellowish, so the test is relative to the paper's chroma; ink
    is excluded by lightness before smoothing so text does not smear into a halo.
    """
    distance = np.hypot(lab[..., 1] - paper[1], lab[..., 2] - paper[2])
    colored = (distance > UNLABELED_CHROMA_DISTANCE) & (
        lab[..., 0] > COLORED_MIN_LIGHTNESS
    )
    smooth = cv2.GaussianBlur(colored.astype(np.float32), (0, 0), 2.0)
    return smooth > 0.5


def paper_mask(
    lab: np.ndarray, paper: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Pixels that look like the sheet's paper: light, with chroma within the paper's own spread.

    The spread is measured on ``background`` (the estimated paper), so a foxed sheet
    gets a looser test than a clean one, while pale fills farther from the paper than
    its own variation stay out.
    """
    distance = np.hypot(lab[..., 1] - paper[1], lab[..., 2] - paper[2])
    light = lab[..., 0] >= PAPER_MIN_LIGHTNESS
    sample = distance[background & light]
    spread = (
        float(np.percentile(sample, PAPER_CHROMA_PERCENTILE))
        if sample.size
        else PAPER_CHROMA_FLOOR
    )
    threshold = float(np.clip(spread, PAPER_CHROMA_FLOOR, UNLABELED_CHROMA_DISTANCE))
    return light & (distance <= threshold)


def open_paper(
    paper_like: np.ndarray, outside: np.ndarray, min_pocket_px: float | None = None
) -> np.ndarray:
    """Paper-like pixels connected, through paper-like pixels, to the paper in ``outside``.

    Streets reach the sheet's margins; a pale speckle inside a block does not. The mask
    is eroded by a pixel first so a one-pixel gap in a block's outline is not a leak.
    With ``min_pocket_px``, enclosed paper of at least that many pixels counts too: on
    a stained or ink-dense sheet the streets break into cells that never reach the
    margin, while fill speckles stay small.
    """
    eroded = cv2.erode(paper_like.astype(np.uint8), np.ones((3, 3), np.uint8))
    count, components = cv2.connectedComponents(eroded, connectivity=4)
    labeled = np.asarray(components, dtype=np.int32)
    touching = np.unique(labeled[outside & (eroded > 0)])
    keep = np.zeros(int(count) + 1, bool)
    keep[touching[touching > 0]] = True
    if min_pocket_px is not None:
        sizes = np.bincount(labeled.ravel(), minlength=int(count) + 1)
        keep |= sizes >= min_pocket_px
    keep[0] = False
    return keep[labeled]


@dataclass
class LabelStats:
    """What the label builder did on one key map, for the prep log."""

    footprints: int
    sharp: int  # footprints replaced by an agreeing color polygon
    band_px: int
    shift_px: float = 0.0  # median footprint shift onto the drawn blocks
    colored: int = 0  # footprints whose block is a colored fill


def truth_has_letters(labels: list[str]) -> bool:
    """Whether any truth label carries a letter suffix (``9A``), so suffixes are meaningful."""
    return any(re.search(r"[A-Za-z]", label) for label in labels)


def matching_label(label: str, keep_letters: bool) -> str:
    """The key a predicted label matches truth under.

    OIM's page numbers drop the division letters some key maps print (Chicago's
    ``63W`` is page 63 in the truth), so unless the truth itself uses letters the
    comparison is on the digits.
    """
    label = label.upper()
    return label if keep_letters else re.sub(r"[A-Z]+$", "", label)


def labels_matched_to_truth(predicted_doc: dict, truth_doc: dict) -> dict:
    """A copy of the predicted doc whose labels are rewritten to match the truth's convention."""
    keep = truth_has_letters([str(label) for label in truth_doc["labels"]])
    doc = dict(predicted_doc)
    doc["labels"] = [
        matching_label(str(label), keep) for label in predicted_doc["labels"]
    ]
    return doc


def color_polygons_by_label(
    truth: list[TruthFootprint], color_regions: list[tuple[str, list[Point]]]
) -> dict[str, list[BaseGeometry]]:
    """page_regions' polygons keyed the way the truth labels its pages (``63W`` -> ``63``)."""
    keep = truth_has_letters([footprint.label for footprint in truth])
    by_label: dict[str, list[BaseGeometry]] = {}
    for label, ring in color_regions:
        key = matching_label(label, keep)
        by_label.setdefault(key, []).append(make_valid(Polygon(ring)))
    return by_label


def footprint_shifts(
    truth: list[TruthFootprint],
    color_regions: list[tuple[str, list[Point]]],
    spacing: float,
) -> list[Point]:
    """Per-footprint (dx, dy) moving it onto the drawn blocks: a local median of color-polygon offsets.

    Where page_regions found a polygon with the footprint's page number, overlapping
    it at a plausible size, the centroid offset samples the drawing's local error.
    Every footprint, with or without a sample of its own, takes the median of the
    nearest samples, so one wrong polygon moves nothing much and a block the color
    segmenter missed is still moved with its neighbors.
    """
    truth_polygons = [make_valid(Polygon(footprint.ring)) for footprint in truth]
    keep = truth_has_letters([footprint.label for footprint in truth])
    by_label = color_polygons_by_label(truth, color_regions)
    samples: list[tuple[float, float, float, float]] = []
    for footprint, polygon in zip(truth, truth_polygons):
        if polygon.area <= 0:
            continue
        best: tuple[float, BaseGeometry] | None = None
        for candidate in by_label.get(matching_label(footprint.label, keep), []):
            ratio = candidate.area / polygon.area
            if not SHIFT_CANDIDATE_AREA[0] <= ratio <= SHIFT_CANDIDATE_AREA[1]:
                continue
            overlap = candidate.intersection(polygon).area
            if overlap > 0 and (best is None or overlap > best[0]):
                best = (overlap, candidate)
        if best is not None:
            center, truth_center = best[1].centroid, polygon.centroid
            samples.append(
                (
                    truth_center.x,
                    truth_center.y,
                    center.x - truth_center.x,
                    center.y - truth_center.y,
                )
            )
    if len(samples) < SHIFT_MIN_SAMPLES:
        return [(0.0, 0.0)] * len(truth)
    table = np.asarray(samples, dtype=np.float64)
    tree = KDTree(table[:, :2])
    limit = SHIFT_MAX_FRAC * spacing
    shifts: list[Point] = []
    for polygon in truth_polygons:
        if polygon.area <= 0:
            shifts.append((0.0, 0.0))
            continue
        center = polygon.centroid
        distances, indexes = tree.query(
            [center.x, center.y], k=min(SHIFT_NEIGHBORS, len(table))
        )
        distances, indexes = np.atleast_1d(distances), np.atleast_1d(indexes)
        near = indexes[distances <= SHIFT_RADIUS_FRAC * spacing]
        if len(near) < SHIFT_MIN_NEAR:
            near = np.arange(len(table))  # too sparse here: the sheet-wide median
        dx = float(np.clip(np.median(table[near, 2]), -limit, limit))
        dy = float(np.clip(np.median(table[near, 3]), -limit, limit))
        shifts.append((dx, dy))
    return shifts


def shifted_footprints(
    truth: list[TruthFootprint], shifts: list[Point]
) -> list[TruthFootprint]:
    """The footprints translated by their shifts."""
    return [
        TruthFootprint(footprint.label, [(x + dx, y + dy) for x, y in footprint.ring])
        for footprint, (dx, dy) in zip(truth, shifts)
    ]


def sharp_cores(
    truth: list[TruthFootprint], color_regions: list[tuple[str, list[Point]]]
) -> dict[int, list[Point]]:
    """Truth index -> color polygon that agrees with it well enough to be the exact block."""
    truth_polygons = [make_valid(Polygon(footprint.ring)) for footprint in truth]
    keep = truth_has_letters([footprint.label for footprint in truth])
    by_label = color_polygons_by_label(truth, color_regions)
    cores: dict[int, list[Point]] = {}
    for index, footprint in enumerate(truth):
        for candidate in by_label.get(matching_label(footprint.label, keep), []):
            if candidate.area <= 0 or truth_polygons[index].area <= 0:
                continue
            overlap = truth_polygons[index].intersection(candidate).area
            if (
                overlap / candidate.area < SHARP_PRECISION
                or overlap / truth_polygons[index].area < SHARP_COVERAGE
            ):
                continue
            others = sum(
                candidate.intersection(other).area
                for j, other in enumerate(truth_polygons)
                if j != index
            )
            if others > SHARP_OTHER_OVERLAP * candidate.area:
                continue
            exterior = getattr(candidate, "exterior", None)
            if exterior is None:  # a MultiPolygon after repair: take its largest part
                parts = sorted(getattr(candidate, "geoms", []), key=lambda g: g.area)
                if not parts:
                    continue
                exterior = parts[-1].exterior
            cores[index] = [(float(x), float(y)) for x, y in exterior.coords]
            break
    return cores


@dataclass
class LabelInputs:
    """Everything the label builder needs, in working pixels."""

    rgb: np.ndarray
    truth: list[TruthFootprint]
    color_regions: list[tuple[str, list[Point]]]
    inset_rings: list[list[Point]]
    seeds: list[Point]
    spacing: float


@dataclass
class FootprintMarker:
    """One truth footprint's core and the band it may claim, on its bounding-box window."""

    rows: slice
    cols: slice
    core: np.ndarray  # boolean, window-sized
    allowed: np.ndarray  # boolean, window-sized
    sharp: bool


def ring_window(
    ring: list[Point], shape: tuple[int, int], margin: int
) -> tuple[slice, slice]:
    """Row and column slices of the ring's bounding box grown by ``margin``, clipped to the image."""
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    row0 = max(0, math.floor(min(ys)) - margin)
    row1 = min(shape[0], math.ceil(max(ys)) + margin + 1)
    col0 = max(0, math.floor(min(xs)) - margin)
    col1 = min(shape[1], math.ceil(max(xs)) + margin + 1)
    return slice(row0, max(row0 + 1, row1)), slice(col0, max(col0 + 1, col1))


def window_mask(ring: list[Point], rows: slice, cols: slice) -> np.ndarray:
    """The ring rasterized into a window-sized boolean mask."""
    shifted = [(x - cols.start, y - rows.start) for x, y in ring]
    return rasterize([shifted], (rows.stop - rows.start, cols.stop - cols.start))


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion by a disk, on a small window (cv2, so it's fast)."""
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.erode(mask.astype(np.uint8), element).astype(bool)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a disk, on a small window."""
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), element).astype(bool)


def footprint_marker(
    ring: list[Point],
    sharp_ring: list[Point] | None,
    band: int,
    shape: tuple[int, int],
) -> FootprintMarker:
    """Core and allowed masks for a footprint: tight around a sharp color polygon, else banded."""
    if sharp_ring is not None:
        rows, cols = ring_window(sharp_ring, shape, SHARP_BAND_PX + 1)
        sharp = window_mask(sharp_ring, rows, cols)
        core = erode(sharp, SHARP_BAND_PX)
        if not core.any():
            core = sharp
        return FootprintMarker(rows, cols, core, dilate(sharp, SHARP_BAND_PX), True)
    rows, cols = ring_window(ring, shape, band + 1)
    mask = window_mask(ring, rows, cols)
    core = erode(mask, band)
    if not core.any():
        core = erode(mask, max(1, band // 2))
    if not core.any():
        core = mask
    return FootprintMarker(rows, cols, core, dilate(mask, band), False)


def neighbor_differs(labels: np.ndarray) -> np.ndarray:
    """Pixels whose right or lower 4-neighbor carries a different label (both sides marked)."""
    edge = np.zeros(labels.shape, bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge


def build_label_image(inputs: LabelInputs) -> tuple[np.ndarray, LabelStats]:
    """The uint8 class image (BACKGROUND / INTERIOR / BOUNDARY / IGNORE) for one key map.

    See the module docstring for the construction. The watershed runs over the whole
    sheet with one core marker per (shifted) footprint and a background marker on the
    paper outside every footprint's band plus the open paper inside the bands of
    colored blocks; a footprint keeps only the pixels it won inside its own band, and
    its won pixels outside the band are ignored.
    """
    shape = inputs.rgb.shape[:2]
    band = max(MIN_BAND_PX, round(BAND_FRAC * inputs.spacing))
    lab = to_lab(inputs.rgb)
    strength = edge_strength(lab)
    shifts = footprint_shifts(inputs.truth, inputs.color_regions, inputs.spacing)
    truth = shifted_footprints(inputs.truth, shifts)
    sharp = sharp_cores(truth, inputs.color_regions)
    footprint_markers = [
        footprint_marker(footprint.ring, sharp.get(index), band, shape)
        for index, footprint in enumerate(truth)
    ]
    union_allowed = np.zeros(shape, bool)
    cores = np.zeros(shape, bool)
    for marker in footprint_markers:
        union_allowed[marker.rows, marker.cols] |= marker.allowed
        cores[marker.rows, marker.cols] |= marker.core
    background = ~union_allowed
    paper = paper_color(lab, background)
    colored = colored_mask(lab, paper)
    paper_like = paper_mask(lab, paper, background)
    # Outline-only blocks are paper inside, so paper in their bands marks nothing.
    uncolored_zone = np.zeros(shape, bool)
    colored_blocks = 0
    for marker in footprint_markers:
        core_paper = paper_like[marker.rows, marker.cols][marker.core]
        if core_paper.size and 1.0 - core_paper.mean() >= COLORED_BLOCK_FRACTION:
            colored_blocks += 1
            if CLIP_CORES_TO_FILL:
                marker.core = marker.core & ~paper_like[marker.rows, marker.cols]
        else:
            uncolored_zone[marker.rows, marker.cols] |= marker.allowed
    markers = np.zeros(shape, np.int32)
    cores[:] = False
    for index, marker in enumerate(footprint_markers):
        view = markers[marker.rows, marker.cols]
        view[marker.core] = index + 2
        cores[marker.rows, marker.cols] |= marker.core
    min_pocket = (
        OPEN_PAPER_POCKET_FRAC * inputs.spacing**2
        if OPEN_PAPER_POCKET_FRAC is not None
        else None
    )
    street = open_paper(paper_like & ~cores, background, min_pocket)
    background |= street & ~uncolored_zone
    markers[background & (markers == 0)] = 1
    won = watershed(strength, markers)

    regions = np.zeros(shape, np.int32)
    sharp_zone = np.zeros(shape, bool)
    for index, marker in enumerate(footprint_markers):
        claimed = won[marker.rows, marker.cols] == index + 2
        view = regions[marker.rows, marker.cols]
        view[claimed & marker.allowed] = index + 1
        if marker.sharp:
            sharp_zone[marker.rows, marker.cols] |= marker.allowed
    # Pixels a footprint won outside its own band (or window) are nobody's: ignored.
    ignore = (won >= 2) & (regions == 0)
    edge = neighbor_differs(regions)
    distance = np.asarray(ndi.distance_transform_edt(~edge), dtype=np.float64)
    boundary = (regions > 0) & (distance <= BOUNDARY_WIDTH_PX / 2)
    interior = (regions > 0) & ~boundary
    # A weak edge between two footprints is unsupported by the ink; against the paper
    # the open-paper marker anchors it at the color edge, except for outline-only blocks.
    against_paper = neighbor_differs(regions > 0) & ~uncolored_zone
    weak = edge & ~against_paper & (strength < WEAK_EDGE) & ~sharp_zone & (regions > 0)
    if weak.any():
        weak_distance = np.asarray(ndi.distance_transform_edt(~weak), dtype=np.float64)
        ignore |= (regions > 0) & (weak_distance <= band)

    labels = np.full(shape, IGNORE, np.uint8)
    labels[won == 1] = BACKGROUND
    labels[interior] = INTERIOR
    labels[boundary] = BOUNDARY
    labels[ignore] = IGNORE
    # Unlabeled blocks: colored ground no footprint claimed (outside every footprint, or
    # inside a band but flooded from the paper), or a detected number with no footprint.
    labels[colored & (regions == 0)] = IGNORE
    radius = max(2, round(UNLABELED_SEED_RADIUS_FRAC * inputs.spacing))
    for x, y in inputs.seeds:
        cx, cy = round(x), round(y)
        if 0 <= cy < shape[0] and 0 <= cx < shape[1] and not union_allowed[cy, cx]:
            cv2.circle(labels, (cx, cy), radius, int(IGNORE), -1)
    if inputs.inset_rings:
        labels[rasterize(inputs.inset_rings, shape)] = IGNORE
    shift_px = (
        float(np.median([math.hypot(dx, dy) for dx, dy in shifts])) if shifts else 0.0
    )
    return labels, LabelStats(
        len(truth), len(sharp), band, shift_px=shift_px, colored=colored_blocks
    )


# Prepared example on disk: <name>.jpg (working-scale RGB), <name>.label.png, <name>.json.
@dataclass
class PreparedMeta:
    """Sidecar of a prepared key map: how to map it back and how to score it."""

    volume: str
    key: str
    scale: float
    spacing: float  # working px
    seeds: list[Point]  # working px
    texts: list[str]
    full_size: tuple[int, int]  # (width, height) of the original image
    truth_doc: dict  # full-resolution panels doc of the footprints, for score_regions
    color_regions_path: str | None

    def to_json(self) -> dict:
        return {
            "volume": self.volume,
            "key": self.key,
            "scale": self.scale,
            "spacing": self.spacing,
            "seeds": [[x, y] for x, y in self.seeds],
            "texts": self.texts,
            "full_size": list(self.full_size),
            "truth_doc": self.truth_doc,
            "color_regions_path": self.color_regions_path,
        }

    @staticmethod
    def from_json(data: dict) -> "PreparedMeta":
        return PreparedMeta(
            volume=data["volume"],
            key=data["key"],
            scale=float(data["scale"]),
            spacing=float(data["spacing"]),
            seeds=[(float(x), float(y)) for x, y in data["seeds"]],
            texts=list(data["texts"]),
            full_size=(int(data["full_size"][0]), int(data["full_size"][1])),
            truth_doc=data["truth_doc"],
            color_regions_path=data.get("color_regions_path"),
        )


def prepare_keymap(sources: KeymapSources, out_dir: Path) -> str | LabelStats:
    """Build and write one key map's training example; returns its stats or a skip reason."""
    assert sources.truth is not None
    georef = json.loads(sources.georef.read_text())
    projected = project_truth(sources.truth, georef)
    if projected.total and projected.unclipped / projected.total > MAX_UNCLIPPED_SHARE:
        return f"{projected.unclipped}/{projected.total} footprints unclipped"
    if len(projected.footprints) < MIN_TRUTH_FOOTPRINTS:
        return f"only {len(projected.footprints)} truth footprints"
    rgb = read_rgb(sources.image)
    full_height, full_width = rgb.shape[:2]
    georef_scale = (
        full_width / georef["width"]
    )  # sidecars may describe another resolution
    seeds_full, texts = seed_points(sources.keymap_json)
    keymap_doc_width = (
        json.loads(sources.keymap_json.read_text()).get("width") or full_width
    )
    seeds_full = scale_points(seeds_full, full_width / keymap_doc_width)
    scale = working_scale_for_seeds(rgb.shape, seeds_full)
    working = resize_image(rgb, scale)
    seeds = scale_points(seeds_full, scale)
    spacing = (
        nearest_neighbor_distance(seeds)
        if len(seeds) >= MIN_SEEDS_FOR_SPACING
        else float(TARGET_SPACING_PX)
    )
    truth = [
        TruthFootprint(
            footprint.label, scale_points(footprint.ring, georef_scale * scale)
        )
        for footprint in projected.footprints
    ]
    color_regions: list[tuple[str, list[Point]]] = []
    if sources.regions is not None:
        doc = json.loads(sources.regions.read_text())
        color_regions = panels_doc_rings(doc, full_width / doc["width"] * scale)
    inset_rings: list[list[Point]] = []
    if sources.inset is not None:
        doc = json.loads(sources.inset.read_text())
        inset_rings = [
            ring for _, ring in panels_doc_rings(doc, full_width / doc["width"] * scale)
        ]
    labels, stats = build_label_image(
        LabelInputs(working, truth, color_regions, inset_rings, seeds, spacing)
    )
    truth_polygons = {
        i: scale_points(footprint.ring, georef_scale)
        for i, footprint in enumerate(projected.footprints)
    }
    truth_doc = regions_panels_doc(
        sources.image.name,
        (full_width, full_height),
        truth_polygons,
        [footprint.label for footprint in projected.footprints],
    )
    meta = PreparedMeta(
        volume=sources.volume.name,
        key=sources.key,
        scale=scale,
        spacing=spacing,
        seeds=seeds,
        texts=texts,
        full_size=(full_width, full_height),
        truth_doc=truth_doc,
        color_regions_path=str(sources.regions) if sources.regions else None,
    )
    name = sources.name
    Image.fromarray(working).save(out_dir / f"{name}.jpg", quality=95)
    Image.fromarray(labels).save(out_dir / f"{name}.label.png")
    (out_dir / f"{name}.json").write_text(json.dumps(meta.to_json()))
    return stats


def label_inputs_from_prepared(
    data_dir: Path, name: str
) -> tuple[LabelInputs, PreparedMeta]:
    """A prepared example's label inputs, rebuilt from its working-scale image and sidecars.

    The corpus sheets were deleted after the first prep (the laptop had 6 GB free), so
    a new label recipe is applied to the working-scale JPEG the example keeps, with the
    footprints from its metadata and page_regions' sidecars from where they were.
    """
    rgb, _, meta = load_example(data_dir, name)
    full_width = meta.full_size[0]
    truth = [
        TruthFootprint(str(label), [(x * meta.scale, y * meta.scale) for x, y in ring])
        for ring, label in zip(meta.truth_doc["panels"], meta.truth_doc["labels"])
        if len(ring) >= 3
    ]
    color_regions: list[tuple[str, list[Point]]] = []
    if meta.color_regions_path:
        regions_path = Path(meta.color_regions_path)
        if regions_path.exists():
            doc = json.loads(regions_path.read_text())
            color_regions = panels_doc_rings(
                doc, full_width / doc["width"] * meta.scale
            )
    return LabelInputs(
        rgb, truth, color_regions, inset_rings_for(meta), meta.seeds, meta.spacing
    ), meta


def inset_rings_for(meta: PreparedMeta) -> list[list[Point]]:
    """Inset rings of a prepared key map in working pixels, from page_regions' sidecar."""
    if not meta.color_regions_path:
        return []
    regions_path = Path(meta.color_regions_path)
    inset_path = regions_path.with_name(
        regions_path.name.replace(".regions.panels.json", ".inset.panels.json")
    )
    if not inset_path.exists():
        return []
    doc = json.loads(inset_path.read_text())
    factor = meta.full_size[0] / doc["width"] * meta.scale
    return [ring for _, ring in panels_doc_rings(doc, factor)]


def hand_label_image(
    hand_doc: dict,
    meta: PreparedMeta,
    inset_rings: list[list[Point]],
    shape: tuple[int, int],
) -> np.ndarray:
    """The class image of a hand-traced key map, at the prepared working scale.

    Unlike the footprint recipe, everything outside the traces is background: a
    tracer draws every page block, so colored ground with no trace (a volume border,
    a legend, a water tint) is exactly what the model must learn to leave alone. A
    detected number outside every trace still gets an ignored disk, since it may be
    a block the tracer skipped; insets are ignored as in the footprint labels.
    """
    regions = np.zeros(shape, np.int32)
    factor = meta.full_size[0] / hand_doc["width"] * meta.scale
    for index, (_, ring) in enumerate(panels_doc_rings(hand_doc, factor)):
        regions[rasterize([ring], shape)] = index + 1
    edge = neighbor_differs(regions)
    distance = np.asarray(ndi.distance_transform_edt(~edge), dtype=np.float64)
    labels = np.full(shape, BACKGROUND, np.uint8)
    labels[regions > 0] = INTERIOR
    labels[(regions > 0) & (distance <= BOUNDARY_WIDTH_PX / 2)] = BOUNDARY
    radius = max(2, round(UNLABELED_SEED_RADIUS_FRAC * meta.spacing))
    for x, y in meta.seeds:
        cx, cy = round(x), round(y)
        if 0 <= cy < shape[0] and 0 <= cx < shape[1] and regions[cy, cx] == 0:
            cv2.circle(labels, (cx, cy), radius, int(IGNORE), -1)
    if inset_rings:
        labels[rasterize(inset_rings, shape)] = IGNORE
    return labels


LABEL_TINTS = {
    INTERIOR: (0, 170, 0),
    BOUNDARY: (230, 0, 0),
    IGNORE: (128, 128, 128),
}


def label_overlay(
    rgb: np.ndarray, labels: np.ndarray, long_side: int = 1800
) -> np.ndarray:
    """The image tinted by class (interior green, boundary red, ignore gray), downscaled."""
    tinted = rgb.astype(np.float32)
    for value, color in LABEL_TINTS.items():
        where = labels == value
        tinted[where] = 0.45 * tinted[where] + 0.55 * np.asarray(color, np.float32)
    scale = min(1.0, long_side / max(rgb.shape[:2]))
    return resize_image(tinted.astype(np.uint8), scale)


def load_example(
    data_dir: Path, name: str, label_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, PreparedMeta]:
    """A prepared key map: working-scale RGB, label image and metadata.

    With ``label_dir``, a label image of the same name there (the ``hand-labels``
    output) replaces the prepared one.
    """
    rgb = read_rgb(data_dir / f"{name}.jpg")
    label_path = data_dir / f"{name}.label.png"
    if label_dir is not None and (label_dir / label_path.name).exists():
        label_path = label_dir / label_path.name
    labels = np.asarray(Image.open(label_path))
    meta = PreparedMeta.from_json(json.loads((data_dir / f"{name}.json").read_text()))
    return rgb, labels, meta


def prepared_names(data_dir: Path) -> list[str]:
    """Names of every prepared key map in the directory."""
    return sorted(path.name[: -len(".json")] for path in data_dir.glob("*.json"))


def split_names(
    data_dir: Path, val_volumes: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """(train, val) prepared names; a name is val when its volume is in ``val_volumes``."""
    train: list[str] = []
    val: list[str] = []
    for name in prepared_names(data_dir):
        (val if prepared_volume(name) in val_volumes else train).append(name)
    return train, val


def prepared_volume(name: str) -> str:
    """The volume of a prepared name ``<volume>__<stem>``.

    Split-panel stems carry their own double underscore (``p0__2``), so the
    volume is everything before the first one, not the last.
    """
    return name.split("__", 1)[0]


# ---------------------------------------------------------------------------- training


@dataclass
class Tile:
    """One training tile: uint8 RGB and uint8 labels."""

    rgb: np.ndarray
    labels: np.ndarray


def photometric_jitter(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness, contrast, per-channel gain, saturation, gamma, blur, JPEG and noise.

    The model must give the same regions when the scan is re-encoded or resized, which
    is what moved page_regions' output on 39 of 41 key maps (#640).
    """
    image = rgb.astype(np.float32)
    gray = image.mean(axis=2, keepdims=True)
    image = gray + (image - gray) * rng.uniform(0.6, 1.4)  # saturation
    image = image * rng.uniform(0.85, 1.15) + rng.uniform(
        -18, 18
    )  # contrast, brightness
    image = image * rng.uniform(0.93, 1.07, size=(1, 1, 3))  # white balance
    image = np.clip(image, 0, 255)
    gamma = rng.uniform(0.8, 1.25)
    image = 255.0 * (image / 255.0) ** gamma
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.2))
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        quality = int(rng.integers(35, 96))
        ok, encoded = cv2.imencode(
            ".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
        if decoded is not None:
            image = decoded[..., ::-1]
    noise = rng.normal(0, rng.uniform(0, 5), size=image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def tile_origin(
    seeds: list[Point],
    shape: tuple[int, int],
    window: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Top-left corner of a training window: near a random seed most of the time, else anywhere."""
    height, width = shape
    max_x, max_y = max(0, width - window), max(0, height - window)
    if seeds and rng.random() < SEED_CENTERED_TILE_SHARE:
        x, y = seeds[int(rng.integers(0, len(seeds)))]
        x0 = (
            x
            - window / 2
            + rng.uniform(-SEED_CENTER_JITTER, SEED_CENTER_JITTER) * window
        )
        y0 = (
            y
            - window / 2
            + rng.uniform(-SEED_CENTER_JITTER, SEED_CENTER_JITTER) * window
        )
        return int(np.clip(round(x0), 0, max_x)), int(np.clip(round(y0), 0, max_y))
    return int(rng.integers(0, max_x + 1)), int(rng.integers(0, max_y + 1))


def random_tile(
    rgb: np.ndarray, labels: np.ndarray, meta: PreparedMeta, rng: np.random.Generator
) -> Tile:
    """A randomly placed, scaled, flipped and jittered TILE_SIZE tile."""
    height, width = labels.shape
    zoom = rng.uniform(*SCALE_JITTER)
    window = round(TILE_SIZE / zoom)
    x0, y0 = tile_origin(meta.seeds, (height, width), window, rng)
    crop_rgb = rgb[y0 : y0 + window, x0 : x0 + window]
    crop_labels = labels[y0 : y0 + window, x0 : x0 + window]
    if (
        crop_rgb.shape[0] < window or crop_rgb.shape[1] < window
    ):  # image smaller than the window
        pad_h, pad_w = window - crop_rgb.shape[0], window - crop_rgb.shape[1]
        crop_rgb = np.pad(
            crop_rgb, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255
        )
        crop_labels = np.pad(
            crop_labels, ((0, pad_h), (0, pad_w)), constant_values=IGNORE
        )
    tile_rgb = cv2.resize(
        crop_rgb, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    tile_labels = cv2.resize(
        crop_labels, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST
    )
    tile_rgb = photometric_jitter(tile_rgb, rng)
    turns = int(rng.integers(0, 4))
    tile_rgb, tile_labels = np.rot90(tile_rgb, turns), np.rot90(tile_labels, turns)
    if rng.random() < 0.5:
        tile_rgb, tile_labels = tile_rgb[:, ::-1], tile_labels[:, ::-1]
    return Tile(np.ascontiguousarray(tile_rgb), np.ascontiguousarray(tile_labels))


def tiles_to_tensors(
    tiles: list[Tile], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of tiles as (N,3,H,W) inputs and (N,H,W) int64 targets."""
    inputs = np.ascontiguousarray(np.stack([model_input(tile.rgb) for tile in tiles]))
    targets = np.ascontiguousarray(
        np.stack([tile.labels.astype(np.int64) for tile in tiles])
    )
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


def soft_dice(prob: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss (1 - Dice) of a probability map against a 0/1 target, per sample, averaged."""
    intersection = (prob * truth).sum(dim=(1, 2))
    denominator = prob.sum(dim=(1, 2)) + truth.sum(dim=(1, 2))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def region_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Weighted cross-entropy over labeled pixels plus soft Dice on interior and boundary.

    A batch with no labeled pixel at all has no cross-entropy (it would be NaN, and one
    NaN step poisons every weight); only the Dice terms, which are then 0, remain.
    """
    valid = (targets != IGNORE).float()
    if valid.sum() > 0:
        weights = torch.tensor(CLASS_WEIGHTS, device=logits.device)
        cross_entropy = torch.nn.functional.cross_entropy(
            logits, targets, weight=weights, ignore_index=IGNORE
        )
    else:
        cross_entropy = logits.sum() * 0.0
    probabilities = torch.softmax(logits, dim=1)
    dice = sum(
        soft_dice(probabilities[:, cls] * valid, (targets == cls).float())
        for cls in DICE_CLASSES
    )
    return cross_entropy + DICE_WEIGHT * dice


def tiles_for_epoch(
    data_dir: Path,
    names: list[str],
    tiles_per_megapixel: float,
    rng: np.random.Generator,
    *,
    label_dir: Path | None = None,
):
    """Yield augmented tiles over the prepared maps in random order, loading one map at a time."""
    for index in rng.permutation(len(names)):
        rgb, labels, meta = load_example(data_dir, names[index], label_dir)
        megapixels = labels.shape[0] * labels.shape[1] / 1e6
        count = max(8, round(tiles_per_megapixel * megapixels))
        for _ in range(count):
            yield random_tile(rgb, labels, meta, rng)


def shuffled_batches(tiles, batch_size: int, pool_size: int, rng: np.random.Generator):
    """Group a tile stream into full batches drawn from a shuffled pool spanning several maps.

    Leftover tiles that cannot fill a batch are dropped: BatchNorm cannot train on a
    batch of one, and every batch having one shape keeps the MPS graph stable.
    """
    pool: list[Tile] = []
    for tile in tiles:
        pool.append(tile)
        if len(pool) >= pool_size:
            order = rng.permutation(len(pool))
            for start in range(0, len(order) - batch_size + 1, batch_size):
                yield [pool[i] for i in order[start : start + batch_size]]
            pool = []
    order = rng.permutation(len(pool))
    for start in range(0, len(order) - batch_size + 1, batch_size):
        yield [pool[i] for i in order[start : start + batch_size]]


@dataclass
class DecodeCounts:
    """How many regions had no seed, and how many seeds the paper test dropped."""

    unseeded: int
    dropped_seeds: int


@dataclass
class MapEvaluation:
    """One key map's scores: the model's regions against truth, and the color baseline's.

    ``model``/``color`` match regions to footprints by page number, so they are capped by
    the number reads; ``model_shape``/``color_shape`` match any region to any footprint
    by best IoU and measure the segmentation alone.
    """

    name: str
    model: RegionScore
    color: RegionScore | None
    model_shape: float
    color_shape: float | None
    counts: DecodeCounts
    seconds: float
    # Against the hand-traced blocks, when the volume has them.
    hand: RegionScore | None = None
    hand_shape: float | None = None
    color_hand: RegionScore | None = None
    color_hand_shape: float | None = None


def hand_truth_path(data_dir: Path, meta: PreparedMeta) -> Path | None:
    """The regions labeler's traces for a prepared key map, if the volume has them."""
    path = data_dir / meta.volume / "raw" / "truth" / f"{meta.key}.regions.panels.json"
    return path if path.exists() else None


def score_summary(score: RegionScore) -> tuple[float, int, int]:
    """(mean IoU, count at or above 0.5, count) of a RegionScore."""
    values = [iou for _, iou in score.ious]
    if not values:
        return 0.0, 0, 0
    return float(np.mean(values)), sum(1 for v in values if v >= 0.5), len(values)


def predict_prepared(
    model: UNet, rgb: np.ndarray, meta: PreparedMeta, device: torch.device
) -> np.ndarray:
    """Class probabilities for a prepared working-scale image (flip-averaged)."""
    return predict_class_maps(model, model_input(rgb), device, flips=TTA_FLIPS)


def regions_doc_from_probabilities(
    probabilities: np.ndarray, meta: PreparedMeta, params: DecodeParams
) -> tuple[dict, DecodeCounts]:
    """Decode class maps into a full-resolution panels doc, with the decode's counts."""
    decoded = decode_regions(probabilities, meta.seeds, meta.spacing, params)
    polygons = region_polygons(decoded, meta.spacing, params)
    label_text = {
        label: meta.texts[i] for i, label in enumerate(decoded.seed_labels) if label > 0
    }
    out_polygons: dict[int, list[Point]] = {}
    texts: list[str] = []
    for label, polygon in sorted(polygons.items()):
        out_polygons[len(texts)] = [
            (x / meta.scale, y / meta.scale) for x, y in polygon
        ]
        texts.append(label_text.get(label, UNSEEDED_LABEL))
    doc = regions_panels_doc(f"{meta.key}.jpg", meta.full_size, out_polygons, texts)
    counts = DecodeCounts(
        unseeded=sum(1 for text in texts if text == UNSEEDED_LABEL),
        dropped_seeds=sum(1 for label in decoded.seed_labels if label == 0),
    )
    return doc, counts


def score_doc(truth_doc: dict, predicted_doc: dict) -> RegionScore:
    """score_regions on copies: truth brought to the predicted doc's pixel space, labels matched."""
    truth = json.loads(json.dumps(truth_doc))
    predicted = labels_matched_to_truth(predicted_doc, truth_doc)
    scale_to(truth, predicted)
    return score_regions(truth, predicted)


def shape_score(truth_doc: dict, predicted_doc: dict) -> float:
    """Mean over truth footprints of the IoU with their best one-to-one matched region, any label.

    Unseeded "?" regions count too (the label regex is bypassed by relabeling), so this
    is the segmentation's quality with the page numbers taken out of the picture.
    """
    truth = json.loads(json.dumps(truth_doc))
    predicted = json.loads(json.dumps(predicted_doc))
    predicted["labels"] = ["0"] * len(predicted["panels"])
    scale_to(truth, predicted)
    truths = [polygon for _, polygon in panel_polygons(truth)]
    predictions = [polygon for _, polygon in panel_polygons(predicted)]
    if not truths:
        return 0.0
    pairs = sorted(
        (
            (
                t.intersection(q).area / t.union(q).area if t.union(q).area else 0.0,
                i,
                j,
            )
            for i, t in enumerate(truths)
            for j, q in enumerate(predictions)
            if t.intersects(q)
        ),
        reverse=True,
    )
    best = [0.0] * len(truths)
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    for iou, i, j in pairs:
        if i in used_truth or j in used_prediction:
            continue
        best[i] = iou
        used_truth.add(i)
        used_prediction.add(j)
    return float(np.mean(best))


def evaluate_map(
    model: UNet,
    data_dir: Path,
    name: str,
    device: torch.device,
    params: DecodeParams,
    hand_dir: Path | None = None,
) -> MapEvaluation:
    """Decode one prepared key map and score it, with the color segmentation beside it.

    With ``hand_dir`` (the ``data/`` root), also scores both against the hand traces.
    """
    started = time.time()
    rgb, _, meta = load_example(data_dir, name)
    probabilities = predict_prepared(model, rgb, meta, device)
    doc, counts = regions_doc_from_probabilities(probabilities, meta, params)
    evaluation = MapEvaluation(
        name,
        score_doc(meta.truth_doc, doc),
        None,
        shape_score(meta.truth_doc, doc),
        None,
        counts,
        0.0,
    )
    color_doc = None
    if meta.color_regions_path and Path(meta.color_regions_path).exists():
        color_doc = json.loads(Path(meta.color_regions_path).read_text())
        evaluation.color = score_doc(meta.truth_doc, color_doc)
        evaluation.color_shape = shape_score(meta.truth_doc, color_doc)
    hand_path = hand_truth_path(hand_dir, meta) if hand_dir else None
    if hand_path is not None:
        hand_doc = json.loads(hand_path.read_text())
        evaluation.hand = score_doc(hand_doc, doc)
        evaluation.hand_shape = shape_score(hand_doc, doc)
        if color_doc is not None:
            evaluation.color_hand = score_doc(hand_doc, color_doc)
            evaluation.color_hand_shape = shape_score(hand_doc, color_doc)
    evaluation.seconds = time.time() - started
    return evaluation


def format_evaluation(evaluation: MapEvaluation) -> str:
    """One line: model mean IoU / >=0.5 vs the color baseline, overlap, unseeded count."""
    mean, good, count = score_summary(evaluation.model)
    line = (
        f"{evaluation.name:50s} unet {mean:.3f} ({good:3d}/{count:3d}) "
        f"shape {evaluation.model_shape:.3f} overlap {evaluation.model.predicted_overlap:.1%} "
        f"unseeded {evaluation.counts.unseeded:2d} dropped {evaluation.counts.dropped_seeds:2d}"
    )
    if evaluation.color is not None and evaluation.color_shape is not None:
        color_mean, color_good, _ = score_summary(evaluation.color)
        line += (
            f" | k-means {color_mean:.3f} ({color_good:3d}) shape {evaluation.color_shape:.3f} "
            f"overlap {evaluation.color.predicted_overlap:.1%}"
        )
    if evaluation.hand is not None and evaluation.hand_shape is not None:
        hand_mean, hand_good, hand_count = score_summary(evaluation.hand)
        line += (
            f" | HAND unet {hand_mean:.3f} ({hand_good:3d}/{hand_count:3d}) "
            f"shape {evaluation.hand_shape:.3f}"
        )
        if (
            evaluation.color_hand is not None
            and evaluation.color_hand_shape is not None
        ):
            color_mean, color_good, _ = score_summary(evaluation.color_hand)
            line += (
                f", k-means {color_mean:.3f} ({color_good:3d}) "
                f"shape {evaluation.color_hand_shape:.3f}"
            )
    return line


def hand_mean(evaluations: list[MapEvaluation]) -> tuple[float | None, int]:
    """(mean hand-traced IoU over the maps that have traces, their count)."""
    values = [score_summary(e.hand)[0] for e in evaluations if e.hand is not None]
    return (float(np.mean(values)) if values else None), len(values)


def cmd_prep(args: argparse.Namespace) -> None:
    out_dir = args.training_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.training_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    sources = discover_keymap_sources(args.data_dir, require_truth=True)
    if args.only:
        sources = [
            s for s in sources if s.volume.name in args.only or s.name in args.only
        ]
    print(f"{len(sources)} key maps with truth", flush=True)
    kept = 0
    for index, source in enumerate(sources, 1):
        if (out_dir / f"{source.name}.json").exists() and not args.force:
            kept += 1
            continue
        started = time.time()
        result = prepare_keymap(source, out_dir)
        if isinstance(result, str):
            print(f"[{index}/{len(sources)}] skip {source.name}: {result}", flush=True)
            continue
        kept += 1
        rgb, labels, _ = load_example(out_dir, source.name)
        Image.fromarray(label_overlay(rgb, labels)).save(
            overlay_dir / f"{source.name}.jpg", quality=85
        )
        print(
            f"[{index}/{len(sources)}] {source.name}: {labels.shape[1]}x{labels.shape[0]} "
            + label_counts_text(labels)
            + f" footprints {result.footprints} sharp {result.sharp}"
            + f" colored {result.colored} shift {result.shift_px:.0f}px band {result.band_px}px"
            + f" [{time.time() - started:.0f}s]",
            flush=True,
        )
    print(f"{kept} prepared key maps in {out_dir}")


def label_counts_text(labels: np.ndarray) -> str:
    """Class shares of a label image, for the prep and relabel logs."""
    return " ".join(
        f"{name} {float((labels == value).mean()):.2f}"
        for name, value in (
            ("bg", BACKGROUND),
            ("in", INTERIOR),
            ("bd", BOUNDARY),
            ("ign", IGNORE),
        )
    )


def cmd_relabel(args: argparse.Namespace) -> None:
    data_dir = args.training_dir / "data"
    overlay_dir = args.training_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    names = args.only or prepared_names(data_dir)
    print(f"relabeling {len(names)} prepared key maps", flush=True)
    for index, name in enumerate(names, 1):
        started = time.time()
        try:
            inputs, _ = label_inputs_from_prepared(data_dir, name)
            labels, stats = build_label_image(inputs)
        except Exception as error:  # noqa: BLE001 - one bad map must not stop the pass
            print(
                f"[{index}/{len(names)}] {name}: ERROR {type(error).__name__}: {error}"
            )
            continue
        Image.fromarray(labels).save(data_dir / f"{name}.label.png")
        Image.fromarray(label_overlay(inputs.rgb, labels)).save(
            overlay_dir / f"{name}.jpg", quality=85
        )
        print(
            f"[{index}/{len(names)}] {name}: {label_counts_text(labels)}"
            f" footprints {stats.footprints} sharp {stats.sharp} colored {stats.colored}"
            f" shift {stats.shift_px:.0f}px [{time.time() - started:.0f}s]",
            flush=True,
        )
    print("RELABEL-DONE", flush=True)


def cmd_hand_labels(args: argparse.Namespace) -> None:
    data_dir = args.training_dir / "data"
    out_dir = args.training_dir / "hand-labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in args.only or prepared_names(data_dir):
        meta = PreparedMeta.from_json(
            json.loads((data_dir / f"{name}.json").read_text())
        )
        hand_path = hand_truth_path(args.data_dir, meta)
        if hand_path is None:
            continue
        rgb = read_rgb(data_dir / f"{name}.jpg")
        hand_doc = json.loads(hand_path.read_text())
        labels = hand_label_image(hand_doc, meta, inset_rings_for(meta), rgb.shape[:2])
        Image.fromarray(labels).save(out_dir / f"{name}.label.png")
        Image.fromarray(label_overlay(rgb, labels)).save(
            out_dir / f"{name}.jpg", quality=85
        )
        written += 1
        print(
            f"{name}: {len(hand_doc['panels'])} traces, {label_counts_text(labels)}",
            flush=True,
        )
    print(f"HAND-LABELS-DONE {written} written to {out_dir}", flush=True)


def cmd_train(args: argparse.Namespace) -> None:
    data_dir = args.training_dir / "data"
    device = select_device()
    train, val = split_names(data_dir, tuple(args.val_volumes))
    print(
        f"train {len(train)} key maps, val {len(val)} ({', '.join(val)}) on {device}",
        flush=True,
    )
    if args.label_dir:
        # Hand-labeled maps replace their footprint labels and are visited more often,
        # since a handful of them would otherwise vanish among 500 maps.
        overridden = [n for n in train if (args.label_dir / f"{n}.label.png").exists()]
        train = train + overridden * (args.label_dir_repeat - 1)
        print(
            f"labels from {args.label_dir} for {len(overridden)} training maps "
            f"({', '.join(overridden)}), each visited {args.label_dir_repeat}x per epoch",
            flush=True,
        )
    if args.init:
        model = load_region_unet(args.init, device)  # warm start
        print(f"initialized from {args.init}", flush=True)
    else:
        model = build_model(args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    params = DecodeParams()
    history: list[dict] = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        losses: list[float] = []
        tiles = tiles_for_epoch(
            data_dir, train, args.tiles_per_megapixel, rng, label_dir=args.label_dir
        )
        for batch in shuffled_batches(tiles, args.batch_size, args.pool_size, rng):
            inputs, targets = tiles_to_tensors(batch, device)
            loss = region_loss(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) % 50 == 0:
                print(
                    f"  epoch {epoch} step {len(losses)}: loss {np.mean(losses[-50:]):.4f}"
                    f"  [{time.time() - started:.0f}s]",
                    flush=True,
                )
        schedule.step()
        model.eval()
        evaluations = [
            evaluate_map(model, data_dir, name, device, params, hand_dir=args.data_dir)
            for name in val
        ]
        means = [score_summary(e.model)[0] for e in evaluations]
        val_iou = float(np.mean(means)) if means else 0.0
        val_hand, hand_count = hand_mean(evaluations)
        # The hand traces are the benchmark; the footprints stand in where there are none.
        selector = (
            val_hand
            if val_hand is not None and hand_count >= MIN_HAND_VAL_MAPS
            else val_iou
        )
        saved = ""
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output.with_suffix(".last.pt"))
        if args.checkpoint_dir:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(), args.checkpoint_dir / f"epoch_{epoch:02d}.pt"
            )
        if selector > best:
            best = selector
            torch.save(model.state_dict(), args.output)
            saved = "  (saved)"
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_iou": val_iou,
                "val_hand": val_hand,
                "per_map": dict(zip(val, means)),
                "per_map_hand": {
                    e.name: score_summary(e.hand)[0]
                    for e in evaluations
                    if e.hand is not None
                },
            }
        )
        args.output.with_suffix(".history.json").write_text(
            json.dumps(history, indent=1)
        )
        hand_text = (
            f"  hand IoU {val_hand:.3f} ({hand_count} maps)"
            if val_hand is not None
            else ""
        )
        print(
            f"epoch {epoch:2d}: loss {np.mean(losses):.4f}  val IoU {val_iou:.3f} "
            f"({' '.join(f'{m:.3f}' for m in means)}){hand_text}"
            f"  [{time.time() - started:.0f}s]{saved}",
            flush=True,
        )
    print(f"best validation score {best:.3f} -> {args.output}")


def stability_iou(
    model: UNet, rgb: np.ndarray, meta: PreparedMeta, device: torch.device
) -> float:
    """Mean per-page IoU between regions decoded from the image and from it one pixel wider.

    The #640 perturbation: a one-pixel change in the resized sheet. 1.0 means the
    partition did not move.
    """
    params = DecodeParams()
    base_doc, _ = regions_doc_from_probabilities(
        predict_prepared(model, rgb, meta, device), meta, params
    )
    height, width = rgb.shape[:2]
    wider = cv2.resize(rgb, (width + 1, height), interpolation=cv2.INTER_LINEAR)
    factor = (width + 1) / width
    perturbed = PreparedMeta(
        **{
            **meta.__dict__,
            "seeds": [(x * factor, y) for x, y in meta.seeds],
            "scale": meta.scale * factor,
        }
    )
    wide_doc, _ = regions_doc_from_probabilities(
        predict_prepared(model, wider, perturbed, device), perturbed, params
    )
    # Compare in the base doc's full-resolution frame: both docs carry full_size.
    score = score_doc(base_doc, wide_doc)
    values = [iou for _, iou in score.ious]
    return float(np.mean(values)) if values else 0.0


def average_checkpoints(paths: list[Path]) -> dict:
    """The element-wise mean of several checkpoints' tensors (counters take the last).

    Averaging the weights of the late epochs of one run gives a model as good as the
    best single epoch on the held-out maps without picking it by those same maps; the
    BatchNorm running statistics are averaged too.
    """
    states = [torch.load(str(path), map_location="cpu") for path in paths]
    averaged: dict = {}
    for key in states[0]:
        tensors = [state[key] for state in states]
        if tensors[0].dtype.is_floating_point:
            averaged[key] = torch.stack(tensors).mean(dim=0)
        else:
            averaged[key] = tensors[-1]
    return averaged


def cmd_average(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(average_checkpoints(args.checkpoints), args.output)
    print(f"averaged {len(args.checkpoints)} checkpoints -> {args.output}")


def cmd_eval(args: argparse.Namespace) -> None:
    global TTA_FLIPS
    TTA_FLIPS = not args.no_flips
    data_dir = args.training_dir / "data"
    device = select_device()
    model = load_region_unet(args.weights, device)
    names = args.names or split_names(data_dir, tuple(args.val_volumes))[1]
    params = DecodeParams()
    model_means: list[float] = []
    color_means: list[float] = []
    model_shapes: list[float] = []
    color_shapes: list[float] = []
    evaluations: list[MapEvaluation] = []
    for name in names:
        evaluation = evaluate_map(
            model, data_dir, name, device, params, hand_dir=args.data_dir
        )
        evaluations.append(evaluation)
        line = format_evaluation(evaluation)
        model_means.append(score_summary(evaluation.model)[0])
        model_shapes.append(evaluation.model_shape)
        if evaluation.color is not None and evaluation.color_shape is not None:
            color_means.append(score_summary(evaluation.color)[0])
            color_shapes.append(evaluation.color_shape)
        if args.stability:
            rgb, _, meta = load_example(data_dir, name)
            line += f" | stability {stability_iou(model, rgb, meta, device):.3f}"
        print(line, flush=True)
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            rgb, _, meta = load_example(data_dir, name)
            probabilities = predict_prepared(model, rgb, meta, device)
            doc, _ = regions_doc_from_probabilities(probabilities, meta, params)
            (args.out_dir / f"{name}.regions.panels.json").write_text(json.dumps(doc))
            from mapsnap.keymap.region_unet import class_map_image

            Image.fromarray(class_map_image(probabilities)).save(
                args.out_dir / f"{name}.maps.png"
            )
    print(
        f"mean over {len(names)} key maps: unet {np.mean(model_means):.3f} "
        f"(shape {np.mean(model_shapes):.3f})"
        + (
            f", k-means {np.mean(color_means):.3f} (shape {np.mean(color_shapes):.3f})"
            if color_means
            else ""
        )
    )
    traced = [e for e in evaluations if e.hand is not None and e.hand_shape is not None]
    if traced:
        unet_hand = [score_summary(e.hand)[0] for e in traced if e.hand is not None]
        unet_shape = [e.hand_shape for e in traced if e.hand_shape is not None]
        line = (
            f"vs hand traces on {len(traced)} key maps: unet {np.mean(unet_hand):.3f} "
            f"(shape {np.mean(unet_shape):.3f})"
        )
        color_hand = [
            score_summary(e.color_hand)[0] for e in traced if e.color_hand is not None
        ]
        color_shape = [
            e.color_hand_shape for e in traced if e.color_hand_shape is not None
        ]
        if color_hand:
            line += f", k-means {np.mean(color_hand):.3f} (shape {np.mean(color_shape):.3f})"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser(
        "prep", help="Project truth, build label images, write examples."
    )
    prep.add_argument("--data-dir", type=Path, default=Path("data"))
    prep.add_argument(
        "--only", nargs="*", help="Volume names or <volume>__<key> names to prepare."
    )
    prep.add_argument(
        "--force", action="store_true", help="Rebuild examples that already exist."
    )
    prep.set_defaults(func=cmd_prep)

    relabel = subparsers.add_parser(
        "relabel",
        help="Rebuild the label images of prepared examples (no full-res sheets needed).",
    )
    relabel.add_argument("--only", nargs="*", help="Prepared names to relabel.")
    relabel.set_defaults(func=cmd_relabel)

    train = subparsers.add_parser(
        "train", help="Train the UNet on the prepared examples."
    )
    train.add_argument("--output", type=Path, default=MODEL_PATH)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--base", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--tiles-per-megapixel", type=float, default=4.0)
    train.add_argument("--pool-size", type=int, default=192)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--val-volumes", nargs="*", default=list(DEFAULT_VAL_VOLUMES))
    train.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Also keep every epoch's weights here (epoch_NN.pt), for averaging.",
    )
    train.add_argument(
        "--init", type=Path, help="Start from these weights instead of at random."
    )
    train.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Volume root holding raw/truth/<key>.regions.panels.json hand traces.",
    )
    train.add_argument(
        "--label-dir",
        type=Path,
        help="Label images (the hand-labels output) that replace the prepared ones "
        "for the training maps found there.",
    )
    train.add_argument(
        "--label-dir-repeat",
        type=int,
        default=1,
        help="Times per epoch each map with a --label-dir label is visited.",
    )
    train.set_defaults(func=cmd_train)

    hand = subparsers.add_parser(
        "hand-labels",
        help="Build label images from the regions labeler's hand traces, into "
        "<training-dir>/hand-labels/.",
    )
    hand.add_argument("--data-dir", type=Path, default=Path("data"))
    hand.add_argument("--only", nargs="*", help="Prepared names to build.")
    hand.set_defaults(func=cmd_hand_labels)

    average = subparsers.add_parser(
        "average", help="Average several checkpoints' weights into one."
    )
    average.add_argument("output", type=Path)
    average.add_argument("checkpoints", type=Path, nargs="+")
    average.set_defaults(func=cmd_average)

    evaluate = subparsers.add_parser(
        "eval", help="Score held-out key maps against truth and k-means."
    )
    evaluate.add_argument("--weights", type=Path, default=MODEL_PATH)
    evaluate.add_argument(
        "--names", nargs="*", help="Prepared names to score (default: the val split)."
    )
    evaluate.add_argument("--val-volumes", nargs="*", default=list(DEFAULT_VAL_VOLUMES))
    evaluate.add_argument(
        "--no-flips",
        action="store_true",
        help="Predict without flip averaging (an ablation of the default).",
    )
    evaluate.add_argument(
        "--stability",
        action="store_true",
        help="Also decode a 1 px wider image and compare.",
    )
    evaluate.add_argument(
        "--out-dir", type=Path, help="Write decoded regions and class maps here."
    )
    evaluate.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Volume root holding raw/truth/<key>.regions.panels.json hand traces.",
    )
    evaluate.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
