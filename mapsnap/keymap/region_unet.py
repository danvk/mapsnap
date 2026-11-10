"""Learned key-map page regions: a boundary + interior UNet over the key-map sheet.

A key map tiles its city into colored blocks, one per page, with the page number at
the block's center. ``page_regions`` recovers those blocks by color (k-means and flood
fills), which fails whenever color stops meaning "same block": two neighboring blocks
printed in one color merge, a block printed in two colors splits, a pale fill lands in
the paper cluster, and a one-pixel change to the input moves regions on most key maps
(#640). This module replaces the color rules with a small UNet that labels every pixel
background (paper, cartouche, inset), block interior, or block boundary (the heavy
line or color change between two pages), and recovers the blocks as a planar partition
by a watershed on P(boundary) seeded from the detected page numbers.

Input is the RGB sheet at a *working scale* chosen so the median spacing between
detected page numbers is TARGET_SPACING_PX. The detected numbers are not an input: a
seed channel taught the model "number here, so block here" and lit up every stray
read (a legend's "27", an unlabeled rail yard); the numbers are printed on the sheet
anyway. They enter only at decode time, as the watershed's markers and the regions'
labels, and a block no detected number sits in becomes a region labeled "?".

Training labels come from OldInsuranceMaps footprints projected through the key map's
own georeference (mapsnap.keymap.train_region_unet). Inference:

    uv run python -m mapsnap.keymap.region_unet data/<vol>/raw/p0.keymap.json \\
        [--output p0.regions.panels.json] [--overlay p0.regions.png] [--image raw/p0.jpg]
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from mapsnap.keymap.page_regions import (
    Point,
    box_center,
    load_seeds,
    mask_to_polygon,
    nearest_neighbor_distance,
    regions_panels_doc,
    render_overlay,
    working_scale,
)
from mapsnap.road_model import UNet

Image.MAX_IMAGE_PIXELS = None

MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "keymap_region_unet.pt"
)
WEIGHTS_ENV_VAR = "MAPSNAP_KEYMAP_REGION_WEIGHTS"  # other weights than MODEL_PATH

# The model works at the scale where neighboring page numbers are this far apart, so a
# block is ~100-200 px across and a tile holds a few blocks with their surroundings.
TARGET_SPACING_PX = 128
FALLBACK_LONG_SIDE = 3000  # working scale when too few seeds exist to measure spacing
MIN_SEEDS_FOR_SPACING = 4
TILE_SIZE = 512
TILE_OVERLAP = 64
# Softmax classes.
BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2
NUM_CLASSES = 3
IGNORE = 255  # label value excluded from the loss
BOUNDARY_WIDTH_PX = 5  # the boundary class's drawn width in working pixels
UNSEEDED_LABEL = "?"  # label of a decoded block no detected number sits in
RUN_TAGS = (
    "corpus-v2",
    "corpus-v1",
)  # corpus runs whose raw/ sidecars stand in for raw/


@dataclass(frozen=True)
class KeymapSources:
    """The files a key map's region model reads: its image and the pipeline's sidecars."""

    volume: Path
    key: str
    image: Path
    georef: Path
    keymap_json: Path
    regions: Path | None  # page_regions' k-means sidecar, if any
    inset: Path | None  # <key>.inset.panels.json, if any
    truth: Path | None  # the volume's main.iiif.json, if it has OIM truth

    @property
    def name(self) -> str:
        """``<volume>__<key>``, unique across the data directory."""
        return f"{self.volume.name}__{self.key}"


def keymap_keys(volume: Path) -> list[str]:
    """The volume's identified key-map page keys, from keymaps.json in the root or a run."""
    candidates = [volume / "keymaps.json"] + [
        volume / "runs" / tag / "keymaps.json" for tag in RUN_TAGS
    ]
    for path in candidates:
        if path.exists():
            return list(json.loads(path.read_text()).get("keys", []))
    return []


def sidecar_dir(volume: Path, key: str) -> Path | None:
    """The directory holding ``<key>.georef.json`` and ``<key>.keymap.json``, if either has both."""
    candidates = [volume / "raw"] + [volume / "runs" / tag / "raw" for tag in RUN_TAGS]
    for directory in candidates:
        if (directory / f"{key}.georef.json").exists() and (
            directory / f"{key}.keymap.json"
        ).exists():
            return directory
    return None


def keymap_sources(volume: Path, key: str) -> KeymapSources | None:
    """A key map's source files, or None when its image or georef is missing."""
    image = volume / "raw" / f"{key}.jpg"
    directory = sidecar_dir(volume, key)
    if directory is None or not image.exists():
        return None
    regions = directory / f"{key}.regions.panels.json"
    inset = directory / f"{key}.inset.panels.json"
    truth = volume / "main.iiif.json"
    return KeymapSources(
        volume=volume,
        key=key,
        image=image,
        georef=directory / f"{key}.georef.json",
        keymap_json=directory / f"{key}.keymap.json",
        regions=regions if regions.exists() else None,
        inset=inset if inset.exists() else None,
        truth=truth if truth.exists() else None,
    )


def discover_keymap_sources(
    data_dir: Path, *, require_truth: bool = False
) -> list[KeymapSources]:
    """Every georeferenced key map under ``data_dir``, optionally only those with truth."""
    found: list[KeymapSources] = []
    for volume in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for key in keymap_keys(volume):
            sources = keymap_sources(volume, key)
            if sources is None or (require_truth and sources.truth is None):
                continue
            found.append(sources)
    return found


def seed_points(keymap_json: Path) -> tuple[list[Point], list[str]]:
    """Centers and texts of the live page-number detections, in full-resolution pixels."""
    boxes, texts = load_seeds(keymap_json)
    return [box_center(box) for box in boxes], texts


def working_scale_for_seeds(image_shape: tuple[int, ...], seeds: list[Point]) -> float:
    """Downscale factor putting the median seed spacing at TARGET_SPACING_PX (never upscales).

    With too few seeds to measure a spacing, falls back to page_regions' long-side rule.
    """
    if len(seeds) < MIN_SEEDS_FOR_SPACING:
        return working_scale(image_shape, FALLBACK_LONG_SIDE)
    spacing = nearest_neighbor_distance(seeds)
    if spacing <= 0:
        return working_scale(image_shape, FALLBACK_LONG_SIDE)
    return min(1.0, TARGET_SPACING_PX / spacing)


def scale_points(points: list[Point], scale: float) -> list[Point]:
    """Points multiplied by ``scale``."""
    return [(x * scale, y * scale) for x, y in points]


def resize_image(rgb: np.ndarray, scale: float) -> np.ndarray:
    """The RGB image resized by ``scale`` (area interpolation when shrinking)."""
    if scale == 1.0:
        return rgb
    size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(rgb, size, interpolation=interpolation)


def read_rgb(path: Path) -> np.ndarray:
    """A JPEG as an HxWx3 uint8 RGB array (PIL, so outsized key maps load)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def model_input(rgb: np.ndarray) -> np.ndarray:
    """The 3xHxW float32 input: RGB on a fixed scale (paper ~+1.6, ink ~-2).

    A fixed normalization, not per-image standardization: block fills are told apart
    by their absolute colors, which per-image statistics would shift from map to map.
    The result is C-contiguous: a channels-last layout left over from the transpose
    makes MPS convolutions four times slower.
    """
    colors = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.25
    return np.ascontiguousarray(colors.transpose(2, 0, 1))


def build_model(base: int = 16, norm: str = "batch") -> UNet:
    """An untrained region UNet: RGB in, three class logits out.

    Base 16 with BatchNorm trains 3x faster per step on MPS than base 24 with
    GroupNorm and predicts 4x faster; tiles are mixed across maps within a batch, so
    the batch statistics are well behaved (unlike the whole-page models').
    """
    return UNet(base=base, in_channels=3, norm=norm, out_channels=NUM_CLASSES)


def load_region_unet(path: Path, device: torch.device) -> UNet:
    """A trained region UNet from its checkpoint, sized from the weights, in eval mode."""
    state_dict = torch.load(str(path), map_location=device)
    first = state_dict["enc1.block.0.weight"]
    head = state_dict["head.weight"]
    norm = "batch" if "enc1.block.1.running_mean" in state_dict else "group"
    model = UNet(
        base=int(first.shape[0]),
        in_channels=int(first.shape[1]),
        norm=norm,
        out_channels=int(head.shape[0]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def tile_origins(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of ``tile``-long windows covering ``length`` with at least ``overlap``."""
    if length <= tile:
        return [0]
    step = tile - overlap
    origins = list(range(0, length - tile, step))
    origins.append(length - tile)
    return origins


def blend_window(tile: int) -> np.ndarray:
    """A separable raised-cosine weight so overlapping tiles blend without seams."""
    ramp = 0.5 - 0.5 * np.cos(
        np.linspace(0, 2 * np.pi, tile, endpoint=False) + np.pi / tile
    )
    window = np.outer(ramp, ramp).astype(np.float32)
    return np.maximum(window, 1e-3)


@torch.no_grad()
def predict_class_maps(
    model: UNet,
    inputs: np.ndarray,
    device: torch.device,
    *,
    flips: bool = True,
) -> np.ndarray:
    """Class probabilities for a 3xHxW input, averaged over the four flips when ``flips``.

    Flip averaging smooths the maps and makes the decoded regions less sensitive to a
    one-pixel change in the input (#640), at four times the inference cost (seconds per
    key map).
    """
    if not flips:
        return predict_maps(model, inputs, device)
    total = np.zeros((NUM_CLASSES,) + inputs.shape[1:], np.float32)
    for flip_x, flip_y in ((False, False), (True, False), (False, True), (True, True)):
        view = inputs
        if flip_x:
            view = view[:, :, ::-1]
        if flip_y:
            view = view[:, ::-1, :]
        probabilities = predict_maps(model, np.ascontiguousarray(view), device)
        if flip_x:
            probabilities = probabilities[:, :, ::-1]
        if flip_y:
            probabilities = probabilities[:, ::-1, :]
        total += probabilities
    return total / 4.0


@torch.no_grad()
def predict_maps(
    model: UNet,
    inputs: np.ndarray,
    device: torch.device,
    *,
    tile: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> np.ndarray:
    """Class probabilities (3xHxW float32) for a 3xHxW input, by blended overlapping tiles.

    Images smaller than a tile are padded with paper-colored input and cropped back.
    """
    _, height, width = inputs.shape
    pad_h, pad_w = max(0, tile - height), max(0, tile - width)
    if pad_h or pad_w:
        paper = model_input(np.full((1, 1, 3), 255, np.uint8))
        padded = np.empty((inputs.shape[0], height + pad_h, width + pad_w), np.float32)
        padded[:] = paper[:, 0, 0][:, None, None]
        padded[:, :height, :width] = inputs
        inputs = padded
    _, full_h, full_w = inputs.shape
    probabilities = np.zeros((NUM_CLASSES, full_h, full_w), np.float32)
    weights = np.zeros((full_h, full_w), np.float32)
    window = blend_window(tile)
    model.eval()
    for y0 in tile_origins(full_h, tile, overlap):
        for x0 in tile_origins(full_w, tile, overlap):
            patch = torch.from_numpy(
                np.ascontiguousarray(inputs[:, y0 : y0 + tile, x0 : x0 + tile][None])
            ).to(device)
            prob = torch.softmax(model(patch), dim=1)[0].cpu().numpy()
            probabilities[:, y0 : y0 + tile, x0 : x0 + tile] += prob * window
            weights[y0 : y0 + tile, x0 : x0 + tile] += window
    probabilities /= weights
    return probabilities[:, :height, :width]


@dataclass
class DecodeParams:
    """Knobs of the watershed that turns class maps and seeds into one region per block."""

    block_prob: float = 0.5  # P(background) below this is block (interior or boundary)
    # The flood domain is the block mask closed by a disk of this radius (fraction of
    # spacing) and with enclosed pockets below this fraction of spacing^2 filled: the
    # model may paint a paper bubble over a page number, and a marker in an unfilled
    # pocket could never flood its block. Regions are clipped back to the block mask.
    close_radius_frac: float = 0.1
    hole_area_frac: float = 0.5
    # A seed's marker is a disk of this radius (fraction of spacing): wide enough to
    # survive the number glyph, which the model may paint as background, and the
    # block-mask erasure around it.
    seed_radius_frac: float = 0.1
    claim_radius_frac: float = (
        0.35  # an interior component this close to a seed is that seed's
    )
    # A seed is dropped (it sits on paper, a legend or text) when the filled block mask
    # over a disk of this radius covers less than this fraction.
    seed_test_radius_frac: float = 0.2
    seed_min_block: float = 0.05
    # An unseeded block is a component of P(interior) above this, after smoothing by a
    # fraction of the spacing and an opening of that radius (cuts bridges and specks),
    # and at least this fraction of spacing^2 (a block is about spacing^2).
    unseeded_interior_prob: float = 0.6
    unseeded_smooth_frac: float = 0.04
    unseeded_open_frac: float = 0.06
    unseeded_min_area_frac: float = 0.3
    # Predicted boundary at or above this splits the confident-interior mask before the
    # unclaimed components are found: a block joined to a seeded neighbor through a gap
    # in the boundary line would otherwise be one component with it, get no marker, and
    # be flooded by the neighbor.
    wall_prob: float = 0.5
    simplify_frac: float = (
        0.015  # polygon simplification tolerance, as a fraction of spacing
    )
    min_region_area_frac: float = (
        0.02  # regions smaller than this fraction of spacing^2 are dropped
    )
    # Sanity caps. A seeded region above this many spacing^2 is nonsense (a half-scale
    # page is ~4); an unseeded one has no number vouching for it, so its cap is tighter.
    # A region whose hole-filled polygon dwarfs its own pixels is a ring (the scan's
    # dark border, a flood that wrapped around a block) and is dropped.
    max_region_area_frac: float = 60.0
    max_unseeded_area_frac: float = 12.0
    max_fill_ratio: float = 1.4


@dataclass
class DecodedRegions:
    """The watershed's result: a label image and which labels are seeded or unseeded."""

    labels: np.ndarray  # int32 HxW, 0 = not a block
    seed_labels: list[
        int
    ]  # labels[i] is seed i's region label, 0 if the seed was dropped
    unseeded_labels: list[int]


def disk(radius: int) -> np.ndarray:
    """A boolean disk structuring element of the given radius."""
    ys, xs = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return xs**2 + ys**2 <= radius**2


def disk_view(
    center: Point, radius: int, shape: tuple[int, int]
) -> tuple[tuple[slice, slice], np.ndarray] | None:
    """Image slices and the matching disk mask for a disk at ``center``, or None if off-image."""
    height, width = shape
    cx, cy = round(center[0]), round(center[1])
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(width, cx + radius + 1), min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    element = disk(radius)
    patch = element[
        y0 - (cy - radius) : y1 - (cy - radius), x0 - (cx - radius) : x1 - (cx - radius)
    ]
    return (slice(y0, y1), slice(x0, x1)), patch


def seed_markers(
    block: np.ndarray, seeds: list[Point], spacing: float, params: DecodeParams
) -> tuple[np.ndarray, list[int]]:
    """Marker image with a disk per seed (label i+1), skipping seeds that sit on paper.

    ``block`` is the hole-filled block mask. The paper test is block coverage over a wide
    disk: the model may call the number's own black glyph background, and a test at the
    glyph would drop the seed and let a neighbor's flood take the block.
    """
    shape = block.shape
    markers = np.zeros(shape, np.int32)
    marker_radius = max(2, round(params.seed_radius_frac * spacing))
    test_radius = max(marker_radius, round(params.seed_test_radius_frac * spacing))
    coverage = block.astype(np.float32)
    labels: list[int] = []
    for index, seed in enumerate(seeds):
        test = disk_view(seed, test_radius, shape)
        marker = disk_view(seed, marker_radius, shape)
        if test is None or marker is None:
            labels.append(0)
            continue
        (rows, cols), patch = test
        if float(coverage[rows, cols][patch].mean()) < params.seed_min_block:
            labels.append(0)
            continue
        (rows, cols), patch = marker
        view = markers[rows, cols]
        view[patch & (view == 0)] = index + 1
        labels.append(index + 1)
    return markers, labels


def fill_small_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """The mask with enclosed holes (components of its complement off the border) below ``max_area`` filled."""
    count, labeled = cv2.connectedComponents((~mask).astype(np.uint8))
    components = np.asarray(labeled, dtype=np.int32)
    if count <= 1:
        return mask
    sizes = np.bincount(components.ravel(), minlength=count)
    border = np.zeros(count, bool)
    for edge in (components[0], components[-1], components[:, 0], components[:, -1]):
        border[np.unique(edge)] = True
    fill = (sizes < max_area) & ~border
    fill[0] = False
    return mask | fill[components]


def unseeded_markers(
    probabilities: np.ndarray, markers: np.ndarray, spacing: float, params: DecodeParams
) -> list[int]:
    """Add a marker for every confident interior component no seed is near; returns their labels.

    These are blocks whose page number the localizer missed (or read as an inset): the
    model saw a block, nobody claimed it. A later pass can look for a number there. A
    component counts as claimed when it comes within ``claim_radius_frac`` of a seed, not
    only when the seed's disk lies inside it: P(interior) dips on the number's own ink,
    and a marker planted beside the seed there would cut its block in two. Candidates
    are smoothed, opened and held to a block-sized area floor, since a marker on a speck
    would carve a piece out of a real block.
    """
    sigma = max(1.0, params.unseeded_smooth_frac * spacing)
    smooth = cv2.GaussianBlur(probabilities[INTERIOR], (0, 0), sigma)
    radius = max(1, round(params.unseeded_open_frac * spacing))
    opening = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    candidate = (smooth > params.unseeded_interior_prob) & (
        probabilities[BOUNDARY] < params.wall_prob
    )
    confident = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_OPEN, opening
    ).astype(bool)
    count, components = cv2.connectedComponents(confident.astype(np.uint8))
    components = components.astype(np.int32)
    count -= 1  # cv2 counts the background as a component
    if count == 0:
        return []
    reach = max(1, round(params.claim_radius_frac * spacing))
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)
    )
    near_seed = cv2.dilate((markers > 0).astype(np.uint8), element).astype(bool)
    claimed = {int(c) for c in np.unique(components[near_seed]) if c > 0}
    sizes = ndi.sum(confident, components, index=np.arange(1, count + 1))
    # The floor must not follow the seeded components' own sizes, which an early or
    # uncertain model fragments into specks.
    min_area = params.unseeded_min_area_frac * spacing**2
    next_label = int(markers.max()) + 1
    added: list[int] = []
    for component, size in enumerate(sizes, start=1):
        if component in claimed or size < min_area:
            continue
        markers[components == component] = next_label
        added.append(next_label)
        next_label += 1
    return added


def flood_domain(block: np.ndarray, spacing: float, params: DecodeParams) -> np.ndarray:
    """The block mask closed and with small pockets filled: where the watershed may flood.

    Closing seals the thin channels through which a paper bubble over a page number
    reaches the paper between blocks, so that the bubble is an enclosed pocket and gets
    filled. Blocks that abut across a narrow paper gap are joined too, which is safe: the
    elevation stays high in the gap, so floods from the two blocks' markers meet there,
    and the clip to the block mask removes the gap again.
    """
    radius = max(1, round(params.close_radius_frac * spacing))
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    closed = cv2.morphologyEx(block.astype(np.uint8), cv2.MORPH_CLOSE, element).astype(
        bool
    )
    return fill_small_holes(closed, params.hole_area_frac * spacing**2)


def decode_regions(
    probabilities: np.ndarray, seeds: list[Point], spacing: float, params: DecodeParams
) -> DecodedRegions:
    """One region per block: a watershed on P(boundary) from seed and unseeded markers.

    The flood runs over the block mask's closed, pocket-filled domain, so paper between
    blocks never joins a region while a paper bubble over a page number does not strand
    its marker; the elevation adds P(background) so that a region which must cross a gap
    does so along the thinnest paper. Regions are clipped to the block mask with its
    pockets filled.
    """
    block = probabilities[BACKGROUND] < params.block_prob
    domain = flood_domain(block, spacing, params)
    markers, seed_labels = seed_markers(domain, seeds, spacing, params)
    unseeded = unseeded_markers(probabilities, markers, spacing, params)
    markers[~domain] = 0
    if markers.max() == 0:
        return DecodedRegions(np.zeros(block.shape, np.int32), seed_labels, [])
    elevation = probabilities[BOUNDARY] + probabilities[BACKGROUND]
    labels = watershed(elevation, markers, mask=domain).astype(np.int32)
    labels[~fill_small_holes(block, params.hole_area_frac * spacing**2)] = 0
    return DecodedRegions(labels, seed_labels, unseeded)


def region_polygons(
    decoded: DecodedRegions, spacing: float, params: DecodeParams
) -> dict[int, list[Point]]:
    """Label -> simplified outer polygon (working pixels) of each region's largest blob.

    Regions that fail the size sanity caps, and rings whose filled outline would cover
    far more than their pixels, are left out.
    """
    tolerance = max(1.0, params.simplify_frac * spacing)
    min_area = params.min_region_area_frac * spacing**2
    unseeded = set(decoded.unseeded_labels)
    polygons: dict[int, list[Point]] = {}
    # Work on each label's bounding box: a whole-image pass per region is minutes per map.
    for index, window in enumerate(ndi.find_objects(decoded.labels), start=1):
        if window is None:
            continue
        rows, cols = window
        crop = decoded.labels[rows, cols] == index
        pixels = int(crop.sum())
        cap = (
            params.max_unseeded_area_frac
            if index in unseeded
            else params.max_region_area_frac
        ) * spacing**2
        if pixels < min_area or pixels > cap:
            continue
        filled = np.asarray(ndi.binary_fill_holes(crop), dtype=bool)
        if int(filled.sum()) > params.max_fill_ratio * pixels:
            continue
        polygon = mask_to_polygon(filled, tolerance)
        if len(polygon) >= 3:
            polygons[index] = [(x + cols.start, y + rows.start) for x, y in polygon]
    return polygons


@dataclass
class SegmentedKeymap:
    """A key map's learned regions in full-resolution pixels, with their page numbers."""

    polygons: dict[int, list[Point]]  # index -> polygon, indexes dense from 0
    texts: list[
        str
    ]  # polygon index -> page number text, UNSEEDED_LABEL for unseeded blocks
    probabilities: np.ndarray  # 3xHxW at the working scale, for debugging
    scale: float


def segment_keymap_image(
    rgb: np.ndarray,
    seeds: list[Point],
    texts: list[str],
    model: UNet,
    *,
    device: torch.device,
    params: DecodeParams | None = None,
) -> SegmentedKeymap:
    """Segment a full-resolution key map given its detected page numbers."""
    params = params or DecodeParams()
    scale = working_scale_for_seeds(rgb.shape, seeds)
    working = resize_image(rgb, scale)
    working_seeds = scale_points(seeds, scale)
    spacing = (
        nearest_neighbor_distance(working_seeds)
        if len(working_seeds) >= MIN_SEEDS_FOR_SPACING
        else float(TARGET_SPACING_PX)
    )
    if spacing <= 0:
        spacing = float(TARGET_SPACING_PX)
    probabilities = predict_class_maps(model, model_input(working), device)
    decoded = decode_regions(probabilities, working_seeds, spacing, params)
    polygons = region_polygons(decoded, spacing, params)
    label_text = {
        label: texts[i] for i, label in enumerate(decoded.seed_labels) if label > 0
    }
    out_polygons: dict[int, list[Point]] = {}
    out_texts: list[str] = []
    for label, polygon in sorted(polygons.items()):
        out_polygons[len(out_texts)] = [(x / scale, y / scale) for x, y in polygon]
        out_texts.append(label_text.get(label, UNSEEDED_LABEL))
    return SegmentedKeymap(out_polygons, out_texts, probabilities, scale)


def segment_keymap(
    keymap_json: Path,
    model: UNet,
    device: torch.device,
    params: DecodeParams | None = None,
    *,
    image_path: Path | None = None,
) -> tuple[SegmentedKeymap, Path]:
    """Segment the key map a ``<stem>.keymap.json`` describes; returns regions and the image path.

    The image defaults to the sibling ``<stem>.jpg``; a mirror run keeps its sidecars
    under ``runs/<tag>/raw/`` with the sheet in the volume's ``raw/``, so callers can
    name it.
    """
    if image_path is None:
        image_path = keymap_json.with_name(
            keymap_json.name.replace(".keymap.json", ".jpg")
        )
    rgb = read_rgb(image_path)
    seeds, texts = seed_points(keymap_json)
    return segment_keymap_image(
        rgb, seeds, texts, model, device=device, params=params
    ), image_path


def class_map_image(probabilities: np.ndarray) -> np.ndarray:
    """An RGB rendering of the class maps: interior green, boundary red, background white."""
    interior = probabilities[INTERIOR]
    boundary = probabilities[BOUNDARY]
    rgb = np.empty(probabilities.shape[1:] + (3,), np.float32)
    rgb[..., 0] = 1.0 - interior
    rgb[..., 1] = 1.0 - boundary
    rgb[..., 2] = 1.0 - interior - boundary
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


DEVICE_ENV_VAR = "MAPSNAP_DEVICE"  # force a torch device, e.g. "cpu" beside a GPU job


def select_device() -> torch.device:
    """MPS, else CUDA, else CPU; MAPSNAP_DEVICE overrides."""
    forced = os.environ.get(DEVICE_ENV_VAR)
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "keymap_json", type=Path, nargs="+", help="<stem>.keymap.json file(s)."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(os.environ.get(WEIGHTS_ENV_VAR, MODEL_PATH)),
        help=f"Checkpoint to load (default: ${WEIGHTS_ENV_VAR} or {MODEL_PATH.name}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Regions sidecar to write (default: <stem>.regions.panels.json beside the input; one input only).",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        help="Write a labeled overlay PNG here (one input only).",
    )
    parser.add_argument(
        "--maps", type=Path, help="Write the class maps as a PNG here (one input only)."
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="The key-map JPEG (default: <stem>.jpg beside the input; one input only).",
    )
    args = parser.parse_args()
    if len(args.keymap_json) > 1 and (
        args.output or args.overlay or args.maps or args.image
    ):
        sys.exit("--output/--overlay/--maps/--image take a single input.")
    device = select_device()
    model = load_region_unet(args.weights, device)
    for keymap_json in args.keymap_json:
        segmented, image_path = segment_keymap(
            keymap_json, model, device, image_path=args.image
        )
        with Image.open(image_path) as image:
            size = image.size
        doc = regions_panels_doc(
            image_path.name, size, segmented.polygons, segmented.texts
        )
        output = args.output or keymap_json.with_name(
            keymap_json.name.replace(".keymap.json", ".regions.panels.json")
        )
        output.write_text(json.dumps(doc))
        unseeded = sum(1 for text in segmented.texts if text == UNSEEDED_LABEL)
        print(
            f"{keymap_json}: {len(segmented.texts)} regions ({unseeded} unseeded) "
            f"at scale {segmented.scale:.3f} -> {output}"
        )
        if args.overlay:
            render_overlay(
                image_path, segmented.polygons, segmented.texts, args.overlay
            )
        if args.maps:
            Image.fromarray(class_map_image(segmented.probabilities)).save(args.maps)


if __name__ == "__main__":
    main()
