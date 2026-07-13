"""Find the colored area drawn around each page number on a Sanborn key map.

On a key map every page is a saturated colored block (yellow, green, blue, pink, tan) with
its page number printed in black at the centre. Blocks are separated either by a black
boundary line or by a direct change of colour; thinner black street-grid lines run *inside* a
block. The white/tan paper between blocks is unsaturated.

Recovering one polygon per page is a colour-cluster segmentation. The (line-smoothed) image is
quantized into a handful of colours; the largest cluster(s) are the "background"
(paper/streets) and the rest are blocks. By default the quantization palette is *seeded*: the
median colour around each page number gives the block fills actually in use (deduped to the
few distinct ones), and global k-means centres fill in the rest — so on a pale, aged map a
block fill that plain k-means would merge with the paper still gets its own cluster, and a
seed-palette cluster is never flagged as background (``--global-kmeans`` restores plain
k-means). Each remaining cluster's mask is morphologically closed
then opened — the opening severs the hairline slivers along which one block would otherwise bleed
into a neighbour — and split into connected components. Each page number is assigned to the
component its box sits on (preferring the larger component when the box straddles two near-identical
colours, i.e. a block whose colour k-means split across two clusters); when two page numbers land in
one component it is split between them by a distance watershed. A number whose box lands on
background paper (not a colour block) is dropped. Interior holes (the black page number, an enclosed
courtyard) are filled; edge concavities are kept, since some blocks are genuinely concave.

    uv run python -m mapsnap.keymap.page_regions data/chicago_il_1950_vol_1/p0b.keymap.json
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.color import lab2rgb, rgb2lab
from skimage.segmentation import watershed

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 (full-res pixels)
ScaledBox = tuple[int, int, int, int]  # col0, row0, col1, row1 in label-image pixels


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
    anywhere from ~2k to ~7k px). The line-smooth window is then derived from the median seed
    spacing rather than set in absolute pixels, so the same params work across maps of different
    resolution and block density (Chicago vs New Orleans).

    target_long_side: the segmentation runs on a copy scaled so max(width, height) is this many
        pixels (never upscaled); polygons are returned in full-resolution coordinates.
    n_clusters / cluster_seed: the image is k-means colour-quantized into this many clusters;
        ~8 separate the pastel block colours from the few background/paper tints. See cluster_image.
    line_smooth_frac: median-blur window (before clustering) as a fraction of the median seed
        spacing; it erases black grid lines thinner than the window so they do not split a block.
    background_area_frac: a cluster is "background" (paper/streets/margin) if its pixel area is at
        least this fraction of the largest cluster's. See background_clusters.
    cluster_close_radius: a cluster mask is morphologically closed by this radius (scaled pixels)
        before the flood fill, so it can bridge a stray pixel or two of another colour.
    cluster_open_radius: the cluster mask is then morphologically opened by this radius, deleting
        thin slivers/arms (narrower than 2x the radius) that would otherwise let one block's flood
        leak through a hairline connection into a neighbour. Keep it well below a block half-width.
    cluster_tie_frac: when a seed's box straddles two clusters with comparable pixel counts (the
        smaller within this fraction of the larger), the block was split across two near-identical
        colours, so the seed is assigned to whichever candidate has the larger connected component
        (its real block) rather than the pixel-majority fragment.
    simplify_tolerance: Douglas-Peucker tolerance (in scaled pixels) for the output polygon.
    use_global_kmeans: skip the seed-colour palette and quantize with plain k-means (the
        pre-palette behaviour; an ablation/debugging switch).
    palette_dedup_distance: seed colours closer than this (CIELAB) are one palette entry. Sweeping
        4-20 against truth IoU: coarse (16) wins — the palette collapses to the handful of fills
        actually printed, while finer spacing splits one fill across near-identical centres and
        fragments its blocks.
    palette_global_min_distance: global k-means centres closer than this to a palette colour are
        dropped (the palette entry owns those pixels).
    palette_paper_distance: a seed colour this close to the dominant (paper) centre is a number
        printed on open paper; it is excluded so it cannot drag the paper into the foreground.
    family_chroma_distance / family_contact_frac: after quantization, two foreground clusters
        are merged into one "family" mask when their centres are within family_chroma_distance
        in Lab (a, b) — aging mottle and stains shift a fill's lightness, not its hue — AND
        their pixels interleave (mutual contact at least family_contact_frac of the smaller
        cluster). The contact gate is what separates mottle (interleaved within blocks) from a
        palette's genuine light/dark variants of one hue (distinct fills that only touch along
        block borders). See merge_cluster_families.
    family_min_lightness: clusters whose centre L is below this never join a family. The
        black line-work is a low-chroma cluster too, so pure-chroma matching would merge it
        into any pale-tint family — and the ink lattice connects the whole map, ballooning
        one region to the full sheet (Hudson p44, 99.8% of the key map).
    max_region_area_factor: a region whose mask exceeds this multiple of the median seed
        spacing squared is dropped as a runaway (leaves room for giant waterfront sheets at
        ~9% of a key map).
    """

    target_long_side: int = 3000
    n_clusters: int = 8
    cluster_seed: int = 0
    line_smooth_frac: float = 0.08
    background_area_frac: float = 0.5
    cluster_close_radius: int = 2
    cluster_open_radius: int = 3
    cluster_tie_frac: float = 0.7
    simplify_tolerance: float = 4.0
    use_global_kmeans: bool = False
    palette_dedup_distance: float = 16.0
    palette_global_min_distance: float = 8.0
    palette_paper_distance: float = 3.0
    family_chroma_distance: float = 12.0
    family_contact_frac: float = 0.10
    family_min_lightness: float = 30.0
    max_region_area_factor: float = 60.0


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


def cluster_image(
    rgb_u8: np.ndarray, n_clusters: int, line_smooth_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """k-means colour-cluster the median-blurred key map.

    Median-blur first (window ``line_smooth_size``) so a thin black grid line — the local minority
    — is replaced by the surrounding block colour and does not split the block. Fit k-means on a
    random subsample of the CIELAB pixels (for speed), then assign every pixel to its nearest
    centre. Returns the per-pixel cluster-label image (H x W) and the CIELAB centres (n_clusters x 3).
    """
    lab = rgb2lab(cv2.medianBlur(rgb_u8, line_smooth_size) / 255.0)
    flat = lab.reshape(-1, 3).astype(np.float32)
    centers = kmeans_centers(flat, n_clusters, seed)
    return assign_to_centers(flat, centers).reshape(rgb_u8.shape[:2]), centers


def kmeans_centers(flat: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """k-means centres of (a 40k subsample of) flattened CIELAB pixels."""
    generator = np.random.default_rng(seed)
    if len(flat) > 40000:
        sample = flat[generator.choice(len(flat), size=40000, replace=False)]
    else:
        sample = flat
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # bestLabels is an output here; pass an empty array (cv2's stub rejects None).
    best_labels = np.empty((0,), dtype=np.int32)
    _, _, centers = cv2.kmeans(
        sample, n_clusters, best_labels, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    return centers


def assign_to_centers(flat: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest centre for every flattened pixel."""
    labels = np.zeros(len(flat), dtype=np.int32)
    best = np.full(len(flat), np.inf, dtype=np.float32)
    for index, center in enumerate(centers):
        distance = ((flat - center) ** 2).sum(axis=1)
        closer = distance < best
        best[closer] = distance[closer]
        labels[closer] = index
    return labels


def seed_palette(
    blurred_lab: np.ndarray, scaled_boxes: list["ScaledBox"]
) -> np.ndarray:
    """Median CIELAB colour in a 1.5x-expanded window around each seed box.

    The page number is printed on its block, so after the line-smoothing blur the window
    is dominated by the block's fill colour.
    """
    height, width = blurred_lab.shape[:2]
    colors = []
    for col0, row0, col1, row1 in scaled_boxes:
        half_w = max(2, (col1 - col0) // 2)
        half_h = max(2, (row1 - row0) // 2)
        c0, c1 = max(0, col0 - half_w), min(width, col1 + half_w + 1)
        r0, r1 = max(0, row0 - half_h), min(height, row1 + half_h + 1)
        patch = blurred_lab[r0:r1, c0:c1].reshape(-1, 3)
        colors.append(np.median(patch, axis=0))
    return np.asarray(colors, dtype=np.float32)


def dedup_palette(colors: np.ndarray, min_distance: float) -> np.ndarray:
    """Greedy dedup: keep a colour only if no already-kept colour is within min_distance."""
    kept: list[np.ndarray] = []
    for color in colors:
        if all(np.linalg.norm(color - k) >= min_distance for k in kept):
            kept.append(color)
    return np.asarray(kept, dtype=np.float32)


def cluster_image_seeded(
    rgb_u8: np.ndarray,
    scaled_boxes: list["ScaledBox"],
    line_smooth_size: int,
    params: "RegionParams",
) -> tuple[np.ndarray, set[int], np.ndarray] | None:
    """Quantize with a seed-colour palette plus filtered global k-means centres.

    The palette entries are the block fills actually in use, so a pale fill that global
    k-means would merge with the paper keeps its own cluster, and palette clusters are
    never flagged background — the area rule applies to the global centres only. Returns
    (label image, background cluster ids, all centres), or None when no usable palette
    colour survives (every seed sits on paper), in which case the caller should fall back
    to plain k-means.
    """
    lab = rgb2lab(cv2.medianBlur(rgb_u8, line_smooth_size) / 255.0)
    flat = lab.reshape(-1, 3).astype(np.float32)
    global_centers = kmeans_centers(flat, params.n_clusters, params.cluster_seed)
    counts = np.bincount(
        assign_to_centers(flat, global_centers), minlength=params.n_clusters
    )
    paper = global_centers[counts.argmax()]

    palette = seed_palette(lab, scaled_boxes)
    palette = palette[
        np.linalg.norm(palette - paper, axis=1) >= params.palette_paper_distance
    ]
    palette = dedup_palette(palette, params.palette_dedup_distance)
    if not len(palette):
        return None

    kept_global = [
        center
        for center in global_centers
        if np.linalg.norm(palette - center, axis=1).min()
        >= params.palette_global_min_distance
    ]
    centers = np.vstack([palette, *kept_global]).astype(np.float32)
    labels = assign_to_centers(flat, centers).reshape(rgb_u8.shape[:2])

    cluster_counts = np.bincount(labels.ravel(), minlength=len(centers))
    threshold = params.background_area_frac * cluster_counts.max()
    background = {
        cluster
        for cluster in range(len(palette), len(centers))
        if cluster_counts[cluster] >= threshold
    }
    return labels, background, centers


def background_clusters(
    labels: np.ndarray, n_clusters: int, area_frac: float
) -> set[int]:
    """Cluster ids that are background (paper/streets/margin): those covering a large area.

    The paper around and between blocks is the most extensive colour on a key map, so the biggest
    cluster(s) are background. A cluster counts as background if its pixel area is at least
    ``area_frac`` of the largest cluster's — this catches the two or three paper / aged-paper tints
    while leaving the smaller per-colour block clusters as foreground.
    """
    counts = np.bincount(labels.ravel(), minlength=n_clusters)
    threshold = area_frac * counts.max()
    return {cluster for cluster in range(n_clusters) if counts[cluster] >= threshold}


def clean_cluster_mask(
    cluster_mask: np.ndarray, close_radius: int, open_radius: int
) -> np.ndarray:
    """Close then open a cluster's boolean mask: bridge faint gaps, then delete thin slivers/arms.

    Closing (radius ``close_radius``) bridges a stray pixel or two of another colour inside a block
    (a faint grid line). Opening (radius ``open_radius``) then erodes away any connection thinner
    than ``2 * open_radius`` — the hairline slivers along which one block's flood would otherwise
    leak into a neighbour — while leaving the compact blocks (bar slight corner rounding).
    """
    mask = cluster_mask.astype(np.uint8)
    if close_radius > 0:
        size = 2 * close_radius + 1
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    if open_radius > 0:
        size = 2 * open_radius + 1
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    return mask.astype(bool)


def merge_cluster_families(
    labels: np.ndarray,
    background: set[int],
    centers: np.ndarray,
    params: RegionParams,
) -> tuple[np.ndarray, set[int], int]:
    """Merge interleaved same-hue foreground clusters; return (labels, background, count).

    On an aged map one block's fill drifts in lightness (mottle, stains, fading), so its
    pixels split across several quantization clusters and the block's region is clipped to
    the seed's share. Such phases share a hue — Lab (a, b) — and interleave spatially,
    whereas a palette's genuine light/dark variants of one hue are separate blocks that only
    touch along borders. Clusters are therefore unioned when their centres are close in
    (a, b) AND either's pixels lie within a few pixels of at least
    ``params.family_contact_frac`` of the smaller one. Background clusters never merge,
    and neither do near-black ones (below ``params.family_min_lightness``): the ink
    line-work is itself a low-chroma cluster whose lattice touches everything, and a
    family containing it becomes one component spanning the whole map.
    """
    fg_ids = [c for c in range(len(centers)) if c not in background]
    mergeable = [c for c in fg_ids if centers[c][0] >= params.family_min_lightness]
    parent = {c: c for c in mergeable}

    def find(c: int) -> int:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = {c: cv2.dilate((labels == c).astype(np.uint8), kernel) for c in mergeable}
    areas = {c: int((labels == c).sum()) for c in mergeable}
    for i, a in enumerate(mergeable):
        for b in mergeable[i + 1 :]:
            chroma = float(np.linalg.norm(centers[a][1:] - centers[b][1:]))
            if chroma >= params.family_chroma_distance:
                continue
            smaller = min(areas[a], areas[b])
            if smaller == 0:
                continue
            contact = max(
                int((dilated[a].astype(bool) & (labels == b)).sum()),
                int((dilated[b].astype(bool) & (labels == a)).sum()),
            )
            if contact / smaller < params.family_contact_frac:
                continue
            parent[find(a)] = find(b)

    roots = sorted({find(c) for c in mergeable})
    mapping = np.zeros(len(centers), dtype=np.int32)
    for c in mergeable:
        mapping[c] = roots.index(find(c))
    next_id = len(roots)
    for c in fg_ids:  # too-dark clusters stay foreground, each its own singleton
        if c not in parent:
            mapping[c] = next_id
            next_id += 1
    new_background: set[int] = set()
    for c in sorted(background):
        mapping[c] = next_id
        new_background.add(next_id)
        next_id += 1
    return mapping[labels], new_background, next_id


def single_seed_region(
    labels: np.ndarray,
    background: set[int],
    scaled_box: ScaledBox,
    params: RegionParams,
) -> np.ndarray | None:
    """Best-effort region for one seed from a label image: its majority cluster's component.

    Used as the fallback when a seed's family region blows past the area sanity cap (a
    family can absorb a map-spanning pale tint); the unmerged cluster is what the seed
    would have gotten before family merging.
    """
    col0, row0, col1, row1 = scaled_box
    patch = labels[row0 : row1 + 1, col0 : col1 + 1]
    if patch.size == 0:
        return None
    counts = np.bincount(patch.ravel())
    for cluster in background:
        if cluster < len(counts):
            counts[cluster] = 0
    if counts.max() == 0:
        return None
    cluster = int(counts.argmax())
    cleaned = clean_cluster_mask(
        labels == cluster, params.cluster_close_radius, params.cluster_open_radius
    )
    components, _ = ndi.label(cleaned)  # type: ignore[misc]
    component = box_component(components, scaled_box)
    if component is None:
        return None
    return components == component


def scale_boxes(
    seeds: list[Box], scale: float, shape: tuple[int, ...]
) -> list[ScaledBox]:
    """Seed boxes in label-image (scaled) pixels, clamped to the image."""
    height, width = shape[:2]
    return [
        (
            max(0, int(round(box[0] * scale))),
            max(0, int(round(box[1] * scale))),
            min(width - 1, int(round(box[2] * scale))),
            min(height - 1, int(round(box[3] * scale))),
        )
        for box in seeds
    ]


def box_component(components: np.ndarray, box: tuple[int, int, int, int]) -> int | None:
    """Label of the connected component holding the most of ``box`` (col0,row0,col1,row1), or None."""
    col0, row0, col1, row1 = box
    patch = components[row0 : row1 + 1, col0 : col1 + 1]
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return int(np.bincount(patch).argmax())


def split_component(
    component_mask: np.ndarray, boxes: list[tuple[int, int, int, int]]
) -> list[np.ndarray]:
    """Partition one component among several seed boxes (a distance-transform watershed).

    When two page numbers both land in the same connected component their blocks abut without a
    separating street (or a hairline join survived the opening); rather than let one swallow the
    other, split the component by flooding from each seed box and meeting along the distance-
    transform ridge between them. Returns one boolean mask per input box, in order.
    """
    markers = np.zeros(component_mask.shape, dtype=np.int32)
    for label, (col0, row0, col1, row1) in enumerate(boxes, start=1):
        inside = component_mask[row0 : row1 + 1, col0 : col1 + 1]
        markers[row0 : row1 + 1, col0 : col1 + 1][inside] = label
    distance: np.ndarray = ndi.distance_transform_edt(component_mask)  # type: ignore[assignment]
    basins = watershed(-distance, markers, mask=component_mask)
    return [basins == label for label in range(1, len(boxes) + 1)]


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


def working_scale(image_shape: tuple[int, ...], target_long_side: int) -> float:
    """Downscale factor so the image's longer side is ``target_long_side`` (never upscales)."""
    height, width = image_shape[:2]
    return min(1.0, target_long_side / max(height, width))


def working_geometry(
    image_shape: tuple[int, ...], seeds: list[Box], params: RegionParams
) -> tuple[float, int]:
    """The downscale factor and median-blur window for an image and its seeds.

    The blur window is derived from the median seed spacing (in scaled pixels) so it tracks the
    map's drawing scale; returns it as an odd size of at least 3. Shared by the segmentation and
    the debug-image outputs so they stay in lockstep.
    """
    scale = working_scale(image_shape, params.target_long_side)
    spacing = nearest_neighbor_distance([box_center(box) for box in seeds]) * scale
    line_smooth_size = max(3, int(round(params.line_smooth_frac * spacing)) | 1)
    return scale, line_smooth_size


def segment_page_regions(
    rgb: np.ndarray, seeds: list[Box], params: RegionParams
) -> dict[int, list[Point]]:
    """Polygon (full-res pixels) of the colored block around each seeded page number.

    rgb is a float image in [0, 1] (H x W x 3). ``seeds`` are page-number bounding boxes in
    full-resolution coordinates. The image is colour-clustered — seed-palette quantization by
    default, plain k-means with ``params.use_global_kmeans`` (or as the fallback when no
    palette colour survives the paper filter); each cluster mask is cleaned
    (closed then opened, so hairline slivers between blocks are severed) and split into connected
    components. Each page number's block is the component its box sits in; if two page numbers land
    in the same component their blocks abut, so the component is split between them by a distance
    watershed. A page number whose box lands on a background cluster (printed on open paper, not a
    block) is dropped. Returns a map from seed index to its block polygon, omitting dropped seeds.
    """
    scale, line_smooth_size = working_geometry(rgb.shape, seeds, params)
    height, width = rgb.shape[:2]
    scaled_u8 = np.clip(
        cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    scaled_boxes = scale_boxes(seeds, scale, scaled_u8.shape)
    seeded = (
        None
        if params.use_global_kmeans or not seeds
        else cluster_image_seeded(scaled_u8, scaled_boxes, line_smooth_size, params)
    )
    unmerged: tuple[np.ndarray, set[int]] | None = None
    if seeded is not None:
        raw_labels, raw_background, centers = seeded
        unmerged = (raw_labels, raw_background)
        labels, background, n_clusters = merge_cluster_families(
            raw_labels, raw_background, centers, params
        )
    else:
        labels, _ = cluster_image(
            scaled_u8, params.n_clusters, line_smooth_size, params.cluster_seed
        )
        background = background_clusters(
            labels, params.n_clusters, params.background_area_frac
        )
        n_clusters = params.n_clusters

    # Clean (close then open) and connected-component each non-background cluster's mask once.
    components: dict[int, np.ndarray] = {}
    component_sizes: dict[int, np.ndarray] = {}
    for cluster in range(n_clusters):
        if cluster in background:
            continue
        mask = clean_cluster_mask(
            labels == cluster, params.cluster_close_radius, params.cluster_open_radius
        )
        labelled, _ = ndi.label(mask)  # type: ignore[misc]
        components[cluster] = labelled
        component_sizes[cluster] = np.bincount(labelled.ravel())

    # Assign each page number to one (cluster, component) = its block, grouping seeds that land
    # in the same one. The seed's cluster is its box's pixel majority, except that near-tied
    # clusters (a block split across two similar colours) defer to whichever has the larger
    # component, so both numbers on such a block land in the same component and get split below.
    members_by_region: dict[tuple[int, int], list[tuple[int, ScaledBox]]] = defaultdict(
        list
    )
    for index, scaled_box in enumerate(scaled_boxes):
        col0, row0, col1, row1 = scaled_box
        patch = labels[row0 : row1 + 1, col0 : col1 + 1]
        if patch.size == 0:
            continue
        counts = np.bincount(patch.ravel(), minlength=n_clusters)
        for cluster in background:
            counts[cluster] = 0  # the page number sits on paper, not a colour block
        if counts.max() == 0:
            continue
        candidates = np.where(counts >= params.cluster_tie_frac * counts.max())[0]
        choice: tuple[int, int] | None = None
        best_size = -1
        for cluster in candidates:
            component = box_component(components[cluster], scaled_box)
            if component is None:
                continue
            size = int(component_sizes[cluster][component])
            if size > best_size:
                best_size = size
                choice = (int(cluster), component)
        if choice is not None:
            members_by_region[choice].append((index, scaled_box))

    inverse_scale = 1.0 / scale
    region_masks: dict[int, np.ndarray] = {}
    for (cluster, component), members in members_by_region.items():
        component_mask = components[cluster] == component
        if len(members) == 1:
            region_masks[members[0][0]] = component_mask
        else:  # two+ page numbers share a component: split it between them
            boxes = [scaled_box for _, scaled_box in members]
            for (index, _), piece in zip(
                members, split_component(component_mask, boxes)
            ):
                region_masks[index] = piece

    spacing = nearest_neighbor_distance([box_center(box) for box in seeds]) * scale
    max_region_area = params.max_region_area_factor * spacing**2
    polygons: dict[int, list[Point]] = {}
    for index, mask in region_masks.items():
        if spacing > 0 and int(mask.sum()) > max_region_area:
            # Runaway region (a family that swallowed a map-spanning tint): retry with
            # this seed's unmerged cluster, and drop the seed only if that is huge too.
            fallback = (
                single_seed_region(*unmerged, scaled_boxes[index], params)
                if unmerged is not None
                else None
            )
            if fallback is None or int(fallback.sum()) > max_region_area:
                continue
            mask = fallback
        filled: np.ndarray = ndi.binary_fill_holes(mask)  # type: ignore[assignment]
        polygon = mask_to_polygon(filled, params.simplify_tolerance)
        if len(polygon) >= 3:
            polygons[index] = [
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
        ring = [[round(x, 1), round(y, 1)] for x, y in polygon]
        ring.append(ring[0])  # GeoJSON-style rings are explicitly closed
        panels.append(ring)
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
        help="Also write the colour-quantized image (the elevation's basis) to this path.",
    )
    parser.add_argument(
        "--global-kmeans",
        action="store_true",
        help="Quantize with plain k-means instead of the seed-colour palette (ablation).",
    )
    args = parser.parse_args()

    image_path = args.image or keymap_image_path(args.keymap)
    output = args.output or args.keymap.with_name(
        image_path.stem + ".regions.panels.json"
    )

    params = RegionParams(use_global_kmeans=args.global_kmeans)
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
        scale, line_smooth_size = working_geometry(rgb.shape, seeds, params)
        scaled = cv2.resize(
            rgb,
            (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))),
        )
        if args.blur_debug:
            Image.fromarray(cv2.medianBlur(scaled, line_smooth_size)).save(
                args.blur_debug
            )
            print(f"Wrote {args.blur_debug}")
        if args.cluster_debug:
            seeded = (
                None
                if params.use_global_kmeans or not seeds
                else cluster_image_seeded(
                    scaled,
                    scale_boxes(seeds, scale, scaled.shape),
                    line_smooth_size,
                    params,
                )
            )
            if seeded is not None:
                labels, _, centers = seeded
            else:
                labels, centers = cluster_image(
                    scaled, params.n_clusters, line_smooth_size, params.cluster_seed
                )
            quant_rgb = (np.clip(lab2rgb(centers[labels]), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(quant_rgb).save(args.cluster_debug)
            print(f"Wrote {args.cluster_debug}")


if __name__ == "__main__":
    main()
