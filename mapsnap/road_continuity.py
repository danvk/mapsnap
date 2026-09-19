"""Place a page by continuing its neighbours' roads.

A Sanborn street is straight and long: it runs off one sheet and onto the
next. So a page that could not be fitted from its own street names can still
be placed by asking where its road lines line up with the roads of the pages
already placed around it, *extended* past their sheet edges into the gap. This
is a whole-road constraint rather than a seam match: a neighbour two sheets
away still votes, and a street that the neighbour has read a name for pins the
target's same-named line to it, which is what defeats the block-grid aliasing
that sank the earlier neighbour-fitting attempts (#165, #125).

Pipeline, all truth-free:

1. ``extract_road_lines``: straight road ridges from a page's P(road) map. A
   rotation-projection search (a Radon transform in effect) finds candidate
   (angle, offset) peaks at quarter resolution; each is verified and refined
   along a band at full resolution and kept where the ridge runs for at least
   ``min_length_px``.
2. ``attach_label_names``: a page's admitted street labels name the lines they
   sit on.
3. ``world_lines``: the placed pages' lines in a metre frame, extended by
   ``extension_m`` past any endpoint that reaches the sheet edge. A street
   that ends mid-sheet really ends, and is not extrapolated.
4. ``place_page``: rotation from the local grid orientation (up to two modes
   of the neighbours' lines near the prior against the target's own), then a
   translation vote from every pair of non-parallel line correspondences that
   are name-compatible, then Nelder-Mead refinement on a continuity score
   minus raster contradictions. The key-map prior weights the vote softly:
   a page with a single through-street is pinned across it but free along
   it, and the prior settles that.

Status: experimental. Nothing in ``fit`` consumes the ``*.georef-continuity.json``
sidecars this writes; the channel needs an A/B over the corpus first. Measured
on the 19 truth volumes (2026-09-19): 35 of 123 unplaced or >50 ft pages land
within 50 ft at rank 1, 53 within 100 ft; the accept gate takes 37 of them with
24 within 50 ft. Strong on regular grids where the page has named
through-streets (Detroit's factory cluster: 9 of 9 within 80 ft), useless on
winding suburbs (Asheville) and on pages with no road content.

Usage::

    mapsnap road-continuity DIR --pages p6,p7,p8 --eval
    mapsnap road-continuity DIR --unplaced --chain --write
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from mapsnap.compare_iiif_georef import extract_gcps, fit_affine
from mapsnap.georef_from_labels import LabelFeature, prepare_label_features
from mapsnap.keymap.locate import KeymapLocator
from mapsnap.osm_to_centerlines import load_centerlines
from mapsnap.streets import build_block_index

EARTH_M_PER_DEG_LAT = 110540.0
EARTH_M_PER_DEG_LON = 111320.0
FT_PER_M = 3.280839895

# Label admission for naming lines: permissive on size (25%-scale pages have
# 12-30 px labels) and canonicalised against the block index like georef does.
MIN_LABEL_CONFIDENCE = 0.5
MIN_LABEL_LONG_SIDE = 40.0
MIN_LABEL_SHORT_SIDE = 12.0
MIN_LABEL_ASPECT = 2.0


@dataclass
class RoadLine:
    """A straight road ridge on one page, in that page's pixel frame."""

    start: np.ndarray
    end: np.ndarray
    angle: float  # radians in [0, pi)
    names: set[str] = field(default_factory=set)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


@dataclass
class WorldLine:
    """A placed page's road line in the metre frame, with its extrapolation."""

    origin: np.ndarray
    direction: np.ndarray  # unit vector
    length: float
    angle: float
    names: set[str]
    page: str
    extend_before: float
    extend_after: float


@dataclass
class PlacedPage:
    """A page with a pose: pixel -> metre-frame affine, size, and its road lines."""

    stem: str
    affine: np.ndarray  # 2x3, page px -> frame xy
    width: int
    height: int
    lines: list[RoadLine]

    def to_xy(self, px: float, py: float) -> np.ndarray:
        return self.affine @ np.array([px, py, 1.0])

    @property
    def centre(self) -> np.ndarray:
        return self.to_xy(self.width / 2, self.height / 2)


@dataclass
class Prior:
    """Where the key map says the page is: frame-xy centre and search radius."""

    centre: np.ndarray
    radius_m: float


@dataclass
class PlacementOptions:
    """Knobs of the search; the defaults are the measured ones."""

    scale_m_per_px: float
    extension_m: float = 450.0
    prior_sigma_m: float | None = 60.0
    angle_tolerance_deg: float = 2.5
    max_seeds: int = 24
    contradiction_weight: float = 1.5


@dataclass
class Candidate:
    """One refined pose: affine (px -> frame xy), its scores, the names it matched."""

    affine: np.ndarray
    rotation_deg: float
    vote: float
    score: float
    names: set[str]
    rmse_ft: float | None = None

    @property
    def translation(self) -> np.ndarray:
        return self.affine[:, 2]


@dataclass
class Placement:
    """The ranked candidates for one page and the evidence behind them."""

    stem: str
    candidates: list[Candidate]
    anchors: list[str]
    target_lines: int
    named_lines: int
    truth_score: float | None = None

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def margin(self) -> float:
        """Best score over the next candidate more than 60 ft away (inf if none)."""
        if not self.candidates:
            return 0.0
        best = self.candidates[0]
        for other in self.candidates[1:]:
            apart = np.linalg.norm(other.translation - best.translation) > 18.0
            if apart and other.score > 0:
                return best.score / other.score
        return float("inf")

    def accepted(self, *, min_score: float = 10.0, min_margin: float = 1.15) -> bool:
        """The gate: a confident, unambiguous best candidate."""
        best = self.best
        return (
            best is not None and best.score >= min_score and self.margin >= min_margin
        )


class Frame:
    """The volume's metre frame: x east, y south, about a lon/lat origin."""

    def __init__(self, lon0: float, lat0: float) -> None:
        self.lon0 = lon0
        self.lat0 = lat0
        self.kx = EARTH_M_PER_DEG_LON * math.cos(math.radians(lat0))
        self.ky = EARTH_M_PER_DEG_LAT

    def affine_to_xy(self, affine: np.ndarray) -> np.ndarray:
        """Convert a px -> lon/lat affine into a px -> frame-xy affine."""
        scale = np.array([[self.kx, 0.0], [0.0, -self.ky]])
        shift = np.array(
            [[0.0, 0.0, -self.kx * self.lon0], [0.0, 0.0, self.ky * self.lat0]]
        )
        return scale @ affine + shift

    def affine_to_lonlat(self, affine_xy: np.ndarray) -> np.ndarray:
        """Convert a px -> frame-xy affine back into px -> lon/lat."""
        scale = np.array([[1.0 / self.kx, 0.0], [0.0, -1.0 / self.ky]])
        shift = np.array([[0.0, 0.0, self.lon0], [0.0, 0.0, self.lat0]])
        return scale @ affine_xy + shift

    def lonlat_to_xy(self, lon: float, lat: float) -> np.ndarray:
        return np.array([(lon - self.lon0) * self.kx, -(lat - self.lat0) * self.ky])


def rotation_of(affine: np.ndarray) -> float:
    """Rotation (radians) of a px -> frame-xy affine."""
    return math.atan2(affine[1, 0], affine[0, 0])


def scale_of(affine: np.ndarray) -> float:
    """Metres per pixel of a px -> frame-xy affine."""
    return float(math.hypot(affine[0, 0], affine[1, 0]))


def pose_affine(translation: np.ndarray, rotation: float, scale: float) -> np.ndarray:
    """A px -> frame-xy affine at a fixed scale from (translation, rotation)."""
    cos, sin = math.cos(rotation), math.sin(rotation)
    return np.array(
        [
            [scale * cos, -scale * sin, translation[0]],
            [scale * sin, scale * cos, translation[1]],
        ]
    )


def corners_xy(page_size: tuple[int, int], affine: np.ndarray) -> np.ndarray:
    """The four page corners under an affine, in the order georef writes them."""
    width, height = page_size
    pixels = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], float
    )
    return (affine @ pixels.T).T


def corner_rmse_ft(
    page_size: tuple[int, int], affine: np.ndarray, truth: np.ndarray
) -> float:
    """RMS corner disagreement (feet) between two px -> frame-xy affines."""
    delta = corners_xy(page_size, affine) - corners_xy(page_size, truth)
    return float(np.sqrt((delta**2).sum(axis=1).mean())) * FT_PER_M


# ---------------------------------------------------------------------------
# 1. Road lines from P(road)
# ---------------------------------------------------------------------------


def extract_road_lines(
    prob: np.ndarray,
    *,
    min_length_px: int = 300,
    downscale: int = 4,
    angle_step_deg: float = 0.5,
    peak_min: float = 0.22,
) -> list[RoadLine]:
    """Straight road ridges of a P(road) map (float32 in [0, 1], page resolution).

    Candidate lines are peaks of the mean P(road) along every direction and
    offset (the map rotated so a direction becomes horizontal, then summed per
    row), found at ``downscale``. Each candidate is verified at full resolution
    along a +-5 px band: runs where the band's mean P(road) stays above 0.35,
    bridging gaps up to 40 px, become lines if at least ``min_length_px`` long.
    The line is then refined to the P(road)-weighted ridge centre by least
    squares, and near-duplicates (within 1.5 degrees and 8 px) are merged.
    """
    height, width = prob.shape
    small = cv2.resize(
        prob, (width // downscale, height // downscale), interpolation=cv2.INTER_AREA
    )
    small_h, small_w = small.shape
    diag = math.ceil(math.hypot(small_w, small_h)) + 2
    canvas = np.zeros((diag, diag), np.float32)
    mask = np.zeros((diag, diag), np.float32)
    off_y, off_x = (diag - small_h) // 2, (diag - small_w) // 2
    canvas[off_y : off_y + small_h, off_x : off_x + small_w] = small
    mask[off_y : off_y + small_h, off_x : off_x + small_w] = 1.0
    angles = np.arange(0.0, 180.0, angle_step_deg)
    profile = np.zeros((len(angles), diag), np.float32)
    counts = np.zeros_like(profile)
    centre = (diag / 2.0, diag / 2.0)
    for i, angle in enumerate(angles):
        rotation = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        profile[i] = cv2.warpAffine(canvas, rotation, (diag, diag)).sum(axis=1)
        counts[i] = cv2.warpAffine(mask, rotation, (diag, diag)).sum(axis=1)
    mean = np.where(
        counts > min_length_px / downscale, profile / np.maximum(counts, 1.0), 0.0
    )
    dilated = cv2.dilate(mean, np.ones((7, 7), np.uint8))
    peaks = np.argwhere((mean >= dilated - 1e-6) & (mean > peak_min))
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for i, row in peaks:
        inverse = cv2.invertAffineTransform(
            cv2.getRotationMatrix2D(centre, float(angles[i]), 1.0)
        )
        start = (inverse @ np.array([0.0, row, 1.0]) - [off_x, off_y]) * downscale
        end = (inverse @ np.array([diag, row, 1.0]) - [off_x, off_y]) * downscale
        candidates.append((float(mean[i, row]), start, end))
    lines: list[RoadLine] = []
    for _, start, end in sorted(candidates, key=lambda c: -c[0]):
        for line in _verified_lines(prob, start, end, min_length_px):
            if not any(_same_line(line, other) for other in lines):
                lines.append(line)
    return lines


def _clip_to_page(
    start: np.ndarray, direction: np.ndarray, page_size: tuple[int, int]
) -> tuple[float, float] | None:
    """The [s0, s1] parameter span where ``start + s * direction`` lies on the page."""
    width, height = page_size
    edges = [
        (np.array([0.0, 0.0]), np.array([1.0, 0.0])),
        (np.array([0.0, height - 1.0]), np.array([1.0, 0.0])),
        (np.array([0.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([width - 1.0, 0.0]), np.array([0.0, 1.0])),
    ]
    params = []
    for point, edge in edges:
        normal = np.array([-edge[1], edge[0]])
        denominator = float(direction @ normal)
        if abs(denominator) > 1e-9:
            s = float(normal @ (point - start)) / denominator
            hit = start + s * direction
            if -1 <= hit[0] <= width and -1 <= hit[1] <= height:
                params.append(s)
    if len(params) < 2:
        return None
    return min(params), max(params)


def _verified_lines(
    prob: np.ndarray, start: np.ndarray, end: np.ndarray, min_length_px: int
) -> list[RoadLine]:
    """Full-resolution runs of road along a candidate line, refined to the ridge."""
    height, width = prob.shape
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    span = _clip_to_page(start, direction, (width, height))
    if span is None or span[1] - span[0] < min_length_px:
        return []
    length = int(span[1] - span[0])
    along = np.arange(length)
    centres = start[None, :] + (span[0] + along)[:, None] * direction[None, :]
    offsets = np.arange(-10, 11)
    grid = centres[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    xs = np.clip(grid[..., 0], 0, width - 1).astype(np.float32)
    ys = np.clip(grid[..., 1], 0, height - 1).astype(np.float32)
    band = cv2.remap(prob, xs, ys, cv2.INTER_LINEAR)
    core = band[:, 5:16].mean(axis=1)
    on = np.flatnonzero(core > 0.35)
    if len(on) == 0:
        return []
    runs: list[tuple[int, int]] = []
    run_start = previous = int(on[0])
    for index in on[1:]:
        if index - previous > 40:
            runs.append((run_start, previous))
            run_start = int(index)
        previous = int(index)
    runs.append((run_start, previous))
    lines = []
    for r0, r1 in runs:
        if r1 - r0 < min_length_px:
            continue
        segment = band[r0 : r1 + 1]
        weights = segment.sum(axis=1)
        ridge = (segment * offsets[None, :]).sum(axis=1) / np.maximum(weights, 1e-6)
        good = segment[:, 5:16].mean(axis=1) > 0.35
        slope, intercept = 0.0, 0.0
        if good.sum() > 10:
            design = np.vstack([along[r0 : r1 + 1][good], np.ones(int(good.sum()))]).T
            slope, intercept = np.linalg.lstsq(design, ridge[good], rcond=None)[0]
        a = centres[r0] + intercept * normal
        b = centres[r1] + (intercept + slope * (r1 - r0)) * normal
        angle = math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi
        lines.append(RoadLine(start=a, end=b, angle=angle))
    return lines


def _same_line(line: RoadLine, other: RoadLine) -> bool:
    """Whether two lines are the same road ridge (angle, offset and extent overlap)."""
    delta = abs(line.angle - other.angle)
    if min(delta, math.pi - delta) >= math.radians(1.5):
        return False
    direction = other.end - other.start
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    if abs(float(normal @ (line.start - other.start))) >= 8.0:
        return False
    s0 = float(direction @ (line.start - other.start))
    s1 = float(direction @ (line.end - other.start))
    return min(s0, s1) < other.length + 40 and max(s0, s1) > -40


def orientation_modes(
    angles: list[float],
    weights: list[float],
    *,
    max_modes: int = 2,
    min_separation_deg: float = 6.0,
    min_fraction: float = 0.15,
) -> list[float]:
    """Dominant grid directions (radians, mod 90 degrees) of weighted line angles.

    Peaks of a 1-degree histogram with circular 3-bin smoothing; a second mode
    must be ``min_separation_deg`` from the first and carry ``min_fraction`` of
    its weight. Two modes cover a page or a neighbourhood that spans two grids.
    """
    histogram = np.zeros(90)
    for angle, weight in zip(angles, weights):
        histogram[int(math.degrees(angle) % 90)] += weight
    smoothed = np.array(
        [
            histogram[(i - 1) % 90] + 2 * histogram[i] + histogram[(i + 1) % 90]
            for i in range(90)
        ]
    )
    if smoothed.max() <= 0:
        return []
    modes: list[int] = []
    for bin_index in np.argsort(-smoothed):
        if smoothed[bin_index] < min_fraction * smoothed.max():
            break
        separated = all(
            min(abs(bin_index - m), 90 - abs(bin_index - m)) >= min_separation_deg
            for m in modes
        )
        if separated:
            modes.append(int(bin_index))
        if len(modes) >= max_modes:
            break
    return [math.radians(m + 0.5) for m in modes]


# ---------------------------------------------------------------------------
# 2. Names
# ---------------------------------------------------------------------------


def attach_label_names(
    lines: list[RoadLine],
    features: list[LabelFeature],
    *,
    max_perpendicular_px: float = 14.0,
    max_angle_deg: float = 12.0,
) -> None:
    """Name each line with the canonical streets of the labels printed along it."""
    for feature in features:
        centre = np.array(feature.center)
        for line in lines:
            direction = line.end - line.start
            length = float(np.linalg.norm(direction))
            direction = direction / length
            normal = np.array([-direction[1], direction[0]])
            delta = abs((feature.dir_pix - line.angle) % math.pi)
            if min(delta, math.pi - delta) >= math.radians(max_angle_deg):
                continue
            along = float(direction @ (centre - line.start))
            perpendicular = abs(float(normal @ (centre - line.start)))
            if perpendicular < max_perpendicular_px and -60 < along < length + 60:
                line.names.add(feature.text)


# ---------------------------------------------------------------------------
# 3. Neighbours' lines in the metre frame
# ---------------------------------------------------------------------------


def reaches_edge(point: np.ndarray, page: PlacedPage, margin_px: float) -> bool:
    """Whether a line endpoint sits within ``margin_px`` of the sheet edge (the street runs off)."""
    return (
        min(point[0], point[1], page.width - point[0], page.height - point[1])
        < margin_px
    )


def world_lines(
    pages: list[PlacedPage], *, extension_m: float = 450.0, edge_margin_px: float = 40.0
) -> list[WorldLine]:
    """Placed pages' road lines in frame xy, extrapolated past sheet-edge endpoints only."""
    result = []
    for page in pages:
        for line in page.lines:
            a = page.to_xy(*line.start)
            b = page.to_xy(*line.end)
            direction = b - a
            length = float(np.linalg.norm(direction))
            if length <= 0:
                continue
            direction = direction / length

            result.append(
                WorldLine(
                    origin=a,
                    direction=direction,
                    length=length,
                    angle=math.atan2(direction[1], direction[0]) % math.pi,
                    names=set(line.names),
                    page=page.stem,
                    extend_before=extension_m
                    if reaches_edge(line.start, page, edge_margin_px)
                    else 0.0,
                    extend_after=extension_m
                    if reaches_edge(line.end, page, edge_margin_px)
                    else 0.0,
                )
            )
    return result


class WorldIndex:
    """World lines as arrays, with name compatibility precomputed per target line."""

    def __init__(self, lines: list[WorldLine], target_lines: list[RoadLine]) -> None:
        self.lines = lines
        self.origin = np.array([w.origin for w in lines]).reshape(-1, 2)
        self.direction = np.array([w.direction for w in lines]).reshape(-1, 2)
        self.normal = np.stack([-self.direction[:, 1], self.direction[:, 0]], axis=1)
        self.length = np.array([w.length for w in lines])
        self.angle = np.array([w.angle for w in lines])
        self.extend_before = np.array([w.extend_before for w in lines])
        self.extend_after = np.array([w.extend_after for w in lines])
        self.allowed: list[np.ndarray] = []
        self.bonus: list[np.ndarray] = []
        for line in target_lines:
            conflict = [
                bool(line.names and w.names and not (line.names & w.names))
                for w in lines
            ]
            self.allowed.append(~np.array(conflict, dtype=bool))
            self.bonus.append(
                np.array([4.0 if (line.names & w.names) else 1.0 for w in lines])
            )


def continuity_score(
    affine: np.ndarray,
    target_lines: list[RoadLine],
    index: WorldIndex,
    *,
    sigma_m: float = 4.0,
) -> tuple[float, set[str]]:
    """How well the target's lines, under ``affine``, continue name-compatible world lines.

    Each target line is sampled every 10 m; each sample scores a Gaussian kernel
    of its perpendicular distance to the best angle-matched world line whose
    (extended) extent covers it, x4 when the names agree. A 150 m line that is
    continued perfectly scores 1.
    """
    total = 0.0
    matched: set[str] = set()
    if not index.lines:
        return 0.0, matched
    for i, line in enumerate(target_lines):
        a = affine @ np.append(line.start, 1.0)
        b = affine @ np.append(line.end, 1.0)
        direction = b - a
        length = float(np.linalg.norm(direction))
        direction = direction / length
        angle = math.atan2(direction[1], direction[0]) % math.pi
        delta = np.abs(angle - index.angle)
        delta = np.minimum(delta, math.pi - delta)
        usable = (delta <= math.radians(2.5)) & index.allowed[i]
        if not usable.any():
            continue
        samples = max(3, int(length / 10))
        points = (
            a[None, :] + np.linspace(0.0, length, samples)[:, None] * direction[None, :]
        )
        relative = points[:, None, :] - index.origin[None, usable, :]
        along = (relative * index.direction[None, usable, :]).sum(axis=2)
        perpendicular = np.abs((relative * index.normal[None, usable, :]).sum(axis=2))
        inside = (along > -index.extend_before[None, usable]) & (
            along < index.length[None, usable] + index.extend_after[None, usable]
        )
        kernel = (np.exp(-(perpendicular**2) / (2 * sigma_m**2)) * inside).sum(axis=0)
        kernel = kernel * index.bonus[i][usable]
        best = int(np.argmax(kernel))
        total += float(kernel[best]) * 10.0 / 150.0
        if index.bonus[i][usable][best] > 1:
            matched |= line.names & index.lines[int(np.flatnonzero(usable)[best])].names
    return total, matched


# ---------------------------------------------------------------------------
# Raster contradictions
# ---------------------------------------------------------------------------


def content_box(
    prob: np.ndarray, *, threshold: float = 0.5, shrink_px: int = 30
) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the map content: the bbox of P(road) > threshold, shrunk.

    Sanborn margins are blank in P(road); a road that lands there is unknown,
    not contradicted, so contradiction checks stay inside this box.
    """
    ys, xs = np.nonzero(prob > threshold)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (
        int(xs.min()) + shrink_px,
        int(ys.min()) + shrink_px,
        int(xs.max()) - shrink_px,
        int(ys.max()) - shrink_px,
    )


def tolerant_raster(
    prob: np.ndarray, *, downscale: int = 4, dilate_cells: int = 3
) -> np.ndarray:
    """A quarter-resolution P(road) dilated so a few metres of pose error is not a contradiction."""
    small = cv2.resize(
        prob,
        (prob.shape[1] // downscale, prob.shape[0] // downscale),
        interpolation=cv2.INTER_AREA,
    )
    size = 2 * dilate_cells + 1
    return cv2.dilate(small, np.ones((size, size), np.uint8))


class RoadRasters:
    """Anchors' tolerant P(road) rasters with inverse poses, and their road samples in frame xy."""

    def __init__(
        self,
        anchors: list[PlacedPage],
        rasters: dict[str, np.ndarray],
        lines: list[WorldLine],
    ) -> None:
        self.items = []
        for page in anchors:
            raster = rasters.get(page.stem)
            if raster is None:
                continue
            full = cv2.resize(
                raster, (page.width, page.height), interpolation=cv2.INTER_NEAREST
            )
            self.items.append(
                (cv2.invertAffineTransform(page.affine), raster, content_box(full))
            )
        points = []
        for line in lines:
            samples = max(2, int(line.length / 10))
            points.append(
                line.origin[None, :]
                + np.linspace(0.0, line.length, samples)[:, None]
                * line.direction[None, :]
            )
        self.world_points = np.vstack(points) if points else np.zeros((0, 2))

    def anchor_road(self, xy: np.ndarray) -> np.ndarray:
        """Max anchor P(road) at frame points; NaN where no anchor's content covers them."""
        out = np.full(len(xy), np.nan)
        for inverse, raster, (x0, y0, x1, y1) in self.items:
            pixels = inverse @ np.vstack([xy.T, np.ones(len(xy))])
            x, y = pixels[0], pixels[1]
            covered = (x > x0) & (x < x1) & (y > y0) & (y < y1)
            if not covered.any():
                continue
            values = cv2.remap(
                raster,
                (x[covered] / 4).astype(np.float32),
                (y[covered] / 4).astype(np.float32),
                cv2.INTER_LINEAR,
            )[:, 0]
            current = out[covered]
            out[covered] = np.where(
                np.isnan(current), values, np.maximum(current, values)
            )
        return out


def contradiction(
    affine: np.ndarray,
    target_lines: list[RoadLine],
    rasters: RoadRasters,
    target: tuple[np.ndarray, tuple[int, int, int, int]],
) -> float:
    """Road samples the other side denies: target lines inside anchors, anchor lines inside the target.

    ``target`` is the target's tolerant raster and content box. Each denied
    10 m sample counts 10/150, the unit of ``continuity_score``.
    """
    target_raster, (x0, y0, x1, y1) = target
    points = []
    for line in target_lines:
        a = affine @ np.append(line.start, 1.0)
        b = affine @ np.append(line.end, 1.0)
        samples = max(2, int(np.linalg.norm(b - a) / 10))
        points.append(
            a[None, :] + np.linspace(0.0, 1.0, samples)[:, None] * (b - a)[None, :]
        )
    denied = 0.0
    if points:
        values = rasters.anchor_road(np.vstack(points))
        denied += float(np.sum(values[~np.isnan(values)] < 0.25))
    if len(rasters.world_points):
        inverse = cv2.invertAffineTransform(affine)
        pixels = inverse @ np.vstack(
            [rasters.world_points.T, np.ones(len(rasters.world_points))]
        )
        x, y = pixels[0], pixels[1]
        covered = (x > x0) & (x < x1) & (y > y0) & (y < y1)
        if covered.any():
            values = cv2.remap(
                target_raster,
                (x[covered] / 4).astype(np.float32),
                (y[covered] / 4).astype(np.float32),
                cv2.INTER_LINEAR,
            )[:, 0]
            denied += float(np.sum(values < 0.25))
    return denied * 10.0 / 150.0


# ---------------------------------------------------------------------------
# 4. Search
# ---------------------------------------------------------------------------


def vote_translations(
    target_lines: list[RoadLine],
    index: WorldIndex,
    rotation_scale: tuple[float, float],
    prior: Prior,
    *,
    prior_sigma_m: float | None = None,
    page_size: tuple[int, int] = (0, 0),
    cell_m: float = 4.0,
    max_peaks: int = 8,
    angle_tolerance_deg: float = 2.5,
) -> list[tuple[float, np.ndarray]]:
    """Translation peaks from pairs of non-parallel line correspondences at a fixed rotation.

    A target line matched to a world line fixes the translation across that
    line; two such correspondences from non-parallel lines fix it fully, and
    every such pair casts a point vote (weighted by line lengths, x6 per name
    match). Peaks of the accumulator, restricted to the prior disc and softly
    weighted toward the prior when ``prior_sigma_m`` is set, seed refinement.
    Returns (vote, translation) pairs, strongest first.
    """
    rotation, scale = rotation_scale
    rotate = scale * np.array(
        [
            [math.cos(rotation), -math.sin(rotation)],
            [math.sin(rotation), math.cos(rotation)],
        ]
    )
    half = int((prior.radius_m + 350.0) / cell_m)
    size = 2 * half + 1
    origin = prior.centre - half * cell_m
    order = np.argsort([-line.length for line in target_lines])[:16]
    correspondences = []
    for i in order:
        line = target_lines[i]
        a = rotate @ line.start
        b = rotate @ line.end
        direction = b - a
        length = float(np.linalg.norm(direction))
        direction = direction / length
        angle = math.atan2(direction[1], direction[0]) % math.pi
        delta = np.abs(angle - index.angle)
        delta = np.minimum(delta, math.pi - delta)
        for j in np.flatnonzero(
            (delta <= math.radians(angle_tolerance_deg)) & index.allowed[i]
        ):
            world_direction = (
                index.direction[j]
                if (index.direction[j] @ direction) > 0
                else -index.direction[j]
            )
            normal = np.array([-world_direction[1], world_direction[0]])
            along_origin = float(world_direction @ index.origin[j])
            along_a, along_b = float(world_direction @ a), float(world_direction @ b)
            low = (
                along_origin
                - index.extend_before[j]
                - min(length, 50.0)
                - min(along_a, along_b)
            )
            high = (
                along_origin
                + index.length[j]
                + index.extend_after[j]
                + min(length, 50.0)
                - max(along_a, along_b)
            )
            weight = min(length, 150.0) / 150.0 * min(index.length[j], 150.0) / 150.0
            weight *= 6.0 if index.bonus[i][j] > 1 else 1.0
            correspondences.append(
                (
                    int(i),
                    world_direction,
                    normal,
                    float(normal @ (index.origin[j] - a)),
                    low,
                    high,
                    weight,
                )
            )
    if len(correspondences) < 2:
        return []
    target_index = np.array([c[0] for c in correspondences])
    directions = np.array([c[1] for c in correspondences])
    normals = np.array([c[2] for c in correspondences])
    offsets = np.array([c[3] for c in correspondences])
    lows = np.array([c[4] for c in correspondences])
    highs = np.array([c[5] for c in correspondences])
    weights = np.array([c[6] for c in correspondences])
    angles = np.array([math.atan2(d[1], d[0]) % math.pi for d in directions])
    accumulator = np.zeros((size, size), np.float32)
    for p in range(len(correspondences)):
        delta = np.abs(angles - angles[p])
        delta = np.minimum(delta, math.pi - delta)
        partner = (
            (delta > math.radians(30))
            & (target_index != target_index[p])
            & (np.arange(len(correspondences)) > p)
        )
        if not partner.any():
            continue
        n1, n2 = normals[p], normals[partner]
        determinant = n1[0] * n2[:, 1] - n1[1] * n2[:, 0]
        tx = (offsets[p] * n2[:, 1] - n1[1] * offsets[partner]) / determinant
        ty = (n1[0] * offsets[partner] - offsets[p] * n2[:, 0]) / determinant
        s1 = tx * directions[p][0] + ty * directions[p][1]
        s2 = tx * directions[partner][:, 0] + ty * directions[partner][:, 1]
        inside = (
            (s1 > lows[p])
            & (s1 < highs[p])
            & (s2 > lows[partner])
            & (s2 < highs[partner])
        )
        if not inside.any():
            continue
        gx = ((tx[inside] - origin[0]) / cell_m).astype(int)
        gy = ((ty[inside] - origin[1]) / cell_m).astype(int)
        valid = (gx >= 0) & (gx < size) & (gy >= 0) & (gy < size)
        np.add.at(
            accumulator,
            (gy[valid], gx[valid]),
            (weights[p] * weights[partner][inside])[valid],
        )
    accumulator = cv2.GaussianBlur(accumulator, (0, 0), 1.5)
    ys, xs = np.mgrid[0:size, 0:size]
    disc = ((xs - half) ** 2 + (ys - half) ** 2) * cell_m**2 <= (
        prior.radius_m + 300.0
    ) ** 2
    accumulator[~disc] = 0.0
    if prior_sigma_m:
        # The accumulator is over the pixel-origin translation; the prior describes the page centre.
        centre_translation = prior.centre - rotate @ np.array(
            [page_size[0] / 2.0, page_size[1] / 2.0]
        )
        gx0, gy0 = (centre_translation - origin) / cell_m
        accumulator *= np.exp(
            -(((xs - gx0) ** 2 + (ys - gy0) ** 2) * cell_m**2) / (2 * prior_sigma_m**2)
        ).astype(np.float32)
    peaks = []
    for _ in range(max_peaks):
        iy, ix = np.unravel_index(int(np.argmax(accumulator)), accumulator.shape)
        vote = float(accumulator[iy, ix])
        if vote <= 0:
            break
        peaks.append((vote, origin + np.array([ix, iy]) * cell_m))
        cv2.circle(accumulator, (int(ix), int(iy)), int(25 / cell_m), 0, -1)
    return peaks


@dataclass
class TargetPage:
    """The page being placed: its lines, size, and (optionally) its tolerant raster."""

    stem: str
    width: int
    height: int
    lines: list[RoadLine]
    raster: np.ndarray | None = None
    box: tuple[int, int, int, int] = (0, 0, 0, 0)


def place_page(
    target: TargetPage,
    anchors: list[PlacedPage],
    prior: Prior,
    options: PlacementOptions,
    *,
    rasters: dict[str, np.ndarray] | None = None,
    truth: np.ndarray | None = None,
) -> Placement:
    """Rank poses for ``target`` by how its roads continue the anchors' roads.

    ``truth`` (a px -> frame-xy affine) only annotates candidates with their
    corner error and scores the truth pose; it never steers the search.
    """
    lines = world_lines(anchors, extension_m=options.extension_m)
    index = WorldIndex(lines, target.lines)
    road_rasters = None
    if rasters is not None and target.raster is not None:
        road_rasters = RoadRasters(anchors, rasters, lines)

    def objective(affine: np.ndarray, sigma_m: float = 4.0) -> tuple[float, set[str]]:
        score, names = continuity_score(affine, target.lines, index, sigma_m=sigma_m)
        if road_rasters is not None and target.raster is not None:
            score -= options.contradiction_weight * contradiction(
                affine, target.lines, road_rasters, (target.raster, target.box)
            )
        if options.prior_sigma_m:
            centre = affine @ np.array([target.width / 2.0, target.height / 2.0, 1.0])
            score -= (
                0.5
                * (float(np.linalg.norm(centre - prior.centre)) / options.prior_sigma_m)
                ** 2
            )
        return score, names

    placement = Placement(
        stem=target.stem,
        candidates=[],
        anchors=[a.stem for a in anchors],
        target_lines=len(target.lines),
        named_lines=sum(1 for line in target.lines if line.names),
        truth_score=objective(truth)[0] if truth is not None else None,
    )
    if not target.lines or not lines:
        return placement
    near = [
        w
        for w in lines
        if np.linalg.norm(w.origin + w.direction * w.length / 2 - prior.centre)
        < prior.radius_m + 400
    ] or lines
    world_modes = orientation_modes([w.angle for w in near], [w.length for w in near])
    target_modes = orientation_modes(
        [line.angle for line in target.lines], [line.length for line in target.lines]
    )
    bases = sorted(
        {
            round((wm - tm) % (math.pi / 2), 3)
            for wm in world_modes
            for tm in target_modes
        }
    )
    seeds: list[tuple[float, np.ndarray, float]] = []
    for base in bases:
        for quadrant in range(4):
            for delta_deg in (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0):
                rotation = base + quadrant * math.pi / 2 + math.radians(delta_deg)
                for vote, translation in vote_translations(
                    target.lines,
                    index,
                    (rotation, options.scale_m_per_px),
                    prior,
                    prior_sigma_m=options.prior_sigma_m,
                    page_size=(target.width, target.height),
                    angle_tolerance_deg=options.angle_tolerance_deg,
                ):
                    seeds.append((vote, translation, rotation))
    seeds.sort(key=lambda s: -s[0])
    kept: list[tuple[float, np.ndarray, float]] = []
    for vote, translation, rotation in seeds:
        duplicate = any(
            np.linalg.norm(translation - t) < 15
            and abs((rotation - r + math.pi) % (2 * math.pi) - math.pi)
            < math.radians(1.5)
            for _, t, r in kept
        )
        if not duplicate:
            kept.append((vote, translation, rotation))
        if len(kept) >= options.max_seeds:
            break
    candidates = []
    for vote, translation, rotation in kept:
        x = refine_pose(
            np.array([translation[0], translation[1], rotation]),
            lambda x, sigma: objective(
                pose_affine(x[:2], x[2], options.scale_m_per_px), sigma
            )[0],
        )
        affine = pose_affine(x[:2], x[2], options.scale_m_per_px)
        score, names = objective(affine)
        rmse = (
            corner_rmse_ft((target.width, target.height), affine, truth)
            if truth is not None
            else None
        )
        candidates.append(
            Candidate(
                affine=affine,
                rotation_deg=math.degrees(x[2]),
                vote=vote,
                score=score,
                names=names,
                rmse_ft=rmse,
            )
        )
    candidates.sort(key=lambda c: -c.score)
    distinct: list[Candidate] = []
    for candidate in candidates:
        if not any(
            np.linalg.norm(candidate.translation - d.translation) < 20 for d in distinct
        ):
            distinct.append(candidate)
    placement.candidates = distinct
    return placement


def refine_pose(x0: np.ndarray, objective) -> np.ndarray:
    """Coarse-to-fine Nelder-Mead on (tx, ty, rotation): sigma 12 m to reach, then 4 m to settle."""
    coarse = np.array(
        [x0, x0 + [8, 0, 0], x0 + [0, 8, 0], x0 + [0, 0, math.radians(0.8)]]
    )
    result = minimize(
        lambda x: -objective(x, 12.0),
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": coarse,
            "maxiter": 120,
            "xatol": 0.5,
            "fatol": 1e-3,
        },
    )
    x1 = result.x
    fine = np.array(
        [x1, x1 + [3, 0, 0], x1 + [0, 3, 0], x1 + [0, 0, math.radians(0.3)]]
    )
    result = minimize(
        lambda x: -objective(x, 4.0),
        x1,
        method="Nelder-Mead",
        options={"initial_simplex": fine, "maxiter": 150, "xatol": 0.2, "fatol": 1e-3},
    )
    return result.x


# ---------------------------------------------------------------------------
# Volume IO
# ---------------------------------------------------------------------------


def page_size_of(volume: Path, stem: str) -> tuple[int, int]:
    """(width, height) of a page from its streets.json."""
    doc = json.loads((volume / f"{stem}.streets.json").read_text())
    return int(doc["width"]), int(doc["height"])


def load_road_probability(volume: Path, stem: str) -> np.ndarray | None:
    """A page's P(road) as float32 in [0, 1] at page resolution, from the sidecar or the snap artifact."""
    for path in (
        volume / f"{stem}.roadprob.jpg",
        volume / "artifacts" / "edge_join" / "roadprob" / f"{stem}.png",
    ):
        if path.exists():
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                width, height = page_size_of(volume, stem)
                if raw.shape != (height, width):
                    raw = cv2.resize(
                        raw, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                return raw.astype(np.float32) / 255.0
    return None


def fitted_affines(volume: Path) -> dict[str, np.ndarray]:
    """Every page's published px -> lon/lat affine from its georef-final corners."""
    affines = {}
    for path in volume.glob("p*.georef-final.json"):
        stem = path.name[: -len(".georef-final.json")]
        if "__" in stem:
            continue
        doc = json.loads(path.read_text())
        corners = doc.get("corners")
        if not corners:
            continue
        width, height = doc["width"], doc["height"]
        pairs = [
            ((0, 0), tuple(corners[0])),
            ((width, 0), tuple(corners[1])),
            ((width, height), tuple(corners[2])),
            ((0, height), tuple(corners[3])),
        ]
        affines[stem] = fit_affine(pairs)
    return affines


def truth_affines(volume: Path) -> dict[str, np.ndarray]:
    """Truth px -> lon/lat affines in the 25%-page frame, keyed by page stem."""
    path = volume / "main.iiif.json"
    if not path.exists():
        return {}
    result = {}
    for item in json.loads(path.read_text()).get("items", []):
        label = str(item.get("label", ""))
        match = re.search(r"\bp(\d+[a-zA-Z]?)\b", label)
        if not match or "[" in label or len(item["body"]["features"]) < 3:
            continue
        stem = f"p{match.group(1)}"
        if not (volume / f"{stem}.streets.json").exists():
            continue
        affine = fit_affine(extract_gcps(item)).copy()
        source = item["target"]["source"]
        width, height = page_size_of(volume, stem)
        affine[:, 0] *= source["width"] / width
        affine[:, 1] *= source["height"] / height
        result[stem] = affine
    return result


def centerlines_path(volume: Path) -> Path | None:
    """The volume's centerlines file, geojson or OSM extract."""
    for name in ("centerlines.geojson", "centerlines.osm.pbf"):
        if (volume / name).exists():
            return volume / name
    return None


class Volume:
    """A volume directory: frame, placed pages, P(road) rasters and label features on demand."""

    def __init__(self, volume: Path) -> None:
        self.path = volume
        lonlat = fitted_affines(volume)
        if not lonlat:
            sys.exit(f"{volume}: no fitted pages (p*.georef-final.json with corners)")
        self.frame = Frame(
            float(np.mean([a[0, 2] for a in lonlat.values()])),
            float(np.mean([a[1, 2] for a in lonlat.values()])),
        )
        self.affines = {stem: self.frame.affine_to_xy(a) for stem, a in lonlat.items()}
        self.scale = float(np.median([scale_of(a) for a in self.affines.values()]))
        self.truth = {
            stem: self.frame.affine_to_xy(a)
            for stem, a in truth_affines(volume).items()
        }
        path = centerlines_path(volume)
        self.centerlines = load_centerlines(path)["features"] if path else []
        self.block_index = build_block_index(
            {"type": "FeatureCollection", "features": self.centerlines}
        )
        self.lines: dict[str, list[RoadLine]] = {}
        self.rasters: dict[str, np.ndarray] = {}

    def features(self, stem: str, block_index: dict) -> list[LabelFeature]:
        """The page's admitted, canonicalised street labels against ``block_index``."""
        width, height = page_size_of(self.path, stem)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return prepare_label_features(
                str(self.path / f"{stem}.streets.json"),
                block_index,
                (width, height),
                min_confidence=MIN_LABEL_CONFIDENCE,
                min_long_side=MIN_LABEL_LONG_SIDE,
                min_short_side=MIN_LABEL_SHORT_SIDE,
                min_aspect_ratio=MIN_LABEL_ASPECT,
            )

    def lines_of(self, stem: str) -> list[RoadLine]:
        """The page's named road lines (cached)."""
        if stem not in self.lines:
            prob = load_road_probability(self.path, stem)
            lines = extract_road_lines(prob) if prob is not None else []
            if lines:
                attach_label_names(lines, self.features(stem, self.block_index))
            self.lines[stem] = lines
        return self.lines[stem]

    def raster_of(self, stem: str) -> np.ndarray | None:
        if stem not in self.rasters:
            prob = load_road_probability(self.path, stem)
            if prob is None:
                return None
            self.rasters[stem] = tolerant_raster(prob)
        return self.rasters[stem]

    def placed(self, stem: str) -> PlacedPage:
        width, height = page_size_of(self.path, stem)
        return PlacedPage(
            stem=stem,
            affine=self.affines[stem],
            width=width,
            height=height,
            lines=self.lines_of(stem),
        )

    def prior_of(self, stem: str) -> tuple[Prior, list[tuple[float, float]]] | None:
        """The page's key-map prior (centre, radius) and its lon/lat centres, if located."""
        path = self.path / f"{stem}.georef.json"
        if not path.exists():
            return None
        keymap = json.loads(path.read_text()).get("keymap") or {}
        if "lat" not in keymap:
            return None
        centres = [
            (float(c[0]), float(c[1]))
            for c in keymap.get("centers") or [[keymap["lon"], keymap["lat"]]]
        ]
        return Prior(
            self.frame.lonlat_to_xy(keymap["lon"], keymap["lat"]),
            float(keymap.get("radius_m", 500.0)),
        ), centres

    def target(
        self, stem: str, prior_centres: list[tuple[float, float]], radius_m: float
    ) -> TargetPage:
        """The page as a target: lines named from the vocabulary near its prior."""
        width, height = page_size_of(self.path, stem)
        prob = load_road_probability(self.path, stem)
        lines = extract_road_lines(prob) if prob is not None else []
        near = (
            KeymapLocator({"1": prior_centres}, radius_m).restricted_features(
                "1", self.centerlines
            )
            if self.centerlines
            else None
        )
        if lines and near:
            attach_label_names(
                lines,
                self.features(
                    stem,
                    build_block_index({"type": "FeatureCollection", "features": near}),
                ),
            )
        raster = tolerant_raster(prob) if prob is not None else None
        box = content_box(prob) if prob is not None else (0, 0, 0, 0)
        return TargetPage(
            stem=stem, width=width, height=height, lines=lines, raster=raster, box=box
        )

    def anchors_near(
        self, centre: np.ndarray, radius_m: float, *, exclude: str, limit: int = 60
    ) -> list[PlacedPage]:
        """The nearest placed pages within the prior disc plus a margin."""
        stems = [s for s in self.affines if s != exclude and s != "p0"]
        near = [
            (float(np.linalg.norm(self.placed_centre(s) - centre)), s) for s in stems
        ]
        return [self.placed(s) for d, s in sorted(near) if d < radius_m + 900][:limit]

    def placed_centre(self, stem: str) -> np.ndarray:
        width, height = page_size_of(self.path, stem)
        return self.affines[stem] @ np.array([width / 2.0, height / 2.0, 1.0])


def sidecar_document(volume: Volume, target: TargetPage, placement: Placement) -> dict:
    """The ``<stem>.georef-continuity.json`` body: corners in lon/lat plus the evidence."""
    best = placement.best
    assert best is not None
    lonlat = volume.frame.affine_to_lonlat(best.affine)
    return {
        "status": "continuity",
        "width": target.width,
        "height": target.height,
        "corners": corners_xy((target.width, target.height), lonlat).tolist(),
        "score": round(best.score, 3),
        "margin": None if math.isinf(placement.margin) else round(placement.margin, 3),
        "rotation_deg": round(best.rotation_deg, 2),
        "names": sorted(best.names),
        "anchors": placement.anchors,
        "target_lines": placement.target_lines,
        "named_lines": placement.named_lines,
    }


def unplaced_pages(volume: Volume) -> list[str]:
    """Pages with a streets.json but no published pose (split panels excluded)."""
    stems = []
    for path in volume.path.glob("p*.streets.json"):
        stem = path.name[: -len(".streets.json")]
        if "__" not in stem and stem != "p0" and stem not in volume.affines:
            stems.append(stem)
    return sorted(stems, key=lambda s: (int(re.sub(r"\D", "", s) or 0), s))


def run_page(
    volume: Volume, stem: str, options: PlacementOptions, *, evaluate: bool = False
) -> tuple[TargetPage, Placement] | None:
    """Place one page against the volume's current anchors; None without a key-map prior."""
    located = volume.prior_of(stem)
    if located is None:
        print(f"{stem}: no key-map prior; skipped")
        return None
    prior, centres = located
    target = volume.target(stem, centres, prior.radius_m)
    anchors = volume.anchors_near(prior.centre, prior.radius_m, exclude=stem)
    rasters = {
        a.stem: r for a in anchors if (r := volume.raster_of(a.stem)) is not None
    }
    truth = volume.truth.get(stem) if evaluate else None
    placement = place_page(
        target, anchors, prior, options, rasters=rasters, truth=truth
    )
    return target, placement


def report(placement: Placement, *, evaluate: bool) -> None:
    """Print a page's candidate table."""
    print(
        f"{placement.stem}: {placement.target_lines} lines ({placement.named_lines} named), {len(placement.anchors)} anchors"
        + (
            f", truth-pose score {placement.truth_score:.2f}"
            if placement.truth_score is not None
            else ""
        )
    )
    if not placement.candidates:
        print("  no candidates")
        return
    for rank, candidate in enumerate(placement.candidates[:5], start=1):
        rmse = (
            ""
            if not evaluate or candidate.rmse_ft is None
            else f" rmse {candidate.rmse_ft:6.0f} ft"
        )
        print(
            f"  #{rank} score {candidate.score:6.2f} vote {candidate.vote:6.1f} theta {candidate.rotation_deg:7.1f}{rmse}  {sorted(candidate.names)}"
        )
    verdict = "ACCEPT" if placement.accepted() else "abstain"
    print(f"  margin {placement.margin:.2f} -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place pages by continuing their placed neighbours' roads (experimental, not consumed by fit)."
    )
    parser.add_argument("dir", type=Path, metavar="DIR", help="Volume directory")
    parser.add_argument(
        "--pages",
        metavar="LIST",
        help="Comma-separated page stems to place (default: --unplaced)",
    )
    parser.add_argument(
        "--unplaced",
        action="store_true",
        help="Place every page that has no published pose",
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help="Accepted placements become anchors for later pages (up to 3 rounds)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write <stem>.georef-continuity.json for accepted placements",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Score candidates against main.iiif.json truth (never steers the search)",
    )
    parser.add_argument(
        "--prior-sigma",
        type=float,
        default=60.0,
        metavar="M",
        help="Key-map prior weight (metres; 0 disables)",
    )
    args = parser.parse_args()

    volume = Volume(args.dir)
    options = PlacementOptions(
        scale_m_per_px=volume.scale, prior_sigma_m=args.prior_sigma or None
    )
    pending = args.pages.split(",") if args.pages else unplaced_pages(volume)
    print(
        f"{args.dir.name}: {len(volume.affines)} placed pages, scale {volume.scale:.3f} m/px; placing {pending}"
    )
    for round_number in range(1, 4 if args.chain else 2):
        if args.chain:
            print(f"--- round {round_number} ---")
        still = []
        for stem in pending:
            result = run_page(volume, stem, options, evaluate=args.eval)
            if result is None:
                continue
            target, placement = result
            report(placement, evaluate=args.eval)
            if not placement.accepted():
                still.append(stem)
                continue
            best = placement.best
            assert best is not None
            if args.write:
                (args.dir / f"{stem}.georef-continuity.json").write_text(
                    json.dumps(sidecar_document(volume, target, placement), indent=2)
                )
            if args.chain:
                volume.affines[stem] = best.affine
                volume.lines.pop(stem, None)
        if not args.chain or still == pending:
            break
        pending = still
    if args.chain:
        print(f"unplaced after chaining: {pending}")


if __name__ == "__main__":
    main()
