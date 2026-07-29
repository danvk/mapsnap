"""Locate volume pages from a georeferenced key map, to restrict OCR/georef to nearby streets.

A key map is a schematic showing where each numbered page sits. Once the key map itself is
georeferenced (its ``<stem>.georef.json``), a page-number detection's pixel maps straight to
that page's real-world location. That lets the main OCR and georeference steps swap the
ambiguous city-wide street vocabulary for the handful of streets actually near a page —
dropping false matches (e.g. a second "Canal St" across town) and, for OCR, driving up
recognizer confidence on the correct names.
"""

import itertools
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mapsnap.feature_index import FeatureIndex
from mapsnap.keymap.fit_keymap import (
    key_stem,
    load_detections,
    page_key,
    page_number,
    volume_page_keys,
)

Point = tuple[float, float]

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQUATOR = 111320.0

# Two key maps that index this much of the smaller one's page set are indexing the
# same volume rather than complementary parts of it. Measured across the seven
# multi-key-map volumes on hand, the complementary pairs (left/right or north/south
# halves) share 3-20% of their keys, and the one redundant pair shares 83%.
REDUNDANT_KEY_OVERLAP = 0.5
# How much more of a redundant sheet's page set must be foreign to this volume before
# the other sheet is preferred. Atlanta's multi-volume index reads 48% pages this
# volume does not have against its counterpart's 0%; no other sheet exceeds 29%.
FOREIGN_KEY_GAP = 0.2

__all__ = [
    "KeymapLocator",
    "discover_keymaps",
    "page_key",
    "page_number",
    "redundant_keymaps",
    "resolve_keymaps",
    "usable_keymaps",
]


def canonical_page_key(text: str) -> str | None:
    """Uppercase page key from a printed/decoded text, or None if not a page key."""
    return text.upper() if re.fullmatch(r"\d+[A-Za-z]{0,2}", text) else None


def keymap_georef_path(keymap_json: Path) -> Path:
    """Sibling ``<stem>.georef.json`` for a key-map detections file (``<stem>.keymap.json``)."""
    return keymap_json.with_name(
        keymap_json.name.replace(".keymap.json", ".georef.json")
    )


def usable_keymaps(directory: Path) -> list[Path]:
    """Key-map detections files in one directory that a locator can actually load.

    A key map whose own georeferencing failed leaves a bare ``<stem>.keymap.json``
    beside a ``-misscale``/``-nofit`` sidecar rather than the ``<stem>.georef.json``
    :func:`KeymapLocator.from_keymap` reads, so it is skipped here instead of
    raising when the locator gets around to opening it.
    """
    return [
        keymap_json
        for keymap_json in sorted(directory.glob("*.keymap.json"))
        if keymap_georef_path(keymap_json).exists()
    ]


def foreign_key_fraction(keys: set[str], volume_keys: set[str]) -> float:
    """Share of a key map's page numbers that are not pages of this volume."""
    if not keys:
        return 0.0
    return sum(1 for key in keys if key.upper() not in volume_keys) / len(keys)


def redundant_keymaps(key_sets: list[set[str]], volume_keys: set[str]) -> set[int]:
    """Indices of key maps to ignore as redundant duplicates of a better sheet.

    A volume with several key maps normally has *complementary* ones — left/right or
    north/south halves that each index their own pages. Two sheets that index largely
    the *same* pages are a different situation: one of them is a multi-volume index,
    covering this volume plus others, and every page it shares gets a second search
    centre that may be far from the first. Atlanta 1911 is the case in point: its
    city-wide sheet placed pages a median 779 m from where they actually fit, against
    26 m for the volume's own sheet, and dropping it gained eight pages.

    Redundancy alone does not say which sheet to keep — the two disagree, but nothing
    local says who is right. What does is that a multi-volume index reads page numbers
    this volume does not have: 48% of Atlanta's foreign against 0% for its counterpart.
    So a sheet is dropped only when it both duplicates another's page set and is the
    more foreign of the two by :data:`FOREIGN_KEY_GAP`; ties keep everything.

    Note that *disagreement* deliberately plays no part. Complementary sheets disagree
    far more than the redundant pair does (5.2 km and 7.4 km medians against Atlanta's
    445 m), because their handful of shared keys are misreads on opposite sides of a
    city — so gating on it would reject exactly the wrong sheets.

    Returns an empty set when the volume's page keys are unknown: with nothing to call
    foreign, there is no evidence to act on.
    """
    if not volume_keys:
        return set()
    foreign = [foreign_key_fraction(keys, volume_keys) for keys in key_sets]
    drop: set[int] = set()
    for i, j in itertools.combinations(range(len(key_sets)), 2):
        if i in drop or j in drop:
            continue
        smaller = min(len(key_sets[i]), len(key_sets[j]))
        if not smaller:
            continue
        if len(key_sets[i] & key_sets[j]) / smaller < REDUNDANT_KEY_OVERLAP:
            continue  # complementary sheets; both are needed
        if foreign[i] - foreign[j] >= FOREIGN_KEY_GAP:
            drop.add(i)
        elif foreign[j] - foreign[i] >= FOREIGN_KEY_GAP:
            drop.add(j)
    return drop


def discover_keymaps(image_paths: list[str]) -> list[Path]:
    """Usable ``<stem>.keymap.json`` files near the input images, for default key-map use.

    Searches each image's own directory and its ``raw/`` subdirectory (where ``mapsnap keymap``
    writes them) and keeps only detections files that have the sibling ``<stem>.georef.json`` a
    locator needs — a bare ``.keymap.json`` from a key map whose georeferencing failed is skipped.
    Returns them de-duplicated in directory-then-name order; empty when none are found.
    """
    directories: list[Path] = []
    for image_path in image_paths:
        parent = Path(image_path).parent
        for directory in (parent, parent / "raw"):
            if directory.is_dir() and directory not in directories:
                directories.append(directory)
    found: list[Path] = []
    for directory in directories:
        for keymap_json in usable_keymaps(directory):
            if keymap_json not in found:
                found.append(keymap_json)
    return found


def resolve_keymaps(
    explicit: list[str] | None, ignore: bool, image_paths: list[str]
) -> list[Path]:
    """The key-map files ``ocr``/``georef`` should use, applying the shared flag precedence.

    ``--ignore-keymap`` (``ignore``) turns the feature off; otherwise an explicit ``--keymap``
    list wins, and with neither the key maps are auto-discovered next to the images (see
    :func:`discover_keymaps`). Centralized so both commands resolve key maps identically.
    """
    if ignore:
        return []
    if explicit:
        return [Path(path) for path in explicit]
    return discover_keymaps(image_paths)


def keymap_regions_path(keymap_json: Path) -> Path:
    """Sibling ``<stem>.regions.panels.json`` (written by ``mapsnap.keymap.page_regions``)."""
    return keymap_json.with_name(
        keymap_json.name.replace(".keymap.json", ".regions.panels.json")
    )


def load_regions(
    keymap_json: Path, corners: list[Point], width: int, height: int
) -> dict[str, list[list[Point]]]:
    """World-space page-region polygons from a key map's ``<stem>.regions.panels.json``.

    Each panel's pixel ring is mapped to (lon, lat) via the key map's georeferenced corners.
    Rings are scaled if the regions file was computed at a different resolution than the
    georef. Returns page key -> list of rings; empty if no regions sidecar exists.
    """
    regions_path = keymap_regions_path(keymap_json)
    if not regions_path.exists():
        return {}
    doc = json.loads(regions_path.read_text())
    scale_x = width / doc["width"]
    scale_y = height / doc["height"]
    regions: dict[str, list[list[Point]]] = {}
    for ring, label in zip(doc["panels"], doc["labels"]):
        key = canonical_page_key(str(label))
        if key is None:
            continue
        world_ring = [
            bilinear_pixel_to_world(corners, width, height, (x * scale_x, y * scale_y))
            for x, y in ring
        ]
        regions.setdefault(key, []).append(world_ring)
    return regions


def bilinear_pixel_to_world(
    corners: list[Point], width: int, height: int, pixel: Point
) -> Point:
    """Bilinearly map an image pixel to (lon, lat) using a TL, TR, BR, BL corner quad."""
    top_left, top_right, bottom_right, bottom_left = corners
    u = pixel[0] / width
    v = pixel[1] / height
    top = (
        top_left[0] + (top_right[0] - top_left[0]) * u,
        top_left[1] + (top_right[1] - top_left[1]) * u,
    )
    bottom = (
        bottom_left[0] + (bottom_right[0] - bottom_left[0]) * u,
        bottom_left[1] + (bottom_right[1] - bottom_left[1]) * u,
    )
    return (top[0] + (bottom[0] - top[0]) * v, top[1] + (bottom[1] - top[1]) * v)


def geometry_vertices(geometry: dict) -> list[Point]:
    """Flatten a GeoJSON geometry's coordinates to a list of (lon, lat) vertices."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "LineString":
        return [(c[0], c[1]) for c in coords]
    if kind == "MultiLineString":
        return [(c[0], c[1]) for line in coords for c in line]
    if kind == "Point":
        return [(coords[0], coords[1])]
    if kind == "Polygon":
        return [(c[0], c[1]) for ring in coords for c in ring]
    return []


def geometry_segments(geometry: dict) -> list[tuple[Point, Point]]:
    """A GeoJSON geometry's edges as (start, end) (lon, lat) pairs, per line/ring.

    Consecutive vertices within each LineString / MultiLineString line / Polygon ring, so a long
    street segment that crosses a neighborhood without a vertex inside it is still testable (see
    :func:`KeymapLocator.restricted_features`). A Point yields one degenerate (p, p) segment so
    isolated points still register.
    """
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "Point":
        return [((coords[0], coords[1]), (coords[0], coords[1]))]
    lines: list[list] = []
    if kind == "LineString":
        lines = [coords]
    elif kind in ("MultiLineString", "Polygon"):
        lines = list(coords)
    segments: list[tuple[Point, Point]] = []
    for line in lines:
        for a, b in itertools.pairwise(line):
            segments.append(((a[0], a[1]), (b[0], b[1])))
    return segments


def meters_between(a: Point, b: Point) -> float:
    """Approximate distance in metres between two (lon, lat) points (equirectangular)."""
    mid_lat = math.radians((a[1] + b[1]) / 2)
    dx = (a[0] - b[0]) * M_PER_DEG_LON_EQUATOR * math.cos(mid_lat)
    dy = (a[1] - b[1]) * M_PER_DEG_LAT
    return math.hypot(dx, dy)


def segment_point_distance_m(segment: tuple[Point, Point], point: Point) -> float:
    """Distance in metres from ``point`` to a (lon, lat) segment, equirectangular local frame.

    Uses a frame anchored at ``point`` (so the point is the origin), scaling degrees to metres
    at the point's latitude, then the standard distance from the origin to the segment. Exact
    regardless of how far apart the segment's endpoints are, so a street crossing a neighborhood
    with no vertex inside still measures within range.
    """
    scale_x = M_PER_DEG_LON_EQUATOR * math.cos(math.radians(point[1]))
    ax = (segment[0][0] - point[0]) * scale_x
    ay = (segment[0][1] - point[1]) * M_PER_DEG_LAT
    bx = (segment[1][0] - point[0]) * scale_x
    by = (segment[1][1] - point[1]) * M_PER_DEG_LAT
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / seg_len_sq))
    return math.hypot(ax + t * dx, ay + t * dy)


def ring_area_m2(ring: list[Point]) -> float:
    """Shoelace area (m²) of a (lon, lat) ring, in a local equirectangular frame."""
    mid_lat = math.radians(sum(p[1] for p in ring) / len(ring))
    scale_x = M_PER_DEG_LON_EQUATOR * math.cos(mid_lat)
    points = [(p[0] * scale_x, p[1] * M_PER_DEG_LAT) for p in ring]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def region_scale_m_per_px(
    rings: list[list[Point]], width_px: float, height_px: float
) -> float | None:
    """Approximate page scale (metres per pixel) from its key-map region footprint.

    A page image of width x height px drawn at s m/px covers width*height*s² m² of
    ground, so s = sqrt(region area / pixel area). Area-based, so the page's rotation
    doesn't matter. Multiple rings for one number (a block split by watershed between
    duplicate detections) sum back to the full block. Returns None for degenerate input.
    """
    total = sum(ring_area_m2(ring) for ring in rings if len(ring) >= 3)
    if total <= 0 or width_px <= 0 or height_px <= 0:
        return None
    return math.sqrt(total / (width_px * height_px))


def estimate_radius(locations: dict[str, list[Point]]) -> float:
    """A neighborhood radius (metres) ~2x the key map's page-to-page spacing.

    Uses one representative point per page number (the mean of its detections); the median
    nearest-neighbor distance between pages approximates a single page's own extent, and 2x
    that comfortably covers a page plus a margin.
    """
    reps = [
        (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
        for pts in locations.values()
    ]
    if len(reps) < 2:
        return 1000.0
    nearest = [
        min(meters_between(p, q) for j, q in enumerate(reps) if j != i)
        for i, p in enumerate(reps)
    ]
    return 2.0 * float(np.median(nearest))


@dataclass
class KeymapLocator:
    """Per-page-key world locations read off a georeferenced key map."""

    locations: dict[str, list[Point]]  # page key -> world (lon, lat) of each detection
    radius_m: float
    # One image-corner quad (lon/lat) per key map; a volume can have several key maps whose
    # rectangles together cover it (e.g. Brooklyn's p0 = SW half, p0b = NE half).
    rectangles: list[list[Point]] = field(default_factory=list)
    # Page key -> world (lon, lat) rings of the colored block(s) drawn around that number
    # on the key map (from page_regions segmentation) — the page's approximate ground footprint.
    regions: dict[str, list[list[Point]]] = field(default_factory=dict)
    # Spatial index over the last feature list the cull methods were handed, kept
    # because callers pass the same volume-wide centerlines list once per page and
    # rebuilding the tree every time would cost what the scan it replaces cost. The
    # strong reference keeps that list alive, so its identity can never be reused.
    index_cache: tuple[list[dict], FeatureIndex] | None = field(
        default=None, repr=False, compare=False
    )

    def feature_index(self, features: list[dict]) -> FeatureIndex:
        """Spatial index over ``features``, reused across calls with the same list."""
        if self.index_cache is None or self.index_cache[0] is not features:
            self.index_cache = (features, FeatureIndex(features))
        return self.index_cache[1]

    @classmethod
    def from_keymap(
        cls, keymap_json: Path, radius_m: float | None = None
    ) -> "KeymapLocator":
        """Build a locator from a ``<stem>.keymap.json`` and its sibling ``<stem>.georef.json``."""
        doc = json.loads(keymap_georef_path(keymap_json).read_text())
        corners = [(float(c[0]), float(c[1])) for c in doc["corners"]]
        width, height = int(doc["width"]), int(doc["height"])
        locations: dict[str, list[Point]] = {}
        for detection in load_detections(keymap_json):
            world = bilinear_pixel_to_world(corners, width, height, detection.pixel)
            locations.setdefault(detection.key or str(detection.number), []).append(
                world
            )
        return cls(
            locations,
            radius_m if radius_m is not None else estimate_radius(locations),
            [corners],
            load_regions(keymap_json, corners, width, height),
        )

    @classmethod
    def from_keymaps(
        cls, keymap_jsons: list[Path], radius_m: float | None = None
    ) -> "KeymapLocator":
        """Combine several key maps of one volume into a single locator.

        A page key is placed by whichever key map(s) detect it (locations are unioned), and
        the fallback rectangle is the union of all the key maps' rectangles. The radius is the
        median of the per-key-map estimates unless overridden. Sheets that
        :func:`redundant_keymaps` rejects are left out of the union entirely.
        """
        # Imported here, as volume_pages_for does, to keep the pipeline module out of
        # this one's import graph.
        from mapsnap.keymap.pipeline import keymap_volume_dir

        locators = [cls.from_keymap(path) for path in keymap_jsons]
        volume_keys = (
            volume_page_keys(keymap_volume_dir(keymap_jsons[0]))
            if keymap_jsons
            else set()
        )
        for index in sorted(
            redundant_keymaps([set(loc.locations) for loc in locators], volume_keys),
            reverse=True,
        ):
            print(
                f"Ignoring key map {keymap_jsons[index].name}: it indexes the same pages "
                "as another sheet in this volume, and more of its page numbers are "
                "foreign to this volume.",
                file=sys.stderr,
            )
            del locators[index]
            del keymap_jsons[index]
        locations: dict[str, list[Point]] = {}
        rectangles: list[list[Point]] = []
        regions: dict[str, list[list[Point]]] = {}
        for locator in locators:
            for key, points in locator.locations.items():
                locations.setdefault(key, []).extend(points)
            rectangles.extend(locator.rectangles)
            for key, rings in locator.regions.items():
                regions.setdefault(key, []).extend(rings)
        radius = (
            radius_m
            if radius_m is not None
            else float(np.median([locator.radius_m for locator in locators]))
        )
        return cls(locations, radius, rectangles, regions)

    def located_keys(self) -> set[str]:
        """The page keys the key map places."""
        return set(self.locations)

    def matching_keys(self, page: int | str | None, mapping: dict) -> list[str]:
        """The stored keys that answer a lookup for ``page``.

        A string key matches exactly when stored; otherwise — and always for a
        bare int — it falls back to the key's whole stem family. The family
        fallback bridges the two mismatch directions: a key map that printed
        bare ``35`` serving disk page ``35A``, and a key map read as ``51N``
        serving a caller that only knows the number 51.
        """
        if page is None:
            return []
        key = canonical_page_key(str(page))
        if key is None:
            return []
        if key in mapping:
            return [key]
        stem = key_stem(key)
        return [stored for stored in mapping if key_stem(stored) == stem]

    def centers_for(self, page: int | str | None) -> list[Point]:
        """Every key-map detection location answering a lookup for ``page``."""
        return [
            point
            for key in self.matching_keys(page, self.locations)
            for point in self.locations[key]
        ]

    def regions_for(self, page: int | str | None) -> list[list[Point]]:
        """The segmented region rings answering a lookup for ``page``."""
        return [
            ring
            for key in self.matching_keys(page, self.regions)
            for ring in self.regions[key]
        ]

    def regions_by_number(self) -> dict[int, list[list[Point]]]:
        """Regions re-keyed by page number, letter families merged (int-consumer view)."""
        merged: dict[int, list[list[Point]]] = {}
        for key, rings in self.regions.items():
            merged.setdefault(int(key_stem(key)), []).extend(rings)
        return merged

    def page_keymap(self, page: int | str | None) -> dict | None:
        """The georef.json ``keymap`` entry: ``{lat, lon, radius_m, centers[, regions]}``.

        ``centers`` holds every key-map detection of the page as [lon, lat] —
        authoritative for display and matching. A page can legitimately appear twice
        (a split sheet has one block per panel), and the blocks can be far apart, so
        lat/lon — their mean, kept for compatibility — can land between them, inside
        neither; prefer ``centers``. radius_m is the neighborhood radius the page's
        OCR/fit was restricted to. ``regions`` (when the key map has a regions sidecar)
        is the page's segmented block(s) as world-space rings of [lon, lat] pairs,
        GeoJSON-style. None if unplaced.
        """
        centers = self.centers_for(page)
        if not centers:
            return None
        entry: dict = {
            "lat": round(sum(c[1] for c in centers) / len(centers), 7),
            "lon": round(sum(c[0] for c in centers) / len(centers), 7),
            "radius_m": round(self.radius_m, 1),
            "centers": [[round(c[0], 7), round(c[1], 7)] for c in centers],
        }
        rings = self.regions_for(page)
        if rings:
            entry["regions"] = [
                [[round(lon, 7), round(lat, 7)] for lon, lat in ring] for ring in rings
            ]
        return entry

    def rectangle_features(self, features: list[dict]) -> list[dict] | None:
        """Features inside the union of the key maps' georeferenced rectangles (+ radius margin).

        Every page sits *somewhere* on one of the key maps, so this volume-wide region is a valid
        — and often much tighter than the full centerlines — fallback vocabulary for a page whose
        own neighborhood came up empty (or that no key map places). Returns None if no key-map
        rectangle is known.
        """
        if not self.rectangles:
            return None
        boxes = []  # (min_lon, max_lon, min_lat, max_lat) per rectangle, with a radius margin
        for corners in self.rectangles:
            lons = [c[0] for c in corners]
            lats = [c[1] for c in corners]
            mid_lat = math.radians(sum(lats) / len(lats))
            margin_lon = self.radius_m / (M_PER_DEG_LON_EQUATOR * math.cos(mid_lat))
            margin_lat = self.radius_m / M_PER_DEG_LAT
            boxes.append(
                (
                    min(lons) - margin_lon,
                    max(lons) + margin_lon,
                    min(lats) - margin_lat,
                    max(lats) + margin_lat,
                )
            )
        return [
            feature
            for feature in self.feature_index(features).near_bboxes(boxes)
            if any(
                min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
                for (min_lon, max_lon, min_lat, max_lat) in boxes
                for lon, lat in geometry_vertices(feature.get("geometry", {}))
            )
        ]

    def restricted_features(
        self, page: int | str | None, features: list[dict]
    ) -> list[dict] | None:
        """Features with a vertex within ``radius_m`` of page ``page``'s location(s).

        Returns None if the key map does not place ``page`` (the caller should fall back to
        the full vocabulary), or a possibly-empty list of nearby features otherwise.
        """
        centers = self.centers_for(page)
        if not centers:
            return None
        kept = []
        for feature in self.feature_index(features).near_points(centers, self.radius_m):
            segments = geometry_segments(feature.get("geometry", {}))
            if any(
                segment_point_distance_m(segment, center) <= self.radius_m
                for center in centers
                for segment in segments
            ):
                kept.append(feature)
        return kept
