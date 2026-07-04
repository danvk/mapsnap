"""Locate volume pages from a georeferenced key map, to restrict OCR/georef to nearby streets.

A key map is a schematic showing where each numbered page sits. Once the key map itself is
georeferenced (its ``<stem>.georef.json``), a page-number detection's pixel maps straight to
that page's real-world location. That lets the main OCR and georeference steps swap the
ambiguous city-wide street vocabulary for the handful of streets actually near a page —
dropping false matches (e.g. a second "Canal St" across town) and, for OCR, driving up
recognizer confidence on the correct names.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mapsnap.keymap.fit_keymap import load_detections, page_number

Point = tuple[float, float]

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQUATOR = 111320.0

__all__ = ["KeymapLocator", "page_number"]


def keymap_georef_path(keymap_json: Path) -> Path:
    """Sibling ``<stem>.georef.json`` for a key-map detections file (``<stem>.keymap.json``)."""
    return keymap_json.with_name(
        keymap_json.name.replace(".keymap.json", ".georef.json")
    )


def keymap_regions_path(keymap_json: Path) -> Path:
    """Sibling ``<stem>.regions.panels.json`` (written by ``mapsnap.keymap.page_regions``)."""
    return keymap_json.with_name(
        keymap_json.name.replace(".keymap.json", ".regions.panels.json")
    )


def load_regions(
    keymap_json: Path, corners: list[Point], width: int, height: int
) -> dict[int, list[list[Point]]]:
    """World-space page-region polygons from a key map's ``<stem>.regions.panels.json``.

    Each panel's pixel ring is mapped to (lon, lat) via the key map's georeferenced corners.
    Rings are scaled if the regions file was computed at a different resolution than the
    georef. Returns page number -> list of rings; empty if no regions sidecar exists.
    """
    regions_path = keymap_regions_path(keymap_json)
    if not regions_path.exists():
        return {}
    doc = json.load(open(regions_path))
    scale_x = width / doc["width"]
    scale_y = height / doc["height"]
    regions: dict[int, list[list[Point]]] = {}
    for ring, label in zip(doc["panels"], doc["labels"]):
        if not str(label).isdigit():
            continue
        world_ring = [
            bilinear_pixel_to_world(corners, width, height, (x * scale_x, y * scale_y))
            for x, y in ring
        ]
        regions.setdefault(int(label), []).append(world_ring)
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


def meters_between(a: Point, b: Point) -> float:
    """Approximate distance in metres between two (lon, lat) points (equirectangular)."""
    mid_lat = math.radians((a[1] + b[1]) / 2)
    dx = (a[0] - b[0]) * M_PER_DEG_LON_EQUATOR * math.cos(mid_lat)
    dy = (a[1] - b[1]) * M_PER_DEG_LAT
    return math.hypot(dx, dy)


def estimate_radius(locations: dict[int, list[Point]]) -> float:
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
    """Per-page-number world locations read off a georeferenced key map."""

    locations: dict[
        int, list[Point]
    ]  # page number -> world (lon, lat) of each detection
    radius_m: float
    # One image-corner quad (lon/lat) per key map; a volume can have several key maps whose
    # rectangles together cover it (e.g. Brooklyn's p0 = SW half, p0b = NE half).
    rectangles: list[list[Point]] = field(default_factory=list)
    # Page number -> world (lon, lat) rings of the colored block(s) drawn around that number
    # on the key map (from page_regions segmentation) — the page's approximate ground footprint.
    regions: dict[int, list[list[Point]]] = field(default_factory=dict)

    @classmethod
    def from_keymap(
        cls, keymap_json: Path, radius_m: float | None = None
    ) -> "KeymapLocator":
        """Build a locator from a ``<stem>.keymap.json`` and its sibling ``<stem>.georef.json``."""
        doc = json.load(open(keymap_georef_path(keymap_json)))
        corners = [(float(c[0]), float(c[1])) for c in doc["corners"]]
        width, height = int(doc["width"]), int(doc["height"])
        locations: dict[int, list[Point]] = {}
        for detection in load_detections(keymap_json):
            world = bilinear_pixel_to_world(corners, width, height, detection.pixel)
            locations.setdefault(detection.number, []).append(world)
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

        A page number is placed by whichever key map(s) detect it (locations are unioned), and
        the fallback rectangle is the union of all the key maps' rectangles. The radius is the
        median of the per-key-map estimates unless overridden.
        """
        locators = [cls.from_keymap(path) for path in keymap_jsons]
        locations: dict[int, list[Point]] = {}
        rectangles: list[list[Point]] = []
        regions: dict[int, list[list[Point]]] = {}
        for locator in locators:
            for number, points in locator.locations.items():
                locations.setdefault(number, []).extend(points)
            rectangles.extend(locator.rectangles)
            for number, rings in locator.regions.items():
                regions.setdefault(number, []).extend(rings)
        radius = (
            radius_m
            if radius_m is not None
            else float(np.median([locator.radius_m for locator in locators]))
        )
        return cls(locations, radius, rectangles, regions)

    def located_numbers(self) -> set[int]:
        """The page numbers the key map places."""
        return set(self.locations)

    def page_keymap(self, number: int | None) -> dict | None:
        """The georef.json ``keymap`` entry for a page: ``{lat, lon, radius_m[, regions]}``.

        lat/lon is the mean of the page number's key-map detections; radius_m is the
        neighborhood radius the page's OCR/fit was restricted to. ``regions`` (when the key
        map has a regions sidecar) is the page's segmented key-map block(s) as world-space
        rings of [lon, lat] pairs, GeoJSON-style. None if unplaced.
        """
        centers = self.locations.get(number) if number is not None else None
        if not centers:
            return None
        entry: dict = {
            "lat": round(sum(c[1] for c in centers) / len(centers), 7),
            "lon": round(sum(c[0] for c in centers) / len(centers), 7),
            "radius_m": round(self.radius_m, 1),
        }
        rings = self.regions.get(number) if number is not None else None
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
            for feature in features
            if any(
                min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
                for (min_lon, max_lon, min_lat, max_lat) in boxes
                for lon, lat in geometry_vertices(feature.get("geometry", {}))
            )
        ]

    def restricted_features(
        self, number: int | None, features: list[dict]
    ) -> list[dict] | None:
        """Features with a vertex within ``radius_m`` of page ``number``'s location(s).

        Returns None if the key map does not place ``number`` (the caller should fall back to
        the full vocabulary), or a possibly-empty list of nearby features otherwise.
        """
        centers = self.locations.get(number) if number is not None else None
        if not centers:
            return None
        kept = []
        for feature in features:
            vertices = geometry_vertices(feature.get("geometry", {}))
            if any(
                meters_between(center, vertex) <= self.radius_m
                for center in centers
                for vertex in vertices
            ):
                kept.append(feature)
        return kept
