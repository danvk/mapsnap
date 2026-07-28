"""Spatial index over GeoJSON features, so "what's near here" stops being a linear scan.

A volume's centerlines.geojson runs to hundreds of thousands of features (Los
Angeles county: 201k). Consumers that repeatedly ask for the features near a
point or a frame — the OSM rasterizer's per-frame bbox cull, the key map's
per-page vocabulary restriction — used to walk the entire list per query, which
costs 0.5-2 s each and dominated their runtime.

The index is a pure *cull*: every query returns a superset of the features whose
geometry could satisfy the caller's own (exact, unchanged) test, in the original
feature order. Callers keep their test and their answers, and only stop paying
for the features that could never have passed it.
"""

import math

import numpy as np
import shapely
from shapely.strtree import STRtree

Bounds = tuple[float, float, float, float]  # (min_lon, max_lon, min_lat, max_lat)

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQUATOR = 111320.0


def geometry_bbox(geometry: dict) -> Bounds | None:
    """(min_lon, max_lon, min_lat, max_lat) of a geometry's vertices, or None if it has none."""
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return None
    if kind == "Point":
        points = [coordinates]
    elif kind in ("LineString", "MultiPoint"):
        points = coordinates
    elif kind in ("MultiLineString", "Polygon"):
        points = [point for part in coordinates for point in part]
    elif kind == "MultiPolygon":
        points = [point for part in coordinates for ring in part for point in ring]
    else:
        return None
    if not points:
        return None
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return min(lons), max(lons), min(lats), max(lats)


def point_bounds(point: tuple[float, float], radius_m: float) -> Bounds:
    """The lon/lat box of everything within ``radius_m`` of a lon/lat point.

    Sized with the same equirectangular scales the exact distance tests use (see
    :func:`mapsnap.keymap.locate.segment_point_distance_m`), so nothing genuinely
    within the radius can fall outside the box.
    """
    lon, lat = point
    d_lat = radius_m / M_PER_DEG_LAT
    d_lon = radius_m / max(M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat)), 1e-6)
    return (lon - d_lon, lon + d_lon, lat - d_lat, lat + d_lat)


class FeatureIndex:
    """An STRtree over GeoJSON features' bounding boxes; build one per feature list."""

    def __init__(self, features: list[dict]) -> None:
        self.features = features
        min_lons: list[float] = []
        min_lats: list[float] = []
        max_lons: list[float] = []
        max_lats: list[float] = []
        # Features with no usable geometry stay out of the tree: they can never
        # be near anything, and every caller already skips them.
        self.positions: list[int] = []
        for position, feature in enumerate(features):
            bbox = geometry_bbox(feature.get("geometry") or {})
            if bbox is None:
                continue
            self.positions.append(position)
            min_lons.append(bbox[0])
            max_lons.append(bbox[1])
            min_lats.append(bbox[2])
            max_lats.append(bbox[3])
        self.tree = STRtree(
            shapely.box(
                np.array(min_lons),
                np.array(min_lats),
                np.array(max_lons),
                np.array(max_lats),
            )
        )

    def near_bboxes(self, boxes: list[Bounds]) -> list[dict]:
        """Features whose bounding box touches any of the lon/lat boxes, in feature order."""
        if not boxes or not self.positions:
            return []
        query = shapely.box(
            np.array([b[0] for b in boxes]),
            np.array([b[2] for b in boxes]),
            np.array([b[1] for b in boxes]),
            np.array([b[3] for b in boxes]),
        )
        # A multi-geometry query returns (query index, tree index) pairs; only
        # the tree side matters, deduped since a feature can hit several boxes.
        hits = sorted(set(self.tree.query(query)[1].tolist()))
        return [self.features[self.positions[i]] for i in hits]

    def near_bbox(self, bounds: Bounds) -> list[dict]:
        """Features whose bounding box touches the lon/lat box, in feature order."""
        return self.near_bboxes([bounds])

    def near_points(
        self, points: list[tuple[float, float]], radius_m: float
    ) -> list[dict]:
        """Features whose bounding box comes within ``radius_m`` of any lon/lat point."""
        return self.near_bboxes([point_bounds(point, radius_m) for point in points])
