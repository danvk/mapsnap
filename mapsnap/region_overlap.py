"""Content-region geometry for the reconciler's overlap factor (#352).

A page's content region is the part of its sheet exclusive to that page:
truth regions tile the ground with no pair overlapping by more than 5%, so
two placed pages whose regions overlap substantially cannot both be right.
The region model's prediction (``mapsnap region`` -> ``artifacts/region/
<stem>.png``) stands in for the truth region here; at the 2026-09-03
published poses its overlap caught every ≥200 ft disaster on detroit while
good pages sat at a median of 3%.

This module turns a P(region) map into a polygon in page pixels, poses it
through a hypothesis affine into local metres, and measures overlap the way
the corpus measurement did: intersection over the smaller region.
"""

from pathlib import Path

import cv2
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from mapsnap.region_predict import region_prob_path

# P(region) at or above this is content.
REGION_THRESHOLD = 0.5
# A predicted region smaller than this share of its page carries no usable
# tiling signal: intersection over the smaller area explodes against it
# (detroit p69__2, 7% of its panel, "overlapped" its siblings 89%).
REGION_MIN_FRAC = 0.15
# Contour components below this share of the page are speckle.
COMPONENT_MIN_FRAC = 0.01


def region_polygon_px(
    volume: Path, stem: str, page_size: tuple[int, int]
) -> Polygon | MultiPolygon | None:
    """The page's predicted content region as a polygon in page pixels.

    None when the map is missing, does not match the page's size, or covers
    less than REGION_MIN_FRAC of the page.
    """
    path = region_prob_path(volume, stem)
    if not path.exists():
        return None
    prob = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if prob is None:
        return None
    height, width = prob.shape
    if (width, height) != tuple(page_size):
        return None
    mask = (prob >= REGION_THRESHOLD * 255).astype(np.uint8)
    if float(mask.mean()) < REGION_MIN_FRAC:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if (
            len(contour) < 3
            or cv2.contourArea(contour) < COMPONENT_MIN_FRAC * width * height
        ):
            continue
        polygons.append(make_valid(Polygon(contour[:, 0, :].astype(np.float64))))
    if not polygons:
        return None
    region = unary_union(polygons).buffer(0)
    return region if not region.is_empty else None


def posed_region_metres(
    region_px: Polygon | MultiPolygon,
    affine: np.ndarray,
    origin: tuple[float, float],
) -> Polygon | MultiPolygon:
    """Pose a page-pixel region through ``affine`` (px -> lon/lat) into metres about origin."""
    lon0, lat0 = origin
    kx = 111_320.0 * np.cos(np.radians(lat0))
    ky = 110_540.0
    matrix = [
        affine[0, 0] * kx,
        affine[0, 1] * kx,
        affine[1, 0] * ky,
        affine[1, 1] * ky,
        (affine[0, 2] - lon0) * kx,
        (affine[1, 2] - lat0) * ky,
    ]
    posed = affine_transform(region_px, matrix)
    return make_valid(posed).buffer(0)


def overlap_over_min(a: Polygon | MultiPolygon, b: Polygon | MultiPolygon) -> float:
    """Intersection area over the smaller area; 0 when either is empty or they are apart."""
    if a.is_empty or b.is_empty or not a.intersects(b):
        return 0.0
    smaller = min(a.area, b.area)
    if smaller <= 0:
        return 0.0
    return float(a.intersection(b).area / smaller)
