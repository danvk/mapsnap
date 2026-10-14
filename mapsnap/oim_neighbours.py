"""Predict the adjacent-sheet references a page prints, from OldInsuranceMaps georeferencing.

A Sanborn sheet prints each neighbouring sheet's number in the margin along the
edge they share. OIM's truth exports give every georeferenced sheet a content
region (the annotation's selector, in canvas pixels) and GCPs, so the neighbours
of sheet A, and where along A's outline each one borders it, follow from mapping
every other sheet's content region into A's pixels.

Measured against the hand labels on Hudson vol 9, LA vol 14 and Nashville 1a
(1,132 printed references): a neighbour is predicted for 98% of the printed
numbers, 96.8% of predictions match a printed number, and the printed number
lies on the predicted stretch of outline 98.5% of the time. The stretch is long
(median ~1,100 px at the mirror's 25% scale), so this gives the text and the
edge, not a box. Corner-touching neighbours, which Sanborn does not label, are
what MIN_SHARE removes.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry

from mapsnap.compare_iiif_georef import GCP, extract_gcps, fit_affine
from mapsnap.utils import source_id_to_page_key

# A neighbour counts when it borders at least this share of the page's content outline.
MIN_SHARE = 0.04
# The neighbour's footprint is buffered by this share of the canvas width before
# intersecting the outline, absorbing small gaps and overlaps between OIM's masks.
BORDER_BUFFER = 0.03


@dataclass
class OimPage:
    """One georeferenced OIM annotation: its page key, content region and pixel/world mapping."""

    key: str  # page key as OIM names it, e.g. 'p12' or 'p12__2'
    to_pixels: np.ndarray  # 3x3 lon/lat -> canvas px
    selector: Polygon  # content region, canvas px
    world: Polygon  # content region, lon/lat
    width: int  # canvas width
    height: int  # canvas height


def affine_3x3(gcps: list[GCP]) -> np.ndarray:
    """The least-squares px -> lon/lat affine of a page's GCPs, as a 3x3 matrix."""
    return np.vstack([fit_affine(gcps), [0.0, 0.0, 1.0]])


def selector_polygon(item: dict) -> Polygon:
    """An annotation's content region in canvas pixels (the whole canvas when unmasked)."""
    value = (item["target"].get("selector") or {}).get("value") or ""
    points = re.findall(r"([\d.]+),([\d.]+)", value)
    source = item["target"]["source"]
    if len(points) < 3:
        width, height = source["width"], source["height"]
        return Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    region = Polygon([(float(x), float(y)) for x, y in points]).buffer(0)
    if isinstance(region, Polygon):
        return region
    return max(getattr(region, "geoms", []), key=lambda g: g.area)


def transform(polygon: Polygon, matrix: np.ndarray) -> Polygon:
    """A polygon's exterior through a 3x3 affine (validity repaired)."""
    points = [tuple((matrix @ [x, y, 1.0])[:2]) for x, y in polygon.exterior.coords]
    repaired = Polygon(points).buffer(0)
    if isinstance(repaired, Polygon):
        return repaired
    return max(getattr(repaired, "geoms", [Polygon()]), key=lambda g: g.area)


def load_pages(export: Path) -> list[OimPage]:
    """Every annotation with at least three GCPs in an OIM export AnnotationPage."""
    pages = []
    for item in json.loads(export.read_text()).get("items", []):
        source = item["target"]["source"]
        gcps = extract_gcps(item)
        if len(gcps) < 3:
            continue
        to_world = affine_3x3(gcps)
        try:
            to_pixels = np.linalg.inv(to_world)
        except np.linalg.LinAlgError:
            continue
        selector = selector_polygon(item)
        key = source_id_to_page_key(source.get("id"), item.get("label") or "")
        pages.append(
            OimPage(
                key=key,
                to_pixels=to_pixels,
                selector=selector,
                world=transform(selector, to_world),
                width=source["width"],
                height=source["height"],
            )
        )
    return pages


def sheet_number(key: str) -> str:
    """The printed form of a page key: 'p12' -> '12', 'p381a__2' -> '381A'."""
    return key.split("__")[0].removeprefix("p").upper()


def longest_line(geometry: BaseGeometry) -> LineString | None:
    """The longest LineString piece of an outline intersection, or None."""
    if geometry.is_empty:
        return None
    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        return max(geometry.geoms, key=lambda g: g.length)
    lines = [g for g in getattr(geometry, "geoms", []) if isinstance(g, LineString)]
    return max(lines, key=lambda g: g.length) if lines else None


def predicted_neighbours(page: OimPage, others: list[OimPage]) -> dict[str, LineString]:
    """Neighbour sheet number -> the stretch of the page's content outline it borders.

    Coordinates are the page's canvas pixels. Panels of one neighbouring sheet
    share its number; the panel with the longest border wins. The page's own
    panels are never its neighbours.
    """
    boundary = page.selector.exterior
    perimeter = boundary.length
    best: dict[str, tuple[float, LineString]] = {}
    for other in others:
        number = sheet_number(other.key)
        if number == sheet_number(page.key):
            continue
        in_page = transform(other.world, page.to_pixels)
        if in_page.is_empty:
            continue
        shared = boundary.intersection(in_page.buffer(BORDER_BUFFER * page.width))
        if shared.is_empty:
            continue
        share = shared.length / perimeter
        line = longest_line(shared)
        if share < MIN_SHARE or line is None:
            continue
        if number not in best or share > best[number][0]:
            best[number] = (share, line)
    return {number: line for number, (_, line) in best.items()}
