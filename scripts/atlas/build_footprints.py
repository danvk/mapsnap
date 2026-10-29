#!/usr/bin/env python
"""Build the atlas's volume footprints from a published run.

    uv run python scripts/atlas/build_footprints.py \\
        --run-dir ~/Documents/mapsnap/data.mapsnap.org/runs/v2.0

The atlas makes the volume its unit. A volume here is a series of editions
of the same ground: Chicago's vol. 5 of 1914 and of 1950, or the whole of a
small town, surveyed every few years. Zoomed in, the map draws one shape per
volume, and a click on it lists every edition. That needs the ground each
edition's placed sheets cover, and how the editions group into volumes
(group_series). This writes one file per town, beside the volume lists
build_places.py writes:

    app/public/atlas/footprints/<state>/<town>.json   (one town, e.g. illinois/chicago)
      {"volumes": {"<item>": {"year": 1914,
                              "anchor": [lon, lat],     # a point inside the footprint
                              "footprint": <MultiPolygon coordinates>}},
       "series": [{"volume": "5",                       # absent when unnumbered
                   "items": ["<item>", ...],            # every edition, placed or not
                   "overview": true,                    # only for an overview atlas
                   "footprint": <MultiPolygon coordinates> | null,
                   "anchor": [lon, lat] | null,
                   "display": true | <MultiPolygon coordinates> | null}]}
    app/public/atlas/footprints/index.json
      {"run": {"version": "v2.0", "url": "https://data.mapsnap.org/runs/v2.0"},
       "towns": {"<state>/<town>": [west, south, east, north]}}

A file per town rather than per state because the map, zoomed in far enough
to show footprints, has only a few towns in view: Wernersville's file is a few
kilobytes, Chicago's (97 volumes) about a megabyte, where all of New York
state is six. The index says which towns a viewport needs, and which run the
app reads annotations from.

A series' ``display`` is the part of its footprint no smaller series of the
town covers (``true`` when that is all of it), which is what the map draws: a
patchwork of volumes rather than a stack of them. It is null for a series
smaller ones cover entirely, which the app still lists for the spot.

A run places a few pages badly, and one wild page would stretch a footprint
across a county, so pages are dropped before the union when their sheet is
larger than any sheet the truth set holds (3.12 km corner to corner), or when
they sit far from every other page of their volume.
"""

import argparse
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely import STRtree, affinity
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

DEFAULT_ATLAS_DIR = Path(__file__).resolve().parents[2] / "app/public/atlas"
DATA_URL = "https://data.mapsnap.org"
EARTH_RADIUS_M = 6371008.8
METRES_PER_DEGREE = math.pi * EARTH_RADIUS_M / 180
# The largest sheet in the truth set (Hudson Co. 1950 vol 9 p92, an inset).
MAX_SHEET_DIAGONAL_M = 3120.0
# Seams between neighbouring sheets are closed by growing the union this much
# and shrinking it back.
SEAM_M = 50.0
SIMPLIFY_M = 10.0
# A series that smaller ones leave less than this share of uncovered is hidden
# from the map.
MIN_VISIBLE_SHARE = 0.05
# Smaller series' coverage grows by this before it is subtracted, and what is
# left is opened by it, so misaligned volumes leave no fringe of slivers.
MARGIN_M = 60.0
# A displayed piece smaller than this (2 ha, about a city block) is dropped.
MIN_PART_M2 = 20_000.0
# A displayed part this close to the whole footprint is drawn as the whole.
WHOLE_SHARE = 0.99
# Two unnumbered editions are one volume when they share this much of the
# smaller one's ground.
SAME_VOLUME_SHARE = 0.3
# An unnumbered edition covering more than this share of each of
# OVERVIEW_MIN_VOLUMES numbered volumes is an overview atlas. In v2.0 this
# finds only Brooklyn's 1905 and 1909 street-cleaning atlases.
OVERVIEW_SHARE = 0.5
OVERVIEW_MIN_VOLUMES = 3
# The P(good) a page needs to count toward its volume's shape (page_bar). A
# page in the main body of its edition, on ground no other volume covers,
# needs none: if it is misplaced it is misplaced in the right part of town,
# and barring it would leave 1,313 towns with no shape at all. Any other page
# needs BASE_BAR, raised for a page apart from the main body, for each
# edition of another volume covering it (up to MAX_CLAIMS), and for a page no
# other edition of its own volume covers. Tuned so that Brooklyn vol. 1
# 1886's misplaced p27L (0.56) and its cluster p14R, p22L and p2R__1 (0.61,
# 0.14, 0.73) drop out, while Poughkeepsie 1952's Hyde Park pages p86 and p87
# (0.85, 0.92), apart from town and in no other edition, stay. In v2.0 this
# leaves 11.5% of pages out of the shapes.
MAIN_BODY_BAR = 0.0
BASE_BAR = 0.7
DISCONNECTED_STEP = 0.1
CLAIM_STEP = 0.1
MAX_CLAIMS = 2
UNCORROBORATED_STEP = 0.05


def page_transform(item: dict):
    """pixel (x, y) -> (lon, lat) from an annotation item's GCPs, or None.

    Similarity for two GCPs (a Helmert fit, with the page's downward y
    flipped to north), affine for more, both fitted in a local metric frame --
    a similarity in raw degrees would stretch the page north-south by
    1/cos(latitude).
    """
    gcps = [
        (feature["properties"]["resourceCoords"], feature["geometry"]["coordinates"])
        for feature in item.get("body", {}).get("features", [])
        if feature.get("properties", {}).get("resourceCoords")
        and feature.get("geometry")
    ]
    if len(gcps) < 2:
        return None
    pixels = np.array([pixel for pixel, _ in gcps], float)
    geo = np.array([point for _, point in gcps], float)
    if not np.all(np.abs(geo[:, 1]) <= 90) or not np.all(np.abs(geo[:, 0]) <= 180):
        return None
    k = math.cos(math.radians(geo[:, 1].mean()))
    metric = np.c_[geo[:, 0] * k, geo[:, 1]]
    if len(gcps) == 2:
        # Pixel y runs down the page and latitude runs north, so y is flipped
        # (x - iy) before the fit: a rotation and scale alone would lay the
        # page mirrored across the line between its two GCPs.
        z = pixels[:, 0] - 1j * pixels[:, 1]
        w = metric[:, 0] + 1j * metric[:, 1]
        if z[1] == z[0]:
            return None
        a = (w[1] - w[0]) / (z[1] - z[0])
        b = w[0] - a * z[0]

        def similarity(x: float, y: float) -> tuple[float, float]:
            c = a * (x - 1j * y) + b
            return (c.real / k, c.imag)

        return similarity
    coefficients, *_ = np.linalg.lstsq(
        np.c_[pixels, np.ones(len(pixels))], metric, rcond=None
    )

    def affine(x: float, y: float) -> tuple[float, float]:
        mx, my = np.array([x, y, 1.0]) @ coefficients
        return (mx / k, my)

    return affine


def ground_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lon, lat) points."""
    lat1, lat2 = math.radians(a[1]), math.radians(b[1])
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(math.radians(b[0] - a[0]) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


@dataclass
class Page:
    """A placed image on the ground: its page key, clip outline, sheet size and P(good)."""

    key: str
    outline: Polygon
    # The sheet's diagonal on the ground, in meters.
    diagonal: float
    # The confidence model's P(good), or None for an annotation without one.
    confidence: float | None = None


def page_key(label: str) -> str:
    """The page key a published label ends with: "... sanborn00001_001 p2 [1]" -> "p2__1"."""
    tokens = label.split()
    if len(tokens) >= 2 and re.fullmatch(r"\[\d+\]", tokens[-1]):
        return f"{tokens[-2]}__{tokens[-1][1:-1]}"
    return tokens[-1] if tokens else ""


def page_confidence(item: dict) -> float | None:
    """An annotation item's P(good), from its ``confidence`` metadata entry."""
    for entry in item.get("metadata") or []:
        if entry.get("label") == "confidence":
            try:
                return float(entry["value"])
            except (TypeError, ValueError):
                return None
    return None


def page_outline(item: dict) -> Page | None:
    """A placed page's clip outline on the ground, its sheet's diagonal and its P(good)."""
    to_geo = page_transform(item)
    source = item.get("target", {}).get("source", {})
    width, height = source.get("width"), source.get("height")
    if to_geo is None or not width or not height:
        return None
    selector = (item["target"].get("selector") or {}).get("value", "")
    points = re.search(r'points="([^"]+)"', selector)
    if points:
        pixels = [
            tuple(map(float, pair.split(","))) for pair in points.group(1).split()
        ]
    else:
        pixels = [(0, 0), (width, 0), (width, height), (0, height)]
    ring = [to_geo(x, y) for x, y in pixels]
    corners = [to_geo(0, 0), to_geo(width, 0), to_geo(width, height), to_geo(0, height)]
    diagonal = max(
        ground_distance_m(corners[0], corners[2]),
        ground_distance_m(corners[1], corners[3]),
    )
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or not math.isfinite(diagonal):
        return None
    return Page(
        page_key(str(item.get("label") or "")),
        polygon,
        diagonal,
        page_confidence(item),
    )


def plausible_pages(pages: list[Page]) -> list[Page]:
    """The pages worth a footprint: sheets of a drawable size, near the rest of the volume.

    A page is isolated when its center is farther than 2 km, or three of the
    volume's median sheets, from every other page's. That only means something
    with three or more pages, so smaller volumes keep every sized page.
    """
    sized = [page for page in pages if page.diagonal <= MAX_SHEET_DIAGONAL_M]
    if len(sized) < 3:
        return sized
    reach = max(2000.0, 3 * float(np.median([page.diagonal for page in sized])))
    centers = [(page.outline.centroid.x, page.outline.centroid.y) for page in sized]
    kept = []
    for i, page in enumerate(sized):
        nearest = min(
            ground_distance_m(centers[i], centers[j])
            for j in range(len(sized))
            if j != i
        )
        if nearest <= reach:
            kept.append(page)
    return kept


def in_metres(geometry, latitude: float):
    """Scale lon/lat so a degree is the same ground distance both ways (for buffers)."""
    return affinity.scale(
        geometry, xfact=math.cos(math.radians(latitude)), yfact=1.0, origin=(0, 0)
    )


def in_degrees(geometry, latitude: float):
    """Undo in_metres."""
    return affinity.scale(
        geometry, xfact=1 / math.cos(math.radians(latitude)), yfact=1.0, origin=(0, 0)
    )


def volume_footprint(pages: list[Polygon]) -> MultiPolygon | None:
    """The union of a volume's pages, seams closed, holes filled and simplified, or None."""
    if not pages:
        return None
    latitude = unary_union(pages).centroid.y
    seam = SEAM_M / METRES_PER_DEGREE
    union = unary_union([in_metres(page, latitude) for page in pages])
    closed = union.buffer(seam, join_style="mitre").buffer(-seam, join_style="mitre")
    simplified = closed.simplify(SIMPLIFY_M / METRES_PER_DEGREE, preserve_topology=True)
    footprint = as_multipolygon(make_valid(in_degrees(simplified, latitude)))
    # A sheet the run could not place leaves a hole, but the ground is still
    # this volume's: a click there should find it, not fall through.
    if footprint is None:
        return None
    filled = unary_union(
        [make_valid(Polygon(part.exterior)) for part in footprint.geoms]
    )
    return as_multipolygon(make_valid(filled))


def as_multipolygon(geometry) -> MultiPolygon | None:
    """The polygons of a geometry as one MultiPolygon, or None when there are none."""
    if geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    if isinstance(geometry, MultiPolygon):
        return geometry
    polygons = [
        part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)
    ]
    return MultiPolygon(polygons) if polygons else None


def overlay(a, b, operation: str):
    """``a.<operation>(b)`` (difference, union, intersection), repairing both and retrying once on a topology error."""
    try:
        return getattr(a, operation)(b)
    except GEOSException:
        return getattr(make_valid(a).buffer(0), operation)(make_valid(b).buffer(0))


def visible_parts[K](
    shapes: Mapping[K, MultiPolygon], order: list[K]
) -> dict[K, MultiPolygon | None]:
    """The part of each shape that no shape before it in ``order`` covers.

    This turns overlapping shapes into a patchwork: each spot belongs to the
    first shape that covers it. Two shapes of one district never line up
    exactly, so a bare difference leaves a fringe of slivers where a later shape
    pokes past an earlier one. To avoid that, the earlier coverage is grown by
    MARGIN_M before it is subtracted, what is left is opened by the same amount
    (dropping anything narrower than twice that), and pieces under MIN_PART_M2
    are dropped. A shape left with under MIN_VISIBLE_SHARE of its area, or with
    nothing, gets None, and so does any shape that is not in ``order``.
    """
    display: dict[K, MultiPolygon | None] = {key: None for key in shapes}
    if not order:
        return display
    latitude = unary_union([shapes[key] for key in order]).centroid.y
    margin = MARGIN_M / METRES_PER_DEGREE
    min_part = MIN_PART_M2 / METRES_PER_DEGREE**2
    covered = None
    for key in order:
        shape = in_metres(shapes[key], latitude)
        grown = None if covered is None else covered.buffer(margin, join_style="mitre")
        if grown is None or not grown.intersects(shape):
            # Nothing to subtract, so nothing to open either: opening would
            # erase a one-sheet town narrower than twice the margin.
            display[key] = shapes[key]
            covered = shape if covered is None else overlay(covered, shape, "union")
            continue
        remainder = overlay(shape, grown, "difference")
        opened = overlay(
            remainder.buffer(-margin, join_style="mitre").buffer(
                margin, join_style="mitre"
            ),
            shape,
            "intersection",
        ).simplify(SIMPLIFY_M / METRES_PER_DEGREE, preserve_topology=True)
        parts = as_multipolygon(opened)
        kept = (
            MultiPolygon([part for part in parts.geoms if part.area >= min_part])
            if parts is not None
            else None
        )
        if kept is not None and kept.area >= MIN_VISIBLE_SHARE * shape.area:
            # Nearly all of it is the whole of it: the caller then gets the
            # shape itself rather than one the opening only nicked.
            whole = kept.area >= WHOLE_SHARE * shape.area
            display[key] = (
                shapes[key]
                if whole
                else as_multipolygon(make_valid(in_degrees(kept, latitude)))
            )
        covered = shape if covered is None else overlay(covered, shape, "union")
    return display


@dataclass
class Edition:
    """One catalogued volume of a town, and its footprint when the run placed any of it."""

    item: str
    year: int | None
    # The catalog's volume number, or "" when it gives none.
    number: str = ""
    footprint: MultiPolygon | None = None
    # Its plausible placed pages, which the footprint is the union of.
    pages: list[Page] = field(default_factory=list)


@dataclass
class Series:
    """One volume of a town across its editions: Chicago's vol. 5 of 1914 and 1950.

    ``number`` is the catalogue's volume number, or "" for editions the
    catalog does not number, which are grouped by where they lie instead.
    An ``overview`` is a small-scale atlas of a whole city (Brooklyn's 1905 and
    1909 street-cleaning atlases), which the map leaves out.
    """

    number: str
    items: list[str] = field(default_factory=list)
    overview: bool = False


def shares(a: MultiPolygon, b: MultiPolygon) -> float:
    """How much of the smaller of two shapes the other covers, from 0 to 1."""
    smaller = min(a.area, b.area)
    return overlay(a, b, "intersection").area / smaller if smaller > 0 else 0.0


def overlap_clusters(editions: list[Edition]) -> list[list[Edition]]:
    """Group placed editions that cover the same ground (SAME_VOLUME_SHARE), oldest first.

    An edge joins two editions when they share SAME_VOLUME_SHARE of the
    smaller one. A cluster is a connected component of that graph.
    """
    parent = list(range(len(editions)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(editions):
        for j in range(i + 1, len(editions)):
            b = editions[j]
            assert a.footprint is not None and b.footprint is not None
            if shares(a.footprint, b.footprint) >= SAME_VOLUME_SHARE:
                parent[root(i)] = root(j)
    groups: dict[int, list[Edition]] = defaultdict(list)
    for i, edition in enumerate(editions):
        groups[root(i)].append(edition)
    return sorted(groups.values(), key=lambda group: min(year_key(e) for e in group))


def year_key(edition: Edition) -> int:
    """Sort key for an edition's year, undated last."""
    return edition.year if edition.year is not None else 10_000


def number_key(number: str) -> tuple[int, str]:
    """Sort key for a volume number: "2" < "10" < "10A" < "A"."""
    match = re.match(r"(\d+)(.*)", number)
    return (int(match.group(1)), match.group(2)) if match else (1_000_000, number)


def is_overview(footprint: MultiPolygon, numbered: Mapping[str, BaseGeometry]) -> bool:
    """Whether an unnumbered edition covers most of OVERVIEW_MIN_VOLUMES or more numbered volumes."""
    covered = sum(
        1
        for union in numbered.values()
        if union.area > 0
        and overlay(footprint, union, "intersection").area > OVERVIEW_SHARE * union.area
    )
    return covered >= OVERVIEW_MIN_VOLUMES


def group_series(editions: list[Edition]) -> list[Series]:
    """Group a town's editions into volumes, each one series of editions of the same ground.

    - Editions with a volume number are grouped by number. Numbered volumes are
      spatially consistent across editions: for the median town, the sum of
      their areas is 1.06 times the area of their union.
    - Placed editions without a number are clustered by overlap
      (overlap_clusters). For most towns this gives one series, the town.
      Overview atlases are clustered on their own and flagged.
    - Unplaced editions without a number join the largest cluster when the
      town has no numbered volumes, or form a series of their own when it has
      no cluster either. In a numbered town there is no telling which volume
      they belong to, so they are left out of every series.

    Numbered series come first, in number order, then the clusters, oldest first.
    """
    by_number: dict[str, list[Edition]] = defaultdict(list)
    loose_placed: list[Edition] = []
    loose_unplaced: list[Edition] = []
    for edition in editions:
        if edition.number:
            by_number[edition.number].append(edition)
        elif edition.footprint is not None:
            loose_placed.append(edition)
        else:
            loose_unplaced.append(edition)
    numbered_unions = {
        number: unary_union([e.footprint for e in group if e.footprint is not None])
        for number, group in by_number.items()
        if any(e.footprint is not None for e in group)
    }
    overviews = [
        e
        for e in loose_placed
        if e.footprint is not None and is_overview(e.footprint, numbered_unions)
    ]
    groups = overlap_clusters([e for e in loose_placed if e not in overviews])
    clusters = [Series("", [e.item for e in group]) for group in groups]
    if loose_unplaced and not by_number:
        if clusters:
            areas = [unary_union([e.footprint for e in group]).area for group in groups]
            largest = clusters[areas.index(max(areas))]
            largest.items.extend(e.item for e in loose_unplaced)
        else:
            clusters.append(Series("", [e.item for e in loose_unplaced]))
    numbered = [
        Series(number, [e.item for e in by_number[number]])
        for number in sorted(by_number, key=number_key)
    ]
    overview_series = [
        Series("", [e.item for e in group], overview=True)
        for group in overlap_clusters(overviews)
    ]
    return numbered + clusters + overview_series


def page_bar(*, disconnected: bool, claims: int, uncorroborated: bool) -> float:
    """The P(good) a page needs to count toward its volume's shape on the map.

    A page in the main body of its edition that no other volume claims needs
    only MAIN_BODY_BAR. For any other, the bar starts at BASE_BAR and rises
    for a page that is less likely to be where it seems: one apart from the
    main body of its edition, one on ground another volume's editions cover
    (more so for each, up to MAX_CLAIMS), and one no other edition of its own
    volume covers.
    """
    if not disconnected and claims == 0:
        return MAIN_BODY_BAR
    return (
        BASE_BAR
        + (DISCONNECTED_STEP if disconnected else 0.0)
        + CLAIM_STEP * min(claims, MAX_CLAIMS)
        + (UNCORROBORATED_STEP if uncorroborated else 0.0)
    )


def page_components(pages: list[Page]) -> list[list[int]]:
    """Group an edition's pages into connected clusters: pages within SEAM_M of each other."""
    if not pages:
        return []
    seam = SEAM_M / METRES_PER_DEGREE
    tree = STRtree([page.outline for page in pages])
    parent = list(range(len(pages)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, page in enumerate(pages):
        for j in tree.query(page.outline.buffer(seam), predicate="intersects"):
            parent[root(int(j))] = root(i)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(pages)):
        groups[root(i)].append(i)
    return list(groups.values())


def confidence_of(page: Page) -> float:
    """A page's P(good); a page without one is taken to be just good enough."""
    return BASE_BAR if page.confidence is None else page.confidence


@dataclass
class PageVerdict:
    """Whether a page counts toward its volume's shape, and why."""

    item: str
    page: Page
    bar: float
    disconnected: bool
    claims: int
    uncorroborated: bool

    @property
    def kept(self) -> bool:
        """Whether the page's P(good) clears its bar."""
        return confidence_of(self.page) >= self.bar


def judge_pages(editions: list[Edition], series: list[Series]) -> list[PageVerdict]:
    """Decide which of a town's placed pages count toward their volumes' shapes.

    A volume's shape is what the map draws and a click finds, so a misplaced
    page in it carves another volume's ground away, or adds an island where
    there is nothing. The imagery still shows every page; only the shape is
    judged, by each page's P(good) against page_bar. Another edition "covers"
    a page when its pages with P(good) of at least BASE_BAR hold the page's
    center.
    """
    series_of = {item: i for i, entry in enumerate(series) for item in entry.items}
    placed = [e for e in editions if e.pages]
    covers: list[tuple[str, BaseGeometry]] = []
    for edition in placed:
        confident = [p.outline for p in edition.pages if confidence_of(p) >= BASE_BAR]
        if confident:
            covers.append((edition.item, unary_union(confident)))
    tree = STRtree([shape for _, shape in covers])
    verdicts = []
    for edition in placed:
        own = series_of.get(edition.item)
        siblings = [
            e.item
            for e in placed
            if e.item != edition.item and series_of.get(e.item) == own
        ]
        components = page_components(edition.pages)
        main = max(
            components,
            key=lambda c: sum(confidence_of(edition.pages[i]) for i in c),
        )
        for component in components:
            for i in component:
                page = edition.pages[i]
                center = page.outline.representative_point()
                coverers = [
                    covers[int(j)][0]
                    for j in tree.query(center, predicate="within")
                    if covers[int(j)][0] != edition.item
                ]
                claims = sum(1 for item in coverers if series_of.get(item) != own)
                corroborated = any(series_of.get(item) == own for item in coverers)
                disconnected = component is not main
                uncorroborated = bool(siblings) and not corroborated
                verdicts.append(
                    PageVerdict(
                        edition.item,
                        page,
                        page_bar(
                            disconnected=disconnected,
                            claims=claims,
                            uncorroborated=uncorroborated,
                        ),
                        disconnected,
                        claims,
                        uncorroborated,
                    )
                )
    return verdicts


def series_footprint(pages: list[Polygon]) -> MultiPolygon | None:
    """The ground a series' editions cover between them, seams closed and holes filled."""
    return volume_footprint(pages)


def rounded(geometry: MultiPolygon | None) -> list | None:
    """GeoJSON MultiPolygon coordinates at 5 decimals (about a metre)."""
    if geometry is None:
        return None
    return json.loads(
        json.dumps(mapping(geometry)["coordinates"]),
        parse_float=lambda text: round(float(text), 5),
    )


def anchor_of(geometry: MultiPolygon) -> list[float]:
    """A point inside a shape, at 5 decimals."""
    point = geometry.representative_point()
    return [round(point.x, 5), round(point.y, 5)]


def town_record(
    editions: list[Edition], verdicts: list[PageVerdict] | None = None
) -> dict:
    """One town's footprint file: its placed editions, and its volumes as series.

    ``volumes`` maps each placed edition to its year, anchor and footprint.
    Each entry of ``series`` lists its editions, the ground they cover
    between them, and ``display``: what the map draws, the part no smaller
    series covers (``true`` when that is all of it). The smallest series goes
    first because it is the most specific map of its ground. Overviews and
    series with no placed edition have no ``display``.

    Only the pages judge_pages keeps count toward a series' shape and an
    edition's footprint; an edition with none kept keeps its whole footprint,
    so it can still be framed and found. ``verdicts`` are judge_pages' for
    these editions, computed here when not given.
    """
    series = group_series(editions)
    if verdicts is None:
        verdicts = judge_pages(editions, series)
    kept: dict[str, list[Polygon]] = defaultdict(list)
    for verdict in verdicts:
        if verdict.kept:
            kept[verdict.item].append(verdict.page.outline)
    footprints = {
        e.item: volume_footprint(kept[e.item]) or e.footprint
        for e in editions
        if e.footprint is not None
    }
    shapes: dict[int, MultiPolygon] = {}
    for i, entry in enumerate(series):
        shape = series_footprint([page for item in entry.items for page in kept[item]])
        if shape is not None:
            shapes[i] = shape
    order = sorted(
        (i for i in shapes if not series[i].overview), key=lambda i: shapes[i].area
    )
    display = visible_parts(shapes, order)
    series_records = []
    for i, entry in enumerate(series):
        shape = shapes.get(i)
        shown = display.get(i)
        record: dict = {"items": entry.items}
        if entry.number:
            record["volume"] = entry.number
        if entry.overview:
            record["overview"] = True
        record["footprint"] = rounded(shape)
        record["anchor"] = anchor_of(shown or shape) if shape is not None else None
        record["display"] = (
            True if shown is not None and shown is shape else rounded(shown)
        )
        series_records.append(record)
    years = {e.item: e.year for e in editions}
    return {
        "volumes": {
            item: {
                "year": years[item],
                "anchor": anchor_of(footprint),
                "footprint": rounded(footprint),
            }
            for item, footprint in footprints.items()
            if footprint is not None
        },
        "series": series_records,
    }


def town_bounds(footprints: list[MultiPolygon]) -> list[float]:
    """[west, south, east, north] around every footprint of a town."""
    west, south, east, north = unary_union(footprints).bounds
    return [round(west, 4), round(south, 4), round(east, 4), round(north, 4)]


def read_town_editions(volumes_dir: Path) -> dict[str, list[Edition]]:
    """Place id -> its catalogued volumes, from build_places.py's per-state volume files."""
    towns: dict[str, list[Edition]] = {}
    for path in volumes_dir.glob("*.json"):
        for place, volumes in json.loads(path.read_text()).items():
            towns[place] = [
                Edition(v["item"], v.get("year"), v.get("volume", "")) for v in volumes
            ]
    return towns


def run_annotations(run_dir: Path) -> list[Path]:
    """A published run's main annotations, one per item (publish_run.py's iiif/loc/)."""
    return sorted((run_dir / "iiif" / "loc").glob("*.main.iiif.json"))


def explanation(verdict: PageVerdict) -> str:
    """One line on why a page counts toward its volume's shape, or not."""
    reasons = [
        reason
        for reason, applies in (
            ("disconnected", verdict.disconnected),
            (f"claimed by {verdict.claims} other-volume edition(s)", verdict.claims),
            ("in no other edition of its volume", verdict.uncorroborated),
        )
        if applies
    ]
    confidence = verdict.page.confidence
    return (
        f"  {verdict.item} {verdict.page.key:10} "
        f"P(good) {'?' if confidence is None else f'{confidence:.2f}'} "
        f"bar {verdict.bar:.2f} {'kept' if verdict.kept else 'LEFT OUT'}"
        + (f" ({', '.join(reasons)})" if reasons else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="A published run, as publish_run.py writes it (holding iiif/loc/)",
    )
    parser.add_argument(
        "--run-url",
        help="Where the run is published (default: https://data.mapsnap.org/runs/<run-dir name>)",
    )
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument(
        "--limit", type=int, help="Only the first N items, for trying things out"
    )
    parser.add_argument(
        "--explain",
        nargs="*",
        default=[],
        help="Print why each page of these items counts toward its volume's shape, or not",
    )
    args = parser.parse_args()
    version = args.run_dir.name
    run_url = args.run_url or f"{DATA_URL}/runs/{version}"

    towns = read_town_editions(args.atlas_dir / "volumes")
    edition_of = {e.item: e for editions in towns.values() for e in editions}
    paths = run_annotations(args.run_dir)[: args.limit]
    counts = {
        "volumes": 0,
        "no place": 0,
        "no footprint": 0,
        "pages kept": 0,
        "pages dropped": 0,
    }
    for n, path in enumerate(paths, start=1):
        item = path.name.split(".")[0]
        edition = edition_of.get(item)
        if edition is None:
            counts["no place"] += 1
            continue
        items = json.loads(path.read_text()).get("items", [])
        outlines = [page for page in map(page_outline, items) if page is not None]
        edition.pages = plausible_pages(outlines)
        counts["pages kept"] += len(edition.pages)
        counts["pages dropped"] += len(items) - len(edition.pages)
        edition.footprint = volume_footprint([p.outline for p in edition.pages])
        if edition.footprint is None:
            counts["no footprint"] += 1
            continue
        counts["volumes"] += 1
        if n % 5000 == 0:
            print(f"  {n:,}/{len(paths):,}", file=sys.stderr)

    out_dir = args.atlas_dir / "footprints"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    bounds: dict[str, list[float]] = {}
    shapeless: list[str] = []
    series_counts = {"series": 0, "on the map": 0, "overviews": 0}
    trust = {"pages in shapes": 0, "pages left out": 0, "series without a shape": 0}
    explain = set(args.explain)
    for place, editions in sorted(towns.items()):
        placed = [e.footprint for e in editions if e.footprint is not None]
        if not placed:
            continue
        verdicts = judge_pages(editions, group_series(editions))
        record = town_record(editions, verdicts)
        for verdict in verdicts:
            trust["pages in shapes" if verdict.kept else "pages left out"] += 1
            if verdict.item in explain:
                print(explanation(verdict), file=sys.stderr)
        trust["series without a shape"] += sum(
            1
            for entry in record["series"]
            if entry["footprint"] is None
            and any(item in record["volumes"] for item in entry["items"])
        )
        if not any(entry["display"] for entry in record["series"]):
            shapeless.append(place)
        series_counts["series"] += len(record["series"])
        series_counts["on the map"] += sum(1 for s in record["series"] if s["display"])
        series_counts["overviews"] += sum(
            1 for s in record["series"] if s.get("overview")
        )
        for entry in record["series"]:
            if entry.get("overview"):
                print(f"  overview: {place} {entry['items']}", file=sys.stderr)
        path = out_dir / f"{place}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, separators=(",", ":")))
        bounds[place] = town_bounds(placed)
    index = {
        "run": {"version": version, "url": run_url},
        "towns": bounds,
        "shapeless": shapeless,
    }
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    size_mb = sum(path.stat().st_size for path in out_dir.rglob("*.json")) / 1e6
    print(counts, series_counts, trust, file=sys.stderr)
    print(
        f"{len(shapeless):,} towns with nothing on the map but a dot", file=sys.stderr
    )
    print(
        f"{counts['volumes']:,} footprints in {len(bounds):,} towns: {size_mb:.1f} MB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
