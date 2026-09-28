#!/usr/bin/env python
"""Build each digitized volume's footprint for the atlas, from a run's annotations.

    uv run python scripts/atlas/build_footprints.py \\
        --iiif-dir ~/Documents/mapsnap/corpus-run/iiif

The atlas makes the volume its unit: zoomed in, the map shows where each
volume's sheets lie, and a click picks the newest volume covering that spot.
That needs, for every volume, the ground its sheets cover -- the union of its
placed pages -- which is what this writes, one file per state beside the
volume lists build_places.py writes:

    app/public/atlas/footprints/<state>/<town>.json   (one town, e.g. illinois/chicago)
      {"<item>": {"year": 1914,
                  "anchor": [lon, lat],          # a point inside the footprint
                  "footprint": <MultiPolygon coordinates>,
                  "display": true | <MultiPolygon coordinates> | null}}
    app/public/atlas/footprints/index.json   {"<state>/<town>": [west, south, east, north]}

A file per town rather than per state because the map, zoomed in far enough
to show footprints, has only a few towns in view: Wernersville's file is a few
kilobytes, Chicago's (97 volumes) about a megabyte, where all of New York
state is six. The index says which towns a viewport needs.

``display`` is the part of the footprint no later volume of the same town
covers (``true`` when that is all of it), which is what the map draws: a town's newest coverage, rather than a
stack of editions over its downtown. It is null for a volume later editions
cover entirely -- it stays reachable through the year buttons.

A run places a few pages badly, and one wild page would stretch a footprint
across a county, so pages are dropped before the union when their sheet is
larger than any sheet the truth set holds (3.12 km corner to corner), or when
they sit far from every other page of their volume.

``--iiif-dir`` holds one ``<item>.iiif.json`` per item: a run's published
annotations, downloaded from the mirror.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

DEFAULT_ATLAS_DIR = Path(__file__).resolve().parents[2] / "app/public/atlas"
EARTH_RADIUS_M = 6371008.8
METRES_PER_DEGREE = math.pi * EARTH_RADIUS_M / 180
# The largest sheet in the truth set (Hudson Co. 1950 vol 9 p92, an inset).
MAX_SHEET_DIAGONAL_M = 3120.0
# Seams between neighbouring sheets are closed by growing the union this much
# and shrinking it back.
SEAM_M = 50.0
SIMPLIFY_M = 10.0
# A later volume that leaves less than this share of an earlier one uncovered
# hides it from the map.
MIN_VISIBLE_SHARE = 0.05
# Later coverage grows by this before it is subtracted, and what is left is
# opened by it, so misaligned editions leave no fringe of slivers.
MARGIN_M = 60.0
# A displayed piece smaller than this (2 ha, about a city block) is dropped.
MIN_PART_M2 = 20_000.0
# A displayed part this close to the whole footprint is drawn as the whole.
WHOLE_SHARE = 0.99


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


def page_outline(item: dict) -> tuple[Polygon, float] | None:
    """A placed page's clip outline on the ground, and its sheet's diagonal in metres."""
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
    return polygon, diagonal


def plausible_pages(outlines: list[tuple[Polygon, float]]) -> list[Polygon]:
    """The outlines worth a footprint: sheets of a drawable size, near the rest of the volume.

    A page is isolated when its centre is farther than 2 km, or three of the
    volume's median sheets, from every other page's. That only means something
    with three or more pages, so smaller volumes keep every sized page.
    """
    sized = [
        (polygon, diagonal)
        for polygon, diagonal in outlines
        if diagonal <= MAX_SHEET_DIAGONAL_M
    ]
    if len(sized) < 3:
        return [polygon for polygon, _ in sized]
    reach = max(2000.0, 3 * float(np.median([diagonal for _, diagonal in sized])))
    centres = [(polygon.centroid.x, polygon.centroid.y) for polygon, _ in sized]
    kept = []
    for i, (polygon, _) in enumerate(sized):
        nearest = min(
            ground_distance_m(centres[i], centres[j])
            for j in range(len(sized))
            if j != i
        )
        if nearest <= reach:
            kept.append(polygon)
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


def newest_coverage(
    footprints: dict[str, tuple[int | None, MultiPolygon]],
) -> dict[str, MultiPolygon | None]:
    """For one town's volumes, the part of each footprint no later volume covers.

    Volumes are taken newest first (undated last). Two editions of a district
    never line up exactly, so a bare difference leaves a fringe of slivers
    wherever an older footprint pokes past a newer one. Later coverage is grown
    by MARGIN_M before it is subtracted, what is left is opened by the same
    amount (dropping anything narrower than twice that), and pieces under
    MIN_PART_M2 are dropped. A volume left with under MIN_VISIBLE_SHARE of its
    footprint, or nothing, gets None.
    """
    latitude = unary_union([f for _, f in footprints.values()]).centroid.y
    metric = {item: in_metres(f, latitude) for item, (_, f) in footprints.items()}
    order = sorted(footprints, key=lambda item: (-(footprints[item][0] or 0), item))
    margin = MARGIN_M / METRES_PER_DEGREE
    min_part = MIN_PART_M2 / METRES_PER_DEGREE**2
    covered = None
    display: dict[str, MultiPolygon | None] = {}
    for item in order:
        footprint = metric[item]
        remainder = (
            footprint
            if covered is None
            else overlay(
                footprint, covered.buffer(margin, join_style="mitre"), "difference"
            )
        )
        opened = overlay(
            remainder.buffer(-margin, join_style="mitre").buffer(
                margin, join_style="mitre"
            ),
            footprint,
            "intersection",
        ).simplify(SIMPLIFY_M / METRES_PER_DEGREE, preserve_topology=True)
        parts = as_multipolygon(opened)
        kept = (
            MultiPolygon([part for part in parts.geoms if part.area >= min_part])
            if parts is not None
            else None
        )
        if (
            kept is None
            or kept.is_empty
            or kept.area < MIN_VISIBLE_SHARE * footprint.area
        ):
            display[item] = None
        else:
            # Nearly all of it is the whole of it: the file then says `true`
            # rather than repeating a shape the opening only nicked.
            whole = kept.area >= WHOLE_SHARE * footprint.area
            display[item] = (
                footprints[item][1]
                if whole
                else as_multipolygon(make_valid(in_degrees(kept, latitude)))
            )
        covered = footprint if covered is None else overlay(covered, footprint, "union")
    return display


def rounded(geometry: MultiPolygon | None) -> list | None:
    """GeoJSON MultiPolygon coordinates at 5 decimals (about a metre)."""
    if geometry is None:
        return None
    return json.loads(
        json.dumps(mapping(geometry)["coordinates"]),
        parse_float=lambda text: round(float(text), 5),
    )


def town_records(
    footprints: dict[str, tuple[int | None, MultiPolygon]],
) -> dict[str, dict]:
    """One town's footprint file: each volume's year, anchor, footprint and map shape.

    ``display`` is ``true`` where the map shows the whole footprint -- the
    newest volume of every town, and most others -- rather than repeating it.
    """
    display = newest_coverage(footprints)
    records = {}
    for item, (year, footprint) in footprints.items():
        anchor = footprint.representative_point()
        shown = display[item]
        records[item] = {
            "year": year,
            "anchor": [round(anchor.x, 5), round(anchor.y, 5)],
            "footprint": rounded(footprint),
            "display": True
            if shown is not None and shown.equals(footprint)
            else rounded(shown),
        }
    return records


def town_bounds(footprints: dict[str, tuple[int | None, MultiPolygon]]) -> list[float]:
    """[west, south, east, north] around every footprint of a town."""
    west, south, east, north = unary_union([f for _, f in footprints.values()]).bounds
    return [round(west, 4), round(south, 4), round(east, 4), round(north, 4)]


def read_volume_places(volumes_dir: Path) -> dict[str, tuple[str, int | None]]:
    """item -> (place id, year), from build_places.py's per-state volume files."""
    places: dict[str, tuple[str, int | None]] = {}
    for path in volumes_dir.glob("*.json"):
        for place, volumes in json.loads(path.read_text()).items():
            for volume in volumes:
                places[volume["item"]] = (place, volume.get("year"))
    return places


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--iiif-dir", type=Path, required=True, help="One <item>.iiif.json per item"
    )
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument(
        "--limit", type=int, help="Only the first N items, for trying things out"
    )
    args = parser.parse_args()

    places = read_volume_places(args.atlas_dir / "volumes")
    paths = sorted(args.iiif_dir.glob("*.iiif.json"))[: args.limit]
    by_town: dict[str, dict[str, tuple[int | None, MultiPolygon]]] = defaultdict(dict)
    counts = {
        "volumes": 0,
        "no place": 0,
        "no footprint": 0,
        "pages kept": 0,
        "pages dropped": 0,
    }
    for n, path in enumerate(paths, start=1):
        item = path.name.split(".")[0]
        if item not in places:
            counts["no place"] += 1
            continue
        items = json.loads(path.read_text()).get("items", [])
        outlines = [
            outline for outline in map(page_outline, items) if outline is not None
        ]
        pages = plausible_pages(outlines)
        counts["pages kept"] += len(pages)
        counts["pages dropped"] += len(items) - len(pages)
        footprint = volume_footprint(pages)
        if footprint is None:
            counts["no footprint"] += 1
            continue
        place, year = places[item]
        by_town[place][item] = (year, footprint)
        counts["volumes"] += 1
        if n % 5000 == 0:
            print(f"  {n:,}/{len(paths):,}", file=sys.stderr)

    out_dir = args.atlas_dir / "footprints"
    index: dict[str, list[float]] = {}
    shown = 0
    for place, footprints in by_town.items():
        records = town_records(footprints)
        shown += sum(1 for record in records.values() if record["display"])
        path = out_dir / f"{place}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, separators=(",", ":")))
        index[place] = town_bounds(footprints)
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    size_mb = sum(path.stat().st_size for path in out_dir.rglob("*.json")) / 1e6
    print(counts, file=sys.stderr)
    print(
        f"{counts['volumes']:,} footprints ({shown:,} on the map) in "
        f"{len(index):,} towns: {size_mb:.1f} MB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
