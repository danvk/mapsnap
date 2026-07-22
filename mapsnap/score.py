"""Land-weighted success score for georeferenced volumes.

The project's success metric: **the share of truth land area on pages
georeferenced to within a good-fit threshold (default 25ft RMSE), minus the
share placed disastrously (default >=200ft)** — placing a page 2000ft off is
worse than not placing it at all, so a disaster subtracts what a success adds.
Unplaced truth pages stay in the denominator and earn nothing.

Weighting by *land* area (rather than counting pages) downweights small split
panels and waterfront sheets that are mostly water. Land is approximated by
street proximity: the fraction of a page's footprint within STREET_NEAR_M of an
OSM street centerline. Streets only exist on usable land, so this needs no
separate water dataset and also discounts rail yards and other unmapped ground.

Each GENERATED_IIIF argument is scored against the ``main.iiif.json`` truth and
``centerlines.geojson`` in its own directory; pages are matched and RMSE
computed exactly as ``mapsnap compare`` does. With several volumes an aggregate
row is printed — the project scoreboard.

    uv run mapsnap score data/*/2026-07-19-familyscale.iiif.json
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    compare_pages,
    extract_gcps,
    fit_transform,
    truth_polygon_world,
)
from mapsnap.utils import default_centerlines, source_id_to_page_key

GOOD_FT = 25.0
DISASTER_FT = 200.0
STREET_NEAR_M = 120.0
GRID_N = 15  # land sampling grid per footprint (GRID_N x GRID_N candidate points)

METERS_PER_DEGREE_LAT = 110_540.0
METERS_PER_DEGREE_LON_EQUATOR = 111_320.0


@dataclass
class LocalFrame:
    """Equirectangular lon/lat -> local metres about a reference point."""

    lon0: float
    lat0: float

    def to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        kx = METERS_PER_DEGREE_LON_EQUATOR * math.cos(math.radians(self.lat0))
        return ((lon - self.lon0) * kx, (lat - self.lat0) * METERS_PER_DEGREE_LAT)


@dataclass
class PageScore:
    """One truth page's contribution to the score."""

    page_key: str
    area_m2: float
    land_m2: float
    rmse_ft: float | None  # None = truth page with no generated fit


@dataclass
class ScoreSummary:
    """Land-weighted totals for a set of pages (one volume or the aggregate)."""

    n_pages: int
    n_placed: int
    land_m2: float
    good_m2: float  # rmse <= good threshold
    disaster_m2: float  # placed at rmse >= disaster threshold

    @property
    def good_share(self) -> float:
        return self.good_m2 / self.land_m2 if self.land_m2 else 0.0

    @property
    def disaster_share(self) -> float:
        return self.disaster_m2 / self.land_m2 if self.land_m2 else 0.0

    @property
    def net_score(self) -> float:
        """The success metric: good land share minus disaster land share."""
        return self.good_share - self.disaster_share


def truth_footprint_ring(item: dict) -> list[list[float]] | None:
    """World [lon, lat] ring of a truth item's footprint, or None.

    Prefers the clip-selector polygon (via ``truth_polygon_world``); truth
    annotations without a selector (e.g. Grand Rapids) fall back to the full
    source rectangle mapped through the item's own GCP transform.
    """
    ring = truth_polygon_world(item)
    if ring:
        return ring
    gcps = extract_gcps(item)
    source = item.get("target", {}).get("source", {})
    width, height = source.get("width"), source.get("height")
    if len(gcps) < 2 or not width or not height:
        return None
    transform = fit_transform(gcps, annotation_transform_type(item))
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    return [list(transform @ np.array([x, y, 1.0])) for x, y in corners]


def street_tree(centerlines_path: Path, frame: LocalFrame) -> STRtree:
    """STRtree of the volume's street centerlines in local metres."""
    lines = []
    for feature in json.loads(centerlines_path.read_text())["features"]:
        geometry = feature.get("geometry", {})
        kind = geometry.get("type")
        parts = (
            [geometry["coordinates"]]
            if kind == "LineString"
            else geometry.get("coordinates", [])
            if kind == "MultiLineString"
            else []
        )
        for part in parts:
            if len(part) < 2:
                continue
            lines.append(
                shapely.LineString([frame.to_xy(lon, lat) for lon, lat in part])
            )
    return STRtree(lines)


def land_fraction(
    footprint: Polygon, streets: STRtree, *, near_m: float = STREET_NEAR_M
) -> float:
    """Fraction of a footprint that lies near a street ("on usable land").

    Samples a GRID_N x GRID_N grid over the footprint's bounds, keeps the points
    inside the polygon, and counts the share within ``near_m`` of any street.
    Returns 0.0 when the tree is empty or no grid point lands inside.
    """
    if len(streets.geometries) == 0:
        return 0.0
    min_x, min_y, max_x, max_y = footprint.bounds
    xs = np.linspace(min_x, max_x, GRID_N)
    ys = np.linspace(min_y, max_y, GRID_N)
    grid = shapely.points([[x, y] for x in xs for y in ys])
    inside = grid[shapely.contains(footprint, grid)]
    if len(inside) == 0:
        return 0.0
    hits, _ = streets.query(inside, predicate="dwithin", distance=near_m)
    return float(len(np.unique(hits)) / len(inside))


def volume_page_scores(
    generated_iiif: Path, *, street_near_m: float = STREET_NEAR_M
) -> list[PageScore]:
    """Per-truth-page scores for one volume's generated annotation file.

    Matching and RMSE come from ``compare_pages`` (identical to ``mapsnap
    compare``, including the skeleton rule); footprints and land fractions are
    computed here from the truth items and the volume's centerlines.
    """
    volume = generated_iiif.parent
    truth_path = volume / "main.iiif.json"
    if not truth_path.exists():
        sys.exit(f"{volume} has no main.iiif.json truth data.")
    centerlines = default_centerlines(volume)
    if centerlines is None:
        sys.exit(f"{volume} has no centerlines.geojson (needed for land weights).")

    rows, missing = compare_pages(truth_path, generated_iiif)
    rmse_by_key: dict[str, float | None] = {}
    for row in missing:
        rmse_by_key.setdefault(row["page_key"], None)
    for row in rows:
        if row["page_key"] in rmse_by_key and rmse_by_key[row["page_key"]] is not None:
            print(
                f"  duplicate page key {row['page_key']}; keeping first",
                file=sys.stderr,
            )
            continue
        rmse_by_key[row["page_key"]] = row["rmse_ft"]

    rings: dict[str, list[list[float]]] = {}
    for item in json.loads(truth_path.read_text()).get("items", []):
        key = source_id_to_page_key(
            item["target"]["source"].get("id"), item.get("label") or ""
        )
        if key not in rmse_by_key or key in rings:
            continue
        ring = truth_footprint_ring(item)
        if ring:
            rings[key] = ring

    dropped = set(rmse_by_key) - set(rings)
    if dropped:
        print(
            f"  {len(dropped)} truth page(s) lack a usable footprint; "
            "excluded: " + ", ".join(sorted(dropped)),
            file=sys.stderr,
        )

    points = [p for ring in rings.values() for p in ring]
    frame = LocalFrame(
        lon0=sum(p[0] for p in points) / len(points),
        lat0=sum(p[1] for p in points) / len(points),
    )
    streets = street_tree(centerlines, frame)

    scores = []
    for key, ring in sorted(rings.items()):
        footprint = Polygon([frame.to_xy(lon, lat) for lon, lat in ring]).buffer(0)
        if footprint.is_empty:
            continue
        fraction = land_fraction(footprint, streets, near_m=street_near_m)
        scores.append(
            PageScore(
                page_key=key,
                area_m2=footprint.area,
                land_m2=footprint.area * fraction,
                rmse_ft=rmse_by_key[key],
            )
        )
    return scores


def summarize(
    pages: list[PageScore],
    *,
    good_ft: float = GOOD_FT,
    disaster_ft: float = DISASTER_FT,
) -> ScoreSummary:
    """Land-weighted totals over a set of page scores."""
    return ScoreSummary(
        n_pages=len(pages),
        n_placed=sum(1 for p in pages if p.rmse_ft is not None),
        land_m2=sum(p.land_m2 for p in pages),
        good_m2=sum(
            p.land_m2 for p in pages if p.rmse_ft is not None and p.rmse_ft <= good_ft
        ),
        disaster_m2=sum(
            p.land_m2
            for p in pages
            if p.rmse_ft is not None and p.rmse_ft >= disaster_ft
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Land-weighted success score: share of truth land area georeferenced "
            "to <=GOOD ft RMSE, minus the share placed at >=DISASTER ft."
        )
    )
    parser.add_argument(
        "generated",
        nargs="+",
        metavar="GEN_IIIF",
        help=(
            "Generated IIIF AnnotationPage file(s); each is scored against the "
            "main.iiif.json and centerlines.geojson in its own directory."
        ),
    )
    parser.add_argument(
        "--good-ft",
        type=float,
        default=GOOD_FT,
        help="RMSE at or below this earns full credit (default: %(default)s)",
    )
    parser.add_argument(
        "--disaster-ft",
        type=float,
        default=DISASTER_FT,
        help="Placed pages at or above this subtract credit (default: %(default)s)",
    )
    parser.add_argument(
        "--street-near-m",
        type=float,
        default=STREET_NEAR_M,
        help="Footprint counts as land within this range of a street (default: %(default)s)",
    )
    parser.add_argument(
        "--csv", metavar="FILE", help="Also write per-page scores to a CSV file"
    )
    args = parser.parse_args()

    header = (
        f"{'volume':<30} {'pages':>5} {'placed':>6} "
        f"{'<=' + format(args.good_ft, 'g') + 'ft':>8} "
        f"{'>=' + format(args.disaster_ft, 'g') + 'ft':>8} {'score':>6}"
    )
    print(header)
    print("-" * len(header))
    all_pages: list[tuple[str, PageScore]] = []
    for path in args.generated:
        generated = Path(path)
        pages = volume_page_scores(generated, street_near_m=args.street_near_m)
        all_pages.extend((generated.parent.name, p) for p in pages)
        s = summarize(pages, good_ft=args.good_ft, disaster_ft=args.disaster_ft)
        print(
            f"{generated.parent.name:<30} {s.n_pages:>5} {s.n_placed:>6} "
            f"{s.good_share:>7.1%} {s.disaster_share:>7.1%} {s.net_score:>6.1%}"
        )
    if len(args.generated) > 1:
        s = summarize(
            [p for _, p in all_pages],
            good_ft=args.good_ft,
            disaster_ft=args.disaster_ft,
        )
        print("-" * len(header))
        print(
            f"{'AGGREGATE':<30} {s.n_pages:>5} {s.n_placed:>6} "
            f"{s.good_share:>7.1%} {s.disaster_share:>7.1%} {s.net_score:>6.1%}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["volume", "page_key", "area_m2", "land_m2", "rmse_ft"])
            for volume_name, p in all_pages:
                writer.writerow(
                    [
                        volume_name,
                        p.page_key,
                        round(p.area_m2, 1),
                        round(p.land_m2, 1),
                        "" if p.rmse_ft is None else round(p.rmse_ft, 1),
                    ]
                )
        print(f"\nCSV written to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
