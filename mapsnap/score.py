"""Success score for georeferenced volumes: every sheet counts once.

The project's success metric: **the weighted share of truth pages georeferenced
to within a good-fit threshold (default 25ft RMSE), minus the share placed
disastrously (default >=200ft)** — placing a page 2000ft off is worse than not
placing it at all, so a disaster subtracts what a success adds. Unplaced truth
pages stay in the denominator and earn nothing.

A page's weight is::

    sheet_portion x land_fraction

**sheet_portion** normalizes each SHEET to weight 1, split across its panels by
the paper area each occupies (so a four-way split still totals one sheet, and a
95/5 split weighs 0.95/0.05). Portions come from the OIM region boundaries in
``oim/pN.panels.json`` — the same source ``mapsnap compare`` matches splits
with — not from the truth SvgSelector, which is a clip MASK and so conflates
masking with splitting.

**land_fraction** is the share of the page's footprint within STREET_NEAR_M of
an OSM street centerline. Streets only exist on usable land, so this needs no
separate water dataset and also discounts rail yards and other unmapped ground.
A sheet that is mostly harbour is worth proportionally less than a dense
downtown sheet.

This replaced weighting by absolute land AREA, which let a physically enormous
low-density sheet outweigh a compact dense one. Hudson scored 50.5% with 76% of
its pages placed, because its unplaced sheets carried 1.45x the land of its
typical placed sheet: 24% of the sheets consumed 43% of the weight. Normalizing
per sheet scores it 72.9% — much closer to what the volume actually looks like —
while keeping the water discount that the land fraction was already providing
(brooklyn moves only -1.5 with the fraction retained, against -9.3 without it).

Across volumes the aggregate is a SIMPLE MEAN of per-volume scores: a volume is
the unit of work, and pooling pages let the biggest volumes set the corpus
number.

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
    load_split_polygons,
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
    sheet_portion: float = 1.0  # share of its SHEET's weight (1.0 if unsplit)

    @property
    def land_fraction(self) -> float:
        """Share of this page's footprint that is usable land."""
        return self.land_m2 / self.area_m2 if self.area_m2 else 0.0

    @property
    def weight(self) -> float:
        """This page's weight in the score: its slice of a sheet, less its water."""
        return self.sheet_portion * self.land_fraction


@dataclass
class ScoreSummary:
    """Weighted totals for a set of pages (one volume)."""

    n_pages: int
    n_placed: int
    weight: float
    good_weight: float  # rmse <= good threshold
    disaster_weight: float  # placed at rmse >= disaster threshold

    @property
    def good_share(self) -> float:
        return self.good_weight / self.weight if self.weight else 0.0

    @property
    def disaster_share(self) -> float:
        return self.disaster_weight / self.weight if self.weight else 0.0

    @property
    def net_score(self) -> float:
        """The success metric: good share minus disaster share."""
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


def sheet_of(page_key: str) -> str:
    """The sheet a page key belongs to ('p59__2' -> 'p59')."""
    return page_key.split("__")[0]


def split_index(page_key: str) -> int | None:
    """1-based panel index in a split page key, or None when the page is unsplit."""
    if "__" not in page_key:
        return None
    try:
        return int(page_key.split("__")[1])
    except ValueError:
        return None


def sheet_portions(oim_dir: Path, page_keys: list[str]) -> dict[str, float]:
    """Each page key's share of its own sheet's weight, summing to 1 per sheet.

    Split portions are the PAPER area of each OIM region boundary
    (``oim/pN.panels.json``), normalized within the sheet. Paper, not ground:
    an inset is small on the sheet however much ground it covers, and that is
    what "a fraction of a page" means.

    Panels are normalized to sum to 1 rather than divided by the raw sheet
    rectangle, so that a sheet whose panels do not quite tile it still carries
    the same total weight as an unsplit one. Measured across the 18 truth
    volumes, the OIM panels tile their sheet almost exactly (median coverage
    1.000), so the normalization is a safeguard rather than a correction.
    """
    by_sheet: dict[str, list[str]] = {}
    for key in page_keys:
        by_sheet.setdefault(sheet_of(key), []).append(key)

    portions: dict[str, float] = {}
    for sheet, keys in by_sheet.items():
        if len(keys) == 1 and split_index(keys[0]) is None:
            portions[keys[0]] = 1.0
            continue
        polygons = load_split_polygons(oim_dir / f"{sheet}.panels.json")
        areas = {}
        for key in keys:
            index = split_index(key)
            polygon = polygons.get(index) if index is not None else None
            areas[key] = polygon.area if polygon is not None else 0.0
        total = sum(areas.values())
        if total <= 0:
            # No usable boundaries: split the sheet evenly rather than drop it.
            print(
                f"  {sheet}: no panel boundaries; weighting its "
                f"{len(keys)} panels equally",
                file=sys.stderr,
            )
        for key in keys:
            portions[key] = areas[key] / total if total > 0 else 1.0 / len(keys)
    return portions


def volume_page_scores(
    generated_iiif: Path,
    *,
    street_near_m: float = STREET_NEAR_M,
    truth: Path | None = None,
) -> list[PageScore]:
    """Per-truth-page scores for one volume's generated annotation file.

    Matching and RMSE come from ``compare_pages`` (identical to ``mapsnap
    compare``, including the skeleton rule); footprints and land fractions are
    computed here from the truth items and the volume's centerlines.

    ``truth`` overrides the volume's own main.iiif.json, so a revised truth set
    can be scored against the same generated pages as the one it replaces. Note
    that the truth defines the denominator as well as the per-page RMSE, so two
    truth sets are only comparable when they cover the same sheets.
    """
    volume = generated_iiif.parent
    truth_path = truth if truth is not None else volume / "main.iiif.json"
    if not truth_path.exists():
        sys.exit(f"No truth data at {truth_path}.")
    centerlines = default_centerlines(volume)
    if centerlines is None:
        sys.exit(f"{volume} has no centerlines.geojson (needed for land weights).")

    # The panels live with the volume, not with whichever truth file is in use.
    rows, missing = compare_pages(truth_path, generated_iiif, oim_dir=volume / "oim")
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

    portions = sheet_portions(volume / "oim", sorted(rings))
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
                sheet_portion=portions.get(key, 1.0),
            )
        )
    return scores


def summarize(
    pages: list[PageScore],
    *,
    good_ft: float = GOOD_FT,
    disaster_ft: float = DISASTER_FT,
) -> ScoreSummary:
    """Weighted totals over a set of page scores (one volume)."""
    return ScoreSummary(
        n_pages=len(pages),
        n_placed=sum(1 for p in pages if p.rmse_ft is not None),
        weight=sum(p.weight for p in pages),
        good_weight=sum(
            p.weight for p in pages if p.rmse_ft is not None and p.rmse_ft <= good_ft
        ),
        disaster_weight=sum(
            p.weight
            for p in pages
            if p.rmse_ft is not None and p.rmse_ft >= disaster_ft
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Success score: weighted share of truth pages georeferenced to "
            "<=GOOD ft RMSE, minus the share placed at >=DISASTER ft. Each "
            "sheet weighs 1 (split across its panels by paper area), less its "
            "water. Several volumes print a simple mean."
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
        "--truth",
        metavar="IIIF",
        type=Path,
        help=(
            "Truth AnnotationPage to score against, instead of the "
            "main.iiif.json in the generated file's own directory. Truth is "
            "per-volume, so this may only be used with generated file(s) from "
            "a single volume — handy for measuring what a truth revision moved."
        ),
    )
    parser.add_argument(
        "--csv", metavar="FILE", help="Also write per-page scores to a CSV file"
    )
    args = parser.parse_args()

    if args.truth is not None:
        volumes = {Path(p).parent for p in args.generated}
        if len(volumes) > 1:
            sys.exit(
                "--truth applies to one volume, but the generated files span "
                + ", ".join(sorted(str(v) for v in volumes))
            )

    header = (
        f"{'volume':<30} {'pages':>5} {'placed':>6} "
        f"{'<=' + format(args.good_ft, 'g') + 'ft':>8} "
        f"{'>=' + format(args.disaster_ft, 'g') + 'ft':>8} {'score':>6}"
    )
    print(header)
    print("-" * len(header))
    all_pages: list[tuple[str, PageScore]] = []
    summaries: list[ScoreSummary] = []
    for path in args.generated:
        generated = Path(path)
        # A truth set with split panels REQUIRES the volume's oim/ panels dir:
        # without it every split panel goes unmatched and counts unplaced,
        # silently cratering split-heavy volumes (champaign scored 70.0 instead
        # of 97.1 when the 2026-08-14 archives were scored from a directory
        # with no oim/ -- the #330 scare). Fail loudly instead.
        volume_dir = generated.parent
        truth_path = Path(args.truth) if args.truth else volume_dir / "main.iiif.json"
        if truth_path.exists() and not (volume_dir / "oim").is_dir():
            truth_doc = json.loads(truth_path.read_text())
            has_splits = any(
                "__" in str(item.get("label") or "")
                or "[" in str(item.get("label") or "")
                for item in truth_doc.get("items", [])
            )
            if has_splits:
                sys.exit(
                    f"{volume_dir} has no oim/ panels directory, but its truth "
                    f"contains split panels -- scoring here would silently count "
                    f"every split as unplaced. Score the volume-root IIIF (which "
                    f"sits beside oim/), not an archive copy."
                )
        pages = volume_page_scores(
            generated, street_near_m=args.street_near_m, truth=args.truth
        )
        all_pages.extend((generated.parent.name, p) for p in pages)
        s = summarize(pages, good_ft=args.good_ft, disaster_ft=args.disaster_ft)
        summaries.append(s)
        print(
            f"{generated.parent.name:<30} {s.n_pages:>5} {s.n_placed:>6} "
            f"{s.good_share:>7.1%} {s.disaster_share:>7.1%} {s.net_score:>6.1%}"
        )
    if len(summaries) > 1:
        # A SIMPLE MEAN, not a pooled total: a volume is the unit of work, and
        # pooling let the largest volumes decide the corpus number.
        n = len(summaries)
        print("-" * len(header))
        print(
            f"{'MEAN OF ' + str(n) + ' VOLUMES':<30} "
            f"{sum(s.n_pages for s in summaries):>5} "
            f"{sum(s.n_placed for s in summaries):>6} "
            f"{sum(s.good_share for s in summaries) / n:>7.1%} "
            f"{sum(s.disaster_share for s in summaries) / n:>7.1%} "
            f"{sum(s.net_score for s in summaries) / n:>6.1%}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "volume",
                    "page_key",
                    "area_m2",
                    "land_m2",
                    "sheet_portion",
                    "weight",
                    "rmse_ft",
                ]
            )
            for volume_name, p in all_pages:
                writer.writerow(
                    [
                        volume_name,
                        p.page_key,
                        round(p.area_m2, 1),
                        round(p.land_m2, 1),
                        round(p.sheet_portion, 4),
                        round(p.weight, 4),
                        "" if p.rmse_ft is None else round(p.rmse_ft, 1),
                    ]
                )
        print(f"\nCSV written to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
