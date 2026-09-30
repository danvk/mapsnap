"""Score an annotation's clip masks: agreement with OIM, and truth-free defects (#548).

    mapsnap mask-score data/<volume>/runs/<tag>/mapsnap.iiif.json \\
        --truth data/<volume>/main.iiif.json --oim-dir data/<volume>/oim \\
        --centerlines data/<volume>/centerlines.geojson --allmaps

Agreement (needs OIM truth): the IoU of each of our SvgSelectors with OIM's, in
canvas pixels, averaged with one weight per sheet (a split sheet's panels share
it). Only OIM masks that hold at least half of their own GCPs are used, since
OIM's export writes some split selectors in the crop's frame (OIM#402). OIM
volunteers left some volumes unclipped (whole-sheet masks), which rewards not
clipping, so ``iou_clipped`` counts only truth masks under 95% of their sheet or
panel. Scoring pipeline output against OIM mixes in pose and split errors; see
``mapsnap mask-eval`` to score the masker alone.

Defects (no truth needed), on the ground under each annotation's own transform:

- ``overlap``: the sum of mask areas less their union, as a share of the union;
- ``gaps``: holes enclosed by the mosaic (gaps between pages are closed by 4 m
  first), as a share of the union -- including ground no page maps, so compare
  against the same poses without masks;
- ``slivers``: mask area narrower than 4 m, as a share of mask area;
- ``vertices_median`` / ``vertices_max``: selector complexity;
- ``invalid``: self-intersecting or degenerate selectors;
- ``seam_on_street`` (with ``--centerlines``): the share of seam length (mask
  edges inside the mosaic, not its outline) within 10 m of a street centerline.
  Overlap, gaps and complexity cannot tell a seam that follows a street from one
  that cuts through a block. It means little where masks overlap much, since
  an overlapped mask's edge is then not a seam;
- ``allmaps_failures`` (with ``--allmaps``): maps Allmaps fails to triangulate,
  as the viewer draws them (app/scripts/triangulate-masks.ts).
"""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from mapsnap.annotation_transform import page_transform
from mapsnap.osm_to_centerlines import load_centerlines
from mapsnap.utils import source_id_to_page_key

M_PER_DEGREE = 111_195.0
# A truth mask covering at least this share of its sheet (or panel) is unclipped.
UNCLIPPED_SHARE = 0.95
# Gaps between neighbouring masks up to about twice this wide count as enclosed.
GAP_CLOSE_M = 4.0
# Mask area narrower than twice this is a sliver.
SLIVER_HALF_WIDTH_M = 2.0
# A seam within this distance of a street centerline follows the street.
STREET_TOLERANCE_M = 10.0
# Mask edges within this distance of the mosaic's outline are outline, not seams.
OUTLINE_MARGIN_M = 5.0
APP_DIR = Path(__file__).resolve().parent.parent / "app"
POINTS = re.compile(r'points="([^"]*)"')


def selector_points(item: dict) -> list[tuple[float, float]]:
    """The canvas-pixel vertices of an annotation's SvgSelector polygon."""
    selector = (item.get("target") or {}).get("selector") or {}
    match = POINTS.search(selector.get("value", ""))
    if not match:
        return []
    return [
        (float(x), float(y))
        for x, y in (point.split(",") for point in match.group(1).split())
    ]


def image_key(item: dict) -> str:
    """The page key of the image an annotation places ('p12', or 'p12__2' for a panel)."""
    source = item["target"]["source"]
    return source_id_to_page_key(source.get("id"), item.get("label", "")).lower()


def gcp_pixels(item: dict) -> list[tuple[float, float]]:
    """The canvas-pixel positions of an annotation's control points."""
    return [
        tuple(feature["properties"]["resourceCoords"])
        for feature in item["body"]["features"]
        if feature.get("properties", {}).get("resourceCoords")
    ]


def trusted_truth_masks(truth_items: list[dict]) -> dict[str, tuple[dict, Polygon]]:
    """OIM masks by image key, leaving out selectors that miss most of their own GCPs."""
    masks: dict[str, tuple[dict, Polygon]] = {}
    for item in truth_items:
        points = selector_points(item)
        if len(points) < 3:
            continue
        polygon = Polygon(points).buffer(0)
        gcps = gcp_pixels(item)
        inside = sum(polygon.covers(Point(gcp)) for gcp in gcps)
        if gcps and inside < 0.5 * len(gcps):
            continue  # written in the crop's frame (OIM#402)
        masks[image_key(item)] = (item, polygon)
    return masks


def reference_outline(key: str, source: dict, oim_dir: Path | None) -> Polygon:
    """What an image could show, in canvas px: its sheet, or its OIM panel outline."""
    sheet = box(0, 0, source["width"], source["height"])
    if "__" not in key or oim_dir is None:
        return sheet
    base, index = key.split("__")
    panels_path = oim_dir / f"{base}.panels.json"
    if not panels_path.exists():
        return sheet
    panels = json.loads(panels_path.read_text())
    scale_x = source["width"] / panels["width"]
    scale_y = source["height"] / panels["height"]
    ring = panels["panels"][int(index) - 1]
    return Polygon([(x * scale_x, y * scale_y) for x, y in ring]).buffer(0)


@dataclass
class ImageAgreement:
    """How well one image's mask matches OIM's."""

    key: str
    iou: float
    # False when OIM's mask is (nearly) the whole sheet or panel.
    clipped: bool

    @property
    def sheet(self) -> str:
        return self.key.split("__")[0]


def image_agreements(
    ours: list[dict], truth: list[dict], oim_dir: Path | None = None
) -> list[ImageAgreement]:
    """Per-image IoU of our masks against OIM's, in OIM's canvas pixels."""
    truth_masks = trusted_truth_masks(truth)
    rows: list[ImageAgreement] = []
    for item in ours:
        key = image_key(item)
        points = selector_points(item)
        if key not in truth_masks or len(points) < 3:
            continue
        truth_item, truth_mask = truth_masks[key]
        ours_source = item["target"]["source"]
        truth_source = truth_item["target"]["source"]
        scale_x = truth_source["width"] / ours_source["width"]
        scale_y = truth_source["height"] / ours_source["height"]
        mask = Polygon([(x * scale_x, y * scale_y) for x, y in points]).buffer(0)
        union = mask.union(truth_mask).area
        iou = mask.intersection(truth_mask).area / union if union else 0.0
        reference = reference_outline(key, truth_source, oim_dir)
        clipped = truth_mask.area < UNCLIPPED_SHARE * reference.area
        rows.append(ImageAgreement(key, iou, clipped))
    return rows


def sheet_equal_mean(rows: list[ImageAgreement]) -> float | None:
    """Mean IoU with one weight per sheet; a split sheet's panels share its weight."""
    by_sheet: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sheet[row.sheet].append(row.iou)
    if not by_sheet:
        return None
    return statistics.mean(statistics.mean(ious) for ious in by_sheet.values())


def agreement_summary(
    ours: list[dict], truth: list[dict], oim_dir: Path | None = None
) -> dict:
    """Sheet-equal IoU over all matched images and over the ones OIM clipped."""
    rows = image_agreements(ours, truth, oim_dir)
    clipped = [row for row in rows if row.clipped]
    return {
        "iou": sheet_equal_mean(rows),
        "iou_clipped": sheet_equal_mean(clipped),
        "images_scored": len(rows),
        "images_clipped": len(clipped),
        "lowest": [
            (row.key, round(row.iou, 3))
            for row in sorted(rows, key=lambda r: r.iou)[:5]
        ],
    }


@dataclass
class LocalFrame:
    """A local metric frame: metres east and north of an origin, longitude scaled by cos(lat)."""

    lon0: float
    lat0: float

    def to_m(self, lon: float, lat: float) -> tuple[float, float]:
        k = math.cos(math.radians(self.lat0))
        return (
            (lon - self.lon0) * k * M_PER_DEGREE,
            (lat - self.lat0) * M_PER_DEGREE,
        )


def ground_masks(items: list[dict]) -> tuple[list[Polygon], LocalFrame | None]:
    """Each annotation's mask on the ground (metres), under its own transform."""
    frame: LocalFrame | None = None
    masks: list[Polygon] = []
    for item in items:
        transform = page_transform(item)
        points = selector_points(item)
        if transform is None or len(points) < 3:
            continue
        lonlat = [transform(x, y) for x, y in points]
        if frame is None:
            frame = LocalFrame(*lonlat[0])
        mask = Polygon([frame.to_m(lon, lat) for lon, lat in lonlat]).buffer(0)
        if not mask.is_empty:
            masks.append(mask)
    return masks, frame


def polygons_of(geometry: BaseGeometry) -> list[Polygon]:
    """The non-empty polygons in a geometry (a polygon, multipolygon or collection)."""
    parts = getattr(geometry, "geoms", [geometry])
    return [part for part in parts if isinstance(part, Polygon) and not part.is_empty]


def filled_mosaic(union: BaseGeometry) -> BaseGeometry:
    """The mosaic with gaps between pages closed and every hole filled."""
    closed = union.buffer(GAP_CLOSE_M).buffer(-GAP_CLOSE_M)
    return unary_union([Polygon(part.exterior) for part in polygons_of(closed)])


def seam_street_share(
    masks: list[Polygon], union: BaseGeometry, streets: list[LineString]
) -> tuple[float, float | None]:
    """Total seam length (m) and the share of it within STREET_TOLERANCE_M of a street.

    Seams are mask edges away from the mosaic's filled outline, so both sides of a
    gap between pages count, and the volume's outer edge does not.
    """
    outline = filled_mosaic(union).boundary.buffer(OUTLINE_MARGIN_M)
    seams = unary_union([mask.boundary for mask in masks]).difference(outline)
    if seams.length == 0:
        return 0.0, None
    near = unary_union(streets).buffer(STREET_TOLERANCE_M) if streets else Polygon()
    return seams.length, seams.intersection(near).length / seams.length


def streets_in_frame(
    centerlines_path: Path, frame: LocalFrame, extent_m: BaseGeometry
) -> list[LineString]:
    """Street centerlines within extent_m (plus a margin), in the local frame."""
    area = extent_m.buffer(2 * STREET_TOLERANCE_M)
    lines: list[LineString] = []
    for feature in load_centerlines(centerlines_path)["features"]:
        geometry = feature.get("geometry")
        if not geometry or geometry["type"] not in ("LineString", "MultiLineString"):
            continue
        parts = getattr(shape(geometry), "geoms", [shape(geometry)])
        for part in parts:
            line = LineString([frame.to_m(lon, lat) for lon, lat, *_ in part.coords])
            if line.intersects(area):
                lines.append(line)
    return lines


def defect_summary(items: list[dict], centerlines_path: Path | None = None) -> dict:
    """Overlap, gaps, slivers, selector complexity and validity, and seams on streets."""
    masks, frame = ground_masks(items)
    if not masks or frame is None:
        return {"images": len(items)}
    union = unary_union(masks)
    total = sum(mask.area for mask in masks)
    gaps = filled_mosaic(union).difference(union).area
    opened = [
        mask.buffer(-SLIVER_HALF_WIDTH_M).buffer(SLIVER_HALF_WIDTH_M) for mask in masks
    ]
    slivers = sum(mask.difference(o).area for mask, o in zip(masks, opened))
    vertices = sorted(max(len(selector_points(item)) - 1, 0) for item in items)
    invalid = sum(
        1
        for item in items
        if len(selector_points(item)) < 4 or not Polygon(selector_points(item)).is_valid
    )
    summary: dict = {
        "images": len(items),
        "overlap": (total - union.area) / union.area,
        "gaps": gaps / union.area,
        "slivers": slivers / total,
        "vertices_median": statistics.median(vertices),
        "vertices_max": vertices[-1],
        "invalid": invalid,
    }
    if centerlines_path is not None:
        streets = streets_in_frame(centerlines_path, frame, union)
        summary["seam_m"], summary["seam_on_street"] = seam_street_share(
            masks, union, streets
        )
    return summary


def allmaps_failures(annotation_path: Path) -> list[dict]:
    """The maps Allmaps fails to triangulate: [{"index", "message"}], via Node."""
    result = subprocess.run(
        ["node", "scripts/triangulate-masks.ts", str(annotation_path.resolve())],
        cwd=APP_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])["failures"]


def score_annotation(
    annotation_path: Path,
    *,
    truth_path: Path | None = None,
    oim_dir: Path | None = None,
    centerlines_path: Path | None = None,
    allmaps: bool = False,
) -> dict:
    """Every score for one annotation file; agreement only when a truth file is given."""
    items = json.loads(annotation_path.read_text())["items"]
    scores = defect_summary(items, centerlines_path)
    if allmaps:
        scores["allmaps_failures"] = len(allmaps_failures(annotation_path))
    if truth_path is not None:
        truth = json.loads(truth_path.read_text())["items"]
        scores.update(agreement_summary(items, truth, oim_dir))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mapsnap mask-score",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("annotation", type=Path, help="An IIIF AnnotationPage")
    parser.add_argument("--truth", type=Path, help="OIM's main.iiif.json")
    parser.add_argument(
        "--oim-dir",
        type=Path,
        help="The volume's oim/ directory (panel outlines), for split panels",
    )
    parser.add_argument(
        "--centerlines",
        type=Path,
        help="Street centerlines (GeoJSON or OSM), for seams on streets",
    )
    parser.add_argument(
        "--allmaps",
        action="store_true",
        help="Also count the maps Allmaps fails to triangulate (needs Node and app/)",
    )
    args = parser.parse_args()
    if args.truth is not None and args.oim_dir is None:
        args.oim_dir = args.truth.parent / "oim"
    scores = score_annotation(
        args.annotation,
        truth_path=args.truth,
        oim_dir=args.oim_dir,
        centerlines_path=args.centerlines,
        allmaps=args.allmaps,
    )
    json.dump(scores, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
