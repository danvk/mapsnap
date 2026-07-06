"""Compare human-generated IIIF georeferencing (truth) to computer-generated IIIF.

For each page present in both files, fits a 2×3 affine transform from each set of GCPs
and measures how far apart the two transforms are at a 7×7 grid of pixel locations.

Metrics per page:
  - RMSE (feet): RMS Haversine error at 49 grid points
  - Max (feet): worst-case Haversine error at any grid point
  - Trans (feet): error at the image center (translation component)
  - Rot (°): signed rotation difference derived from affine linear parts
  - Scale (%): relative scale difference

Pages only in truth are counted as total losses (no fit produced).

Usage:
    python compare_iiif_georef.py <truth.iiif.json> <generated.iiif.json>
    python compare_iiif_georef.py <truth.iiif.json> <generated.iiif.json> --csv out.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box

from mapsnap.utils import source_id_to_page_key

GCP = tuple[tuple[float, float], tuple[float, float]]  # ((px, py), (lon, lat))
EARTH_RADIUS_FT = 20_925_524.0
MIN_SPLIT_IOU = (
    0.1  # minimum panel overlap to treat a truth and generated split as the same region
)


def extract_metadata_int(item: dict, label: str) -> int | None:
    """Extract an integer value from a IIIF annotation's metadata list by label."""
    for entry in item.get("metadata", []):
        if entry.get("label") == label:
            try:
                return int(entry["value"])
            except (KeyError, ValueError):
                return None
    return None


def extract_metadata_float(item: dict, label: str) -> float | None:
    """Extract a float value from a IIIF annotation's metadata list by label."""
    for entry in item.get("metadata", []):
        if entry.get("label") == label:
            try:
                return float(entry["value"])
            except (KeyError, ValueError):
                return None
    return None


def extract_gcps(item: dict) -> list[GCP]:
    """Extract (resourceCoords, (lon, lat)) pairs from a IIIF georeferencing annotation."""
    gcps: list[GCP] = []
    for feat in item["body"]["features"]:
        px, py = feat["properties"]["resourceCoords"]
        lon, lat = feat["geometry"]["coordinates"][:2]
        gcps.append(((float(px), float(py)), (float(lon), float(lat))))
    return gcps


def fit_affine(gcps: list[GCP]) -> np.ndarray:
    """Fit a 2×3 affine transform: [lon, lat] = A @ [px, py, 1].

    Returns A (2×3 ndarray). Requires at least 3 GCPs; underdetermined otherwise.
    """
    X = np.array([[px, py, 1.0] for (px, py), _ in gcps])
    Y = np.array([[lon, lat] for _, (lon, lat) in gcps])
    result, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return result.T  # 2×3


def fit_similarity(gcps: list[GCP]) -> np.ndarray:
    """Fit a 4-parameter similarity (Helmert) transform from 2+ GCPs.

    Enforces equal metric scale in x/y and no shear. cos(lat) is estimated from
    the mean latitude of the GCPs to handle the lon/lat aspect ratio.

    Returns A (2×3 ndarray) where [lon, lat]^T = A @ [px, py, 1]^T.
    """
    mean_lat = sum(lat for _, (_, lat) in gcps) / len(gcps)
    cos_phi = math.cos(math.radians(mean_lat))
    M = []
    b: list[float] = []
    for (px, py), (lon, lat) in gcps:
        M.append([px / cos_phi, py / cos_phi, 1.0, 0.0])
        b.append(lon)
        M.append([-py, px, 0.0, 1.0])
        b.append(lat)
    result, *_ = np.linalg.lstsq(np.array(M), np.array(b), rcond=None)
    alpha, beta, tx, ty = result
    return np.array([[alpha / cos_phi, beta / cos_phi, tx], [beta, -alpha, ty]])


def fit_transform(gcps: list[GCP], transform_type: str) -> np.ndarray:
    """Fit the transform indicated by transform_type.

    'helmert' uses a 4-parameter similarity fit; any other value (e.g. 'polynomial')
    uses a full 6-parameter affine least-squares fit.
    """
    if transform_type == "helmert":
        return fit_similarity(gcps)
    return fit_affine(gcps)


def annotation_transform_type(item: dict) -> str:
    """Return the transformation.type string from a IIIF georef annotation body.

    Defaults to 'polynomial' when the field is absent (older annotations).
    """
    return item.get("body", {}).get("transformation", {}).get("type", "polynomial")


def parse_svg_polygon(svg_value: str) -> list[tuple[float, float]]:
    """Parse the ``points`` of an ``<svg><polygon points="x,y x,y ..."/>`` selector.

    Returns the vertices as (x, y) in the annotation's source-pixel frame — the same
    frame as the GCP resourceCoords — or an empty list if no points are found.
    """
    match = re.search(r'points="([^"]*)"', svg_value)
    if not match:
        return []
    points: list[tuple[float, float]] = []
    for token in match.group(1).split():
        x_str, _, y_str = token.partition(",")
        try:
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return points


def truth_polygon_world(item: dict) -> list[list[float]] | None:
    """World-space [lon, lat] ring of a truth annotation's page footprint, or None.

    Fits the annotation's own transform from its GCPs and applies it to the SvgSelector
    polygon (both live in source-pixel coordinates). Returns None when the annotation
    lacks a usable selector or enough GCPs to fit.
    """
    selector = item.get("target", {}).get("selector", {})
    if selector.get("type") != "SvgSelector":
        return None
    polygon = parse_svg_polygon(selector.get("value", ""))
    gcps = extract_gcps(item)
    if len(polygon) < 3 or len(gcps) < 3:
        return None
    A = fit_transform(gcps, annotation_transform_type(item))
    ring = []
    for px, py in polygon:
        lon, lat = A @ np.array([px, py, 1.0])
        ring.append([round(float(lon), 7), round(float(lat), 7)])
    return ring


def truth_polygons_world(iiif_path: Path) -> list[list[list[float]]]:
    """Every truth annotation's page footprint in world space, from a IIIF file.

    One [lon, lat] ring per annotation (split pages contribute several). Annotations
    without a usable selector/fit are skipped. Order follows file order.
    """
    data: dict = json.loads(iiif_path.read_text())
    rings = []
    for item in data.get("items", []):
        ring = truth_polygon_world(item)
        if ring is not None:
            rings.append(ring)
    return rings


def truth_page_number(item: dict) -> int | None:
    """Page number N from a truth annotation label like '... p156' or '... p73 [1]'."""
    match = re.search(r"\bp(\d+)", str(item.get("label", "")))
    return int(match.group(1)) if match else None


def truth_polygons_by_page(iiif_path: Path) -> dict[int, list[list[list[float]]]]:
    """Truth page footprints grouped by page number.

    A page's entry holds every truth annotation sharing its number — so a split page
    (p73__1, p73__2, …) maps to all of page 73's footprints. Annotations without a
    usable page number, selector, or fit are skipped.
    """
    data: dict = json.loads(iiif_path.read_text())
    by_page: dict[int, list[list[list[float]]]] = {}
    for item in data.get("items", []):
        number = truth_page_number(item)
        ring = truth_polygon_world(item)
        if number is None or ring is None:
            continue
        by_page.setdefault(number, []).append(ring)
    return by_page


def north_angle(A: np.ndarray) -> float:
    """Return north direction in degrees from a 2×3 affine matrix.

    North is the direction of increasing latitude in pixel space.
    Angle convention matches detect_compass.py: measured from image-right, clockwise.
    """
    return math.degrees(math.atan2(A[1, 1], A[1, 0])) % 360.0


def scale_deg_per_px(A: np.ndarray) -> float:
    """Return the (latitude) scale in degrees/pixel from a 2×3 affine matrix."""
    return math.sqrt(A[1, 0] ** 2 + A[1, 1] ** 2)


_FT_PER_DEG_LAT = EARTH_RADIUS_FT * math.pi / 180.0


def px_per_ft(A: np.ndarray) -> float:
    """Pixels per foot of ground distance, from a 2×3 pixel→geo affine matrix."""
    return 1.0 / (scale_deg_per_px(A) * _FT_PER_DEG_LAT)


def haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return Haversine distance in feet between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2.0 * EARTH_RADIUS_FT * math.asin(math.sqrt(a))


def sample_grid(width: float, height: float, n: int = 7) -> list[tuple[float, float]]:
    """Return an n×n grid of pixel points at uniform fractional positions.

    Points lie at fractions 1/(n+1), 2/(n+1), …, n/(n+1) of width and height,
    avoiding the image corners.
    """
    fracs = [(k + 1) / (n + 1) for k in range(n)]
    return [(f_x * width, f_y * height) for f_y in fracs for f_x in fracs]


def truth_distortion(A: np.ndarray) -> tuple[float, float]:
    """Compute skew (°) and anisotropy from the truth affine linear part.

    Works in a metric coordinate system (lon scaled by cos(lat at origin)) so that
    distances in lon and lat are comparable.  The generated 4-parameter similarity
    model is always skew=0°, anisotropy=1.0; these metrics show how much the truth
    fit deviates and therefore how much residual error the generated model *must* have.

    Returns:
        skew_deg: deviation of pixel-x / pixel-y axes from perpendicular (0° = similarity)
        aniso: scale_x / scale_y ratio (1.0 = similarity; >1 means x-pixels cover more ground)
    """
    lat0 = A[1, 2]  # latitude at pixel origin, for lon→arc-degree scaling
    cos_lat = math.cos(math.radians(lat0))

    # Column vectors: change in metric geo per one pixel in each image direction.
    v_x = np.array([A[0, 0] * cos_lat, A[1, 0]])
    v_y = np.array([A[0, 1] * cos_lat, A[1, 1]])

    scale_x = float(np.linalg.norm(v_x))
    scale_y = float(np.linalg.norm(v_y))

    aniso = scale_x / scale_y if scale_y > 0 else 1.0

    if scale_x > 0 and scale_y > 0:
        cos_angle = float(np.dot(v_x, v_y) / (scale_x * scale_y))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        skew_deg = math.degrees(math.acos(cos_angle)) - 90.0
    else:
        skew_deg = 0.0

    return round(skew_deg, 2), round(aniso, 3)


def angle_diff(a: float, b: float) -> float:
    """Signed angular difference (a - b) in degrees, normalized to (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def analyze_pair(truth_item: dict, gen_item: dict) -> dict:
    """Compute accuracy metrics for one page by comparing two IIIF georef annotations.

    Returns a dict with keys: page_key, gen_page_key, n_truth, n_gen, rmse_ft, max_ft,
    trans_ft, rot_err, scale_pct, skew_deg, aniso. page_key uses the truth's split
    numbering; gen_page_key uses the matched generated split's numbering (they can differ
    when OIM and our splitter number panels differently).
    """
    source_id: str = truth_item["target"]["source"]["id"]
    page_key = source_id_to_page_key(source_id, truth_item["label"])
    gen_index = annotation_split_index(gen_item)
    base_key = source_id_to_page_key(source_id, "")
    gen_page_key = f"{base_key}__{gen_index}" if gen_index is not None else base_key

    truth_gcps = extract_gcps(truth_item)
    gen_gcps = extract_gcps(gen_item)

    A_truth = fit_transform(truth_gcps, annotation_transform_type(truth_item))
    A_gen = fit_transform(gen_gcps, annotation_transform_type(gen_item))

    width = float(truth_item["target"]["source"]["width"])
    height = float(truth_item["target"]["source"]["height"])

    # For split sub-images with known canvas placement, sample only within the
    # split region so the error metric covers actual split pixels, not the full canvas.
    split_cx = extract_metadata_float(gen_item, "split_canvas_x")
    split_cy = extract_metadata_float(gen_item, "split_canvas_y")
    split_cw = extract_metadata_float(gen_item, "split_canvas_w")
    split_ch = extract_metadata_float(gen_item, "split_canvas_h")

    if (
        split_cx is not None
        and split_cy is not None
        and split_cw is not None
        and split_ch is not None
    ):
        grid = [
            (x + split_cx, y + split_cy) for x, y in sample_grid(split_cw, split_ch)
        ]
    else:
        grid = sample_grid(width, height)

    # Positional errors at grid sample points.
    errors: list[float] = []
    for px, py in grid:
        v = np.array([px, py, 1.0])
        lon_t, lat_t = A_truth @ v
        lon_g, lat_g = A_gen @ v
        errors.append(haversine_ft(lat_t, lon_t, lat_g, lon_g))

    rmse_ft = math.sqrt(sum(e**2 for e in errors) / len(errors))
    max_ft = max(errors)

    # Translation error: difference at image center.
    cx, cy = width / 2.0, height / 2.0
    v_center = np.array([cx, cy, 1.0])
    lon_t, lat_t = A_truth @ v_center
    lon_g, lat_g = A_gen @ v_center
    trans_ft = haversine_ft(lat_t, lon_t, lat_g, lon_g)

    # Rotation and scale from affine linear parts.
    rot_err = angle_diff(north_angle(A_gen), north_angle(A_truth))
    scale_t = scale_deg_per_px(A_truth)
    scale_g = scale_deg_per_px(A_gen)
    scale_pct = (scale_g / scale_t - 1.0) * 100.0 if scale_t > 0 else 0.0

    # Distortion of the truth fit (skew and anisotropy the generated model can't capture).
    skew_deg, aniso = truth_distortion(A_truth)

    return {
        "page_key": page_key,
        "gen_page_key": gen_page_key,
        "n_truth": len(truth_gcps),
        "n_gen": len(gen_gcps),
        "n_streets": extract_metadata_int(gen_item, "streets"),
        "n_intersections": extract_metadata_int(gen_item, "intersections"),
        "truth_px_per_ft": round(px_per_ft(A_truth), 2),
        "gen_px_per_ft": round(px_per_ft(A_gen), 2),
        "rmse_ft": round(rmse_ft, 1),
        "max_ft": round(max_ft, 1),
        "trans_ft": round(trans_ft, 1),
        "rot_err": round(rot_err, 2),
        "scale_pct": round(scale_pct, 2),
        "skew_deg": skew_deg,
        "aniso": aniso,
    }


def analyze_truth_only(truth_item: dict) -> dict:
    """Compute truth-only stats for a page that has no generated fit.

    Returns a dict with keys: page_key, n_truth, skew_deg, aniso.
    """
    source_id: str = truth_item["target"]["source"]["id"]
    page_key = source_id_to_page_key(source_id, truth_item["label"])
    truth_gcps = extract_gcps(truth_item)
    A_truth = fit_transform(truth_gcps, annotation_transform_type(truth_item))
    skew_deg, aniso = truth_distortion(A_truth)
    return {
        "page_key": page_key,
        "n_truth": len(truth_gcps),
        "skew_deg": skew_deg,
        "aniso": aniso,
    }


def split_numbers_disagree(row: dict) -> bool:
    """True if a paired row's truth and generated split numbering differ."""
    gen_page_key = row.get("gen_page_key")
    return gen_page_key is not None and gen_page_key != row["page_key"]


def page_label(row: dict) -> str:
    """Page column text: the truth page key, marked '(t)' when our numbering disagrees."""
    return f"{row['page_key']} (t)" if split_numbers_disagree(row) else row["page_key"]


def print_table(rows: list[dict], missing: list[dict]) -> None:
    """Print paired results (sorted by RMSE desc) then missing pages, with summary stats."""
    rows_sorted = sorted(rows, key=lambda r: r["rmse_ft"], reverse=True)

    header = (
        f"{'Page':<13} {'n_t':>3} {'n_g':>3} {'str':>4} {'int':>4}  "
        f"{'t.px/ft':>7}  {'g.px/ft':>7}  "
        f"{'rmse_ft':>8}  {'max_ft':>8}  {'trans_ft':>9}  "
        f"{'rot_err':>8}  {'scale_%':>7}  {'skew°':>6}  {'aniso':>6}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows_sorted:
        n_str = r["n_streets"] if r["n_streets"] is not None else "—"
        n_int = r["n_intersections"] if r["n_intersections"] is not None else "—"
        # When the split numbers disagree, note the matched generated page in the trailing
        # column (where missing rows show "(no fit)").
        trailing = f"  ({r['gen_page_key']})" if split_numbers_disagree(r) else ""
        print(
            f"{page_label(r):<13} {r['n_truth']:>3} {r['n_gen']:>3} {n_str!s:>4} {n_int!s:>4}  "
            f"{r['truth_px_per_ft']:>7.2f}  {r['gen_px_per_ft']:>7.2f}  "
            f"{r['rmse_ft']:>8.1f}  {r['max_ft']:>8.1f}  {r['trans_ft']:>9.1f}  "
            f"{r['rot_err']:>+8.2f}  {r['scale_pct']:>+7.2f}  "
            f"{r['skew_deg']:>+6.2f}  {r['aniso']:>6.3f}{trailing}"
        )
    if missing:
        missing_sorted = sorted(missing, key=lambda r: r["page_key"])
        for r in missing_sorted:
            print(
                f"{r['page_key']:<13} {r['n_truth']:>3} {'—':>3} {'—':>4} {'—':>4}  "
                f"{'—':>7}  {'—':>7}  "
                f"{'—':>8}  {'—':>8}  {'—':>9}  "
                f"{'—':>8}  {'—':>7}  "
                f"{r['skew_deg']:>+6.2f}  {r['aniso']:>6.3f}  (no fit)"
            )
    print(sep)

    n_paired = len(rows)
    n_missing = len(missing)
    n_truth_total = n_paired + n_missing
    pct = 100 * n_paired / n_truth_total
    print(
        f"\n{n_paired}/{n_truth_total} = {pct:.02f}% pages georeferenced ({n_missing} total losses)"
    )

    if rows:
        rmsers = np.array([r["rmse_ft"] for r in rows])
        rot_errs = np.abs([r["rot_err"] for r in rows])
        trans_fts = np.array([r["trans_ft"] for r in rows])
        print(
            f"RMSE:  mean={float(np.mean(rmsers)):.0f} ft  "
            f"median={float(np.median(rmsers)):.0f} ft  "
            f"max={float(np.max(rmsers)):.0f} ft"
        )
        print(
            f"Trans: mean={float(np.mean(trans_fts)):.0f} ft  "
            f"median={float(np.median(trans_fts)):.0f} ft"
        )
        print(
            f"|rot|: mean={float(np.mean(rot_errs)):.1f}°  median={float(np.median(rot_errs)):.1f}°"
        )
        for thresh in (15, 25, 50, 100, 500, 1000):
            count = int(np.sum(rmsers <= thresh))
            print(
                f"  RMSE ≤ {thresh:>4} ft: {count}/{n_paired} ({100 * count / n_paired:.0f}%)"
            )


def print_tsv(rows: list[dict], missing: list[dict]) -> None:
    """Print all results as TSV for copy/paste into a spreadsheet.

    Paired pages come first (sorted by page_key), then missing pages.
    Missing pages have empty strings for comparison-only columns.
    """
    fields = [
        "page_key",
        "n_truth",
        "n_gen",
        "n_streets",
        "n_intersections",
        "truth_px_per_ft",
        "gen_px_per_ft",
        "rmse_ft",
        "max_ft",
        "trans_ft",
        "rot_err",
        "scale_pct",
        "skew_deg",
        "aniso",
    ]
    print("\t".join(fields))
    for r in sorted(rows, key=lambda x: x["page_key"]):
        print("\t".join(str(r[f]) for f in fields))
    for r in sorted(missing, key=lambda x: x["page_key"]):
        row_vals = {f: "" for f in fields}
        row_vals["page_key"] = r["page_key"]
        row_vals["n_truth"] = str(r["n_truth"])
        row_vals["skew_deg"] = str(r["skew_deg"])
        row_vals["aniso"] = str(r["aniso"])
        print("\t".join(row_vals[f] for f in fields))


def annotations_by_source(path: Path) -> dict[str, list[dict]]:
    """Load a IIIF AnnotationPage, grouping items by target.source.id.

    Split pages produce several items per source id (they share the parent canvas);
    full pages produce one. Group order follows file order.
    """
    data: dict = json.loads(path.read_text())
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in data.get("items", []):
        groups[item["target"]["source"]["id"]].append(item)
    return dict(groups)


def label_split_index(item: dict) -> int | None:
    """Return split number N from a label ending in '[N]', or None for full pages."""
    m = re.search(r"\[(\d+)\]$", item.get("label", ""))
    return int(m.group(1)) if m else None


def annotation_split_index(item: dict) -> int | None:
    """Return split number N from a generated annotation id like '...__N/georef'."""
    m = re.search(r"__(\d+)/", item.get("id", ""))
    return int(m.group(1)) if m else None


def ring_to_polygon(ring: list[list[float]]) -> ShapelyPolygon:
    """A panels.json ring as a valid Shapely polygon (buffer(0) repairs self-intersection)."""
    polygon = ShapelyPolygon(ring)
    return polygon if polygon.is_valid else polygon.buffer(0)


def polygon_iou(a: ShapelyPolygon, b: ShapelyPolygon) -> float:
    """Intersection-over-union of two polygons; 0 if disjoint or empty."""
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def load_split_polygons(
    path: Path, source_dims: tuple[float, float] | None = None
) -> dict[int, ShapelyPolygon]:
    """Load panels.json rings as polygons keyed by 1-based split index.

    OIM truth panels (oim/pN.panels.json) are already in canvas coordinates, so pass
    source_dims=None. Our generated panels (pN.panels.json) are in the 25%-scale page
    frame; pass the canvas (source) width/height to scale them up to canvas coordinates.
    Returns {} if the file is absent.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if source_dims is not None:
        scale_x = source_dims[0] / data["width"]
        scale_y = source_dims[1] / data["height"]
    else:
        scale_x = scale_y = 1.0
    return {
        i: ring_to_polygon([[x * scale_x, y * scale_y] for x, y in ring])
        for i, ring in enumerate(data["panels"], start=1)
    }


def match_split_pairs(
    truth_items: list[dict],
    gen_items: list[dict],
    truth_polygons: dict[int, ShapelyPolygon],
    gen_polygons: dict[int, ShapelyPolygon],
    canvas_polygon: ShapelyPolygon | None = None,
) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """Associate truth and generated split annotations by panel-polygon overlap.

    OIM's split numbering need not match ours, so pairs are chosen by greatest panel IoU
    globally: every (truth, generated) pair with IoU >= MIN_SPLIT_IOU is considered, and
    the highest-overlap pairs are assigned first (each split used once). Choosing globally
    rather than per-truth-in-file-order matters — a truth split that merely grazes a
    generated panel must not claim it ahead of the truth split that actually overlaps it.

    When OIM split a page but we kept it whole, the generated annotation has no split index;
    canvas_polygon (the full page) then stands in as its panel, so it pairs with the truth
    split it most overlaps — i.e. the largest split. Both sides' polygons are in canvas
    coordinates. Returns (pairs, unmatched_truth_items).
    """

    def gen_polygon(gen: dict) -> ShapelyPolygon | None:
        gen_index = annotation_split_index(gen)
        if gen_index is not None:
            return gen_polygons.get(gen_index)
        return canvas_polygon  # unsplit page: one panel covering the whole canvas

    candidates: list[tuple[float, int, int]] = []  # (iou, truth_idx, gen_idx)
    for ti, truth in enumerate(truth_items):
        truth_index = label_split_index(truth)
        truth_poly = truth_polygons.get(truth_index) if truth_index else None
        if truth_poly is None:
            continue
        for gi, gen in enumerate(gen_items):
            gen_poly = gen_polygon(gen)
            if gen_poly is None:
                continue
            iou = polygon_iou(truth_poly, gen_poly)
            if iou >= MIN_SPLIT_IOU:
                candidates.append((iou, ti, gi))

    candidates.sort(reverse=True)  # assign highest-overlap pairs first
    used_truth: set[int] = set()
    used_gen: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for _iou, ti, gi in candidates:
        if ti in used_truth or gi in used_gen:
            continue
        pairs.append((truth_items[ti], gen_items[gi]))
        used_truth.add(ti)
        used_gen.add(gi)

    unmatched = [t for ti, t in enumerate(truth_items) if ti not in used_truth]
    return pairs, unmatched


def compare_pages(
    truth_path: Path, generated_path: Path
) -> tuple[list[dict], list[dict]]:
    """Compute per-page comparison rows for a truth/generated IIIF pair.

    Pairs truth and generated annotations by source id (and, for split pages, by
    panel-polygon overlap). Returns (rows, missing): rows are pages present in both,
    missing are truth pages with no matching generated fit. Used by the `compare` CLI
    and by `mapsnap fit` to compute the metrics recorded in an experiment manifest.
    """
    truth_by_source = annotations_by_source(truth_path)
    gen_by_source = annotations_by_source(generated_path)
    oim_dir = truth_path.parent / "oim"
    gen_dir = generated_path.parent

    rows: list[dict] = []
    missing: list[dict] = []
    for source_id, truth_items in sorted(truth_by_source.items()):
        gen_items = gen_by_source.get(source_id, [])
        truth_splits = [t for t in truth_items if label_split_index(t) is not None]
        if not truth_splits:
            # Full page: pair the single truth and generated annotations directly.
            gen_item = gen_items[0] if gen_items else None
            if gen_item is None:
                missing.append(analyze_truth_only(truth_items[0]))
            else:
                rows.append(analyze_pair(truth_items[0], gen_item))
            continue
        # Split page: associate by panel-polygon overlap (numbering may differ).
        page_key = source_id_to_page_key(source_id, "")
        source = truth_items[0]["target"]["source"]
        source_dims = (float(source["width"]), float(source["height"]))
        truth_polygons = load_split_polygons(oim_dir / f"{page_key}.panels.json")
        gen_polygons = load_split_polygons(
            gen_dir / f"{page_key}.panels.json", source_dims
        )
        # If we kept the page whole, a generated annotation with no split index stands in
        # for a panel covering the full canvas, matching the largest truth split.
        canvas_polygon = box(0.0, 0.0, source_dims[0], source_dims[1])
        pairs, unmatched = match_split_pairs(
            truth_splits, gen_items, truth_polygons, gen_polygons, canvas_polygon
        )
        rows.extend(analyze_pair(t, g) for t, g in pairs)
        missing.extend(analyze_truth_only(t) for t in unmatched)
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare human-generated and computer-generated IIIF georeferencing."
    )
    parser.add_argument(
        "truth", metavar="TRUTH_IIIF", help="Human-generated IIIF AnnotationPage"
    )
    parser.add_argument(
        "generated", metavar="GEN_IIIF", help="Computer-generated IIIF AnnotationPage"
    )
    parser.add_argument(
        "--omit-missing",
        action="store_true",
        help="Only print statistics for matched images.",
    )
    parser.add_argument(
        "--csv", metavar="FILE", help="Also write per-page results to a CSV file"
    )
    parser.add_argument(
        "--tabs",
        action="store_true",
        help="Output TSV to stdout instead of the fixed-width table (for spreadsheet paste)",
    )
    args = parser.parse_args()

    truth_by_source = annotations_by_source(Path(args.truth))
    gen_by_source = annotations_by_source(Path(args.generated))
    n_truth_total = sum(len(items) for items in truth_by_source.values())
    n_gen_total = sum(len(items) for items in gen_by_source.values())
    print(
        f"Truth: {n_truth_total} pages  |  Generated: {n_gen_total} pages",
        file=sys.stderr,
    )

    rows, missing = compare_pages(Path(args.truth), Path(args.generated))

    if args.omit_missing:
        missing = []

    if args.tabs:
        print_tsv(rows, missing)
    else:
        print_table(rows, missing)

    if args.csv:
        fields = [
            "page_key",
            "n_truth",
            "n_gen",
            "n_streets",
            "n_intersections",
            "truth_px_per_ft",
            "gen_px_per_ft",
            "rmse_ft",
            "max_ft",
            "trans_ft",
            "rot_err",
            "scale_pct",
            "skew_deg",
            "aniso",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
