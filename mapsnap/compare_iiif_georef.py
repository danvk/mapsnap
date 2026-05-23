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
from pathlib import Path

import numpy as np

GCP = tuple[tuple[float, float], tuple[float, float]]  # ((px, py), (lon, lat))
EARTH_RADIUS_FT = 20_925_524.0


def extract_metadata_int(item: dict, label: str) -> int | None:
    """Extract an integer value from a IIIF annotation's metadata list by label."""
    for entry in item.get("metadata", []):
        if entry.get("label") == label:
            try:
                return int(entry["value"])
            except (KeyError, ValueError):
                return None
    return None


def extract_metadata_bool(item: dict, label: str) -> bool | None:
    """Extract a boolean value from a IIIF annotation's metadata list by label."""
    for entry in item.get("metadata", []):
        if entry.get("label") == label:
            v = entry.get("value", "")
            if v == "true":
                return True
            if v == "false":
                return False
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

    Returns A (2×3 ndarray).
    """
    X = np.array([[px, py, 1.0] for (px, py), _ in gcps])
    Y = np.array([[lon, lat] for _, (lon, lat) in gcps])
    result, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return result.T  # 2×3


def north_angle(A: np.ndarray) -> float:
    """Return north direction in degrees from a 2×3 affine matrix.

    North is the direction of increasing latitude in pixel space.
    Angle convention matches detect_compass.py: measured from image-right, clockwise.
    """
    return math.degrees(math.atan2(A[1, 1], A[1, 0])) % 360.0


def scale_deg_per_px(A: np.ndarray) -> float:
    """Return the (latitude) scale in degrees/pixel from a 2×3 affine matrix."""
    return math.sqrt(A[1, 0] ** 2 + A[1, 1] ** 2)


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
    """Signed angular difference (a − b) in degrees, normalized to (−180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def source_id_to_page_key(source_id: str, label: str) -> str:
    """Extract a short page key like 'p425' from a LOC IIIF image service URL."""
    split_suffix = ""
    if label.endswith("]"):
        m = re.search(r"\[(\d+)\]$", label)
        assert m
        split = m.group(1)
        split_suffix = f"__{split}"
    m = re.search(r"-(\d+)([sNESW]?)(?:/info\.json)?$", source_id)
    if m:
        page_key = f"p{int(m.group(1))}{m.group(2)}"
    else:
        page_key = source_id.split("/")[-2] or source_id
    page_key += split_suffix
    return page_key


def analyze_pair(truth_item: dict, gen_item: dict) -> dict:
    """Compute accuracy metrics for one page by comparing two IIIF georef annotations.

    Returns a dict with keys: page_key, n_truth, n_gen, rmse_ft, max_ft, trans_ft,
    rot_err, scale_pct, skew_deg, aniso.
    """
    source_id: str = truth_item["target"]["source"]["id"]
    page_key = source_id_to_page_key(source_id, truth_item["label"])

    truth_gcps = extract_gcps(truth_item)
    gen_gcps = extract_gcps(gen_item)

    A_truth = fit_affine(truth_gcps)
    A_gen = fit_affine(gen_gcps)

    width = float(truth_item["target"]["source"]["width"])
    height = float(truth_item["target"]["source"]["height"])

    # Positional errors at grid sample points.
    grid = sample_grid(width, height)
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

    is_full_canvas = extract_metadata_bool(gen_item, "is_full_canvas")

    return {
        "page_key": page_key,
        "n_truth": len(truth_gcps),
        "n_gen": len(gen_gcps),
        "n_streets": extract_metadata_int(gen_item, "streets"),
        "n_intersections": extract_metadata_int(gen_item, "intersections"),
        "is_full_canvas": is_full_canvas,
        # Metrics below are only meaningful when is_full_canvas is True.
        # For sub-images, canvas coords were derived from truth GCPs → circular.
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
    A_truth = fit_affine(truth_gcps)
    skew_deg, aniso = truth_distortion(A_truth)
    return {
        "page_key": page_key,
        "n_truth": len(truth_gcps),
        "skew_deg": skew_deg,
        "aniso": aniso,
    }


def print_table(rows: list[dict], missing: list[dict]) -> None:
    """Print paired results (sorted by RMSE desc) then missing pages, with summary stats."""
    rows_sorted = sorted(rows, key=lambda r: r["rmse_ft"], reverse=True)

    header = (
        f"{'Page':<9} {'n_t':>3} {'n_g':>3} {'str':>4} {'int':>4}  "
        f"{'rmse_ft':>8}  {'max_ft':>8}  {'trans_ft':>9}  "
        f"{'rot_err':>8}  {'scale_%':>7}  {'skew°':>6}  {'aniso':>6}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    has_circular = False
    for r in rows_sorted:
        n_str = r["n_streets"] if r["n_streets"] is not None else "—"
        n_int = r["n_intersections"] if r["n_intersections"] is not None else "—"
        # Sub-images: metrics are circular (canvas coords derived from truth GCPs).
        circular = r.get("is_full_canvas") is False
        if circular:
            has_circular = True
        suffix = "*" if circular else ""
        print(
            f"{r['page_key'] + suffix:<9} {r['n_truth']:>3} {r['n_gen']:>3} {n_str!s:>4} {n_int!s:>4}  "
            f"{r['rmse_ft']:>8.1f}  {r['max_ft']:>8.1f}  {r['trans_ft']:>9.1f}  "
            f"{r['rot_err']:>+8.2f}  {r['scale_pct']:>+7.2f}  "
            f"{r['skew_deg']:>+6.2f}  {r['aniso']:>6.3f}"
        )
    if missing:
        missing_sorted = sorted(missing, key=lambda r: r["page_key"])
        for r in missing_sorted:
            print(
                f"{r['page_key']:<9} {r['n_truth']:>3} {'—':>3} {'—':>4} {'—':>4}  "
                f"{'—':>8}  {'—':>8}  {'—':>9}  "
                f"{'—':>8}  {'—':>7}  "
                f"{r['skew_deg']:>+6.2f}  {r['aniso']:>6.3f}  (no fit)"
            )
    print(sep)
    if has_circular:
        print(
            "* sub-image: metrics are circular (canvas coords derived from truth GCPs)"
        )

    n_paired = len(rows)
    n_missing = len(missing)
    n_truth_total = n_paired + n_missing
    print(
        f"\n{n_paired}/{n_truth_total} pages georeferenced ({n_missing} total losses)"
    )

    # Summary statistics exclude sub-image rows (circular comparison).
    valid_rows = [r for r in rows if r.get("is_full_canvas") is not False]
    n_circular = n_paired - len(valid_rows)
    if valid_rows:
        rmsers = np.array([r["rmse_ft"] for r in valid_rows])
        rot_errs = np.abs([r["rot_err"] for r in valid_rows])
        trans_fts = np.array([r["trans_ft"] for r in valid_rows])
        qualifier = f" (excl. {n_circular} circular)" if n_circular else ""
        print(
            f"RMSE{qualifier}:  mean={float(np.mean(rmsers)):.0f} ft  "
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
        n_valid = len(valid_rows)
        for thresh in (15, 25, 50, 100, 500, 1000):
            count = int(np.sum(rmsers <= thresh))
            print(
                f"  RMSE ≤ {thresh:>4} ft: {count}/{n_valid} ({100 * count / n_valid:.0f}%)"
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
        "is_full_canvas",
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


def load_items_by_source(path: Path) -> dict[str, dict]:
    """Load a IIIF AnnotationPage and index items by target.source.id."""
    data: dict = json.loads(path.read_text())
    index: dict[str, dict] = {}
    for item in data.get("items", []):
        source_id: str = item["target"]["source"]["id"]
        if item["label"].endswith("]"):
            m = re.search(r"\[(\d+)\]$", item["label"])
            assert m
            split = m.group(1)
            source_id += f"__{split}"
        index[source_id] = item
    return index


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

    truth_items = load_items_by_source(Path(args.truth))
    gen_items = load_items_by_source(Path(args.generated))

    n_truth_total = len(truth_items)
    print(
        f"Truth: {n_truth_total} pages  |  Generated: {len(gen_items)} pages",
        file=sys.stderr,
    )

    rows: list[dict] = []
    missing: list[dict] = []
    for source_id, truth_item in sorted(truth_items.items()):
        gen_item = gen_items.get(source_id)
        if gen_item is None:
            missing.append(analyze_truth_only(truth_item))
        else:
            rows.append(analyze_pair(truth_item, gen_item))

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
