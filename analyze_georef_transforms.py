"""Compare 6-parameter affine vs 4-parameter similarity transforms for IIIF georef data.

For each annotation in an OIM IIIF AnnotationPage, fits both a full affine transform
and a constrained similarity transform (uniform scale + rotation + translation) to the
GCPs. Reports scale isotropy, axis orthogonality, and the residual cost of the
similarity constraint for each image.

The similarity model used is north-up (pixel-y axis points south):
    px = a*x + b*y + tx
    py = b*x - a*y + ty
where (x, y) are equal-scale projected coordinates (longitude scaled by cos(lat)).

Usage:
    python analyze_georef_transforms.py <iiif_annotation_page.json>
"""

import argparse
import json
import math
import sys

import numpy as np


def project_gcps(gcps: list[tuple]) -> list[tuple]:
    """Convert (lon, lat) to equal-scale flat-Earth km coordinates.

    Uses a single cos(lat) correction at the mean latitude of the GCP cluster so
    that 1 km east and 1 km north have the same coordinate magnitude.
    """
    lats = [lat for _, (lon, lat) in gcps]
    lat0 = math.radians(sum(lats) / len(lats))
    cos_lat = math.cos(lat0)
    KM = 111.32  # km per degree latitude
    return [((px, py), (lon * cos_lat * KM, lat * KM)) for (px, py), (lon, lat) in gcps]


def fit_affine(gcps_proj: list[tuple]) -> tuple[np.ndarray, float]:
    """Fit a 6-parameter affine transform in projected km space.

    Returns (A, rms_px) where A is 2×3 and rms_px is the per-pixel RMS residual.
    The transform is: [px, py] = A @ [x, y, 1]^T.
    """
    X = np.array([[x, y, 1.0] for (px, py), (x, y) in gcps_proj])
    Y = np.array([[px, py] for (px, py), (x, y) in gcps_proj])
    A, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = A.T  # 2×3
    rms = float(np.sqrt(np.mean((X @ A.T - Y) ** 2)))
    return A, rms


def fit_similarity(gcps_proj: list[tuple]) -> tuple[np.ndarray, float]:
    """Fit a 4-parameter north-up similarity transform in projected km space.

    The model is:  px = a*x + b*y + tx
                   py = b*x - a*y + ty
    which preserves orthogonality and uniform scale while allowing rotation and
    translation. The minus sign on a in the py equation reflects the image y-axis
    pointing south (opposite to geographic north).

    Coefficient rows for [a, b, tx, ty]:
        px row: [ x,  y, 1, 0]
        py row: [-y,  x, 0, 1]

    Returns (params, rms_px) where params = [a, b, tx, ty].
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (px, py), (x, y) in gcps_proj:
        rows.append([x, y, 1.0, 0.0])
        rhs.append(px)
        rows.append([-y, x, 0.0, 1.0])
        rhs.append(py)
    M = np.array(rows)
    b = np.array(rhs)
    params, *_ = np.linalg.lstsq(M, b, rcond=None)
    rms = float(np.sqrt(np.mean((M @ params - b) ** 2)))
    return params, rms


def decompose_affine(A: np.ndarray) -> tuple[float, float, float, float]:
    """Decompose the 2×2 linear part of A (2×3) via SVD.

    Returns (s1, s2, scale_ratio, skew_deg):
      - s1, s2: singular values (px/km), s1 ≥ s2
      - scale_ratio: s1/s2 — 1.0 means isotropic (similarity-like)
      - skew_deg: angle between the two projected geo axes in pixel space — 90° means
                  orthogonal (no shear)
    """
    L = A[:, :2]
    _, S, _ = np.linalg.svd(L)
    s1, s2 = float(S[0]), float(S[1])
    scale_ratio = s1 / s2 if s2 > 0 else float("inf")
    col0, col1 = L[:, 0], L[:, 1]
    n0, n1 = np.linalg.norm(col0), np.linalg.norm(col1)
    cos_angle = float(np.dot(col0, col1) / (n0 * n1)) if n0 > 0 and n1 > 0 else 0.0
    skew_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
    return s1, s2, scale_ratio, skew_deg


def analyze_item(item: dict) -> dict | None:
    """Extract GCPs and compute transform metrics for one annotation item.

    Returns None if fewer than 3 GCPs are present (affine needs ≥3).
    """
    feats = item.get("body", {}).get("features", [])
    if len(feats) < 3:
        return None
    gcps = [
        (
            (
                f["properties"]["resourceCoords"][0],
                f["properties"]["resourceCoords"][1],
            ),
            (f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1]),
        )
        for f in feats
    ]
    gcps_proj = project_gcps(gcps)
    A, aff_rms = fit_affine(gcps_proj)
    _, sim_rms = fit_similarity(gcps_proj)
    s1, s2, scale_ratio, skew_deg = decompose_affine(A)
    # Geometric-mean scale (px/km) → px/ft.  s1, s2 are in px/km from the affine fit.
    px_per_ft = math.sqrt(s1 * s2) / 3280.84
    return {
        "label": item.get("label", ""),
        "n_gcps": len(gcps),
        "scale_ratio": scale_ratio,
        "skew_deg": skew_deg,
        "aff_rms": aff_rms,
        "sim_rms": sim_rms,
        "extra_rms": sim_rms - aff_rms,
        "px_per_ft": px_per_ft,
    }


def print_table(rows: list[dict]) -> None:
    """Print a fixed-width table of per-image transform metrics."""
    header = (
        f"{'Label':<40} {'n':>3}  "
        f"{'s1/s2':>6}  {'skew°':>6}  "
        f"{'aff_rms':>8}  {'sim_rms':>8}  {'Δrms':>8}  {'px/ft':>6}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        label = r["label"]
        # Trim label to last 40 chars for readability
        if len(label) > 40:
            label = "…" + label[-39:]
        print(
            f"{label:<40} {r['n_gcps']:>3}  "
            f"{r['scale_ratio']:>6.3f}  {r['skew_deg']:>6.2f}  "
            f"{r['aff_rms']:>8.1f}  {r['sim_rms']:>8.1f}  {r['extra_rms']:>8.1f}  "
            f"{r['px_per_ft']:>6.3f}"
        )
    print(sep)

    # Summary row
    n = len(rows)

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows])

    def med(key: str) -> float:
        return float(np.median(arr(key)))

    def mean(key: str) -> float:
        return float(np.mean(arr(key)))

    print(
        f"{'MEDIAN':<40} {n:>3}  "
        f"{med('scale_ratio'):>6.3f}  {med('skew_deg'):>6.2f}  "
        f"{med('aff_rms'):>8.1f}  {med('sim_rms'):>8.1f}  {med('extra_rms'):>8.1f}  "
        f"{med('px_per_ft'):>6.3f}"
    )
    print(
        f"{'MEAN':<40} {n:>3}  "
        f"{mean('scale_ratio'):>6.3f}  {mean('skew_deg'):>6.2f}  "
        f"{mean('aff_rms'):>8.1f}  {mean('sim_rms'):>8.1f}  {mean('extra_rms'):>8.1f}  "
        f"{mean('px_per_ft'):>6.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 6-param affine vs 4-param similarity transforms for each "
            "image in an OIM IIIF AnnotationPage JSON file."
        )
    )
    parser.add_argument(
        "iiif_json", metavar="FILE", help="OIM IIIF AnnotationPage JSON file"
    )
    parser.add_argument(
        "--sort",
        choices=[
            "label",
            "scale_ratio",
            "skew_deg",
            "aff_rms",
            "sim_rms",
            "extra_rms",
            "px_per_ft",
        ],
        default="label",
        help="Sort column (default: label)",
    )
    parser.add_argument(
        "--min-gcps",
        type=int,
        default=3,
        metavar="N",
        help="Minimum GCPs to include an image (default: 3)",
    )
    args = parser.parse_args()

    data: dict = json.load(open(args.iiif_json))
    items: list[dict] = data.get("items", [])

    rows = []
    skipped = 0
    for item in items:
        result = analyze_item(item)
        if result is None or result["n_gcps"] < args.min_gcps:
            skipped += 1
            continue
        rows.append(result)

    if not rows:
        print("No images with enough GCPs found.", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: r[args.sort])

    print(
        f"# {len(rows)} images analyzed  ({skipped} skipped for <{args.min_gcps} GCPs)\n"
        f"# Columns: label | n_gcps | scale_ratio (s1/s2) | skew° | "
        f"affine_rms_px | similarity_rms_px | Δrms (sim−aff)\n"
    )
    print_table(rows)


if __name__ == "__main__":
    main()
