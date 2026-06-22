"""Refine per-page georeferencing using the key map's expected page locations.

The first-pass georeferencer (mapsnap.georef_from_labels) matches OCR'd street names against
the whole ``centerlines.geojson``, an ambiguous vocabulary that makes outlier and un-fit pages
drift or fail. The key map tells us roughly where each page should be, so for those pages we:

1. fit the key map (mapsnap.keymap.fit_keymap) to get each page number's expected world location;
2. filter the centerlines to a neighborhood around that location (a much tighter, unambiguous
   street vocabulary);
3. re-run OCR over the page's cached CRAFT boxes (no CRAFT, so it's fast) with that vocabulary,
   writing ``<stem>.streets2.json``;
4. re-fit with the filtered centerlines, writing ``<stem>.georef2.json`` (or
   ``<stem>.georef2-reject.json`` if the new fit wanders away from the expected location);
5. emit a new IIIF that prefers the refined fit where it improved on the original.

First-pass ``streets.json`` / ``georef.json`` are never overwritten (the key-map fit reads only
the originals), avoiding circularity.

    uv run mapsnap refine-with-keymap data/<vol>/p1b.keymap.json data/<vol>/p*.jpg
"""

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import easyocr
import numpy as np

from mapsnap.ctc_vocab_decode import generate_vocab_strings
from mapsnap.detect_text import detect_text
from mapsnap.fit import find_centerlines, find_ref_iiif
from mapsnap.georef_from_labels import (
    compute_cos_phi,
    deg_per_px_to_px_per_ft,
    process_deferred_image,
    process_image,
    px_per_ft_to_deg_per_px,
)
from mapsnap.keymap.fit_keymap import (
    Detection,
    Model,
    Point,
    build_correspondences,
    load_detections,
    load_georef_pages,
    page_number,
    project,
    ransac,
    similarity_apply,
    superseded_stems,
    unproject,
)
from mapsnap.streets import build_block_index
from mapsnap.utils import image_stem, run_cmd


# Detection-filter parameters process_image accepts; read from an existing georef.json so the
# re-fit uses the same thresholds the volume was first georeferenced with (its defaults are far
# stricter than the auto-derived ones the `mapsnap georef` CLI actually used).
PROCESS_PARAM_KEYS = (
    "min_confidence",
    "min_long_side",
    "min_short_side",
    "min_aspect_ratio",
    "high_confidence_size_fraction",
)
DEFAULT_PROCESS_PARAMS = {
    "min_confidence": 0.15,
    "min_short_side": 24.0,
    "min_long_side": 48.0,
    "min_aspect_ratio": 1.75,
    "high_confidence_size_fraction": 0.7,
}


def load_georef_params(volume: Path) -> dict:
    """The detection-filter parameters the volume was georeferenced with (from a georef.json)."""
    for path in sorted(volume.glob("*.georef.json")):
        params = json.load(open(path)).get("inputs", {}).get("parameters", {})
        chosen = {k: params[k] for k in PROCESS_PARAM_KEYS if k in params}
        if chosen:
            return chosen
    return dict(DEFAULT_PROCESS_PARAMS)


def frame_diagonal(frame: list[Point]) -> float:
    """Bounding-box diagonal of a polygon (same units as the polygon, e.g. metres)."""
    xs = [p[0] for p in frame]
    ys = [p[1] for p in frame]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def expected_centers(
    number: int, model: Model, detections: list[Detection]
) -> list[Point]:
    """World (metre-frame) locations of every key-map detection of ``number``.

    A page number can appear several times on the key map (one per split), and we don't know
    which split is which, so all occurrences are returned and treated as a union downstream.
    """
    return [similarity_apply(model, d.pixel) for d in detections if d.number == number]


def near_any(point: Point, centers: list[Point], radius: float) -> bool:
    """Whether ``point`` is within ``radius`` of any of ``centers`` (Euclidean, metres)."""
    return any(math.dist(point, c) <= radius for c in centers)


def _geometry_vertices(geometry: dict) -> list[Point]:
    """Flatten a GeoJSON geometry's coordinates to a list of (lon, lat) vertices."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "LineString":
        return [(c[0], c[1]) for c in coords]
    if kind == "MultiLineString":
        return [(c[0], c[1]) for line in coords for c in line]
    if kind == "Point":
        return [(coords[0], coords[1])]
    if kind == "Polygon":
        return [(c[0], c[1]) for ring in coords for c in ring]
    return []


def filter_centerlines(
    geojson: dict,
    centers: list[Point],
    radius: float,
    lon0: float,
    lat0: float,
) -> dict:
    """Subset ``geojson`` to features with any vertex within ``radius`` (metres) of a center.

    ``centers`` are in the local metre frame (origin lon0/lat0); feature vertices are lon/lat
    and projected with the same origin before the distance test.
    """
    kept = []
    for feature in geojson.get("features", []):
        for lon, lat in _geometry_vertices(feature.get("geometry", {})):
            if near_any(project(lon, lat, lon0, lat0), centers, radius):
                kept.append(feature)
                break
    return {"type": "FeatureCollection", "features": kept}


def refine_page(
    image_path: str,
    centers: list[Point],
    radius: float,
    geojson: dict,
    origin: Point,
    reader: easyocr.Reader,
    centerlines_path: str,
    params: dict,
    scale_deg_per_px: float | None,
    neighbor_rotations: list[tuple[Point, float]],
) -> tuple[str, float | None, Path | None]:
    """Re-OCR + re-fit one page against centerlines near ``centers``.

    Pages with only one intersection GCP are fit like `mapsnap georef`'s deferred path,
    using ``scale_deg_per_px`` (the volume scale) and ``neighbor_rotations`` to resolve
    rotation; unconfirmed 1-GCP fits are rejected.

    Returns (outcome, scale_deg_per_px, georef2_path): outcome is "accepted" (wrote
    <stem>.georef2.json, scale set), "rejected" (fit wandered/unconfirmed/failed; wrote
    <stem>.georef2-reject.json if a fit was produced), or "skipped" (no nearby streets).
    """
    lon0, lat0 = origin
    stem = image_stem(image_path)
    parent = Path(image_path).parent
    streets2 = parent / f"{stem}.streets2.json"
    georef2 = parent / f"{stem}.georef2.json"
    reject = parent / f"{stem}.georef2-reject.json"
    for stale in (georef2, reject):
        stale.unlink(missing_ok=True)

    subset = filter_centerlines(geojson, centers, radius, lon0, lat0)
    block_index = build_block_index(subset)
    if not block_index:
        print(f"  {stem}: no centerlines within radius — skipping", file=sys.stderr)
        return "skipped", None, None
    cos_phi = compute_cos_phi(block_index)
    vocab = generate_vocab_strings(set(block_index.keys()))

    detect_text(
        image_path,
        vocab_strings=vocab,
        reuse_boxes=True,
        reader=reader,
        streets_path=str(streets2),
    )
    # Record the filter region (provenance), so streets2.json is self-describing.
    doc = json.load(open(streets2))
    doc["refine"] = {
        "centers_lonlat": [list(unproject(x, y, lon0, lat0)) for x, y in centers],
        "radius_m": radius,
        "streets_in_vocab": len(block_index),
    }
    with open(streets2, "w") as f:
        json.dump(doc, f, indent=2)

    result = process_image(
        image_path,
        str(streets2),
        str(georef2),
        block_index,
        cos_phi,
        centerlines_path,
        one_gcp_fits=True,
        **params,
    )
    # A 1-GCP page is deferred by process_image; fit it with the volume scale + neighbor
    # rotation voting, exactly like `mapsnap georef` does.
    if (
        not result.success
        and result.deferred is not None
        and scale_deg_per_px is not None
    ):
        result = process_deferred_image(
            result.deferred,
            scale_deg_per_px,
            block_index,
            cos_phi,
            neighbor_rotations,
        )

    if result.success and result.center is not None:
        center_m = project(result.center[0], result.center[1], lon0, lat0)
        if near_any(center_m, centers, radius):
            return "accepted", result.scale_deg_per_px, georef2
        if georef2.exists():
            georef2.replace(reject)  # confirmed fit but wandered — keep for inspection
        return "rejected", None, None
    # Unconfirmed 1-GCP fits write a georef2 file but return success=False with a center;
    # keep them as reject for inspection. Hard failures wrote nothing.
    if result.center is not None and georef2.exists():
        georef2.replace(reject)
        return "rejected", None, None
    georef2.unlink(missing_ok=True)
    return "rejected", None, None


def pages_to_refine(
    pages: list, detections: list[Detection], inliers: set[int]
) -> set[int]:
    """Page numbers worth refining: key-map outliers plus pages absent from any georef."""
    georef_numbers = {p.number for p in pages}
    keymap_numbers = {d.number for d in detections}
    matched = {i for i, p in enumerate(pages) if p.number in keymap_numbers}
    outliers = {pages[i].number for i in (matched - inliers)}
    unfit = keymap_numbers - georef_numbers
    return outliers | unfit


def georef_scale_deg_per_px(corners: list, width: float, height: float) -> float:
    """Scale (degrees latitude per pixel) implied by a georef.json's corner quad.

    corners are [TL, TR, BR, BL] image-corner lon/lats; this is the magnitude of the affine's
    latitude row, matching georef_from_labels.affine_scale_deg_per_px.
    """
    d_lat_dx = (corners[1][1] - corners[0][1]) / width
    d_lat_dy = (corners[3][1] - corners[0][1]) / height
    return math.hypot(d_lat_dx, d_lat_dy)


def georef_rotation(corners: list, width: float, height: float) -> float:
    """Directed rotation (radians) implied by a georef.json's corner quad.

    Matches georef_from_labels' rotation = atan2(A[1,0], -A[1,1]) (the latitude row), so it
    feeds the 1-GCP neighbor-rotation voting consistently.
    """
    d_lat_dx = (corners[1][1] - corners[0][1]) / width
    d_lat_dy = (corners[3][1] - corners[0][1]) / height
    return math.atan2(d_lat_dx, -d_lat_dy)


def neighbor_rotations_from_inliers(
    pages: list, inliers: set[int], volume: Path
) -> list[tuple[Point, float]]:
    """(center lon/lat, rotation) for each inlier page — neighbors for 1-GCP rotation voting."""
    neighbors: list[tuple[Point, float]] = []
    for i in inliers:
        for stem in pages[i].piece_ids:
            path = volume / f"{stem}.georef.json"
            if path.exists():
                g = json.load(open(path))
                corners = g["corners"]
                center = (
                    sum(c[0] for c in corners) / len(corners),
                    sum(c[1] for c in corners) / len(corners),
                )
                neighbors.append(
                    (center, georef_rotation(corners, g["width"], g["height"]))
                )
                break
    return neighbors


def reference_scale_px_per_ft(
    pages: list, inliers: set[int], volume: Path
) -> float | None:
    """Median page scale (px/ft) over the key-map inlier pages — the volume's true scale.

    Reads each inlier page's original georef.json; a far more reliable reference than the
    median of the (biased, mostly-hard) refit batch.
    """
    scales = []
    for i in inliers:
        for stem in pages[i].piece_ids:
            path = volume / f"{stem}.georef.json"
            if path.exists():
                g = json.load(open(path))
                scales.append(
                    deg_per_px_to_px_per_ft(
                        georef_scale_deg_per_px(g["corners"], g["width"], g["height"])
                    )
                )
                break
    return float(np.median(scales)) if scales else None


def scale_outlier_indices(
    scales: list[float], reference_px_per_ft: float | None, threshold: float
) -> set[int]:
    """Indices of refit scales (deg/px) off the reference px/ft by > ``threshold`` (a fraction).

    Mirrors `mapsnap georef`'s misscale filter (ratio to a reference scale), using the volume's
    own scale (reference_scale_px_per_ft) rather than the biased refit-batch median.
    """
    if not scales or threshold <= 0 or not reference_px_per_ft:
        return set()
    return {
        i
        for i, s in enumerate(scales)
        if abs(deg_per_px_to_px_per_ft(s) / reference_px_per_ft - 1.0) > threshold
    }


def keymap_centroid_error(
    model: Model, pages: list, inliers: set[int], detections: list[Detection]
) -> float:
    """Median distance (metres) between the model's predicted page location and the page's
    actual centroid, over inlier pages — i.e. how accurate the key map's expected location is.
    """
    by_number: dict[int, list[Point]] = {}
    for d in detections:
        by_number.setdefault(d.number, []).append(d.pixel)
    residuals = []
    for i in inliers:
        page = pages[i]
        preds = [similarity_apply(model, px) for px in by_number.get(page.number, [])]
        if preds:
            residuals.append(min(math.dist(p, page.centroid) for p in preds))
    return float(np.median(residuals)) if residuals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine georeferencing of outlier/un-fit pages using the key map."
    )
    parser.add_argument(
        "keymap", type=Path, help="Key-map detections JSON (p1b.keymap.json)."
    )
    parser.add_argument("images", nargs="+", help="Page images to consider refining.")
    parser.add_argument(
        "--centerlines", type=Path, help="centerlines.geojson (default: auto)."
    )
    parser.add_argument("--radius-diagonals", type=float, default=2.0)
    parser.add_argument(
        "--scale-outlier-threshold",
        type=float,
        default=0.25,
        help="Reject a refit whose scale deviates from the volume's reference scale by more "
        "than this fraction (default 0.25, matching `mapsnap georef`). 0 disables.",
    )
    parser.add_argument(
        "--manifest", type=Path, help="Source IIIF/manifest (default: auto)."
    )
    parser.add_argument("--output", type=Path, help="Refined IIIF output path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for OCR.")
    args = parser.parse_args()

    volume = args.keymap.parent
    centerlines_path = args.centerlines or find_centerlines(volume)
    geojson = json.load(open(centerlines_path))

    pages, origin = load_georef_pages(volume)
    detections = load_detections(args.keymap)
    correspondences = build_correspondences(pages, detections)
    model, inliers = ransac(
        pages, correspondences, rng=np.random.default_rng(args.seed)
    )
    if model is None:
        sys.exit("Could not fit the key map (too few correspondences).")

    inlier_diagonals = [frame_diagonal(f) for i in inliers for f in pages[i].frames]
    if not inlier_diagonals:
        inlier_diagonals = [frame_diagonal(f) for p in pages for f in p.frames]
    radius = args.radius_diagonals * float(np.median(inlier_diagonals))

    refine_numbers = pages_to_refine(pages, detections, inliers)
    params = load_georef_params(volume)
    keymap_error = keymap_centroid_error(model, pages, inliers, detections)
    ref_scale = reference_scale_px_per_ft(pages, inliers, volume)
    ref_scale_str = f"{ref_scale:.4f} px/ft" if ref_scale else "n/a"
    print(
        f"Key map: {len(detections)} detections, {len(inliers)} inlier pages "
        f"(centroid error median {keymap_error:.0f} m); "
        f"{len(refine_numbers)} page numbers to refine; radius={radius:.0f} m; "
        f"reference scale={ref_scale_str}; params={params}",
        file=sys.stderr,
    )

    # Split panels supersede their un-split original (skip p126.jpg if p126__1.jpg exists).
    superseded = superseded_stems(volume)
    # Inputs for the 1-GCP deferred path (volume scale + neighbor rotations).
    scale_deg_per_px = px_per_ft_to_deg_per_px(ref_scale) if ref_scale else None
    neighbor_rotations = neighbor_rotations_from_inliers(pages, inliers, volume)

    reader = easyocr.Reader(["en"], gpu=not args.no_gpu, verbose=False)
    accepted_fits: list[tuple[Path, float]] = []
    rejected = skipped = 0
    for image_path in args.images:
        stem = image_stem(str(image_path))
        if stem in superseded:
            continue
        number = page_number(stem)
        if number is None or number not in refine_numbers:
            continue
        boxes = Path(image_path).parent / f"{stem}.boxes.json"
        if not boxes.exists():
            print(f"  {image_path}: no boxes.json — skipping", file=sys.stderr)
            skipped += 1
            continue
        centers = expected_centers(number, model, detections)
        if not centers:
            skipped += 1
            continue
        outcome, scale, georef2 = refine_page(
            str(image_path),
            centers,
            radius,
            geojson,
            origin,
            reader,
            str(centerlines_path),
            params,
            scale_deg_per_px,
            neighbor_rotations,
        )
        if outcome == "accepted" and georef2 is not None and scale is not None:
            accepted_fits.append((georef2, scale))
        else:
            rejected += outcome == "rejected"
            skipped += outcome == "skipped"

    # Drop scale outliers vs the refit median, mirroring `mapsnap georef`'s misscale filter.
    outlier_idx = scale_outlier_indices(
        [s for _, s in accepted_fits], ref_scale, args.scale_outlier_threshold
    )
    for i in sorted(outlier_idx):
        georef2, _ = accepted_fits[i]
        reject = georef2.with_name(
            georef2.name.replace(".georef2.json", ".georef2-reject.json")
        )
        georef2.replace(reject)
        print(f"  scale outlier: {georef2.name} -> {reject.name}", file=sys.stderr)
    scale_outliers = len(outlier_idx)
    accepted_fits = [f for i, f in enumerate(accepted_fits) if i not in outlier_idx]
    rejected += scale_outliers

    print(
        f"\nRefined: {len(accepted_fits)} accepted, "
        f"{rejected} unsalvageable ({scale_outliers} scale outliers), {skipped} skipped.",
        file=sys.stderr,
    )

    manifest = args.manifest or find_ref_iiif(volume)
    if manifest is None:
        sys.exit(f"No reference IIIF/manifest in {volume}; pass --manifest.")
    output = args.output or (volume / f"{date.today().isoformat()}-keymap.iiif.json")
    run_cmd(
        [
            "mapsnap",
            "iiif",
            str(manifest),
            str(volume / "*.georef*.json"),
            "--centerlines",
            str(centerlines_path),
            "--output",
            str(output),
        ]
    )
    print(f"Wrote refined IIIF: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
