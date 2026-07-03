"""Refine per-page georeferencing using a georeferenced key map's page locations.

The first-pass georeferencer (mapsnap.georef_from_labels) matches OCR'd street names against
the whole ``centerlines.geojson`` — an ambiguous city-wide vocabulary that makes some pages
fail to fit at all (their street names cross-match streets all over the city, so no consistent
intersection is found). A *georeferenced* key map tells us where each page sits, so for a page
that failed we:

1. map its page-number detection on the key map straight to world coordinates, using the key
   map's own georeference (``raw/p0.georef.json`` corners → bilinear pixel→lon/lat) — no
   page-footprint fit, so nothing here depends on the very page georefs we are about to write;
2. filter the centerlines to a neighborhood around that location — a much smaller, unambiguous
   street vocabulary;
3. re-run OCR over the page's cached CRAFT boxes (``reuse_boxes`` — no CRAFT, so it is fast)
   with that vocabulary. A smaller vocabulary drives up recognizer confidence on the correct
   names, so this changes the OCR step as well as the fit;
4. re-fit against the filtered centerlines.

Because the key map is georeferenced, page locations do not come from the page georefs (unlike
the earlier fit_keymap-based refiner), so there is no circularity and we overwrite
``<stem>.streets.json`` / ``<stem>.georef.json`` directly rather than writing ``.streets2`` /
``.georef2`` sidecars. A refit that lands away from the key-map location is rejected (moved to
``<stem>.georef-reject.json``), leaving the page un-fit.

    uv run mapsnap refine-with-keymap data/<vol>/raw/p0.keymap.json data/<vol>/p*.jpg
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
    GeorefPage,
    Point,
    load_detections,
    load_georef_pages,
    page_number,
    project,
    superseded_stems,
)
from mapsnap.keymap.score_keymap_labels import point_in_polygon
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


@dataclass
class KeymapGeoref:
    """A georeferenced key map's corner quad, for mapping key-map pixels to world coordinates."""

    corners: list[Point]  # image-corner lon/lat in order TL, TR, BR, BL
    width: int
    height: int


def load_keymap_georef(path: Path) -> KeymapGeoref:
    """Load the key map's georeference (corner quad) from its ``<stem>.georef.json``."""
    doc = json.load(open(path))
    corners = [(float(c[0]), float(c[1])) for c in doc["corners"]]
    return KeymapGeoref(corners, int(doc["width"]), int(doc["height"]))


def keymap_pixel_to_world(keymap: KeymapGeoref, pixel: Point) -> Point:
    """Bilinearly map a key-map pixel to (lon, lat) using the corner quad.

    Corners are TL, TR, BR, BL at image (0,0), (w,0), (w,h), (0,h); a page-number detection's
    pixel therefore maps to the real-world location of that page's area on the map.
    """
    top_left, top_right, bottom_right, bottom_left = keymap.corners
    u = pixel[0] / keymap.width
    v = pixel[1] / keymap.height
    top = (
        top_left[0] + (top_right[0] - top_left[0]) * u,
        top_left[1] + (top_right[1] - top_left[1]) * u,
    )
    bottom = (
        bottom_left[0] + (bottom_right[0] - bottom_left[0]) * u,
        bottom_left[1] + (bottom_right[1] - bottom_left[1]) * u,
    )
    return (top[0] + (bottom[0] - top[0]) * v, top[1] + (bottom[1] - top[1]) * v)


def expected_world_centers(
    number: int, detections: list[Detection], keymap: KeymapGeoref
) -> list[Point]:
    """World (lon, lat) location of every key-map detection of ``number``.

    A page number can appear several times on the key map (one per split), and we don't know
    which split is which, so all occurrences are returned and treated as a union downstream.
    """
    return [
        keymap_pixel_to_world(keymap, d.pixel) for d in detections if d.number == number
    ]


def near_any(point: Point, centers: list[Point], radius: float) -> bool:
    """Whether ``point`` is within ``radius`` of any of ``centers`` (Euclidean, metres)."""
    return any(math.dist(point, c) <= radius for c in centers)


def frame_diagonal(frame: list[Point]) -> float:
    """Bounding-box diagonal of a polygon (same units as the polygon, e.g. metres)."""
    xs = [p[0] for p in frame]
    ys = [p[1] for p in frame]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def geometry_vertices(geometry: dict) -> list[Point]:
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
        for lon, lat in geometry_vertices(feature.get("geometry", {})):
            if near_any(project(lon, lat, lon0, lat0), centers, radius):
                kept.append(feature)
                break
    return {"type": "FeatureCollection", "features": kept}


def refit_accepted(
    georef: Path,
    centers: list[Point],
    origin: Point,
    radius: float,
    *,
    strict: bool,
) -> bool:
    """Whether a refined fit is good enough to keep, given its georef.json and expected centers.

    Multi-GCP fits are well-constrained, so the loose gate (centroid within ``radius`` of an
    expected center) suffices. 1-GCP fits are weakly constrained (one anchor + assumed scale +
    voted rotation), so ``strict`` requires an expected point to land INSIDE the refined frame —
    the same in-frame test that defines a key-map inlier — which rejects plausibly-near-but-wrong
    placements.
    """
    lon0, lat0 = origin
    frame = [
        project(c[0], c[1], lon0, lat0) for c in json.load(open(georef))["corners"]
    ]
    if strict:
        return any(point_in_polygon(c, frame) for c in centers)
    centroid = (sum(p[0] for p in frame) / 4, sum(p[1] for p in frame) / 4)
    return near_any(centroid, centers, radius)


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


def neighbor_rotations(
    pages: list[GeorefPage], volume: Path
) -> list[tuple[Point, float]]:
    """(center lon/lat, rotation) for each georeferenced page — for 1-GCP rotation voting."""
    neighbors: list[tuple[Point, float]] = []
    for page in pages:
        for stem in page.piece_ids:
            path = volume / f"{stem}.georef.json"
            if path.exists():
                georef = json.load(open(path))
                corners = georef["corners"]
                center = (
                    sum(c[0] for c in corners) / len(corners),
                    sum(c[1] for c in corners) / len(corners),
                )
                neighbors.append(
                    (
                        center,
                        georef_rotation(corners, georef["width"], georef["height"]),
                    )
                )
                break
    return neighbors


def reference_scale_px_per_ft(pages: list[GeorefPage], volume: Path) -> float | None:
    """Median page scale (px/ft) over the already-georeferenced pages — the volume's scale.

    A far more reliable reference than the median of the (biased, mostly-hard) refit batch.
    """
    scales = []
    for page in pages:
        for stem in page.piece_ids:
            path = volume / f"{stem}.georef.json"
            if path.exists():
                georef = json.load(open(path))
                scales.append(
                    deg_per_px_to_px_per_ft(
                        georef_scale_deg_per_px(
                            georef["corners"], georef["width"], georef["height"]
                        )
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
        for i, scale in enumerate(scales)
        if abs(deg_per_px_to_px_per_ft(scale) / reference_px_per_ft - 1.0) > threshold
    }


def load_georef_params(volume: Path) -> dict:
    """The detection-filter parameters the volume was georeferenced with (from a georef.json)."""
    for path in sorted(volume.glob("*.georef.json")):
        params = json.load(open(path)).get("inputs", {}).get("parameters", {})
        chosen = {key: params[key] for key in PROCESS_PARAM_KEYS if key in params}
        if chosen:
            return chosen
    return dict(DEFAULT_PROCESS_PARAMS)


def unfit_located_numbers(
    pages: list[GeorefPage], detections: list[Detection]
) -> set[int]:
    """Page numbers the key map detects but that have no georeference yet (failed pages)."""
    georeferenced = {page.number for page in pages}
    detected = {detection.number for detection in detections}
    return detected - georeferenced


def refine_page(
    image_path: str,
    world_centers: list[Point],
    metre_centers: list[Point],
    radius: float,
    geojson: dict,
    origin: Point,
    reader: easyocr.Reader,
    centerlines_path: str,
    params: dict,
    scale_deg_per_px: float | None,
    neighbor_rots: list[tuple[Point, float]],
) -> tuple[str, float | None, Path | None]:
    """Re-OCR + re-fit one page against centerlines near its key-map location.

    Overwrites ``<stem>.streets.json`` (restricted-vocab re-OCR of the cached CRAFT boxes) and
    ``<stem>.georef.json`` (re-fit). Pages with a single intersection GCP are fit like
    `mapsnap georef`'s deferred path, using ``scale_deg_per_px`` (the volume scale) and
    ``neighbor_rots`` to resolve rotation; unconfirmed 1-GCP fits are rejected.

    Returns (outcome, scale_deg_per_px, georef_path): outcome is "accepted" (wrote
    <stem>.georef.json, scale set), "rejected" (fit wandered/unconfirmed/failed; wrote
    <stem>.georef-reject.json if a fit was produced), or "skipped" (no nearby streets).
    """
    lon0, lat0 = origin
    stem = image_stem(image_path)
    parent = Path(image_path).parent
    streets = parent / f"{stem}.streets.json"
    georef = parent / f"{stem}.georef.json"
    reject = parent / f"{stem}.georef-reject.json"
    one_gcp_sidecar = parent / f"{stem}.georef-1gcp.json"
    for stale in (georef, reject, one_gcp_sidecar):
        stale.unlink(missing_ok=True)

    subset = filter_centerlines(geojson, metre_centers, radius, lon0, lat0)
    block_index = build_block_index(subset)
    if not block_index:
        print(f"  {stem}: no centerlines within radius — skipping", file=sys.stderr)
        return "skipped", None, None
    cos_phi = compute_cos_phi(block_index)
    vocab = generate_vocab_strings(set(block_index.keys()))

    # Re-recognize the cached CRAFT boxes with the tighter vocabulary; overwrites streets.json.
    # Keep the original OCR so a rejected page is left exactly as we found it (only accepted
    # pages should have their streets.json replaced by the restricted-vocab read).
    original_streets = streets.read_bytes() if streets.exists() else None
    detect_text(image_path, vocab_strings=vocab, reuse_boxes=True, reader=reader)
    doc = json.load(open(streets))
    doc["refine"] = {
        "centers_lonlat": [list(center) for center in world_centers],
        "radius_m": radius,
        "streets_in_vocab": len(block_index),
    }
    with open(streets, "w") as f:
        json.dump(doc, f, indent=2)

    result = process_image(
        image_path,
        str(streets),
        str(georef),
        block_index,
        cos_phi,
        centerlines_path,
        one_gcp_fits=True,
        **params,
    )
    # A 1-GCP page is deferred by process_image; fit it with the volume scale + neighbor
    # rotation voting, exactly like `mapsnap georef` does.
    one_gcp = False
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
            neighbor_rots,
        )
        one_gcp = True

    def as_rejected() -> tuple[str, None, None]:
        """Restore the original OCR (the refine didn't take) and report rejection."""
        if original_streets is not None:
            streets.write_bytes(original_streets)
        return "rejected", None, None

    if result.success and result.center is not None:
        if refit_accepted(georef, metre_centers, origin, radius, strict=one_gcp):
            return "accepted", result.scale_deg_per_px, georef
        if georef.exists():
            georef.replace(
                reject
            )  # fit landed in the wrong place — keep for inspection
        return as_rejected()
    # Unconfirmed 1-GCP fits write a georef file but return success=False with a center;
    # keep them as reject for inspection. Hard failures wrote nothing.
    if result.center is not None and georef.exists():
        georef.replace(reject)
        return as_rejected()
    georef.unlink(missing_ok=True)
    return as_rejected()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine un-fit pages using a georeferenced key map's page locations."
    )
    parser.add_argument(
        "keymap",
        type=Path,
        help="Key-map detections JSON (e.g. raw/p0.keymap.json); its sibling "
        "<stem>.georef.json (the key map's own georeference) must exist.",
    )
    parser.add_argument("images", nargs="+", help="Page images to consider refining.")
    parser.add_argument(
        "--centerlines", type=Path, help="centerlines.geojson (default: auto)."
    )
    parser.add_argument(
        "--radius-diagonals",
        type=float,
        default=2.0,
        help="Neighborhood radius as a multiple of the median page footprint diagonal.",
    )
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
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU for OCR.")
    args = parser.parse_args()

    volume = Path(args.images[0]).parent
    keymap_georef_path = args.keymap.with_name(
        args.keymap.name.replace(".keymap.json", ".georef.json")
    )
    if not keymap_georef_path.exists():
        sys.exit(f"Key map is not georeferenced: {keymap_georef_path} not found.")

    keymap = load_keymap_georef(keymap_georef_path)
    detections = load_detections(args.keymap)
    pages, origin = load_georef_pages(volume)
    lon0, lat0 = origin

    centerlines_path = args.centerlines or find_centerlines(volume)
    geojson = json.load(open(centerlines_path))

    diagonals = [frame_diagonal(f) for p in pages for f in p.frames] or [500.0]
    radius = args.radius_diagonals * float(np.median(diagonals))

    refine_numbers = unfit_located_numbers(pages, detections)
    params = load_georef_params(volume)
    ref_scale = reference_scale_px_per_ft(pages, volume)
    ref_scale_str = f"{ref_scale:.4f} px/ft" if ref_scale else "n/a"
    scale_deg_per_px = px_per_ft_to_deg_per_px(ref_scale) if ref_scale else None
    neighbor_rots = neighbor_rotations(pages, volume)

    print(
        f"Key map: {len(detections)} detections; {len(pages)} pages already georeferenced; "
        f"{len(refine_numbers)} un-fit page numbers to refine; radius={radius:.0f} m; "
        f"reference scale={ref_scale_str}; params={params}",
        file=sys.stderr,
    )

    superseded = superseded_stems(volume)
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
        if not (Path(image_path).parent / f"{stem}.boxes.json").exists():
            print(f"  {image_path}: no boxes.json — skipping", file=sys.stderr)
            skipped += 1
            continue
        world_centers = expected_world_centers(number, detections, keymap)
        if not world_centers:
            skipped += 1
            continue
        metre_centers = [project(lon, lat, lon0, lat0) for lon, lat in world_centers]
        outcome, scale, georef = refine_page(
            str(image_path),
            world_centers,
            metre_centers,
            radius,
            geojson,
            origin,
            reader,
            str(centerlines_path),
            params,
            scale_deg_per_px,
            neighbor_rots,
        )
        if outcome == "accepted" and georef is not None and scale is not None:
            accepted_fits.append((georef, scale))
            print(f"  {stem}: recovered", file=sys.stderr)
        else:
            rejected += outcome == "rejected"
            skipped += outcome == "skipped"

    # Drop scale outliers vs the volume's reference scale, mirroring `mapsnap georef`.
    outlier_idx = scale_outlier_indices(
        [scale for _, scale in accepted_fits], ref_scale, args.scale_outlier_threshold
    )
    for i in sorted(outlier_idx):
        georef, _ = accepted_fits[i]
        reject = georef.with_name(
            georef.name.replace(".georef.json", ".georef-reject.json")
        )
        georef.replace(reject)
        print(f"  scale outlier: {georef.name} -> {reject.name}", file=sys.stderr)
    accepted_fits = [f for i, f in enumerate(accepted_fits) if i not in outlier_idx]
    rejected += len(outlier_idx)

    print(
        f"\nRefined: {len(accepted_fits)} accepted, {rejected} rejected "
        f"({len(outlier_idx)} scale outliers), {skipped} skipped.",
        file=sys.stderr,
    )

    manifest = args.manifest or find_ref_iiif(volume)
    if manifest is None:
        print(
            f"No reference IIIF/manifest in {volume}; skipping IIIF.", file=sys.stderr
        )
        return
    output = args.output or (
        volume / f"{date.today().isoformat()}-keymap-refine.iiif.json"
    )
    run_cmd(
        [
            "mapsnap",
            "iiif",
            str(manifest),
            str(volume / "*.georef.json"),
            "--centerlines",
            str(centerlines_path),
            "--output",
            str(output),
        ]
    )
    print(f"Wrote refined IIIF: {output}", file=sys.stderr)
    truth = volume / "main.iiif.json"
    if truth.exists():
        run_cmd(["mapsnap", "compare", str(truth), str(output)])


if __name__ == "__main__":
    main()
