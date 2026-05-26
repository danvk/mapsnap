"""Combine IIIF annotations with our georeferences into a new IIIF AnnotationPage.

Accepts two input formats for the source IIIF file:

  OIM AnnotationPage (type: AnnotationPage)
    Each item has a label, target.source.{id,width,height}, and body.features (GCPs).
    Split pages (labels ending in "[N]") are supported: the unsplit original is located
    via template matching to determine a non-circular canvas placement.

  LOC sc:Manifest (@type: sc:Manifest)
    Each canvas has a label like "Page 8" and an image service URL. No split pages;
    the script asserts that all matching georef page keys are unsplit.

For images where the raw file covers the full source canvas (raw_dims ≈ source_dims),
resource coordinates are computed by simple proportional scaling:

    resourceCoords = georef_pixel × source_dim / georef_dim

For OIM split sub-images (raw_dims << source_dims), the unsplit original (e.g.
p4.unsplit.jpg) is located via template matching to determine the sub-image's pixel
offset within the full canvas. Raises an error and skips the page if the unsplit file
is absent or the match score is too low.

Usage:
    python make_iiif_georef.py <main.iiif.json> <georef_glob> [--output FILE] [--creator URL]
    python make_iiif_georef.py <loc.iiif.json>  <georef_glob> [--output FILE] [--creator URL]
"""

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import mapping as geom_mapping

from mapsnap.clip_masks import compute_all_clip_masks
from mapsnap.clip_masks import geo_polygon_to_svg
from mapsnap.utils import jpeg_dimensions


def _replace_white(arr: np.ndarray, white_threshold: int = 250) -> np.ndarray:
    """Replace pure-white pixels with the mean of non-white pixels.

    White pixels (≥ white_threshold) are masked-out regions on Sanborn map splits
    that share no content with the corresponding area in the unsplit image.
    Replacing them with the template mean gives zero contribution to TM_CCOEFF_NORMED,
    preventing them from pulling the match towards the wrong location.
    """
    non_white = arr < white_threshold
    if not non_white.any():
        return arr
    mean_val = int(round(float(arr[non_white].mean())))
    out = arr.copy()
    out[~non_white] = mean_val
    return out


def locate_split_in_unsplit(
    split_path: Path,
    unsplit_path: Path,
    min_score: float = 0.90,
    coarse_downsample: int = 4,
    refine_margin: int = 20,
) -> tuple[int, int]:
    """Find the pixel offset of a split sub-image within its unsplit original.

    Uses a two-stage search: coarse normalized cross-correlation at downsampled
    resolution to find the rough location, then full-resolution refinement in a
    small window around that estimate. Pure-white pixels in the split (masked-out
    regions) are replaced with the template mean so they contribute nothing to the
    correlation score.

    Raises ValueError if the refined match score is below min_score (uncertain
    placement) or if the located region overflows the unsplit image bounds.

    Returns (offset_x, offset_y) in unsplit image pixel coordinates (top-left corner).
    """
    split_arr = _replace_white(
        np.array(Image.open(split_path).convert("L"), dtype=np.uint8)
    )
    unsplit_arr = np.array(Image.open(unsplit_path).convert("L"), dtype=np.uint8)

    ds = coarse_downsample
    split_small = split_arr[::ds, ::ds]
    unsplit_small = unsplit_arr[::ds, ::ds]

    if (
        split_small.shape[0] > unsplit_small.shape[0]
        or split_small.shape[1] > unsplit_small.shape[1]
    ):
        raise ValueError(
            f"Split ({split_arr.shape[1]}×{split_arr.shape[0]}) is larger than "
            f"unsplit ({unsplit_arr.shape[1]}×{unsplit_arr.shape[0]}) at coarse resolution"
        )

    coarse_result = cv2.matchTemplate(unsplit_small, split_small, cv2.TM_CCOEFF_NORMED)
    _, _, _, coarse_loc = cv2.minMaxLoc(coarse_result)
    coarse_x = coarse_loc[0] * ds
    coarse_y = coarse_loc[1] * ds

    # Refine at full resolution within a small window around the coarse estimate.
    sh, sw = split_arr.shape
    uh, uw = unsplit_arr.shape
    x1 = max(0, coarse_x - refine_margin)
    x2 = min(uw, coarse_x + sw + refine_margin)
    y1 = max(0, coarse_y - refine_margin)
    y2 = min(uh, coarse_y + sh + refine_margin)

    if x2 - x1 < sw or y2 - y1 < sh:
        raise ValueError(
            f"Refinement window too small ({x2 - x1}×{y2 - y1}) for template ({sw}×{sh})"
        )

    region = unsplit_arr[y1:y2, x1:x2]
    result = cv2.matchTemplate(region, split_arr, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_loc = cv2.minMaxLoc(result)

    if max_score < min_score:
        raise ValueError(
            f"Template match score {max_score:.3f} < threshold {min_score} "
            f"({split_path.name} in {unsplit_path.name})"
        )

    offset_x = x1 + max_loc[0]
    offset_y = y1 + max_loc[1]

    if offset_x + sw > uw or offset_y + sh > uh:
        raise ValueError(
            f"Located position ({offset_x}, {offset_y}) + {sw}×{sh} overflows "
            f"unsplit bounds {uw}×{uh}"
        )

    return offset_x, offset_y


def georef_path_to_page_key(path: str) -> str | None:
    """Extract page key like 'p428__2' from a georef filename.

    Accepts filenames ending in '_p16s.georef.json', '_p16.georef.json', or
    '_p16s.gcps.georef.json' (the '.gcps' infix is optional).
    """
    m = re.search(r"(?:\b|_)(p\d+[snew]?(?:__\d+)?)(?:\.[^.]+)?\.georef\.json$", path)
    return m.group(1) if m else None


def _service_url_to_page_key(url: str) -> str | None:
    """Extract the page key from a LOC IIIF service URL (OIM or LOC manifest format).

    The page key is the trailing segment after the last "-", with leading zeros
    stripped and any letter suffix lowercased:
      "...:01790_01N_1950-0006N/info.json" → "p6n"
      "...:01790_01N_1950-0103W"           → "p103w"
      "...:05791_02_1939-0027s"            → "p27s"

    Non-sheet URLs (covers, indexes: "...-covr", "...-titl", etc.) start with a
    letter after the "-" and return None.
    """
    url = url.removesuffix("/info.json")
    m = re.search(r"-0*(\d+)([a-z]*)$", url, re.IGNORECASE)
    if m is None:
        return None
    return f"p{m.group(1)}{m.group(2).lower()}"


def _load_oim_index(data: dict) -> dict[str, dict]:
    """Build page_key → item dict from an OIM IIIF AnnotationPage.

    Labels ending with "[N]" (e.g. "Page 4 [1]") denote split sub-images; the
    split number is appended to the page key as "__N" (e.g. "p4__1") to match
    the naming convention used by georef_path_to_page_key.
    """
    index: dict[str, dict] = {}
    for item in data.get("items", []):
        source_id: str = item.get("target", {}).get("source", {}).get("id", "")
        page_key = _service_url_to_page_key(source_id)
        if page_key is None:
            continue
        label: str = item.get("label", "")
        m = re.search(r"\[(\d+)\]$", label)
        if m:
            page_key += f"__{m.group(1)}"
        index[page_key] = item
    return index


def _load_loc_index(data: dict, raw_dir: Path) -> dict[str, dict]:
    """Build page_key → normalized item dict from a LOC sc:Manifest.

    Normalizes each canvas into the same shape make_annotation expects:
      {"label": str, "target": {"source": {"id": str, "width": int, "height": int}}}

    Full image dimensions come from the raw.jpg on disk when present; otherwise the
    pct thumbnail scale factor is parsed from the resource URL and applied.
    """
    manifest_label: str = data.get("label", "")
    canvases: list[dict] = data["sequences"][0]["canvases"]
    index: dict[str, dict] = {}
    for canvas in canvases:
        canvas_label: str = canvas.get("label", "")
        resource: dict = canvas["images"][0]["resource"]
        service_id: str = resource["service"]["@id"]
        page_key = _service_url_to_page_key(service_id)
        if page_key is None:
            continue

        raw_path = raw_dir / f"{page_key}.raw.jpg"
        if raw_path.exists():
            source_width, source_height = jpeg_dimensions(raw_path)
        else:
            # Resource URL contains e.g. "/full/pct:25/0/default.jpg"; scale up.
            pct_m = re.search(r"/pct:(\d+)/", resource.get("@id", ""))
            scale = 100.0 / int(pct_m.group(1)) if pct_m else 1.0
            source_width = int(round(resource["width"] * scale))
            source_height = int(round(resource["height"] * scale))

        index[page_key] = {
            "label": f"{manifest_label} | {canvas_label}",
            "target": {
                "source": {
                    "id": f"{service_id}/info.json",
                    "type": "ImageService2",
                    "width": source_width,
                    "height": source_height,
                }
            },
        }
    return index


def _georef_metadata(
    georef: dict,
    split_canvas: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Build IIIF metadata entries for streets, intersections, and split canvas bounds.

    split_canvas, when present, gives the sub-image region within the full canvas as
    (x, y, w, h) in canvas pixel coordinates, derived via template matching.
    """
    n_streets = len(
        set(s["street"] for s in georef.get("streets", []) if s.get("inlier"))
    )
    n_intersections = sum(1 for i in georef.get("intersections", []) if i.get("inlier"))
    entries: list[dict] = [
        {"label": "streets", "value": str(n_streets)},
        {"label": "intersections", "value": str(n_intersections)},
    ]
    if split_canvas is not None:
        x, y, w, h = split_canvas
        entries += [
            {"label": "split_canvas_x", "value": str(x)},
            {"label": "split_canvas_y", "value": str(y)},
            {"label": "split_canvas_w", "value": str(w)},
            {"label": "split_canvas_h", "value": str(h)},
        ]
    return entries


def _two_gcp_affine(
    p1_pixel: tuple[float, float],
    p1_geo: tuple[float, float],
    p2_pixel: tuple[float, float],
    p2_geo: tuple[float, float],
) -> np.ndarray:
    """Compute a 2×3 similarity affine from exactly 2 (pixel, geo) point pairs.

    Enforces equal metric scale in x/y (no shear). cos(lat) is estimated from
    the mean latitude of the two points. Returns A where [lon, lat]^T = A @ [px, py, 1]^T.
    """
    cos_phi = float(np.cos(np.radians((p1_geo[1] + p2_geo[1]) / 2.0)))
    dx = p2_pixel[0] - p1_pixel[0]
    dy = p2_pixel[1] - p1_pixel[1]
    dlon = p2_geo[0] - p1_geo[0]
    dlat = p2_geo[1] - p1_geo[1]
    det = dx * dx + dy * dy
    alpha = (dx * dlon * cos_phi - dy * dlat) / det
    beta = (dy * dlon * cos_phi + dx * dlat) / det
    tx = p1_geo[0] - (alpha * p1_pixel[0] + beta * p1_pixel[1]) / cos_phi
    ty = p1_geo[1] - beta * p1_pixel[0] + alpha * p1_pixel[1]
    return np.array([[alpha / cos_phi, beta / cos_phi, tx], [beta, -alpha, ty]])


GcpPoint = tuple[tuple[float, float], tuple[float, float], str]


def _corner_fallback(
    corners: list,
    width: int,
    height: int,
    intersections: list[dict],
) -> list[GcpPoint]:
    """Return 4 image-corner GCPs plus any detected intersection GCPs.

    Used as a fallback when there are fewer than 2 non-coincident initial
    intersections and a full affine fit is not possible.
    """
    w, h = float(width), float(height)
    corner_pts: list[GcpPoint] = [
        ((0.0, 0.0), (float(corners[0][0]), float(corners[0][1])), "corner"),
        ((w, 0.0), (float(corners[1][0]), float(corners[1][1])), "corner"),
        ((w, h), (float(corners[2][0]), float(corners[2][1])), "corner"),
        ((0.0, h), (float(corners[3][0]), float(corners[3][1])), "corner"),
    ]
    intersection_pts: list[GcpPoint] = [
        ((float(i["x"]), float(i["y"])), (float(i["lon"]), float(i["lat"])), "gcp")
        for i in intersections
    ]
    return corner_pts + intersection_pts


def georef_gcp_points(
    georef: dict,
) -> list[GcpPoint]:
    """Return (pixel, geo, type) triples for the GCPs to embed in the IIIF annotation.

    Uses the two "initial" intersections from the RANSAC fit plus a third point
    selected in priority order:
      1. Another intersection formed by streets already in the initial pair set.
      2. Any other intersection in the georef, preferring inliers.
      3. A perpendicular offset from the midpoint of the two initial intersections.
    Among multiple candidates in each tier, the one closest to the image center
    is chosen.

    The third point's geo coords are computed by projecting its pixel position
    through the similarity affine defined by the two initial GCPs alone, so it
    carries no independent information and cannot shift the fitted transform.

    Type values: "gcp" for the two initial seed intersections that determine the fit,
    "intersection" for the third detected intersection (options 1 & 2), "projected" for
    the synthetic perpendicular third point (option 3), "corner" for image-corner fallback.

    Falls back to 4 image corners (type "corner") plus any inlier intersections
    (type "gcp") when fewer than 2 non-coincident initial intersections exist
    (e.g. georefs produced by deferred single-GCP processing).
    """
    width: int = georef["width"]
    height: int = georef["height"]
    corners: list = georef["corners"]
    image_center = np.array([width / 2.0, height / 2.0])

    all_intersections: list[dict] = georef.get("intersections", [])
    inlier_intersections = [i for i in all_intersections if i.get("inlier")]
    initials = [i for i in all_intersections if i.get("initial")]
    if len(initials) < 2:
        return _corner_fallback(corners, width, height, inlier_intersections)

    int1, int2 = initials[0], initials[1]
    p1_pixel = (float(int1["x"]), float(int1["y"]))
    p2_pixel = (float(int2["x"]), float(int2["y"]))
    p1_geo = (float(int1["lon"]), float(int1["lat"]))
    p2_geo = (float(int2["lon"]), float(int2["lat"]))

    p1 = np.array(p1_pixel)
    p2 = np.array(p2_pixel)
    if float(np.linalg.norm(p2 - p1)) < 1.0:
        return _corner_fallback(corners, width, height, inlier_intersections)

    streets_set = {int1["label_a"], int1["label_b"], int2["label_a"], int2["label_b"]}
    initial_pairs = {
        frozenset({int1["label_a"], int1["label_b"]}),
        frozenset({int2["label_a"], int2["label_b"]}),
    }

    def dist_to_center(x: float, y: float) -> float:
        return float(np.linalg.norm(np.array([x, y]) - image_center))

    # Non-collinearity helpers shared by options 1 and 2.
    dist12 = float(np.linalg.norm(p2 - p1))
    d_unit = (p2 - p1) / dist12
    min_offset = 0.25 * dist12

    def perp_dist(x: float, y: float) -> float:
        v = np.array([x, y]) - p1
        return abs(float(v[0] * d_unit[1] - v[1] * d_unit[0]))

    # Option 1: intersection formed by streets already in the initial pair set.
    # Non-collinearity check prevents degenerate cases where an alias pair (e.g.
    # "KORTE AVENUE x PHILIP STREET" when "KORTE AVENUE" and "KORTE STREET" are the
    # same physical street) duplicates an existing initial GCP pixel position.
    option1 = [
        (float(i["x"]), float(i["y"]))
        for i in all_intersections
        if frozenset({i["label_a"], i["label_b"]}) not in initial_pairs
        and i["label_a"] in streets_set
        and i["label_b"] in streets_set
        and perp_dist(float(i["x"]), float(i["y"])) >= min_offset
    ]

    p3_type: str
    if option1:
        p3_x, p3_y = min(option1, key=lambda c: dist_to_center(*c))
        p3_type = "intersection"
    else:
        # Option 2: any other intersection; inliers preferred, then closest to center.
        # Only consider candidates offset from the P1-P2 line by ≥ 25% of dist(P1, P2)
        # to avoid near-collinear third points that degrade the affine fit.
        others = [
            (float(i["x"]), float(i["y"]), bool(i.get("inlier")))
            for i in all_intersections
            if frozenset({i["label_a"], i["label_b"]}) not in initial_pairs
        ]
        inliers = [
            (x, y)
            for x, y, inlier in others
            if inlier and perp_dist(x, y) >= min_offset
        ]
        non_inliers = [
            (x, y)
            for x, y, inlier in others
            if not inlier and perp_dist(x, y) >= min_offset
        ]
        pool = inliers if inliers else non_inliers

        if pool:
            p3_x, p3_y = min(pool, key=lambda c: dist_to_center(*c))
            p3_type = "intersection"
        else:
            # Option 3: perpendicular offset from midpoint.
            diff = p2 - p1
            perp = np.array([-diff[1], diff[0]])
            mid = (p1 + p2) / 2.0
            p3_a, p3_b = mid + perp, mid - perp
            p3 = (
                p3_a
                if float(np.linalg.norm(p3_a - image_center))
                <= float(np.linalg.norm(p3_b - image_center))
                else p3_b
            )
            p3_x = float(np.clip(p3[0], 0.0, float(width)))
            p3_y = float(np.clip(p3[1], 0.0, float(height)))
            p3_type = "projected"

    # Project P3 through the 2-GCP similarity to get its geo coords.
    A = _two_gcp_affine(p1_pixel, p1_geo, p2_pixel, p2_geo)
    p3_geo = A @ np.array([p3_x, p3_y, 1.0])

    return [
        (p1_pixel, p1_geo, "gcp"),
        (p2_pixel, p2_geo, "gcp"),
        ((p3_x, p3_y), (float(p3_geo[0]), float(p3_geo[1])), p3_type),
    ]


def make_annotation(
    item: dict,
    georef: dict,
    raw_path: Path,
    creator_url: str,
    now: str,
    geo_mask: ShapelyPolygon | None = None,
) -> dict:
    """Build a IIIF georeferencing annotation for one page.

    item must have {"label": str, "target": {"source": {"id", "width", "height"}}};
    both OIM AnnotationPage items and normalized LOC canvas items satisfy this.

    For full-canvas images (raw_dims ≈ source_dims), resource coords are computed
    by simple proportional scaling. For split sub-images, the unsplit original
    (e.g. p4.unsplit.jpg) is located via template matching to determine the non-circular
    canvas placement. Raises ValueError if the unsplit file is missing or the match
    is too uncertain.

    geo_mask, when provided, is a Shapely Polygon in geographic (lon/lat) space used
    as the SvgSelector clipping polygon. Falls back to the full-page rectangle if None.
    """
    source = item["target"]["source"]
    source_id: str = source["id"]
    source_width: int = source["width"]
    source_height: int = source["height"]
    label: str = item["label"]

    # Derive a unique canvas ID from the source URL; append split number if present.
    canvas_id = source_id.removesuffix("/info.json")
    m = re.search(r"\[(\d+)\]$", label)
    if m:
        canvas_id += f"__{m.group(1)}"

    creator = {"id": creator_url, "type": "Person"}
    georef_width = georef["width"]
    georef_height = georef["height"]

    if not raw_path.exists():
        raise ValueError(f"raw image not found: {raw_path}")
    raw_width, raw_height = jpeg_dimensions(raw_path)

    # Use simple scaling when the raw image covers the full canvas (within 2px tolerance).
    is_full_canvas = (
        abs(raw_width - source_width) <= 2 and abs(raw_height - source_height) <= 2
    )
    split_canvas: tuple[float, float, float, float] | None = None

    gcp_pts = georef_gcp_points(georef)

    if is_full_canvas:
        scale_x = source_width / georef_width
        scale_y = source_height / georef_height
        resource_coords_list = [
            [round(px * scale_x, 1), round(py * scale_y, 1)]
            for (px, py), _, _ in gcp_pts
        ]
    else:
        # True sub-image: locate via template matching for a non-circular placement.
        page_key_from_raw = raw_path.name.removesuffix(".raw.jpg")
        if "__" in page_key_from_raw:
            base_key = page_key_from_raw.split("__")[0]
            unsplit_path = raw_path.parent / f"{base_key}.unsplit.jpg"
        else:
            unsplit_path = None

        if unsplit_path is None or not unsplit_path.exists():
            missing = unsplit_path.name if unsplit_path else "unsplit file"
            raise ValueError(
                f"{missing} not found; cannot place sub-image non-circularly"
            )

        # locate_split_in_unsplit raises ValueError if the match is uncertain.
        offset_x_px, offset_y_px = locate_split_in_unsplit(raw_path, unsplit_path)
        unsplit_width, unsplit_height = jpeg_dimensions(unsplit_path)

        # Scale from unsplit pixel coordinates to canvas coordinates.
        scale_x = source_width / unsplit_width
        scale_y = source_height / unsplit_height

        split_cx = offset_x_px * scale_x
        split_cy = offset_y_px * scale_y
        split_cw = raw_width * scale_x
        split_ch = raw_height * scale_y
        split_canvas = (
            round(split_cx, 1),
            round(split_cy, 1),
            round(split_cw, 1),
            round(split_ch, 1),
        )

        resource_coords_list = [
            [
                round(split_cx + px * split_cw / georef_width, 1),
                round(split_cy + py * split_ch / georef_height, 1),
            ]
            for (px, py), _, _ in gcp_pts
        ]

    features = [
        {
            "type": "Feature",
            "properties": {
                "resourceCoords": rc,
                "creator": creator,
                "type": gcp_type,
            },
            "geometry": {
                "type": "Point",
                "coordinates": list(geo),
            },
        }
        for rc, (_, geo, gcp_type) in zip(resource_coords_list, gcp_pts)
    ]

    return {
        "id": f"{canvas_id}/georef",
        "type": "Annotation",
        "@context": [
            "http://iiif.io/api/extension/georef/1/context.json",
            "http://iiif.io/api/presentation/3/context.json",
        ],
        "label": label,
        "metadata": _georef_metadata(georef, split_canvas),
        "created": now,
        "modified": now,
        "creator": [creator],
        "motivation": "georeferencing",
        "target": {
            "id": f"{canvas_id}/selector",
            "type": "SpecificResource",
            "source": {
                "id": source_id,
                "type": "ImageService2",
                "height": source_height,
                "width": source_width,
            },
            "selector": {
                "type": "SvgSelector",
                "value": geo_polygon_to_svg(
                    geo_mask, georef, source_width, source_height, split_canvas
                ),
            },
        },
        "body": {
            "id": f"{canvas_id}/gcps",
            "type": "FeatureCollection",
            "transformation": {
                "type": "polynomial",
                "options": {"order": 1},
            },
            "features": features,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine OIM IIIF annotations with our georeferences into an IIIF AnnotationPage. "
            "Each georef file is matched to the OIM annotation by page key parsed from the label."
        )
    )
    parser.add_argument(
        "oim_iiif",
        metavar="OIM_IIIF",
        help="OIM IIIF AnnotationPage JSON file (e.g. main.iiif.json)",
    )
    parser.add_argument(
        "georef_glob",
        metavar="GEOREF_GLOB",
        help="Glob pattern matching georef JSON files (e.g. 'path/to/*.georef.json')",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write output to this file (default: stdout)",
    )
    parser.add_argument(
        "--creator",
        metavar="URL",
        default="https://oldinsurancemaps.net/profile/danvk",
        help="Creator profile URL (default: %(default)s)",
    )
    parser.add_argument(
        "--centerlines",
        metavar="FILE",
        help="GeoJSON centerlines file for block-based clipping masks",
    )
    parser.add_argument(
        "--debug-blocks",
        metavar="FILE",
        help="Write a GeoJSON file with one Polygon feature per block (requires --centerlines)",
    )
    parser.add_argument(
        "--debug-clip",
        metavar="FILE",
        help="Write a GeoJSON file with one Polygon feature per clipping mask (requires --centerlines)",
    )
    args = parser.parse_args()

    source_data: dict = json.load(open(args.oim_iiif))

    georef_paths = sorted(glob.glob(args.georef_glob))
    if not georef_paths:
        print(f"Error: no files matched '{args.georef_glob}'.", file=sys.stderr)
        sys.exit(1)
    raw_dir = Path(georef_paths[0]).parent

    # Detect format and build the page-key → canvas-item index.
    if source_data.get("type") == "AnnotationPage":
        items_by_key = _load_oim_index(source_data)
        result_id = source_data.get("id", "") + "/generated"
        is_loc = False
        print(f"Loaded {len(items_by_key)} OIM annotations.", file=sys.stderr)
    elif source_data.get("@type") == "sc:Manifest":
        items_by_key = _load_loc_index(source_data, raw_dir)
        result_id = source_data.get("@id", "") + "/generated"
        is_loc = True
        print(f"Loaded {len(items_by_key)} LOC canvases.", file=sys.stderr)
    else:
        print(
            "Error: expected an OIM IIIF AnnotationPage (type: AnnotationPage) "
            "or a LOC manifest (@type: sc:Manifest).",
            file=sys.stderr,
        )
        sys.exit(1)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Pass 1: collect all valid (page_key, canvas_item, georef, raw_path) tuples.
    valid_items: list[tuple[str, dict, dict, Path]] = []
    for path in georef_paths:
        page_key = georef_path_to_page_key(path)
        if not page_key:
            print(f"Warning: could not parse page key from '{path}'", file=sys.stderr)
            continue
        if is_loc:
            assert "__" not in page_key, (
                f"LOC manifests have no split pages; found split key '{page_key}' in {path}"
            )
        canvas_item = items_by_key.get(page_key)
        if not canvas_item:
            print(
                f"Warning: no canvas for page key '{page_key}' ({path})",
                file=sys.stderr,
            )
            continue
        georef = json.load(open(path))
        raw_path = Path(path).parent / f"{page_key}.raw.jpg"
        valid_items.append((page_key, canvas_item, georef, raw_path))

    # Compute block-based clipping masks when a centerlines file is provided.
    geo_masks: list[ShapelyPolygon | None] = [None] * len(valid_items)
    if args.centerlines:
        centerlines_geojson: dict = json.load(open(args.centerlines))
        all_georefs = [georef for _, _, georef, _ in valid_items]
        print("Computing block-based clipping masks...", file=sys.stderr)
        debug_blocks: list[dict] | None = [] if args.debug_blocks else None
        geo_masks = compute_all_clip_masks(
            all_georefs, centerlines_geojson, debug_blocks_out=debug_blocks
        )
        n_masked = sum(m is not None for m in geo_masks)
        print(
            f"Computed masks for {n_masked}/{len(valid_items)} pages.", file=sys.stderr
        )
        if debug_blocks is not None:
            blocks_geojson = {"type": "FeatureCollection", "features": debug_blocks}
            with open(args.debug_blocks, "w") as f:
                json.dump(blocks_geojson, f)
            print(
                f"Wrote {len(debug_blocks)} block features to {args.debug_blocks}",
                file=sys.stderr,
            )
        if args.debug_clip:
            clip_features = [
                {
                    "type": "Feature",
                    "properties": {"page_key": page_key},
                    "geometry": geom_mapping(mask),
                }
                for (page_key, _, _, _), mask in zip(valid_items, geo_masks)
                if mask is not None
            ]
            with open(args.debug_clip, "w") as f:
                json.dump({"type": "FeatureCollection", "features": clip_features}, f)
            print(
                f"Wrote {len(clip_features)} clip features to {args.debug_clip}",
                file=sys.stderr,
            )

    # Pass 2: build annotations with the per-page geo mask.
    annotations: list[dict] = []
    for (page_key, canvas_item, georef, raw_path), geo_mask in zip(
        valid_items, geo_masks
    ):
        try:
            annotations.append(
                make_annotation(
                    canvas_item, georef, raw_path, args.creator, now, geo_mask
                )
            )
        except ValueError as exc:
            print(f"Warning: skipping {page_key}: {exc}", file=sys.stderr)

    print(
        f"Generated {len(annotations)} annotations from {len(georef_paths)} georef files.",
        file=sys.stderr,
    )

    result = {
        "id": result_id,
        "type": "AnnotationPage",
        "@context": ["http://www.w3.org/ns/anno.jsonld"],
        "label": source_data.get("label", ""),
        "items": annotations,
    }

    out_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out_json)
        print(f"Wrote to {args.output}", file=sys.stderr)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
