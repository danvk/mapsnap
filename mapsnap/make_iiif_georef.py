"""Combine IIIF annotations with our georeferences into a new IIIF AnnotationPage.

Accepts two input formats for the source IIIF file:

  OIM AnnotationPage (type: AnnotationPage)
    Each item has a label, target.source.{id,width,height}, and body.features (GCPs).

  LOC sc:Manifest (@type: sc:Manifest)
    Each canvas has a label like "Page 8" and an image service URL.

Georefs are computed in the 25%-scale page-image pixel frame (pN.jpg). Resource
coordinates are mapped into the full source canvas by proportional scaling:

    resourceCoords = georef_pixel × source_dim / georef_dim

Split pages (georef key pN__i) are placed within the parent page's canvas using the
panel polygon recorded in pN.panels.json: the panel's bounding box (in the pN.jpg frame)
is scaled to canvas coordinates to give the sub-image's offset and extent. The split
panels of a page all share the parent page's full canvas (matching how OIM expresses
split annotations).

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

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import mapping as geom_mapping

from mapsnap.clip_masks import compute_all_clip_masks, geo_polygon_to_svg
from mapsnap.split import panels_json_path, read_panels_json
from mapsnap.utils import jpeg_dimensions

# Page images are stored at 25% scale, so the full-resolution canvas is 4× larger.
FULL_RES_FACTOR = 4


def georef_path_to_page_key(path: str) -> str | None:
    """Extract page key like 'p428__2' from a georef filename.

    Accepts filenames ending in '_p16s.georef.json', '_p16.georef.json', or
    '_p16s.gcps.georef.json' (the '.gcps' infix is optional). Also accepts the
    refined '.georef2.json' form; tossed/reject variants (e.g. '.georef-misscale.json',
    '.georef2-reject.json') do not match and are ignored.
    """
    m = re.search(r"(?:\b|_)(p\d+[snew]?(?:__\d+)?)(?:\.[^.]+)?\.georef2?\.json$", path)
    return m.group(1) if m else None


def _service_url_to_page_key(url: str) -> str | None:
    """Extract the page key from a LOC IIIF service URL (OIM or LOC manifest format).

    The page key is the trailing segment after the last "-", with leading zeros
    stripped and any letter suffix lowercased:
      "...:01790_01N_1950-0006N/info.json" → "p6n"
      "...:01790_01N_1950-0103W"           → "p103w"
      "...:05791_02_1939-0027s"            → "p27s"

    Sanborn sb-format (e.g. Washington DC 1916):
      "...:sb001250"                        → "p125"
      "...:sb00154s"                        → "p154s"

    Non-sheet URLs (covers, indexes: "...-covr", "...-titl", etc.) start with a
    letter after the "-" and return None.
    """
    url = url.removesuffix("/info.json")
    # Sanborn sb-format: service:...:sb{5-digit page}{suffix char}
    m = re.search(r":sb(\d{5})([a-z0-9])$", url, re.IGNORECASE)
    if m:
        page_num = int(m.group(1))
        suffix_char = m.group(2).lower()
        suffix = "" if suffix_char == "0" else suffix_char
        return f"p{page_num}{suffix}"
    m = re.search(r"-0*(\d+)([a-z]*)$", url, re.IGNORECASE)
    if m is None:
        return None
    return f"p{m.group(1)}{m.group(2).lower()}"


def _load_oim_index(data: dict) -> dict[str, dict]:
    """Build parent-page_key → item dict from an OIM IIIF AnnotationPage.

    Items are keyed by the unsplit page key (e.g. "p4"), dropping any "[N]" split
    suffix in the label. OIM expresses each split annotation in the parent page's full
    canvas (all splits of a page share the same source id and width/height), so a single
    full-canvas item per page is all make_annotation needs; our own splits are placed
    within it via panels.json.
    """
    index: dict[str, dict] = {}
    for item in data.get("items", []):
        source_id: str = item.get("target", {}).get("source", {}).get("id", "")
        page_key = _service_url_to_page_key(source_id)
        if page_key is None:
            continue
        index[page_key] = item
    return index


def _load_loc_index(data: dict) -> dict[str, dict]:
    """Build page_key → normalized item dict from a LOC sc:Manifest.

    Normalizes each canvas into the same shape make_annotation expects:
      {"label": str, "target": {"source": {"id": str, "width": int, "height": int}}}

    The manifest's resource dimensions are at the requested pct scale (we download at
    pct:25), so the full-resolution source dimensions are recovered by scaling up by
    100/pct (e.g. ×4 for pct:25).
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

        # Resource URL contains e.g. "/full/pct:25/0/default.jpg"; scale up to full res.
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
    (x, y, w, h) in canvas pixel coordinates, derived from the panel polygon.
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


def georef_gcp_points(georef: dict) -> list[GcpPoint]:
    """Return (pixel, geo, type) triples for the GCPs to embed in the IIIF annotation.

    For georefs with two non-coincident initial (RANSAC seed) intersections, returns
    exactly those two points with type "gcp". The caller uses a Helmert transformation,
    which only requires two points.

    Falls back to 4 image corners (type "corner") plus any inlier intersections
    (type "gcp") when fewer than 2 non-coincident initial intersections exist
    (e.g. georefs produced by deferred single-GCP processing). The caller uses a
    polynomial transformation for these.
    """
    corners: list = georef["corners"]
    width: int = georef["width"]
    height: int = georef["height"]

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

    if float(np.linalg.norm(np.array(p2_pixel) - np.array(p1_pixel))) < 1.0:
        return _corner_fallback(corners, width, height, inlier_intersections)

    return [
        (p1_pixel, p1_geo, "gcp"),
        (p2_pixel, p2_geo, "gcp"),
    ]


def make_annotation(
    item: dict,
    georef: dict,
    page_key: str,
    image_path: Path,
    creator_url: str,
    now: str,
    geo_mask: ShapelyPolygon | None = None,
) -> dict:
    """Build a IIIF georeferencing annotation for one page.

    item must have {"label": str, "target": {"source": {"id", "width", "height"}}};
    both OIM AnnotationPage items and normalized LOC canvas items satisfy this. source
    width/height are the full-resolution canvas dimensions.

    page_key is the georef page key (e.g. "p4" or split "p4__2"). The georef ran on the
    25%-scale page image, so full-page coordinates are scaled up to the full canvas. For
    split pages, the parent page's pN.panels.json is read and panel N's bounding box (in
    the pN.jpg frame) is scaled to canvas coordinates to place the sub-image. Raises
    ValueError if the panels.json is missing or has too few panels.

    image_path is the georeferenced image (e.g. dir/p4__2.jpg); its directory is used to
    locate the parent panels.json.

    geo_mask, when provided, is a Shapely Polygon in geographic (lon/lat) space used
    as the SvgSelector clipping polygon. Falls back to the full-page rectangle if None.
    """
    source = item["target"]["source"]
    source_id: str = source["id"]
    source_type: str = source.get("type", "ImageService2")
    source_width: int = source["width"]
    source_height: int = source["height"]
    label: str = item["label"]

    creator = {"id": creator_url, "type": "Person"}
    georef_width = georef["width"]
    georef_height = georef["height"]

    # Derive a unique canvas ID from the source URL; append split number if present.
    canvas_id = source_id.removesuffix("/info.json")
    split_index: int | None = None
    if "__" in page_key:
        split_index = int(page_key.split("__")[1])
        canvas_id += f"__{split_index}"

    gcp_pts = georef_gcp_points(georef)
    split_canvas: tuple[float, float, float, float] | None = None

    if split_index is None:
        # Full page: scale 25%-page coordinates up to the full canvas.
        scale_x = source_width / georef_width
        scale_y = source_height / georef_height
        resource_coords_list = [
            [round(px * scale_x, 1), round(py * scale_y, 1)]
            for (px, py), _, _ in gcp_pts
        ]
    else:
        # Split page: place the panel within the parent canvas using panels.json.
        parent_key = page_key.split("__")[0]
        panels_path = panels_json_path(image_path.parent / f"{parent_key}.jpg")
        if not panels_path.exists():
            raise ValueError(f"panels.json not found: {panels_path}")
        panels_data = read_panels_json(panels_path)
        panels = panels_data["panels"]
        if not (1 <= split_index <= len(panels)):
            raise ValueError(
                f"split index {split_index} out of range for {panels_path.name} "
                f"({len(panels)} panels)"
            )
        ring = panels[split_index - 1]
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)

        # panels.json records polygons in the pN.jpg (25%) frame; scale to full canvas.
        page_scale_x = source_width / panels_data["width"]
        page_scale_y = source_height / panels_data["height"]
        split_cx = minx * page_scale_x
        split_cy = miny * page_scale_y
        split_cw = (maxx - minx) * page_scale_x
        split_ch = (maxy - miny) * page_scale_y
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
                "type": source_type,
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
            "transformation": (
                {"type": "helmert"}
                if len(gcp_pts) == 2
                else {"type": "polynomial", "options": {"order": 1}}
            ),
            "features": features,
        },
    }


def _load_s3_items(
    georef_glob_pattern: str,
    image_base_url: str,
    source_type: str = "ImageService3",
) -> tuple[list[tuple[str, dict, dict, Path, Path]], str, str]:
    """Load volume items for self-hosted images (no IIIF manifest needed).

    Constructs canvas items directly from the 25%-scale page image dimensions on disk
    and a base URL, so no LOC or OIM manifest file is required. The full-resolution
    canvas dimensions are 4× the page image; split pages share their parent's canvas.

    source_type should match the image server in use:
      'ImageService3' — IIIF Image API v3 (e.g. serverless-iiif v3 on Lambda)
      'ImageService2' — IIIF Image API v2
      'Image'         — plain static JPEG (no tile server; limited viewer support)

    The image URL for each page is: {image_base_url}/{parent_key}.jpg
    """
    georef_paths = sorted(glob.glob(georef_glob_pattern))
    if not georef_paths:
        print(f"Error: no files matched '{georef_glob_pattern}'.", file=sys.stderr)
        sys.exit(1)

    base_url = image_base_url.rstrip("/")
    valid_items: list[tuple[str, dict, dict, Path, Path]] = []
    for path in georef_paths:
        page_key = georef_path_to_page_key(path)
        if not page_key:
            print(f"Warning: could not parse page key from '{path}'", file=sys.stderr)
            continue
        # Splits share their parent page's full canvas.
        parent_key = page_key.split("__")[0]
        image_path = Path(path).parent / f"{page_key}.jpg"
        parent_image = Path(path).parent / f"{parent_key}.jpg"
        if not parent_image.exists():
            print(f"Warning: page image not found: {parent_image}", file=sys.stderr)
            continue
        # The page image is at 25% scale; the full-resolution canvas is 4× larger.
        page_w, page_h = jpeg_dimensions(parent_image)
        source_width = page_w * FULL_RES_FACTOR
        source_height = page_h * FULL_RES_FACTOR
        canvas_item = {
            "label": page_key,
            "target": {
                "source": {
                    "id": f"{base_url}/{parent_key}.jpg",
                    "type": source_type,
                    "width": source_width,
                    "height": source_height,
                }
            },
        }
        georef = json.load(open(path))
        valid_items.append((page_key, canvas_item, georef, image_path, Path(path)))

    # Drop skeleton pages within this volume.
    non_skeleton_keys = {pk for pk, _, _, _, _ in valid_items if not pk.endswith("s")}
    skipped = [
        pk
        for pk, _, _, _, _ in valid_items
        if pk.endswith("s") and pk[:-1] in non_skeleton_keys
    ]
    if skipped:
        print(
            f"Dropping {len(skipped)} skeleton page(s) with full-color counterparts: "
            + ", ".join(skipped),
            file=sys.stderr,
        )
        valid_items = [item for item in valid_items if item[0] not in set(skipped)]

    print(
        f"Loaded {len(valid_items)} pages from {len(georef_paths)} georef files.",
        file=sys.stderr,
    )
    return valid_items, base_url, ""


def _load_volume_items(
    iiif_path: str,
    georef_glob_pattern: str,
) -> tuple[list[tuple[str, dict, dict, Path, Path]], str, str]:
    """Load one volume's valid items after per-volume skeleton deduplication.

    Accepts an OIM AnnotationPage or LOC sc:Manifest as the source IIIF, plus a
    glob pattern for georef JSON files. Loads each georef, matches it to a canvas
    item in the manifest, and drops skeleton pages (keys ending in 's') that have
    a full-color counterpart within this volume.

    Returns (valid_items, result_id, label) where valid_items is a list of
    (page_key, canvas_item, georef, raw_path, georef_path) tuples ready for annotation building.
    """
    source_data: dict = json.load(open(iiif_path))

    georef_paths = sorted(glob.glob(georef_glob_pattern))
    if not georef_paths:
        print(f"Error: no files matched '{georef_glob_pattern}'.", file=sys.stderr)
        sys.exit(1)

    # Prefer a refined '<stem>.georef2.json' over the original '<stem>.georef.json' for the
    # same page, and ignore unparseable paths (tossed/reject variants). This lets a caller
    # pass a broad glob like '*.georef*.json' and get the refined fit where one exists.
    best_for_key: dict[str, str] = {}
    for path in georef_paths:
        key = georef_path_to_page_key(path)
        if not key:
            continue
        if key not in best_for_key or ".georef2.json" in Path(path).name:
            best_for_key[key] = path
    georef_paths = sorted(best_for_key.values())

    if source_data.get("type") == "AnnotationPage":
        items_by_key = _load_oim_index(source_data)
        result_id = source_data.get("id", "") + "/generated"
        print(f"Loaded {len(items_by_key)} OIM annotations.", file=sys.stderr)
    elif source_data.get("@type") == "sc:Manifest":
        items_by_key = _load_loc_index(source_data)
        result_id = source_data.get("@id", "") + "/generated"
        print(f"Loaded {len(items_by_key)} LOC canvases.", file=sys.stderr)
    else:
        print(
            "Error: expected an OIM IIIF AnnotationPage (type: AnnotationPage) "
            "or a LOC manifest (@type: sc:Manifest).",
            file=sys.stderr,
        )
        sys.exit(1)

    label: str = source_data.get("label", "")

    valid_items: list[tuple[str, dict, dict, Path, Path]] = []
    for path in georef_paths:
        page_key = georef_path_to_page_key(path)
        if not page_key:
            print(f"Warning: could not parse page key from '{path}'", file=sys.stderr)
            continue
        # Splits (pN__i) share their parent page's canvas item.
        parent_key = page_key.split("__")[0]
        canvas_item = items_by_key.get(parent_key)
        if not canvas_item:
            print(
                f"Warning: no canvas for page key '{parent_key}' ({path})",
                file=sys.stderr,
            )
            continue
        georef = json.load(open(path))
        image_path = Path(path).parent / f"{page_key}.jpg"
        valid_items.append((page_key, canvas_item, georef, image_path, Path(path)))

    # Drop skeleton pages (key ends in 's') when a full-color counterpart exists
    # within this volume. Scoped per-volume so skeletons in one volume are never
    # dropped because another volume has a matching full-color page.
    non_skeleton_keys = {pk for pk, _, _, _, _ in valid_items if not pk.endswith("s")}
    skipped = [
        pk
        for pk, _, _, _, _ in valid_items
        if pk.endswith("s") and pk[:-1] in non_skeleton_keys
    ]
    if skipped:
        print(
            f"Dropping {len(skipped)} skeleton page(s) with full-color counterparts: "
            + ", ".join(skipped),
            file=sys.stderr,
        )
        valid_items = [item for item in valid_items if item[0] not in set(skipped)]

    return valid_items, result_id, label


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine OIM IIIF annotations with our georeferences into an IIIF AnnotationPage. "
            "Each georef file is matched to the OIM annotation by page key parsed from the label."
        )
    )
    parser.add_argument(
        "oim_iiif",
        nargs="?",
        metavar="OIM_IIIF",
        help="OIM IIIF AnnotationPage JSON file (e.g. main.iiif.json)",
    )
    parser.add_argument(
        "georef_glob",
        nargs="?",
        metavar="GEOREF_GLOB",
        help="Glob pattern matching georef JSON files (e.g. 'path/to/*.georef.json')",
    )
    parser.add_argument(
        "--volume",
        action="append",
        nargs=2,
        metavar=("ARG1", "ARG2"),
        help=(
            "Add a volume. Without --image-base-url: ARG1=IIIF manifest, ARG2=georef glob. "
            "With --image-base-url: ARG1=georef glob, ARG2=path suffix appended to the base URL "
            "(e.g. 'vol1' gives {base_url}/vol1/{page}.jpg). "
            "Repeatable for multi-volume output."
        ),
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
        "--image-source-type",
        metavar="TYPE",
        default="ImageService3",
        help=(
            "IIIF source type used with --image-base-url "
            "(default: ImageService3). Use ImageService2 for a v2 endpoint, "
            "or Image for a plain static JPEG with no tile server."
        ),
    )
    parser.add_argument(
        "--image-base-url",
        metavar="URL",
        help=(
            "Base URL for plain-image hosting (e.g. an S3 bucket). "
            "When set, no IIIF manifest is needed: canvas items are built directly "
            "from the page images on disk. The URL for each page is "
            "{URL}/{parent_key}.jpg. Provide the georef glob as the sole "
            "positional argument."
        ),
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

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_valid_items: list[tuple[str, dict, dict, Path, Path]] = []
    result_id = ""
    label = ""
    n_georef_files = 0

    if args.image_base_url:
        # Plain-image (S3) mode: no manifest needed.
        if args.volume:
            # Multi-volume S3: each --volume takes GLOB SUFFIX where SUFFIX is
            # appended to --image-base-url to form the per-volume image base URL.
            if args.oim_iiif or args.georef_glob:
                print(
                    "Warning: positional args are ignored when --volume is used.",
                    file=sys.stderr,
                )
            base = args.image_base_url.rstrip("/")
            for s3_glob, vol_suffix in args.volume:
                vol_base_url = base + "/" + vol_suffix.lstrip("/")
                vol_items, vol_result_id, _ = _load_s3_items(
                    s3_glob, vol_base_url, args.image_source_type
                )
                all_valid_items.extend(vol_items)
                n_georef_files += len(sorted(glob.glob(s3_glob)))
                if not result_id:
                    result_id = vol_result_id
        else:
            # Single-volume S3: georef glob from positional arg.
            # oim_iiif is first positional so it receives the value when only one
            # positional argument is supplied.
            s3_glob = args.georef_glob or args.oim_iiif
            if not s3_glob:
                parser.error(
                    "Provide a georef glob pattern as a positional argument "
                    "when using --image-base-url."
                )
            vol_items, result_id, label = _load_s3_items(
                s3_glob, args.image_base_url, args.image_source_type
            )
            all_valid_items.extend(vol_items)
            n_georef_files = len(sorted(glob.glob(s3_glob)))
    else:
        # Manifest-based mode: resolve volume specs from --volume or positionals.
        if args.volume:
            if args.oim_iiif or args.georef_glob:
                print(
                    "Warning: positional OIM_IIIF/GEOREF_GLOB are ignored when --volume is used.",
                    file=sys.stderr,
                )
            volume_specs: list[list[str]] = args.volume
        elif args.oim_iiif and args.georef_glob:
            volume_specs = [[args.oim_iiif, args.georef_glob]]
        else:
            parser.error(
                "Provide positional OIM_IIIF GEOREF_GLOB for a single volume, "
                "or use --volume IIIF GLOB (repeatable) for multiple volumes, "
                "or use --image-base-url URL for plain-image hosting."
            )

        for iiif_path, georef_glob in volume_specs:
            vol_items, vol_result_id, vol_label = _load_volume_items(
                iiif_path, georef_glob
            )
            all_valid_items.extend(vol_items)
            n_georef_files += len(sorted(glob.glob(georef_glob)))
            if not result_id:
                result_id = vol_result_id
                label = vol_label

    # Compute block-based clipping masks when a centerlines file is provided.
    # All volumes' pages are passed together so blocks at volume boundaries are
    # assigned correctly using color and geometry from all neighboring pages.
    geo_masks: list[ShapelyPolygon | None] = [None] * len(all_valid_items)
    if args.centerlines:
        with open(args.centerlines) as f:
            centerlines_geojson: dict = json.load(f)
        all_georefs = [georef for _, _, georef, _, _ in all_valid_items]
        print("Computing block-based clipping masks...", file=sys.stderr)
        debug_blocks: list[dict] | None = [] if args.debug_blocks else None
        geo_masks = compute_all_clip_masks(
            all_georefs,
            centerlines_geojson,
            debug_blocks_out=debug_blocks,
            raw_paths=[image_path for _, _, _, image_path, _ in all_valid_items],
        )
        n_masked = sum(m is not None for m in geo_masks)
        print(
            f"Computed masks for {n_masked}/{len(all_valid_items)} pages.",
            file=sys.stderr,
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
                for (page_key, _, _, _, _), mask in zip(all_valid_items, geo_masks)
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
    annotation_georef_paths: list[Path] = []
    for (page_key, canvas_item, georef, image_path, georef_path), geo_mask in zip(
        all_valid_items, geo_masks
    ):
        try:
            annotations.append(
                make_annotation(
                    canvas_item,
                    georef,
                    page_key,
                    image_path,
                    args.creator,
                    now,
                    geo_mask,
                )
            )
            annotation_georef_paths.append(georef_path)
        except ValueError as exc:
            print(f"Warning: skipping {page_key}: {exc}", file=sys.stderr)

    n_total = len(annotations)
    one_gcp_paths = [
        str(georef_path)
        for ann, georef_path in zip(annotations, annotation_georef_paths)
        if sum(1 for f in ann["body"]["features"] if f["properties"]["type"] == "gcp")
        == 1
    ]
    print(
        f"Generated {n_total} annotations from {n_georef_files} georef files.",
        file=sys.stderr,
    )
    if one_gcp_paths:
        pct = 100.0 * len(one_gcp_paths) / n_total if n_total else 0.0
        print(
            f"One-GCP fits: {len(one_gcp_paths)}/{n_total} ({pct:.1f}%): "
            + ", ".join(one_gcp_paths),
            file=sys.stderr,
        )

    result = {
        "id": result_id,
        "type": "AnnotationPage",
        "@context": ["http://www.w3.org/ns/anno.jsonld"],
        "label": label,
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
