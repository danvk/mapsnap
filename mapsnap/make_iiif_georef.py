"""Combine OIM IIIF annotations with our georeferences into a new IIIF AnnotationPage.

For each georef JSON file whose page key matches an annotation in the OIM IIIF file
(main.iiif.json), generates one georeferencing annotation with four corner GCPs.

For images where the raw file covers the full LOC canvas (raw_dims ≈ source_dims),
resource coordinates are computed by simple proportional scaling:

    resourceCoords = georef_pixel × source_dim / georef_dim

For true sub-images (raw_dims << source_dims), the OIM annotation's GCPs provide a
geo → canvas-pixel affine that is used to map our georef corner coordinates directly
into the full canvas pixel space.

Usage:
    python make_iiif_georef.py <main.iiif.json> <georef_glob> [--output FILE] [--creator URL]
"""

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mapsnap.utils import jpeg_dimensions, label_to_page_key


def georef_path_to_page_key(path: str) -> str | None:
    """Extract page key like 'p428__2' from a georef filename.

    Accepts filenames ending in '_p16s.georef.json', '_p16.georef.json', or
    '_p16s.gcps.georef.json' (the '.gcps' infix is optional).
    """
    m = re.search(r"(?:\b|_)(p\d+[snew]?(?:__\d)?)(?:\.[^.]+)?\.georef\.json$", path)
    return m.group(1) if m else None


def _canvas_coords_from_oim_gcps(oim_item: dict, georef: dict) -> list[list[float]]:
    """Map georef corner geo coords to full canvas pixel coords using OIM GCPs.

    Used for true sub-images where raw_dims differ significantly from source_dims.
    The OIM GCPs carry both canvas pixel coordinates and geo coordinates; we fit a
    geo → canvas-pixel affine and apply it to our georef corner geo coordinates.
    """
    oim_gcps = oim_item["body"]["features"]
    if len(oim_gcps) < 3:
        raise ValueError(f"Need ≥3 OIM GCPs for affine fit, got {len(oim_gcps)}")

    geo_pts = np.array([feat["geometry"]["coordinates"][:2] for feat in oim_gcps])
    canvas_pts = np.array([feat["properties"]["resourceCoords"] for feat in oim_gcps])

    X = np.column_stack([geo_pts, np.ones(len(geo_pts))])
    coef, *_ = np.linalg.lstsq(X, canvas_pts, rcond=None)

    result: list[list[float]] = []
    for lon, lat in georef["corners"]:
        v = np.array([lon, lat, 1.0])
        cx, cy = coef.T @ v
        result.append([round(float(cx), 1), round(float(cy), 1)])
    return result


def _georef_metadata(georef: dict) -> list[dict]:
    """Build IIIF metadata entries for streets and intersections found in the image."""
    n_streets = len(
        set(s["street"] for s in georef.get("streets", []) if s.get("inlier"))
    )
    n_intersections = sum(1 for i in georef.get("intersections", []) if i.get("inlier"))
    return [
        {"label": "streets", "value": str(n_streets)},
        {"label": "intersections", "value": str(n_intersections)},
    ]


def make_annotation(
    oim_item: dict,
    georef: dict,
    raw_width: int,
    raw_height: int,
    creator_url: str,
    now: str,
) -> dict:
    """Build a IIIF georeferencing annotation in the OIM coordinate space.

    Reuses the OIM annotation's source ID and dimensions so the output is directly
    comparable to main.iiif.json by compare_iiif_georef.py.
    """
    source = oim_item["target"]["source"]
    source_id: str = source["id"]
    source_width: int = source["width"]
    source_height: int = source["height"]
    label: str = oim_item["label"]

    # Derive a unique canvas ID from the source URL; append split number if present.
    canvas_id = source_id.removesuffix("/info.json")
    m = re.search(r"\[(\d+)\]$", label)
    if m:
        canvas_id += f"__{m.group(1)}"

    creator = {"id": creator_url, "type": "Person"}
    georef_width = georef["width"]
    georef_height = georef["height"]
    corners = georef["corners"]
    pixel_corners = [
        (0, 0),
        (georef_width, 0),
        (georef_width, georef_height),
        (0, georef_height),
    ]

    # Use simple scaling when the raw image covers the full canvas (within 2px tolerance).
    is_full_canvas = (
        abs(raw_width - source_width) <= 2 and abs(raw_height - source_height) <= 2
    )
    if is_full_canvas:
        scale_x = source_width / georef_width
        scale_y = source_height / georef_height
        resource_coords_list = [
            [round(px * scale_x, 1), round(py * scale_y, 1)] for px, py in pixel_corners
        ]
    else:
        # True sub-image: use OIM GCPs to establish geo → canvas affine.
        resource_coords_list = _canvas_coords_from_oim_gcps(oim_item, georef)

    features = [
        {
            "type": "Feature",
            "properties": {
                "resourceCoords": rc,
                "creator": creator,
            },
            "geometry": {
                "type": "Point",
                "coordinates": corner,
            },
        }
        for rc, corner in zip(resource_coords_list, corners)
    ]

    return {
        "id": f"{canvas_id}/georef",
        "type": "Annotation",
        "@context": [
            "http://iiif.io/api/extension/georef/1/context.json",
            "http://iiif.io/api/presentation/3/context.json",
        ],
        "label": label,
        "metadata": _georef_metadata(georef),
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
                "value": (
                    f'<svg><polygon points="0,{source_height} 0,0 '
                    f'{source_width},0 {source_width},{source_height} 0,{source_height}" /></svg>'
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
    args = parser.parse_args()

    oim_data: dict = json.load(open(args.oim_iiif))
    if oim_data.get("type") != "AnnotationPage":
        print(
            "Error: expected an OIM IIIF AnnotationPage (type: AnnotationPage).",
            file=sys.stderr,
        )
        sys.exit(1)

    oim_by_key: dict[str, dict] = {}
    for item in oim_data.get("items", []):
        page_key = label_to_page_key(item.get("label", ""))
        if page_key:
            oim_by_key[page_key] = item
    print(f"Loaded {len(oim_by_key)} OIM annotations.", file=sys.stderr)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    georef_paths = sorted(glob.glob(args.georef_glob))
    if not georef_paths:
        print(f"Error: no files matched '{args.georef_glob}'.", file=sys.stderr)
        sys.exit(1)

    annotations: list[dict] = []
    for path in georef_paths:
        page_key = georef_path_to_page_key(path)
        if not page_key:
            print(f"Warning: could not parse page key from '{path}'", file=sys.stderr)
            continue
        oim_item = oim_by_key.get(page_key)
        if not oim_item:
            print(
                f"Warning: no OIM annotation for page key '{page_key}' ({path})",
                file=sys.stderr,
            )
            continue
        georef: dict = json.load(open(path))
        raw_path = Path(path).parent / f"{page_key}.raw.jpg"
        raw_width, raw_height = jpeg_dimensions(raw_path)
        annotations.append(
            make_annotation(oim_item, georef, raw_width, raw_height, args.creator, now)
        )

    print(
        f"Generated {len(annotations)} annotations from {len(georef_paths)} georef files.",
        file=sys.stderr,
    )

    result = {
        "id": oim_data.get("id", "") + "/generated",
        "type": "AnnotationPage",
        "@context": ["http://www.w3.org/ns/anno.jsonld"],
        "label": oim_data.get("label", ""),
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
