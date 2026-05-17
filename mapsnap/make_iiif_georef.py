"""Combine a Library of Congress IIIF manifest with georef JSON files into an IIIF AnnotationPage.

For each georef JSON file whose page key (e.g. 'p16s') matches a canvas in the LOC manifest,
generates one georeferencing annotation with four corner GCPs. The LOC manifest uses IIIF Image
API images served at pct:25 (quarter resolution), so full-resolution pixel coordinates for the
annotation body are computed as:
    resourceCoords = georef_pixel × (canvas_dim × 4) / georef_dim

Usage:
    python make_iiif_georef.py <iiif_manifest.json> <georef_glob> [--output FILE] [--creator URL]
"""

import argparse
import glob
import json
import re
import sys
from datetime import datetime, timezone


def canvas_id_to_page_key(canvas_id: str) -> str | None:
    """Map a LOC canvas @id to a page key like 'p16s'.

    The canvas @id suffix looks like '05791_02_1939-0016s'. We extract the trailing
    numeric-plus-optional-s part and strip leading zeros: '0016s' → 'p16s'.
    """
    m = re.search(r"-(\d+)([sNSEW]?)$", canvas_id)
    if not m:
        return None
    return f"p{int(m.group(1))}{m.group(2)}".lower()


def georef_path_to_page_key(path: str) -> str | None:
    """Extract page key like 'p16s' from a georef filename.

    Accepts filenames ending in '_p16s.georef.json', '_p16.georef.json', or
    '_p16s.gcps.georef.json' (the '.gcps' infix is optional).
    """
    # drop the trailing "__2" in "p428__2" since the LOC images won't have splits.
    m = re.search(r"(?:\b|_)(p\d+[snew]?(?:__\d)?)(?:\.[^.]+)?\.georef\.json$", path)
    return m.group(1) if m else None


def make_annotation(
    canvas: dict,
    georef: dict,
    creator_url: str,
    label: str,
    now: str,
) -> dict:
    """Build a single IIIF georeferencing annotation from a canvas and georef data.

    LOC serves images at pct:25, so the canvas dimensions are 1/4 of the full-resolution
    originals. Scale factors convert from georef pixel space (2048px-wide images) to the
    full-resolution pixel space used by resourceCoords.
    """
    full_width = canvas["width"] * 4
    full_height = canvas["height"] * 4
    georef_width = georef["width"]
    georef_height = georef["height"]
    scale_x = full_width / georef_width
    scale_y = full_height / georef_height

    service_id = canvas["images"][0]["resource"]["service"]["@id"]
    canvas_id = canvas["@id"]
    creator = {"id": creator_url, "type": "Person"}

    # corners[i] = [lon, lat] for pixel (px, py) in the georef image.
    # Order from georef_from_labels.py: (0,0), (w,0), (w,h), (0,h).
    corners = georef["corners"]
    pixel_corners = [
        (0, 0),
        (georef_width, 0),
        (georef_width, georef_height),
        (0, georef_height),
    ]

    features = [
        {
            "type": "Feature",
            "properties": {
                "resourceCoords": [round(px * scale_x, 1), round(py * scale_y, 1)],
                "creator": creator,
            },
            "geometry": {
                "type": "Point",
                "coordinates": corner,
            },
        }
        for (px, py), corner in zip(pixel_corners, corners)
    ]

    return {
        "id": f"{canvas_id}/georef",
        "type": "Annotation",
        "@context": [
            "http://iiif.io/api/extension/georef/1/context.json",
            "http://iiif.io/api/presentation/3/context.json",
        ],
        "label": label,
        "created": now,
        "modified": now,
        "creator": [creator],
        "motivation": "georeferencing",
        "target": {
            "id": f"{canvas_id}/selector",
            "type": "SpecificResource",
            "source": {
                "id": f"{service_id}/info.json",
                "type": "ImageService2",
                "height": full_height,
                "width": full_width,
            },
            "selector": {
                "type": "SvgSelector",
                "value": (
                    f'<svg><polygon points="0,{full_height} 0,0 '
                    f'{full_width},0 {full_width},{full_height} 0,{full_height}" /></svg>'
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
            "Combine a Library of Congress IIIF manifest with georef JSON files "
            "into an IIIF AnnotationPage."
        )
    )
    parser.add_argument("iiif", metavar="MANIFEST", help="LOC IIIF manifest JSON file")
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

    manifest: dict = json.load(open(args.iiif))
    if "sequences" not in manifest:
        print("Error: expected a IIIF v2 manifest with 'sequences'.", file=sys.stderr)
        sys.exit(1)

    canvases: list[dict] = manifest["sequences"][0]["canvases"]
    canvas_by_key: dict[str, dict] = {}
    for canvas in canvases:
        key = canvas_id_to_page_key(canvas["@id"])
        if key:
            canvas_by_key[key] = canvas

    manifest_label: str = manifest.get("label", "")
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
        canvas = canvas_by_key.get(page_key)
        split = None
        if not canvas and "__" in page_key:
            page, split = page_key.split("__")
            canvas = canvas_by_key.get(page)
        if not canvas:
            print(
                f"Warning: no canvas in manifest for page key '{page_key}' ({path})",
                file=sys.stderr,
            )
            continue
        georef: dict = json.load(open(path))
        canvas_label: str = canvas.get("label", page_key)
        if split:
            canvas_label += f" [{split}]"
        label = f"{manifest_label} | {canvas_label}"
        annotations.append(make_annotation(canvas, georef, args.creator, label, now))

    print(
        f"Generated {len(annotations)} annotations from {len(georef_paths)} georef files.",
        file=sys.stderr,
    )

    result = {
        "id": f"{manifest.get('@id', manifest.get('id', ''))}/georef/",
        "type": "AnnotationPage",
        "@context": ["http://www.w3.org/ns/anno.jsonld"],
        "label": manifest_label,
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
