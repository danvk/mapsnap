"""Combine OIM IIIF annotations with our georeferences into a new IIIF AnnotationPage.

For each georef JSON file whose page key matches an annotation in the OIM IIIF file
(main.iiif.json), generates one georeferencing annotation with four corner GCPs.

For images where the raw file covers the full LOC canvas (raw_dims ≈ source_dims),
resource coordinates are computed by simple proportional scaling:

    resourceCoords = georef_pixel × source_dim / georef_dim

For true sub-images (raw_dims << source_dims), the unsplit original (e.g. p4.unsplit.jpg)
is located via template matching to determine the sub-image's pixel offset within the
full canvas. This gives a non-circular placement: the canvas coords derive from image
geometry, not from the truth GCPs. If the unsplit image is absent or the match score
is too low, the script raises an error and skips the page.

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

import cv2
import numpy as np
from PIL import Image

from mapsnap.utils import jpeg_dimensions, label_to_page_key


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
    m = re.search(r"(?:\b|_)(p\d+[snew]?(?:__\d)?)(?:\.[^.]+)?\.georef\.json$", path)
    return m.group(1) if m else None


def _georef_metadata(
    georef: dict,
    is_full_canvas: bool,
    split_canvas: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """Build IIIF metadata entries for streets, intersections, and canvas type.

    split_canvas, when present, gives the sub-image region within the full canvas as
    (x, y, w, h) in canvas pixel coordinates, derived via template matching rather than
    truth GCPs. When absent for a sub-image, the comparison is circular.
    """
    n_streets = len(
        set(s["street"] for s in georef.get("streets", []) if s.get("inlier"))
    )
    n_intersections = sum(1 for i in georef.get("intersections", []) if i.get("inlier"))
    entries: list[dict] = [
        {"label": "streets", "value": str(n_streets)},
        {"label": "intersections", "value": str(n_intersections)},
        {"label": "is_full_canvas", "value": "true" if is_full_canvas else "false"},
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


def make_annotation(
    oim_item: dict,
    georef: dict,
    raw_path: Path,
    creator_url: str,
    now: str,
) -> dict:
    """Build a IIIF georeferencing annotation in the OIM coordinate space.

    Reuses the OIM annotation's source ID and dimensions so the output is directly
    comparable to main.iiif.json by compare_iiif_georef.py.

    For full-canvas images (raw_dims ≈ source_dims), resource coords are computed
    by simple proportional scaling. For split sub-images, the unsplit original
    (e.g. p4.unsplit.jpg) is located via template matching to determine the non-circular
    canvas placement. Raises ValueError if the unsplit file is missing or the match
    is too uncertain.
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

    raw_width, raw_height = jpeg_dimensions(raw_path)

    # Use simple scaling when the raw image covers the full canvas (within 2px tolerance).
    is_full_canvas = (
        abs(raw_width - source_width) <= 2 and abs(raw_height - source_height) <= 2
    )
    split_canvas: tuple[float, float, float, float] | None = None

    if is_full_canvas:
        scale_x = source_width / georef_width
        scale_y = source_height / georef_height
        resource_coords_list = [
            [round(px * scale_x, 1), round(py * scale_y, 1)] for px, py in pixel_corners
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
            for px, py in pixel_corners
        ]

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
        "metadata": _georef_metadata(georef, is_full_canvas, split_canvas),
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
        try:
            annotations.append(
                make_annotation(oim_item, georef, raw_path, args.creator, now)
            )
        except ValueError as exc:
            print(f"Warning: skipping {page_key}: {exc}", file=sys.stderr)

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
