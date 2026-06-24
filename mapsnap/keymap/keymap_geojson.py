"""Emit a debug GeoJSON visualising a key-map fit against the truth georeferencing.

Two feature sources, so a viewer (geojson.io etc.) shows them together:

  * **truth page footprints** — one Polygon per truth annotation in ``main.iiif.json``,
    obtained by transforming the annotation's SvgSelector clip mask through its own GCP
    transform. Because each split panel is its own annotation with its own clip mask, splits
    appear as separate polygons. Stroked green/red/grey for inlier / outlier / not-in-key-map.
  * **key-map predicted centroids** — one Point per (page, key-map detection) correspondence,
    placed where the fitted key-map model says that page number sits. ``marker-color`` is green
    for an inlier page and red for an outlier, so a wrong global fit shows up as a cluster of
    red points pulled away from their truth polygons.
  * **key-map region polygons** (``--regions``) — one Polygon per page number, the colored block
    segmented around it (mapsnap.keymap.page_regions) mapped to lon/lat through the same fit. These
    carry ``type`` = "keymap-region" and a distinct blue ``fill``, versus ``type`` = "truth" on the
    footprints, so the segmented blocks read clearly apart from the truth.

    uv run python -m mapsnap.keymap.keymap_geojson data/chicago_il_1950_vol_1 --regions \
        --output data/chicago_il_1950_vol_1/keymap-debug.geojson
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    extract_gcps,
    fit_transform,
)
from mapsnap.keymap.fit_keymap import (
    Model,
    build_correspondences,
    load_detections,
    load_georef_pages,
    ransac,
    similarity_apply,
    unproject,
)
from mapsnap.keymap.page_regions import (
    RegionParams,
    keymap_image_path,
    load_seeds,
    segment_page_regions,
)
from mapsnap.utils import source_id_to_page_key

INLIER_COLOR = "#2ca02c"  # green
OUTLIER_COLOR = "#d62728"  # red
NEUTRAL_COLOR = "#999999"  # grey: page has no key-map detection
REGION_FILL = (
    "#1f77b4"  # blue: segmented key-map region, to stand apart from truth footprints
)

Point = tuple[float, float]


def parse_svg_polygon(svg: str) -> list[Point]:
    """Pixel vertices of the single <polygon> in an SvgSelector value string."""
    match = re.search(r'points="([^"]+)"', svg)
    if not match:
        return []
    coords = [float(v) for v in match.group(1).replace(",", " ").split()]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]


def transform_ring(ring: list[Point], affine: np.ndarray) -> list[list[float]]:
    """Map pixel vertices to [lon, lat] via a 2x3 affine ([lon, lat] = A @ [px, py, 1])."""
    out: list[list[float]] = []
    for px, py in ring:
        lon, lat = affine @ np.array([px, py, 1.0])
        out.append([float(lon), float(lat)])
    return out


def truth_footprints(iiif_path: Path) -> list[tuple[str, list[list[float]]]]:
    """(page_key, lon/lat ring) for each truth annotation's clip mask.

    Splits are separate annotations and so yield separate footprints.
    """
    data = json.load(open(iiif_path))
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    footprints: list[tuple[str, list[list[float]]]] = []
    for item in items:
        target = item["target"]
        source_id = target["source"]["id"]
        page_key = source_id_to_page_key(source_id, item.get("label", "") or "")
        selector = target.get("selector", {})
        ring = parse_svg_polygon(selector.get("value", ""))
        if not ring:
            continue
        affine = fit_transform(extract_gcps(item), annotation_transform_type(item))
        footprints.append((page_key, transform_ring(ring, affine)))
    return footprints


def page_color(number: int | None, inlier_numbers: set[int], detected: set[int]) -> str:
    """Green if the page number is a key-map inlier, red if detected-but-outlier, else grey."""
    if number in inlier_numbers:
        return INLIER_COLOR
    if number in detected:
        return OUTLIER_COLOR
    return NEUTRAL_COLOR


def region_features(
    keymap_path: Path,
    model: Model,
    origin: Point,
    inlier_numbers: set[int],
    detected: set[int],
    params: RegionParams,
) -> list[dict]:
    """Polygon features for the colored block segmented around each page number.

    Each page number's block is found on the key-map image (page_regions.segment_page_regions),
    then its pixel polygon is mapped to lon/lat through the same key-map fit as the points, so the
    block overlays its truth footprint. The fill is a fixed blue (``REGION_FILL``) so the segmented
    regions stand apart from the truth footprints at a glance; the stroke still encodes
    inlier/outlier. Each feature is tagged ``type`` = "keymap-region".
    """
    lon0, lat0 = origin
    boxes, texts = load_seeds(keymap_path)
    rgb = np.asarray(Image.open(keymap_image_path(keymap_path)).convert("RGB"))
    polygons = segment_page_regions(rgb.astype(np.float64) / 255.0, boxes, params)
    features: list[dict] = []
    for index, pixel_polygon in polygons.items():
        text = texts[index]
        number = int(text) if text.isdigit() else None
        ring = [
            list(unproject(*similarity_apply(model, vertex), lon0, lat0))
            for vertex in pixel_polygon
        ]
        ring.append(ring[0])  # GeoJSON rings are explicitly closed
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "type": "keymap-region",
                    "source": "region",
                    "page": f"p{text}",
                    "inlier": number in inlier_numbers,
                    "stroke": page_color(number, inlier_numbers, detected),
                    "stroke-width": 2,
                    "fill": REGION_FILL,
                    "fill-opacity": 0.25,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return features


def build_geojson(
    volume: Path, keymap_path: Path, truth_iiif: Path, seed: int, regions: bool = False
) -> dict:
    """Assemble the truth-footprint + key-map-prediction FeatureCollection for a volume.

    With ``regions`` set, also segment the colored block around each page number on the key-map
    image and add it as a Polygon feature (``source`` = "region"), to eyeball segmentation quality
    against the truth footprints.
    """
    pages, (lon0, lat0) = load_georef_pages(volume)
    detections = load_detections(keymap_path)
    correspondences = build_correspondences(pages, detections)
    model, inliers = ransac(pages, correspondences, rng=np.random.default_rng(seed))
    if model is None:
        raise SystemExit("Could not fit the key map (too few correspondences).")

    inlier_numbers = {pages[i].number for i in inliers}
    detected = {d.number for d in detections}

    features: list[dict] = []
    if regions:
        features.extend(
            region_features(
                keymap_path,
                model,
                (lon0, lat0),
                inlier_numbers,
                detected,
                RegionParams(),
            )
        )

    # Truth page footprints (clip masks), coloured by inlier/outlier status.
    for page_key, ring in truth_footprints(truth_iiif):
        digits = re.search(r"\d+", page_key)
        number = int(digits.group()) if digits else None
        color = page_color(number, inlier_numbers, detected)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "type": "truth",
                    "source": "truth",
                    "page_key": page_key,
                    "stroke": color,
                    "stroke-width": 2,
                    "fill": color,
                    "fill-opacity": 0.08,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    # Key-map predicted centroids, one per (page, detection) correspondence.
    for page_index, pixel in correspondences:
        page = pages[page_index]
        is_inlier = page_index in inliers
        lon, lat = unproject(*similarity_apply(model, pixel), lon0, lat0)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "type": "keymap-point",
                    "source": "keymap",
                    "page": f"p{page.number}",
                    "inlier": is_inlier,
                    "marker-color": INLIER_COLOR if is_inlier else OUTLIER_COLOR,
                },
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )

    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", type=Path, help="Volume directory.")
    parser.add_argument(
        "--keymap", type=Path, help="Key-map JSON (default <volume>/p0.keymap.json)."
    )
    parser.add_argument(
        "--truth", type=Path, help="Truth IIIF (default <volume>/main.iiif.json)."
    )
    parser.add_argument("--output", type=Path, help="Output GeoJSON path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--regions",
        action="store_true",
        help="Also segment and emit the colored block polygon around each page number.",
    )
    args = parser.parse_args()

    keymap_path = args.keymap or (args.volume / "p0.keymap.json")
    truth_iiif = args.truth or (args.volume / "main.iiif.json")
    output = args.output or (args.volume / "keymap-debug.geojson")

    geojson = build_geojson(
        args.volume, keymap_path, truth_iiif, args.seed, args.regions
    )
    output.write_text(json.dumps(geojson))
    by_source: dict[str, int] = {}
    for feature in geojson["features"]:
        source = feature["properties"]["source"]
        by_source[source] = by_source.get(source, 0) + 1
    summary = ", ".join(
        f"{count} {source}" for source, count in sorted(by_source.items())
    )
    print(f"Wrote {output}: {summary}.")


if __name__ == "__main__":
    main()
