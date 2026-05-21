"""Find the geographic bounding box of all georeferenced images in a IIIF annotation file.

Fits a 2×3 affine transform from each image's GCPs, projects the four image corners to
(lon, lat), and reports the overall SW/NE bounding box in a form ready to pass to
download_osm.py.

Usage:
    python iiif_bbox.py <path/to/main.iiif.json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from compare_iiif_georef import extract_gcps, fit_affine


def image_corners(width: float, height: float) -> list[tuple[float, float]]:
    """Return the four corner pixel coordinates of an image."""
    return [(0, 0), (width, 0), (width, height), (0, height)]


def project_corners(
    item: dict,
) -> list[tuple[float, float]] | None:
    """Project the four corners of an image to (lon, lat) using its GCPs.

    Returns None if the item has fewer than 3 GCPs (can't fit an affine transform).
    """
    gcps = extract_gcps(item)
    if len(gcps) < 3:
        return None

    source = item["target"]["source"]
    width = float(source["width"])
    height = float(source["height"])

    affine = fit_affine(gcps)
    corners = []
    for px, py in image_corners(width, height):
        lon, lat = affine @ np.array([px, py, 1.0])
        corners.append((float(lon), float(lat)))
    return corners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "iiif", metavar="IIIF_FILE", help="Path to IIIF AnnotationPage JSON"
    )
    args = parser.parse_args()

    data = json.loads(Path(args.iiif).read_text())
    items = data.get("items", [])

    all_lons: list[float] = []
    all_lats: list[float] = []
    skipped = 0

    for item in items:
        corners = project_corners(item)
        if corners is None:
            skipped += 1
            continue
        for lon, lat in corners:
            all_lons.append(lon)
            all_lats.append(lat)

    n_georef = len(items) - skipped
    print(
        f"{n_georef}/{len(items)} images georeferenced ({skipped} skipped, <3 GCPs)",
        file=sys.stderr,
    )

    if not all_lons:
        print("No georeferenced images found.", file=sys.stderr)
        sys.exit(1)

    sw_lat, sw_lon = min(all_lats), min(all_lons)
    ne_lat, ne_lon = max(all_lats), max(all_lons)

    print(
        f"Bounding box: SW ({sw_lat:.6f}, {sw_lon:.6f})  NE ({ne_lat:.6f}, {ne_lon:.6f})",
        file=sys.stderr,
    )
    print(f"{sw_lat} {sw_lon} {ne_lat} {ne_lon}")

    geojson = json.dumps(
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[sw_lon, sw_lat], [ne_lon, ne_lat]],
            },
            "properties": {},
        }
    )
    print(geojson, file=sys.stderr)


if __name__ == "__main__":
    main()
