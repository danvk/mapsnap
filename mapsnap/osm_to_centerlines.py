"""Convert an OSM JSON dump to a centerlines GeoJSON compatible with georef_from_labels.py.

Each named drivable way becomes a GeoJSON LineString feature with a `street_name`
property set to the OSM name (e.g. "Hooper Street"). georef_from_labels.py calls
normalize_street on this value, which uppercases and expands abbreviations, so
"HOOPER ST" in a detected label matches "Hooper Street" from OSM.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from mapsnap.streets import normalize_street

# Highway values to skip — non-street features unlikely to appear as labeled
# streets on historic maps.
EXCLUDE_HIGHWAY = {
    "construction",
    "corridor",
    "cycleway",
    "elevator",
    "footway",
    "motorway",
    "motorway_link",
    "path",
    "pedestrian",
    "proposed",
    "steps",
    "track",
}

# highway=service is usually not of interest to us, but there are a
# few cases where it is. highway=service without service=* is a tagging
# error, but we let it slide.
OK_SERVICE = {"alley", None}

_FT_PER_DEG_LAT: float = math.pi * 20_925_524.0 / 180.0
_INTERSECTION_CLUSTER_FT: float = 60.0


def should_drop(tags: dict[str, str]) -> bool:
    highway = tags.get("highway")
    if highway in EXCLUDE_HIGHWAY:
        return True
    if highway == "service":
        return tags.get("service") not in OK_SERVICE
    return False


def osm_to_centerlines(osm_json: str) -> dict:
    """Convert an OSM JSON dump to a GeoJSON FeatureCollection of street centerlines."""
    data = json.load(open(osm_json))
    elements = data["elements"]

    id_to_node = {el["id"]: el for el in elements if el["type"] == "node"}

    features = []
    n_missing = 0
    for way in elements:
        if way["type"] != "way":
            continue
        tags = way.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        if should_drop(tags):
            print(f"Dropping w{way['id']} highway={tags.get('highway')} {name}")
            continue

        coords = []
        missing = False
        for node_id in way["nodes"]:
            node = id_to_node.get(node_id)
            if node is None:
                missing = True
                break
            coords.append([node["lon"], node["lat"]])
        if missing:
            n_missing += 1
            continue
        if len(coords) < 2:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {"street_name": name},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )

    if n_missing:
        print(f"Skipped {n_missing} ways with missing nodes", file=sys.stderr)

    return {"type": "FeatureCollection", "features": features}


def _cluster_coords(
    coords: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Cluster nearby coordinates by 60ft proximity and return one centroid per cluster."""
    if not coords:
        return []
    mean_lat = sum(c[1] for c in coords) / len(coords)
    ft_per_deg_lon = _FT_PER_DEG_LAT * math.cos(math.radians(mean_lat))
    clusters: list[list[tuple[float, float]]] = []
    for pt in coords:
        merged = False
        for cluster in clusters:
            for c in cluster:
                if (
                    math.sqrt(
                        ((pt[0] - c[0]) * ft_per_deg_lon) ** 2
                        + ((pt[1] - c[1]) * _FT_PER_DEG_LAT) ** 2
                    )
                    < _INTERSECTION_CLUSTER_FT
                ):
                    cluster.append(pt)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            clusters.append([pt])
    return [
        (
            sum(c[0] for c in cl) / len(cl),
            sum(c[1] for c in cl) / len(cl),
        )
        for cl in clusters
    ]


def compute_street_intersections(
    features: list[dict],
) -> list[tuple[str, str, float, float]]:
    """Find all street pairs sharing a GeoJSON node, returning one row per intersection.

    Each shared coordinate is a potential georef GCP (two streets physically meeting).
    Nearby shared nodes are clustered by a 60ft threshold (matching the logic in
    georef_from_labels.py) so that a single physical intersection with multiple
    adjacent OSM nodes produces one row rather than many.

    Returns a sorted list of (street_a, street_b, lon, lat) tuples with
    street_a < street_b (both normalized). Coordinates are rounded to 7 decimal places.
    """
    coord_to_streets: dict[tuple[float, float], set[str]] = {}
    for feat in features:
        name = normalize_street(feat["properties"]["street_name"])
        for coord in feat["geometry"]["coordinates"]:
            key = (round(float(coord[0]), 7), round(float(coord[1]), 7))
            coord_to_streets.setdefault(key, set()).add(name)

    pair_coords: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for coord, streets in coord_to_streets.items():
        streets_list = sorted(streets)
        for i in range(len(streets_list)):
            for j in range(i + 1, len(streets_list)):
                pair = (streets_list[i], streets_list[j])
                pair_coords.setdefault(pair, []).append(coord)

    result: list[tuple[str, str, float, float]] = []
    for (street_a, street_b), coords in sorted(pair_coords.items()):
        for lon, lat in _cluster_coords(coords):
            result.append((street_a, street_b, round(lon, 7), round(lat, 7)))
    return sorted(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an OSM JSON dump to a centerlines GeoJSON "
            "for use with georef_from_labels.py."
        )
    )
    parser.add_argument(
        "osm_json", metavar="FILE", help="OSM JSON dump (Overpass API output)"
    )
    parser.add_argument(
        "--output", metavar="FILE", help="Write GeoJSON to this file (default: stdout)"
    )
    args = parser.parse_args()

    geojson = osm_to_centerlines(args.osm_json)

    out_str = json.dumps(geojson, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out_str)
        print(
            f"Wrote {len(geojson['features'])} features to {args.output}",
            file=sys.stderr,
        )

        out_dir = Path(args.output).parent
        features = geojson["features"]

        streets = sorted(
            {normalize_street(f["properties"]["street_name"]) for f in features}
        )
        streets_path = out_dir / "streets.txt"
        streets_path.write_text("\n".join(streets) + "\n")
        print(f"Wrote {len(streets)} streets to {streets_path}", file=sys.stderr)

        intersections = compute_street_intersections(features)
        intersections_path = out_dir / "intersections.csv"
        with open(intersections_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["street_a", "street_b", "lon", "lat"])
            writer.writerows(intersections)
        print(
            f"Wrote {len(intersections)} intersections to {intersections_path}",
            file=sys.stderr,
        )
    else:
        print(out_str)


if __name__ == "__main__":
    main()
