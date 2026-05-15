"""Convert an OSM JSON dump to a centerlines GeoJSON compatible with georef_from_labels.py.

Each named drivable way becomes a GeoJSON LineString feature with a `street_name`
property set to the OSM name (e.g. "Hooper Street"). georef_from_labels.py calls
normalize_street on this value, which uppercases and expands abbreviations, so
"HOOPER ST" in a detected label matches "Hooper Street" from OSM.
"""

import argparse
import json
import sys

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


def should_drop(tags: dict[str, str]):
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
    else:
        print(out_str)


if __name__ == "__main__":
    main()
