"""Download OSM street data for a bounding box using the Overpass API.

Usage:
    python download_osm.py <sw_lat> <sw_lon> <ne_lat> <ne_lon> --output FILE
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def download_osm(
    sw_lat: float,
    sw_lon: float,
    ne_lat: float,
    ne_lon: float,
) -> dict:
    """Query Overpass for named highway ways in a bounding box and return parsed JSON."""
    query = f"""[out:json][timeout:60];
(
  way["highway"]["name"](
    {sw_lat}, {sw_lon},
    {ne_lat}, {ne_lon}
  );
);
out body;
>;
out skel qt;
"""
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        OVERPASS_URL,
        data=data,
        method="get",
        headers={
            "User-Agent": "mapsnap/0.1",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(
            f"Error: Overpass returned HTTP {exc.code}: {exc.reason}", file=sys.stderr
        )
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Error: {exc.reason}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OSM street data for a bounding box via the Overpass API."
    )
    parser.add_argument(
        "sw_lat", type=float, metavar="SW_LAT", help="Southwest latitude"
    )
    parser.add_argument(
        "sw_lon", type=float, metavar="SW_LON", help="Southwest longitude"
    )
    parser.add_argument(
        "ne_lat", type=float, metavar="NE_LAT", help="Northeast latitude"
    )
    parser.add_argument(
        "ne_lon", type=float, metavar="NE_LON", help="Northeast longitude"
    )
    parser.add_argument(
        "--output", metavar="FILE", required=True, help="Output JSON file"
    )
    args = parser.parse_args()

    print(
        f"Querying Overpass for bbox ({args.sw_lat}, {args.sw_lon}) → ({args.ne_lat}, {args.ne_lon})…",
        file=sys.stderr,
    )
    result = download_osm(args.sw_lat, args.sw_lon, args.ne_lat, args.ne_lon)
    n_elements = len(result.get("elements", []))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {n_elements} elements to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
