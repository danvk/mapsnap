"""Download OSM street data using the Overpass API.

Usage:
    python download_osm.py r<relation_id> --output FILE
    python download_osm.py <sw_lat> <sw_lon> <ne_lat> <ne_lon> --output FILE
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def form_osm_query_bbox(sw: tuple[float, float], ne: tuple[float, float]) -> str:
    sw_lat, sw_lon = sw
    ne_lat, ne_lon = ne
    return f"""[out:json][timeout:120];
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


def form_osm_query_relation(relation_id: int) -> str:
    # relation_id = 1836428  # Orleans Parish
    # relation_id = 369518  # Kings County aka Brooklyn
    area_id = 3600000000 + relation_id
    return f"""[out:json][timeout:120];

area({area_id})->.searchArea;

(
way(area.searchArea)["highway"]["name"];
);
out body;
>;
out skel qt;
"""


def download_osm(query: str) -> dict:
    """Submit an Overpass query and return the parsed JSON response."""

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
        description="Download OSM street data via the Overpass API.",
        usage="%(prog)s (r<relation_id> | sw_lat sw_lon ne_lat ne_lon) --output FILE",
    )
    parser.add_argument(
        "args",
        nargs="+",
        metavar="ARG",
        help="Either 'r<relation_id>' or four floats: sw_lat sw_lon ne_lat ne_lon",
    )
    parser.add_argument("--output", required=True, help="Output JSON file")
    parsed = parser.parse_args()

    positional = parsed.args
    if (
        len(positional) == 1
        and positional[0].startswith("r")
        and positional[0][1:].isdigit()
    ):
        relation_id = int(positional[0][1:])
        query = form_osm_query_relation(relation_id)
        print(f"Querying relation r{relation_id}.", file=sys.stderr)
    elif len(positional) == 4:
        try:
            sw_lat, sw_lon, ne_lat, ne_lon = [float(x) for x in positional]
        except ValueError:
            parser.error(
                "Bounding box arguments must be four floats: sw_lat sw_lon ne_lat ne_lon"
            )
        query = form_osm_query_bbox((sw_lat, sw_lon), (ne_lat, ne_lon))
        print(
            f"Querying bbox ({sw_lat}, {sw_lon}) → ({ne_lat}, {ne_lon}).",
            file=sys.stderr,
        )
    else:
        parser.error(
            "Pass either 'r<relation_id>' or four floats: sw_lat sw_lon ne_lat ne_lon"
        )

    print(f"Running Overpass query:\n{query}", file=sys.stderr)
    result = download_osm(query)
    n_elements = len(result.get("elements", []))
    with open(parsed.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {n_elements} elements to {parsed.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
