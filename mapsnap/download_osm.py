"""Download OSM street data using the Overpass API.

Usage:
    python download_osm.py r<relation_id> --output FILE
    python download_osm.py <sw_lat> <sw_lon> <ne_lat> <ne_lon> --output FILE
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

MAX_ATTEMPTS = 5  # total Overpass attempts before giving up
INITIAL_RETRY_DELAY = 2.0  # seconds before the first retry; doubles each attempt
MAX_RETRY_DELAY = 60.0  # cap on the exponential backoff delay


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


def download_osm(
    query: str,
    max_attempts: int = MAX_ATTEMPTS,
    initial_delay: float = INITIAL_RETRY_DELAY,
) -> dict:
    """Submit an Overpass query and return the parsed JSON response.

    Retries with exponential backoff on transient errors: HTTP 429 (Overpass rate-limits
    when busy) and 5xx (e.g. 504 when overloaded), and network/timeout errors. Exits with
    an error message on a non-transient HTTP error or once all attempts are exhausted.
    """
    data = urllib.parse.urlencode({"data": query}).encode()
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
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
            transient = exc.code == 429 or 500 <= exc.code < 600
            if not transient or attempt == max_attempts:
                sys.exit(f"Error: Overpass returned HTTP {exc.code}: {exc.reason}")
            print(
                f"  Overpass HTTP {exc.code} ({exc.reason}); retrying in {delay:.0f}s "
                f"(attempt {attempt}/{max_attempts})",
                file=sys.stderr,
            )
        except urllib.error.URLError as exc:
            if attempt == max_attempts:
                sys.exit(f"Error: {exc.reason}")
            print(
                f"  Overpass request failed ({exc.reason}); retrying in {delay:.0f}s "
                f"(attempt {attempt}/{max_attempts})",
                file=sys.stderr,
            )
        time.sleep(delay)
        delay = min(delay * 2, MAX_RETRY_DELAY)

    # Unreachable: the final attempt always returns or exits above.
    sys.exit("Error: exhausted Overpass retries")


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
