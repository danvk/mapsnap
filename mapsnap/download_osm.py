"""Download OSM street data using the Overpass API.

Writes into a volume directory:

    streets.osm.json   the named highways inside the relation
    r<id>.json         the relation's own boundary geometry

The boundary is what the volume viewer draws as a red ring, so a page whose
ground falls OUTSIDE the download is visible as such -- its streets are absent
from the vocabulary and it cannot be fit at all, however good the reads are
(richmond p383's BENTON/VAWTER/FENWICK appear nowhere in its streets.txt).
Which boundary belongs to a volume is recorded in its mapsnap.json manifest
under params.relation, so a leftover r<id>.json from an earlier relation is
simply ignored rather than needing to be cleaned up.

Usage:
    python download_osm.py r<relation_id> DIR
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

MAX_ATTEMPTS = 5  # total Overpass attempts before giving up
INITIAL_RETRY_DELAY = 2.0  # seconds before the first retry; doubles each attempt
MAX_RETRY_DELAY = 60.0  # cap on the exponential backoff delay


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


def form_relation_geometry_query(relation_id: int) -> str:
    """Query for the relation's own boundary, with each member way's geometry.

    ``out geom`` inlines every way's coordinates on the relation element, which
    is what the viewer reads: it draws one line per member way rather than
    stitching a ring, so no polygon assembly is needed in the browser.
    """
    return f"""[out:json][timeout:180];
rel({relation_id});
out geom;
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
        usage="%(prog)s r<relation_id> DIR",
    )
    parser.add_argument(
        "relation",
        metavar="RELATION",
        help="OSM relation to download, as 'r<relation_id>' (e.g. r3864712)",
    )
    parser.add_argument(
        "output_dir",
        metavar="DIR",
        help="Volume directory to write streets.osm.json and r<id>.json into",
    )
    parsed = parser.parse_args()

    if not (parsed.relation.startswith("r") and parsed.relation[1:].isdigit()):
        parser.error(
            f"Expected an OSM relation like 'r3864712', got {parsed.relation!r}"
        )
    relation_id = int(parsed.relation[1:])

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "streets.osm.json"
    query = form_osm_query_relation(relation_id)
    print(f"Querying relation r{relation_id}.", file=sys.stderr)

    print(f"Running Overpass query:\n{query}", file=sys.stderr)
    result = download_osm(query)
    n_elements = len(result.get("elements", []))
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {n_elements} elements to {output}", file=sys.stderr)

    boundary = download_osm(form_relation_geometry_query(relation_id))
    boundary_path = out_dir / f"r{relation_id}.json"
    boundary_path.write_text(json.dumps(boundary, indent=2))
    members = len((boundary.get("elements") or [{}])[0].get("members", []))
    print(
        f"Wrote the r{relation_id} boundary ({members} ways) to {boundary_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
