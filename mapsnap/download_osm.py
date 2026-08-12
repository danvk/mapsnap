"""Download OSM street data using the Overpass API.

Writes into a volume directory:

    streets.osm.json   the named highways inside the relation, plus a buffer
    r<id>.json         the boundary of the area actually downloaded

The boundary is what the volume viewer draws as a red ring, so a page whose
ground falls OUTSIDE the download is visible as such -- its streets are absent
from the vocabulary and it cannot be fit at all, however good the reads are
(richmond p383's BENTON/VAWTER/FENWICK appear nowhere in its streets.txt).
Which boundary belongs to a volume is recorded in its mapsnap.json manifest
under params.relation, so a leftover r<id>.json from an earlier relation is
simply ignored rather than needing to be cleaned up.

An administrative relation is not the area a volume maps. Sheets routinely
cover ground just outside the modern line -- richmond is an independent city
with no containing county, and its 1925 volume reaches up to 618 m past the
boundary into Henrico. So the download is buffered by BUFFER_M (default 1 km)
and `r<id>.json` describes the BUFFERED extent, because that is what the ring
is claiming: the area whose streets we actually have.

The buffer is deliberately small. Widening it costs same-name ambiguity, not
just bytes: 5 km around Kings County reaches lower Manhattan, where 165 of 482
street names already exist in brooklyn vol 1 -- a second numbered grid across
the river, which is the aliasing behind chicago p53N's 6,735 ft catastrophe.

Usage:
    python download_osm.py r<relation_id> DIR [--buffer-m 1000]
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

BUFFER_M = 1000.0
"""How far past the relation to download, in metres.

Richmond's worst page reaches 618 m beyond its city line; 1 km clears every
affected page in the corpus with margin. Costs +4-7% area on county-seeded
volumes and +29% on a compact one like Kings County."""

SIMPLIFY_M = 50.0
"""Douglas-Peucker tolerance for the buffered ring before it goes into an
Overpass `poly:` filter. A buffered boundary has thousands of vertices and the
query is a URL-encoded string; 50 m is far below the buffer itself, so it
cannot pull the edge inside the unbuffered relation."""

MAX_ATTEMPTS = 5  # total Overpass attempts before giving up
INITIAL_RETRY_DELAY = 2.0  # seconds before the first retry; doubles each attempt
MAX_RETRY_DELAY = 60.0  # cap on the exponential backoff delay


def form_osm_query_polygons(rings: list[list[tuple[float, float]]]) -> str:
    """Named highways inside any of ``rings`` ((lon, lat) sequences).

    A `poly:` filter rather than `area(3600000000 + id)`: the area form can only
    select the relation exactly, and the point of the buffer is to reach past
    it. Several rings because buffering a multi-part boundary (islands, or a
    relation whose parts do not touch) can stay multi-part.
    """
    clauses = "\n".join(
        'way["highway"]["name"](poly:"'
        + " ".join(f"{lat:.6f} {lon:.6f}" for lon, lat in ring)
        + '");'
        for ring in rings
    )
    return f"""[out:json][timeout:180];
(
{clauses}
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


def relation_rings(boundary: dict) -> list[list[tuple[float, float]]]:
    """(lon, lat) rings of a relation's member ways, from an ``out geom`` doc."""
    element = (boundary.get("elements") or [{}])[0]
    return [
        [(p["lon"], p["lat"]) for p in member["geometry"]]
        for member in element.get("members") or []
        if member.get("type") == "way" and member.get("geometry")
    ]


def buffered_rings(
    rings: list[list[tuple[float, float]]], buffer_m: float
) -> list[list[tuple[float, float]]]:
    """Outline(s) of ``rings`` grown by ``buffer_m`` metres.

    The member ways are stitched into whatever polygons they form and buffered
    in a local equirectangular metre frame; a relation whose ways do not close
    (or that has none) falls back to buffering the lines themselves, which
    still yields a usable download area rather than nothing.
    """
    from shapely.geometry import LineString, MultiPolygon, Polygon
    from shapely.ops import linemerge, polygonize, unary_union

    lats = [lat for ring in rings for _, lat in ring]
    lons = [lon for ring in rings for lon, _ in ring]
    if not lats:
        return []
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    lines = [
        LineString([((lon - lon0) * kx, (lat - lat0) * ky) for lon, lat in ring])
        for ring in rings
        if len(ring) >= 2
    ]
    # Stitch the member ways into whatever closed rings they form; a boundary
    # whose ways do not close still buffers usefully as lines.
    polygons = list(polygonize(linemerge(lines)))
    shape = unary_union(polygons) if polygons else unary_union(lines)
    grown = shape.buffer(buffer_m).simplify(SIMPLIFY_M)
    parts = list(grown.geoms) if isinstance(grown, MultiPolygon) else [grown]
    return [
        [(x / kx + lon0, y / ky + lat0) for x, y in part.exterior.coords]
        for part in parts
        if isinstance(part, Polygon)
    ]


def boundary_document(
    relation_id: int,
    boundary: dict,
    rings: list[list[tuple[float, float]]],
    buffer_m: float,
) -> dict:
    """The downloaded area, in the shape the viewer already reads.

    Same schema as an Overpass ``out geom`` relation (one member way per ring)
    so the overlay needs no change, but the geometry is the BUFFERED outline --
    the ring has to describe what was actually downloaded, not the
    administrative line we grew it from.
    """
    tags = dict((boundary.get("elements") or [{}])[0].get("tags") or {})
    tags["mapsnap:buffer_m"] = str(int(buffer_m))
    return {
        "elements": [
            {
                "type": "relation",
                "id": relation_id,
                "tags": tags,
                "members": [
                    {
                        "type": "way",
                        "role": "outer",
                        "geometry": [{"lat": lat, "lon": lon} for lon, lat in ring],
                    }
                    for ring in rings
                ],
            }
        ]
    }


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
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=BUFFER_M,
        metavar="M",
        help=(
            "Download this far past the relation boundary (default: %(default)s). "
            "Sheets routinely map ground just outside the modern line."
        ),
    )
    parsed = parser.parse_args()

    if not (parsed.relation.startswith("r") and parsed.relation[1:].isdigit()):
        parser.error(
            f"Expected an OSM relation like 'r3864712', got {parsed.relation!r}"
        )
    relation_id = int(parsed.relation[1:])

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The boundary comes FIRST now: the streets query is a polygon filter built
    # from it, so the buffer has to exist before we can ask for streets.
    print(f"Fetching the r{relation_id} boundary.", file=sys.stderr)
    boundary = download_osm(form_relation_geometry_query(relation_id))
    rings = relation_rings(boundary)
    if not rings:
        sys.exit(f"Error: relation r{relation_id} has no member way geometry")
    grown = buffered_rings(rings, parsed.buffer_m)
    if not grown:
        sys.exit(f"Error: could not buffer r{relation_id}'s boundary")
    print(
        f"  {len(rings)} member way(s) -> {len(grown)} ring(s) buffered by "
        f"{parsed.buffer_m:.0f} m ({sum(len(r) for r in grown)} vertices)",
        file=sys.stderr,
    )

    query = form_osm_query_polygons(grown)
    result = download_osm(query)
    n_elements = len(result.get("elements", []))
    output = out_dir / "streets.osm.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {n_elements} elements to {output}", file=sys.stderr)

    boundary_path = out_dir / f"r{relation_id}.json"
    boundary_path.write_text(
        json.dumps(
            boundary_document(relation_id, boundary, grown, parsed.buffer_m), indent=2
        )
    )
    print(
        f"Wrote the downloaded area ({parsed.buffer_m:.0f} m past r{relation_id}) "
        f"to {boundary_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
