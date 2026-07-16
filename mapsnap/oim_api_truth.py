"""Build a truth IIIF Georeference AnnotationPage from the OldInsuranceMaps API.

OIM's own AnnotationPage endpoint (/iiif/mosaic/<map>/main-content/) crashes when
any layer uses a helmert (two-GCP) fit: its serializer (ohmg/extensions/iiif.py
get_body) only handles the poly1 and tps transformation types and raises on
anything else. This module reconstructs an equivalent AnnotationPage from the
beta2 API instead: georeference sessions supply GCPs, transformation types, and
region boundaries; documents supply the LOC image service URLs and
full-resolution canvas dimensions.

GCP pixel coordinates from the API are relative to the region image, and region
boundaries use a bottom-left origin, so both are mapped into top-left-origin
canvas coordinates the same way OIM's serializer does:

    resourceCoords = [pixel_x + xmin, pixel_y + (canvas_height - ymax)]

where (xmin, ymax) come from the region boundary's extent.

Split pages: each truth item is labeled "... psbNNNNN [division]", and
oim/<parent>.panels.json files (division-indexed rings in canvas coordinates)
are written so `mapsnap compare` can match OIM's splits to ours by IoU.

Usage:
    OIM_API_KEY=... mapsnap oim-truth data/washington_dc_1916_vol_2
    mapsnap oim-truth data/washington_dc_1916_vol_2 --map-id sanborn01227_003 \\
        --output data/washington_dc_1916_vol_2/main.iiif.json
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from mapsnap.split import PanelsJson
from mapsnap.utils import source_id_to_page_key

API_BASE = "https://oldinsurancemaps.net/api/beta2"
GEOREF_CONTEXT = "http://iiif.io/api/extension/georef/1/context.json"
PRESENTATION_CONTEXT = "http://iiif.io/api/presentation/3/context.json"

# A degenerate stand-in ring for split divisions with no georeferenced truth
# item, keeping panels.json 1-based indexing aligned with OIM division numbers.
PLACEHOLDER_RING = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


def fetch_api(path: str, params: dict[str, str], api_key: str) -> Any:
    """GET an OIM beta2 API endpoint as JSON, retrying transient 5xx errors."""
    url = f"{API_BASE}/{path}/?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"X-API-Key": api_key})
    last_error: Exception | None = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2**attempt)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code < 500:
                raise
            last_error = error
    raise RuntimeError(f"OIM API request failed after retries: {url}") from last_error


def fetch_georef_sessions(map_id: str, api_key: str) -> list[dict[str, Any]]:
    """All successful georeference sessions for a map, following pagination."""
    sessions: list[dict[str, Any]] = []
    while True:
        page = fetch_api(
            "sessions",
            {"map": map_id, "type": "g", "limit": "500", "offset": str(len(sessions))},
            api_key,
        )
        sessions.extend(page["items"])
        if len(sessions) >= page["count"] or not page["items"]:
            break
    return [s for s in sessions if s.get("status") == "success"]


def boundary_extent(boundary: dict[str, Any]) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of a GeoJSON polygon's outer ring."""
    ring = boundary["coordinates"][0]
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return min(xs), min(ys), max(xs), max(ys)


# Convert a bottom-left-origin ring to top-left-origin canvas coordinates.
def flip_ring(ring: list[list[float]], canvas_height: float) -> list[list[float]]:
    return [[x, canvas_height - y] for x, y in ring]


# A region's split division number as an int, or None for unsplit regions.
# The API serializes it as a string ("2"), empty, or null.
def region_division(region: dict[str, Any]) -> int | None:
    value = region.get("division_number")
    return int(value) if value not in (None, "") else None


def transformation_body(name: str | None, gcp_count: int) -> dict[str, Any]:
    """The georef-extension transformation object for an OIM transformation name.

    Unknown or missing names fall back to mapsnap's own convention: helmert for
    exactly two GCPs, first-order polynomial otherwise.
    """
    if name == "helmert":
        return {"type": "helmert"}
    if name == "tps":
        return {"type": "thinPlateSpline"}
    match = re.fullmatch(r"poly(\d)", name or "")
    if match:
        return {"type": "polynomial", "options": {"order": int(match.group(1))}}
    if name:
        print(f"warning: unknown OIM transformation {name!r}", file=sys.stderr)
    if gcp_count == 2:
        return {"type": "helmert"}
    return {"type": "polynomial", "options": {"order": 1}}


def truth_annotation(session: dict[str, Any], document: dict[str, Any]) -> dict:
    """One georef Annotation built from a session and its parent document.

    Mirrors the item shape of OIM's /iiif/mosaic/ output: the target source is
    the document's LOC image service at full-resolution canvas dimensions, the
    selector is the region's cut polygon, and the GCP resourceCoords are the
    region-relative pixels offset into the canvas frame.
    """
    region = session["reg2"]
    layer_id = session["lyr2"]["id"]
    canvas_width, canvas_height = document["image_size"]
    xmin, _, _, ymax = boundary_extent(region["boundary"])
    y_offset = canvas_height - ymax

    features = []
    for gcp in session["data"]["gcps"]["features"]:
        pixel_x, pixel_y = gcp["properties"]["image"]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "resourceCoords": [pixel_x + xmin, pixel_y + y_offset],
                },
                "geometry": gcp["geometry"],
            }
        )

    selector_ring = flip_ring(region["boundary"]["coordinates"][0], canvas_height)
    selector_points = " ".join(f"{x:g},{y:g}" for x, y in selector_ring)

    return {
        "id": f"https://oldinsurancemaps.net/iiif/resource/{layer_id}/",
        "type": "Annotation",
        "@context": [GEOREF_CONTEXT, PRESENTATION_CONTEXT],
        "label": region["title"],
        "creator": [
            {
                "id": f"https://oldinsurancemaps.net/profile/{region['created_by']}",
                "type": "Person",
            }
        ],
        "motivation": "georeferencing",
        "target": {
            "id": f"https://oldinsurancemaps.net/iiif/selector/{layer_id}/",
            "type": "SpecificResource",
            "source": {
                "id": document["iiif_info"],
                "type": "ImageService2",
                "height": canvas_height,
                "width": canvas_width,
            },
            "selector": {
                "type": "SvgSelector",
                "value": f'<svg><polygon points="{selector_points}" /></svg>',
            },
        },
        "body": {
            "id": f"https://oldinsurancemaps.net/iiif/gcps/{layer_id}/",
            "type": "FeatureCollection",
            "transformation": transformation_body(
                session["data"].get("transformation"), len(features)
            ),
            "features": features,
        },
    }


def split_panels(
    sessions: list[dict[str, Any]], document: dict[str, Any], parent_key: str
) -> PanelsJson:
    """Canvas-coordinate panels.json for one split document.

    Rings are indexed by OIM division number (panels[i-1] = division i, matching
    the "[i]" suffix in truth labels); divisions with no georeferenced session
    get a degenerate placeholder ring so the indexing stays aligned.
    """
    canvas_width, canvas_height = document["image_size"]
    by_division = {
        division: s
        for s in sessions
        if (division := region_division(s["reg2"])) is not None
    }
    rings: list[list[list[float]]] = []
    for division in range(1, max(by_division) + 1):
        session = by_division.get(division)
        if session is None:
            rings.append(PLACEHOLDER_RING)
        else:
            ring = session["reg2"]["boundary"]["coordinates"][0]
            rings.append(flip_ring(ring, canvas_height))
    return {
        "image": f"{parent_key}.jpg",
        "width": canvas_width,
        "height": canvas_height,
        "panels": rings,
    }


def build_annotation_page(
    sessions: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    map_id: str,
) -> tuple[dict, dict[str, PanelsJson]]:
    """Assemble the AnnotationPage and per-parent split panels from API data.

    Returns (annotation_page, {parent_page_key: panels_json}). Sessions whose
    document lacks an image service or dimensions are skipped with a warning.
    """
    document_by_id = {d["id"]: d for d in documents}
    sessions_by_document: dict[int, list[dict[str, Any]]] = {}
    items_with_keys: list[tuple[str, int, dict]] = []
    for session in sessions:
        document = document_by_id.get(session["reg2"]["document_id"])
        if (
            not document
            or not document.get("iiif_info")
            or not document.get("image_size")
        ):
            title = session["reg2"]["title"]
            print(
                f"warning: skipping {title!r}: no document image info", file=sys.stderr
            )
            continue
        sessions_by_document.setdefault(document["id"], []).append(session)
        item = truth_annotation(session, document)
        division = region_division(session["reg2"]) or 0
        items_with_keys.append((document["slug"], division, item))

    items_with_keys.sort(key=lambda entry: (entry[0], entry[1]))
    items = [item for _, _, item in items_with_keys]

    panels_by_parent: dict[str, PanelsJson] = {}
    for document_id, document_sessions in sessions_by_document.items():
        divisions = [region_division(s["reg2"]) for s in document_sessions]
        if not any(divisions):
            continue
        document = document_by_id[document_id]
        parent_key = source_id_to_page_key(document["iiif_info"], "")
        panels_by_parent[parent_key] = split_panels(
            document_sessions, document, parent_key
        )

    label = sessions[0]["map"]["title"] if sessions else map_id
    page = {
        "id": f"https://oldinsurancemaps.net/iiif/mosaic/{map_id}/main-content/",
        "type": "AnnotationPage",
        "@context": ["http://www.w3.org/ns/anno.jsonld"],
        "label": f"Mosaic of main content, {label}",
        "items": items,
    }
    return page, panels_by_parent


# The map identifier from an LOC manifest URL like
# "https://www.loc.gov/item/sanborn01227_003/manifest.json".
def map_id_from_manifest(manifest_path: Path) -> str | None:
    data = json.loads(manifest_path.read_text())
    match = re.search(r"/item/([^/]+)/", data.get("@id", ""))
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a truth IIIF AnnotationPage from the OldInsuranceMaps API."
    )
    parser.add_argument(
        "volume_dir",
        type=Path,
        help="Volume directory (e.g. data/washington_dc_1916_vol_2); "
        "the map id is read from its manifest.json unless --map-id is given.",
    )
    parser.add_argument("--map-id", help="OIM map identifier, e.g. sanborn01227_003")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OIM_API_KEY"),
        help="OIM API key (default: $OIM_API_KEY)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output AnnotationPage path (default: <volume_dir>/main.iiif.json)",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("an OIM API key is required (--api-key or $OIM_API_KEY)")
    map_id = args.map_id
    if not map_id:
        manifest_path = args.volume_dir / "manifest.json"
        if manifest_path.exists():
            map_id = map_id_from_manifest(manifest_path)
        if not map_id:
            parser.error(
                f"could not determine map id from {manifest_path}; use --map-id"
            )

    print(f"Fetching OIM data for {map_id}…", file=sys.stderr)
    documents = fetch_api("documents", {"map": map_id}, args.api_key)
    sessions = fetch_georef_sessions(map_id, args.api_key)
    print(
        f"{len(documents)} documents, {len(sessions)} georeference sessions",
        file=sys.stderr,
    )

    page, panels_by_parent = build_annotation_page(sessions, documents, map_id)

    output = args.output or args.volume_dir / "main.iiif.json"
    output.write_text(json.dumps(page, indent=2))
    transform_counts: dict[str, int] = {}
    for item in page["items"]:
        kind = item["body"]["transformation"]["type"]
        transform_counts[kind] = transform_counts.get(kind, 0) + 1
    summary = ", ".join(
        f"{count} {kind}" for kind, count in sorted(transform_counts.items())
    )
    print(f"Wrote {len(page['items'])} annotations ({summary}) to {output}")

    if panels_by_parent:
        oim_dir = args.volume_dir / "oim"
        oim_dir.mkdir(exist_ok=True)
        for parent_key, panels in sorted(panels_by_parent.items()):
            (oim_dir / f"{parent_key}.panels.json").write_text(
                json.dumps(panels, indent=2)
            )
        print(f"Wrote {len(panels_by_parent)} split panels.json files to {oim_dir}/")


if __name__ == "__main__":
    main()
