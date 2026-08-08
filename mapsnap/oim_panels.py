"""Build ``oim/pN.panels.json`` from OIM's own region boundaries (#273).

``oim-split-truth`` reverse-engineers panel polygons by template-matching OIM's
whitened region JPEGs and thresholding "non-white" content. That fails two ways
(#273): regions that were never georeferenced are invisible (they are absent
from main.iiif.json, which drives the crop downloads), and on bright scans the
paper is *whiter than the mask* (fargo p12: paper 87% exactly-255, JPEG-dithered
mask at 253–254), so the extracted polygons collapse onto ink blobs.

OIM publishes the answer directly. The document page embeds, per region, an
exact ``boundary`` polygon in canvas pixel coordinates plus ``division_number``
and ``georeferenced`` — for every region, georeferenced or not — and the
volunteer's ``cutlines``. The map page embeds the full document listing. This
command reads those and writes:

- ``oim/pN.panels.json`` — the standard panels file (canvas coordinates, one
  ring per region, ordered by division number), now exact and complete;
- ``oim/pN.cutlines.json`` — the dividing polylines, kept because they are
  literal ground truth for "split only on dividing lines" (#83 Phase 2).

Only pages OIM actually cut (>= 2 regions) get files. Existing panels.json
files are overwritten: the API boundaries supersede the template-matched ones.

    mapsnap oim-panels data/fargo_nd_1958
    mapsnap oim-panels data/fargo_nd_1958 --pages p12 p59
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from mapsnap.utils import label_to_page_key


def title_page_key(title: str) -> str | None:
    """Page key from an OIM document title.

    Most volumes use plain page labels ("Fargo, N.D. | 1958 p12" -> p12);
    volumes ingested from LOC's sb-format use it in titles too
    ("Washington, D.C. | 1916 | Vol. 2 psb002600" -> p260: sb + 5-digit page +
    suffix char, '0' meaning none), which label_to_page_key does not parse.
    """
    key = label_to_page_key(title)
    if key:
        return key
    m = re.search(r"\bpsb(\d{5})([a-z0-9])$", title.strip(), re.IGNORECASE)
    if m:
        suffix = m.group(2).lower()
        return f"p{int(m.group(1))}{'' if suffix == '0' else suffix}"
    return None


OIM_BASE = "https://oldinsurancemaps.net"
FETCH_DELAY_S = 0.3  # politeness delay between document-page fetches


def fetch(url: str) -> str:
    """Fetch a URL as text (split out for tests to monkeypatch)."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def embedded_json(html: str, key: str):
    """Decode the JSON value following ``"<key>": `` in an OIM page.

    OIM server-renders its Vue props straight into the page, so the arrays we
    need appear as ordinary JSON after their key. ``raw_decode`` parses one
    balanced value and ignores the rest of the document. Returns None if the
    key is absent.
    """
    marker = f'"{key}": '
    index = html.find(marker)
    if index < 0:
        return None
    value, _ = json.JSONDecoder().raw_decode(html[index + len(marker) :])
    return value


def volume_map_slug(volume: Path) -> str:
    """The OIM map slug (e.g. 'sanborn06536_012') for a volume.

    Prefer the volume's own truth file: main.iiif.json's top-level id is the
    OIM mosaic URL (…/iiif/mosaic/<slug>/main-content/), present for both
    OIM-native and LOC-downloaded volumes. Fall back to the LOC manifest for
    volumes without truth.
    """
    truth = volume / "main.iiif.json"
    if truth.exists():
        m = re.search(
            r"/iiif/mosaic/([^/]+)/", json.loads(truth.read_text()).get("id", "")
        )
        if m:
            return m.group(1)
    manifest = json.loads((volume / "manifest.json").read_text())
    m = re.search(r"/item/([^/]+)/", manifest["@id"])
    if not m:
        raise ValueError(f"no OIM slug in {volume}'s main.iiif.json or manifest.json")
    return m.group(1)


def map_documents(slug: str) -> list[dict]:
    """The volume's document listing (id, title, page_number) from its map page."""
    documents = embedded_json(fetch(f"{OIM_BASE}/map/{slug}"), "documents")
    if not documents:
        raise ValueError(f"no document listing on {OIM_BASE}/map/{slug}")
    return documents


def document_regions(doc_id: int) -> tuple[list[dict], list, list | None]:
    """(regions, cutlines, canvas [w, h]) from a document page.

    ``regions`` appears several times in the page (once per Vue component); all
    occurrences describe the same region set, so occurrences are merged by id.
    The canvas size is the document's own image_size — regions carry their crop
    sizes, so it is taken from the document object (the one with no
    division_number) when present, else left None for the caller to fill from
    the local raw scan.
    """
    html = fetch(f"{OIM_BASE}/document/{doc_id}")
    regions: dict[int, dict] = {}
    for match in re.finditer(r'"regions": ', html):
        try:
            value, _ = json.JSONDecoder().raw_decode(html[match.end() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            for region in value:
                # One embedded component lists every region in the VOLUME;
                # keep only this document's.
                if (
                    isinstance(region, dict)
                    and region.get("id") is not None
                    and region.get("document_id") == doc_id
                ):
                    regions.setdefault(region["id"], region)
    cutlines = embedded_json(html, "cutlines") or []
    canvas = None
    doc_marker = f'"id": {doc_id}, '
    index = html.find(doc_marker)
    if index >= 0:
        try:
            doc, _ = json.JSONDecoder().raw_decode(html[html.rfind("{", 0, index) :])
            if isinstance(doc.get("image_size"), list):
                canvas = doc["image_size"]
        except json.JSONDecodeError:
            pass
    return sorted(regions.values(), key=region_division), cutlines, canvas


def region_division(region: dict) -> int:
    """A region's division number, tolerating strings and absences."""
    try:
        return int(region.get("division_number") or 0)
    except (TypeError, ValueError):
        return 0


def region_ring(region: dict, canvas_height: float) -> list[list[float]] | None:
    """A region's boundary as a closed [x, y] ring in image (y-down) coordinates.

    OIM stores boundaries in a GIS-style y-up frame (origin bottom-left):
    kansas_city p493's region-2 crop is provably the TOP-left of the sheet
    (0.956 correlation) while its boundary reads y 4128–7795 of 7795. Flip y
    against the canvas height to get ordinary image coordinates. Full-height
    vertical cuts are flip-invariant, which is what let the bug hide.
    """
    boundary = region.get("boundary") or {}
    coords = boundary.get("coordinates")
    if not coords or boundary.get("type") != "Polygon":
        return None
    ring = [[float(x), canvas_height - float(y)] for x, y in coords[0]]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def write_page_files(
    volume: Path, page_key: str, regions: list[dict], cutlines: list, canvas: list
) -> bool:
    """Write oim/<page>.panels.json (+ cutlines.json); returns False if not cut."""
    width, height = canvas
    rings = [r for r in (region_ring(region, height) for region in regions) if r]
    if len(rings) < 2:
        return False
    oim_dir = volume / "oim"
    oim_dir.mkdir(exist_ok=True)
    (oim_dir / f"{page_key}.panels.json").write_text(
        json.dumps(
            {
                "image": f"{page_key}.jpg",
                "width": width,
                "height": height,
                "panels": rings,
                "source": "oim-region-boundaries",
                "georeferenced": [bool(r.get("georeferenced")) for r in regions],
            },
            indent=1,
        )
    )
    if cutlines:
        (oim_dir / f"{page_key}.cutlines.json").write_text(
            json.dumps(
                {
                    "image": f"{page_key}.jpg",
                    "width": width,
                    "height": height,
                    "cutlines": [
                        [[float(x), height - float(y)] for x, y in line]
                        for line in cutlines
                    ],
                },
                indent=1,
            )
        )
    return True


def split_page_keys(volume: Path) -> set[str]:
    """Parent keys of pages the volume's truth or oim/ dir already marks as split."""
    keys: set[str] = set()
    truth = volume / "main.iiif.json"
    if truth.exists():
        for item in json.loads(truth.read_text()).get("items", []):
            key = label_to_page_key(str(item.get("label", "")))
            if key and "__" in key:
                keys.add(key.split("__")[0])
    for panels in (volume / "oim").glob("p*.panels.json"):
        keys.add(panels.name.split(".")[0])
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build oim/pN.panels.json from OIM's region boundaries (#273)."
    )
    parser.add_argument("volume", type=Path, help="Volume directory (data/<vol>)")
    parser.add_argument(
        "--pages",
        nargs="*",
        default=None,
        metavar="pN",
        help="Only these page keys (default: every known-split page).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check every document in the volume, not just known-split pages "
        "(finds cuts whose regions were never georeferenced).",
    )
    args = parser.parse_args()

    slug = volume_map_slug(args.volume)
    documents = map_documents(slug)
    # Some volumes carry DUPLICATE documents per page (grand_rapids has two
    # uploads of most sheets, both "prepared"); only one holds the cut
    # regions. Keep every candidate and let the fetch loop pick whichever
    # actually has >= 2 region boundaries.
    by_key: dict[str, list[int]] = {}
    for doc in documents:
        key = title_page_key(str(doc.get("title", "")))
        if key:
            by_key.setdefault(key, []).append(doc["id"])

    if args.pages:
        targets = list(args.pages)
    elif args.all:
        targets = sorted(by_key)
    else:
        targets = sorted(split_page_keys(args.volume))
    written = 0
    for page_key in targets:
        doc_ids = by_key.get(page_key)
        if not doc_ids:
            print(f"  {page_key}: no OIM document", file=sys.stderr)
            continue
        best: tuple[list, list, list | None] = ([], [], None)
        for doc_id in doc_ids:
            candidate = document_regions(doc_id)
            if len(candidate[0]) > len(best[0]):
                best = candidate
            if len(best[0]) >= 2:
                break
            time.sleep(FETCH_DELAY_S)
        regions, cutlines, canvas = best
        if canvas is None:
            raw = args.volume / "raw" / f"{page_key}.jpg"
            if raw.exists():
                from PIL import Image

                canvas = list(Image.open(raw).size)
            else:
                print(f"  {page_key}: no canvas size and no raw scan", file=sys.stderr)
                continue
        if write_page_files(args.volume, page_key, regions, cutlines, canvas):
            written += 1
            print(f"  {page_key}: {len(regions)} regions -> panels.json")
        elif not args.all:
            print(f"  {page_key}: fewer than 2 region boundaries", file=sys.stderr)
        time.sleep(FETCH_DELAY_S)
    print(f"{written} page(s) written for {args.volume.name} ({slug})")


if __name__ == "__main__":
    main()
