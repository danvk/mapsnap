"""Build the public dump of a corpus run: its IIIF files, for loc.gov and Chronoscope, and items.tsv.

    uv run python scripts/atlas/publish_run.py \\
        --out ~/Documents/mapsnap/mapsnap.org/runs/v1.3 \\
        --base-url https://mapsnap.org/runs/v1.3

Reads a corpus run's published annotations (one ``<item>.iiif.json`` per item, plus
its key maps) and writes, under ``--out``:

- ``iiif/loc/<item>.main.iiif.json`` and ``<item>.keymap.iiif.json``, pointing at
  loc.gov's image services;
- ``iiif/chronoscope/`` with the same names, each page repointed at the Chronoscope
  CDN, its control points and clip polygon rescaled to the CDN's 25% images;
- ``items.tsv``, one row per item, including those that end up with nothing to
  publish, which get no files.

Pages and items are filtered to match what the pipeline does now:

- a page whose sheet spans more than MAX_PAGE_EXTENT_M of ground is dropped (#503);
- in a volume of at most SMALL_VOLUME_MAX_PAGES sheets with a catalogue location, a
  page centered more than SMALL_VOLUME_RADIUS_M from it is dropped (#531);
- an item whose key map was georeferenced to under 2.5 km corner to corner is
  withheld (#525), from the list of such items (``--small-keymaps``);
- a page that is also the item's key map is dropped: the key-map file has it,
  placed as a key map rather than at street-sheet scale (#542);
- a skeleton sheet's pages are dropped when any page of its full-color sheet is
  kept, even when either one is split into panels (#545);
- an item with no page left is withheld, unless it has a key map, which is
  published on its own.

Each file states its creator and license (the ODbL: the control points are
OpenStreetMap positions) once, at the top, and its ``id`` is its published URL.
Files are minified.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from gazetteer import STATE_CODES_BY_LOWERCASE

from mapsnap.annotation_transform import Transform, page_transform
from mapsnap.compare_iiif_georef import redundant_skeleton_keys
from mapsnap.loc_fit import SMALL_VOLUME_MAX_PAGES, SMALL_VOLUME_RADIUS_M
from mapsnap.loc_mirror import keep_sheet
from mapsnap.make_iiif_georef import CREATOR, RIGHTS
from mapsnap.osm_snap import MAX_PAGE_EXTENT_M

DOCUMENTS = Path.home() / "Documents/mapsnap"
CDN_BASE = "https://cdn.chronoscope.io/mapsnap"
# Where the files for each image source go, and so their URLs.
SOURCES = ("loc", "chronoscope")
EARTH_RADIUS_M = 6_371_008.8

# A loc.gov image service, and what the pipeline appends to it for an
# annotation's own ids: an optional split-panel suffix and a path such as /georef.
LOC_SERVICE = re.compile(
    r"^https://tile\.loc\.gov/image-services/iiif/(service:[^/]+?)(__\d+)?(/.*)?$"
)
POINTS = re.compile(r'points="([^"]*)"')


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lon, lat) points."""
    lat1, lat2 = math.radians(a[1]), math.radians(b[1])
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(math.radians(b[0] - a[0]) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def sheet_extent_m(annotation: dict, transform: Transform) -> float:
    """Ground diagonal of the whole sheet an annotation is drawn on, in metres.

    The sheet, not the split panel: that is what #503 measured its cap against. The
    larger of the two diagonals, since an affine need not keep them equal.
    """
    source = annotation["target"]["source"]
    width, height = source["width"], source["height"]
    corners = [
        transform(0, 0),
        transform(width, 0),
        transform(width, height),
        transform(0, height),
    ]
    return max(haversine_m(corners[0], corners[2]), haversine_m(corners[1], corners[3]))


def page_center(annotation: dict, transform: Transform) -> tuple[float, float]:
    """Ground (lon, lat) of the middle of a page's clip outline, or of its sheet without one."""
    source = annotation["target"]["source"]
    selector = annotation["target"].get("selector") or {}
    match = POINTS.search(selector.get("value", ""))
    if match:
        xy = np.array(
            [[float(v) for v in pair.split(",")] for pair in match.group(1).split()]
        )
        return transform(float(xy[:, 0].mean()), float(xy[:, 1].mean()))
    return transform(source["width"] / 2, source["height"] / 2)


def page_key(annotation: dict) -> str:
    """The page stem an annotation's label names: "... p4" is p4, "... p4 [2]" is p4__2."""
    tokens = str(annotation.get("label", "")).split()
    if len(tokens) >= 2 and re.fullmatch(r"\[\d+\]", tokens[-1]):
        return f"{tokens[-2]}__{tokens[-1][1:-1]}"
    return tokens[-1] if tokens else "?"


def item_id(path: Path) -> str:
    """The LoC item an annotation file is named for; some carry a suffix (sanborn04424_001.5)."""
    return path.name.removesuffix(".iiif.json")


@dataclass
class FilteredPages:
    """What survives the page filters, and what was withheld and why."""

    kept: list[dict] = field(default_factory=list)
    withheld: list[tuple[str, str]] = field(default_factory=list)
    too_large: int = 0
    too_far: int = 0


def filter_pages(
    annotations: list[dict], sheets: int, location: tuple[float, float] | None
) -> FilteredPages:
    """Drop the pages the pipeline would no longer place.

    A page whose sheet spans more than MAX_PAGE_EXTENT_M goes (#503). So, in a volume
    of at most SMALL_VOLUME_MAX_PAGES sheets with a catalogue ``location`` (lon, lat),
    does a page centered more than SMALL_VOLUME_RADIUS_M from it (#531). A volume of
    unknown size (``sheets`` 0) is not treated as small. A page failing both counts
    as too large. A page with too few control points to fit is kept.
    """
    small = 0 < sheets <= SMALL_VOLUME_MAX_PAGES
    result = FilteredPages()
    for annotation in annotations:
        transform = page_transform(annotation)
        if transform is not None:
            extent = sheet_extent_m(annotation, transform)
            if extent > MAX_PAGE_EXTENT_M:
                result.too_large += 1
                result.withheld.append(
                    (page_key(annotation), f"sheet {extent / 1000:.1f} km across")
                )
                continue
            if small and location is not None:
                distance = haversine_m(page_center(annotation, transform), location)
                if distance > SMALL_VOLUME_RADIUS_M:
                    result.too_far += 1
                    reason = f"{distance / 1000:.1f} km from the catalogue location"
                    result.withheld.append((page_key(annotation), reason))
                    continue
        result.kept.append(annotation)
    return result


def drop_keymap_pages(
    annotations: list[dict], keymap: list[dict]
) -> tuple[list[dict], list[str]]:
    """The main-content pages that are not also the item's key map, and the keys of those that are.

    corpus-v1 georeferenced a key-map sheet twice: as a key map, from its
    full-resolution scan, and as an ordinary page, at street-sheet scale (#542).
    The key-map annotation is the right one.
    """
    keymap_keys = {page_key(annotation).lower() for annotation in keymap}
    kept = [a for a in annotations if page_key(a).lower() not in keymap_keys]
    dropped = [page_key(a) for a in annotations if page_key(a).lower() in keymap_keys]
    return kept, dropped


def drop_skeleton_pages(annotations: list[dict]) -> tuple[list[dict], list[str]]:
    """The pages left once skeletons yield to their full-color sheets, and the keys dropped.

    A skeleton sheet (pNs) maps the same ground as its full-color sheet (pN). The
    pipeline paired them by exact key until #547, so a pair where either sheet was
    split into panels (p3__1 beside p3s, or p2 beside p2s__1) was published twice.
    Pairing is make_iiif_georef's, by sheet.
    """
    sheets = {page_key(annotation).split("__")[0] for annotation in annotations}
    skeletons = redundant_skeleton_keys(sheets, sheets)
    kept = [a for a in annotations if page_key(a).split("__")[0] not in skeletons]
    dropped = [
        page_key(a) for a in annotations if page_key(a).split("__")[0] in skeletons
    ]
    return kept, dropped


def item_status(
    annotations: list[dict],
    kept: list[dict],
    small_keymap: bool = False,
    has_keymap: bool = False,
) -> str:
    """Whether an item is published, and if not, why.

    An item with no page left but a key map still publishes its key map.
    """
    if small_keymap:
        return "withheld: key map under 2.5 km"
    if kept:
        return "published"
    if has_keymap:
        return "published: key map only"
    if not annotations:
        return "no page placed"
    return "withheld: every page filtered"


def strip_creators(annotation: dict) -> None:
    """Remove an annotation's creator and rights, and each GCP's creator: the page states them once."""
    annotation.pop("creator", None)
    annotation.pop("rights", None)
    for feature in annotation["body"]["features"]:
        feature.get("properties", {}).pop("creator", None)


def for_cdn(annotation: dict) -> dict:
    """A copy of a loc.gov annotation repointed at the Chronoscope CDN.

    The CDN serves the mirrored 25% scans at ceil(LoC size / 4), keyed by LoC's own
    service id, so the control points and clip polygon are rescaled into that
    frame per axis. The annotation's own ids follow the image they describe.
    Raises ValueError for an annotation whose image is not a loc.gov service.
    """
    result = json.loads(json.dumps(annotation))
    source = result["target"]["source"]
    match = LOC_SERVICE.match(source.get("id") or "")
    if not match:
        raise ValueError(f"not a loc.gov image service: {source.get('id')}")
    width, height = math.ceil(source["width"] / 4), math.ceil(source["height"] / 4)
    scale_x, scale_y = width / source["width"], height / source["height"]
    result["target"]["source"] = {
        "id": f"{CDN_BASE}/{match.group(1)}",
        "type": "ImageService3",
        "width": width,
        "height": height,
    }
    for feature in result["body"]["features"]:
        x, y = feature["properties"]["resourceCoords"]
        feature["properties"]["resourceCoords"] = [
            round(x * scale_x, 1),
            round(y * scale_y, 1),
        ]

    def rescale(points: re.Match) -> str:
        pairs = []
        for pair in points.group(1).split():
            x, y = (float(v) for v in pair.split(","))
            new_x = round(min(max(x * scale_x, 0.0), width), 1)
            new_y = round(min(max(y * scale_y, 0.0), height), 1)
            pairs.append(f"{new_x},{new_y}")
        return f'points="{" ".join(pairs)}"'

    selector = result["target"].get("selector")
    if selector and selector.get("type") == "SvgSelector":
        selector["value"] = POINTS.sub(rescale, selector["value"])
    for holder in (result, result["target"], result["body"]):
        id_match = LOC_SERVICE.match(holder.get("id") or "")
        if id_match:
            service, panel, rest = id_match.groups()
            holder["id"] = f"{CDN_BASE}/{service}{panel or ''}{rest or ''}"
    return result


def with_page_fields(page: dict, url: str) -> dict:
    """The AnnotationPage with its published URL as its id, and its creator and license, at the top."""
    head = {
        "id": url,
        "type": page["type"],
        "@context": page["@context"],
        "label": page.get("label"),
        "creator": CREATOR,
        "rights": RIGHTS,
    }
    return {**head, **{key: value for key, value in page.items() if key not in head}}


def update_report(page: dict, kept_count: int, withheld: list[tuple[str, str]]) -> None:
    """Keep a page's report card true to what is published: its counts, and what was withheld and why."""
    metadata = page.get("metadata") or []
    by_label = {entry.get("label"): entry for entry in metadata}
    if "placed" in by_label:
        by_label["placed"]["value"] = str(kept_count)
    if "unplaced" in by_label and "pages" in by_label:
        try:
            total = int(by_label["pages"]["value"])
            by_label["unplaced"]["value"] = str(total - kept_count)
        except ValueError:
            pass
    if withheld:
        value = ", ".join(f"{key} ({reason})" for key, reason in withheld)
        metadata.append({"label": "withheld", "value": value})
    page["metadata"] = metadata


def write_min(path: Path, doc: dict) -> None:
    """Write minified JSON."""
    path.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))


@dataclass
class Destination:
    """Where a run is written, and the URL it will be served from."""

    out_dir: Path
    base_url: str

    def write(self, name: str, page: dict) -> None:
        """Write one annotation page for every image source, each with its own URL as its id."""
        for source in SOURCES:
            if source == "loc":
                doc = page
            else:
                doc = {**page, "items": [for_cdn(item) for item in page["items"]]}
            url = f"{self.base_url.rstrip('/')}/iiif/{source}/{name}"
            write_min(self.out_dir / "iiif" / source / name, with_page_fields(doc, url))


def read_mapping(path: Path) -> dict[str, dict]:
    """Item -> state and city slugs, the mirror's year, and its sheet counts, from the mirror mapping TSV.

    ``scans`` counts every image in the catalogue. ``sheets`` counts the map sheets
    a reader would: only the pages the pipeline can use (not title or index pages),
    with a skeleton sheet (pNs) and its full-color sheet (pN) counted once, since at
    most one of them is ever published.
    """
    items: dict[str, dict] = {}
    keys: dict[str, set[str]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            entry = items.setdefault(
                row["item"],
                {
                    "state_slug": row["state"],
                    "city_slug": row["city"],
                    "year": row["year"],
                    "scans": 0,
                },
            )
            entry["scans"] += 1
            if keep_sheet(row["page_key"]):
                keys.setdefault(row["item"], set()).add(row["page_key"])
    for item, entry in items.items():
        entry["sheets"] = unique_sheets(keys.get(item, set()))
    return items


def unique_sheets(keys: set[str]) -> int:
    """How many sheets these page keys are, a skeleton and its full-color sheet counting once."""
    return len(keys - redundant_skeleton_keys(keys, keys))


def add_names(items: dict[str, dict], atlas_dir: Path) -> None:
    """Give each item its town's name, state, postal code and catalogue date, from the atlas's index."""
    index = json.loads((atlas_dir / "places.json").read_text())
    places = {place["id"]: place for place in index["places"]}
    for path in (atlas_dir / "volumes").glob("*.json"):
        for place_id, volumes in json.loads(path.read_text()).items():
            place = places.get(place_id)
            for volume in volumes:
                entry = items.get(volume["item"])
                if entry is not None and place is not None:
                    entry["city"] = place["name"]
                    entry["state"] = place["state"]
                    entry["date"] = volume.get("date") or ""
                    entry["title"] = volume.get("title") or ""
    for entry in items.values():
        entry.setdefault("city", entry["city_slug"].replace("-", " ").title())
        entry.setdefault("state", entry["state_slug"].replace("-", " ").title())
        entry.setdefault("date", "")
        entry.setdefault("title", "")
        entry["postal"] = STATE_CODES_BY_LOWERCASE.get(entry["state"].lower(), "")


def volume_number(notes: list[str]) -> str:
    """The volume number a catalogue record's notes give ("Vol. 2, 1915; Republished 1939."), or ""."""
    for note in notes:
        match = re.search(r"\bvol(?:ume)?\.?\s*(\d+[a-z]?)\b", note, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def read_volume_numbers(path: Path) -> dict[str, str]:
    """Item -> its volume number, from the LoC catalogue (metadata.jsonl).

    Catalogue titles never say which volume a record is, so a town-year of
    several volumes would otherwise list them identically.
    """
    numbers: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            match = re.search(r"/item/([^/]+)/?$", str(record.get("Id", "")))
            number = volume_number(record.get("Notes") or [])
            if match and number:
                numbers[match.group(1)] = number
    return numbers


def sheets_placed(annotations: list[dict]) -> int:
    """How many sheets have at least one of these pages; a split sheet's panels count once."""
    return len({page_key(annotation).split("__")[0] for annotation in annotations})


def read_locations(path: Path) -> dict[str, tuple[float, float]]:
    """Item -> catalogue (lon, lat), keyed by the exact item id, as #531 keys it."""
    with path.open() as handle:
        return {
            row["item"]: (float(row["lon"]), float(row["lat"]))
            for row in csv.DictReader(handle, delimiter="\t")
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="The run's output directory"
    )
    parser.add_argument(
        "--base-url", required=True, help="The URL --out will be served at"
    )
    parser.add_argument("--iiif-dir", type=Path, default=DOCUMENTS / "corpus-run/iiif")
    parser.add_argument(
        "--keymap-dir", type=Path, default=DOCUMENTS / "corpus-run/keymap-iiif"
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path.home() / "Downloads/loc-sanborn-maps.mapping.tsv",
    )
    parser.add_argument(
        "--locations",
        type=Path,
        default=DOCUMENTS / "loc-counties/item-locations.tsv",
    )
    parser.add_argument(
        "--small-keymaps",
        type=Path,
        default=DOCUMENTS / "corpus-run/rerun/keymap-under-2500m.ids.txt",
        help="Items whose key map spans under 2.5 km, one per line",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "app/public/atlas",
        help="build_places.py's output, for towns' names",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DOCUMENTS / "metadata.jsonl",
        help="The LoC catalogue, for volume numbers",
    )
    parser.add_argument("--limit", type=int, help="Only the first N items")
    parser.add_argument("--only", nargs="*", help="Only these items")
    args = parser.parse_args()

    items = read_mapping(args.mapping)
    add_names(items, args.atlas_dir)
    volume_numbers = read_volume_numbers(args.metadata)
    locations = read_locations(args.locations)
    small_keymaps = {line.strip() for line in args.small_keymaps.open() if line.strip()}
    destination = Destination(args.out, args.base_url)
    for source in SOURCES:
        (args.out / "iiif" / source).mkdir(parents=True, exist_ok=True)

    paths = sorted(args.iiif_dir.glob("*.iiif.json"))
    if args.only:
        paths = [path for path in paths if item_id(path) in set(args.only)]
    if args.limit:
        paths = paths[: args.limit]

    tally: Counter[str] = Counter()
    rows = []
    for index, path in enumerate(paths, 1):
        item = item_id(path)
        meta = items.get(item, {})
        page = json.loads(path.read_text())
        keymap_path = args.keymap_dir / path.name
        keymap = json.loads(keymap_path.read_text()) if keymap_path.exists() else {}
        keymap_items = keymap.get("items") or []
        placed = page.get("items") or []
        annotations, keymap_pages = drop_keymap_pages(placed, keymap_items)
        report = {
            entry.get("label"): entry.get("value")
            for entry in page.get("metadata") or []
        }
        # #531 counted every scan, so the small-volume test does too.
        pages = filter_pages(annotations, meta.get("scans", 0), locations.get(item))
        pages.kept, skeleton_pages = drop_skeleton_pages(pages.kept)
        pages.withheld[:0] = [
            (key, "the key map, published in the key-map file") for key in keymap_pages
        ] + [
            (key, "a skeleton sheet; its full-color sheet is published")
            for key in skeleton_pages
        ]
        status = item_status(
            placed,
            pages.kept,
            small_keymap=item in small_keymaps,
            has_keymap=bool(keymap_items),
        )
        published = status.startswith("published")
        tally[status] += 1
        tally["pages dropped: sheet too large"] += pages.too_large
        tally["pages dropped: far from the catalogue location"] += pages.too_far
        tally["pages dropped: the key map, published as a page"] += len(keymap_pages)
        tally["pages dropped: skeleton of a published sheet"] += len(skeleton_pages)

        main_name = keymap_name = ""
        if published and pages.kept:
            for annotation in pages.kept:
                strip_creators(annotation)
            page["items"] = pages.kept
            update_report(page, len(pages.kept), pages.withheld)
            main_name = f"{item}.main.iiif.json"
            destination.write(main_name, page)
        if published and keymap_items:
            for annotation in keymap_items:
                strip_creators(annotation)
            keymap_name = f"{item}.keymap.iiif.json"
            destination.write(keymap_name, keymap)
            tally["key maps published"] += 1

        rows.append(
            {
                "item": item,
                "city": meta.get("city", ""),
                "state": meta.get("postal", ""),
                "year": meta.get("year", ""),
                "date": meta.get("date", ""),
                "volume": volume_numbers.get(item, ""),
                "title": meta.get("title", ""),
                "sheets": meta.get("sheets", ""),
                "scans": meta.get("scans", ""),
                "images": report.get("pages", ""),
                "placed": len(placed),
                "published": len(pages.kept) if published else 0,
                "sheets_placed": sheets_placed(pages.kept) if published else 0,
                "dropped_over_6km": pages.too_large,
                "dropped_over_5km_from_location": pages.too_far,
                "dropped_keymap": len(keymap_pages),
                "dropped_skeleton": len(skeleton_pages),
                "status": status,
                "main": main_name,
                "keymap": keymap_name,
            }
        )
        if index % 5000 == 0:
            print(f"{index:,}/{len(paths):,}", file=sys.stderr, flush=True)

    with (args.out / "items.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    for key, value in sorted(tally.items()):
        print(f"{value:8,}  {key}")
    unnamed = [row["item"] for row in rows if not row["state"]]
    if unnamed:
        print(f"{len(unnamed):8,}  rows without a postal code, e.g. {unnamed[:5]}")


if __name__ == "__main__":
    main()
