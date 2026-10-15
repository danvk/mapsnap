"""Build the data behind mapsnap.org/item: one record and one IIIF manifest per volume.

    AWS_PROFILE=mapsnap uv run python scripts/atlas/build_item_pages.py \\
        --run-dir ~/Documents/mapsnap/data.mapsnap.org/runs/v1.3 --version v1.3 \\
        --out ~/Documents/mapsnap/data.mapsnap.org \\
        --site-dir ~/Documents/mapsnap/mapsnap.org

The item page (item.html, beside this script) recreates a loc.gov item page from
these files and Chronoscope's copies of the scans, so a LoC Sanborn id leads to
everything about a volume without visiting loc.gov (#606). It writes, under
``--out``:

- ``items/<item>.json``: the catalogue record (metadata.jsonl), every mirrored
  sheet with its Chronoscope image service, thumbnail and loc.gov sources, and
  what ``--run-dir``'s run did with each sheet;
- ``manifests/<item>.json``: a IIIF Presentation 3 manifest of the Chronoscope
  images, for Mirador, Universal Viewer or the Allmaps editor;
- ``items/index.json``: one row per item, for the page's search box and its
  list of other volumes of the same town.

The sheet table comes from the mirror's ``metadata.json`` for each item, fetched
from S3 once into ``--sheets-cache``. The run's fit is baked in, so the records are
rebuilt with each published run. With ``--site-dir``, item.html is copied there.
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from build_run_site import allmaps_url, display_title, read_items
from publish_run import CDN_BASE, page_key, write_min

DOCUMENTS = Path.home() / "Documents/mapsnap"
DATA_URL = "https://data.mapsnap.org"
ITEM_PAGE = "https://mapsnap.org/item"
LOC_ITEM = "https://www.loc.gov/item/"
LOC_RESOURCE = "https://www.loc.gov/resource/"
LOC_IIIF = "https://tile.loc.gov/image-services/iiif"
LOC_STORAGE = "https://tile.loc.gov/storage-services"
# The CDN's images are 25% copies; its sizes halve from there, rounding up.
THUMBNAIL_SCALE = 4
IMAGE_CREDIT = (
    "Library of Congress, Geography and Map Division, Sanborn Maps Collection; "
    "quarter-size copies served by Chronoscope."
)
INDEX_COLUMNS = ["id", "city", "state", "year", "date", "volume", "sheets", "status"]
# "p9__1 (tier 1, 33.99,-86.02 +/-179m)" or "p0a (the key map, ...)" in an
# annotation page's "unplaced pages" and "withheld" metadata.
PAGE_NOTE = re.compile(r"(p[^\s(;,]+) \(([^)]*)\)")


@dataclass
class Fit:
    """What a run did with one image: a whole sheet or one panel of a split sheet."""

    image: str  # page key, with a panel suffix for a split sheet: p9__2
    status: str  # placed | key map | unplaced | withheld
    note: str = ""

    def as_json(self) -> dict[str, str]:
        """The record's form, leaving out an empty note."""
        doc = {"image": self.image, "status": self.status}
        if self.note:
            doc["note"] = self.note
        return doc


def service_id(storage_dir: str, stem: str) -> str:
    """LoC's image-service id for a sheet: its storage path with colons, then its stem."""
    return "service:" + storage_dir.replace("/", ":") + ":" + stem


def thumbnail_size(width: int, height: int) -> tuple[int, int]:
    """The CDN's pre-rendered size at 1/THUMBNAIL_SCALE of its image, rounding up as it does."""
    return math.ceil(width / THUMBNAIL_SCALE), math.ceil(height / THUMBNAIL_SCALE)


def loc_resource(storage_dir: str) -> str:
    """loc.gov's resource id for a storage directory: .../g4014cm/g032971909 is g4014cm.g032971909.

    Most Sanborn directories already join the two parts with an underscore
    (g3974am_g000011907); the rest nest them.
    """
    parts = storage_dir.rstrip("/").split("/")
    if "_" in parts[-1]:
        return parts[-1].replace("_", ".", 1)
    return f"{parts[-2]}.{parts[-1]}"


def loc_source_url(sheet: dict) -> str:
    """The sheet's original on loc.gov: a JP2 in the service tree, or a TIFF master."""
    if sheet.get("source") == "torrent-master-tif":
        branch, suffix = "master", ".tif"
    else:
        branch, suffix = "service", ".jp2"
    return f"{LOC_STORAGE}/{branch}/{sheet['storage_dir']}/{sheet['stem']}{suffix}"


def sheet_record(sheet: dict, fits: list[Fit]) -> dict:
    """One sheet's entry: its place in LoC's order, its images and what the run did with it."""
    service = service_id(sheet["storage_dir"], sheet["stem"])
    cdn = f"{CDN_BASE}/{service}"
    thumb_width, thumb_height = thumbnail_size(sheet["width"], sheet["height"])
    return {
        "seq": sheet["seq"],
        "key": sheet["key"],
        "stem": sheet["stem"],
        "width": sheet["width"],
        "height": sheet["height"],
        "image": cdn,
        "thumbnail": {
            "url": f"{cdn}/full/{thumb_width},{thumb_height}/0/default.jpg",
            "width": thumb_width,
            "height": thumb_height,
        },
        "loc_page": f"{LOC_RESOURCE}{loc_resource(sheet['storage_dir'])}/?sp={sheet['seq']}",
        "loc_image": f"{LOC_IIIF}/{service}",
        "loc_source": loc_source_url(sheet),
        "fit": [fit.as_json() for fit in fits],
    }


def page_notes(page: dict, label: str) -> list[tuple[str, str]]:
    """The (page key, note) pairs an annotation page's metadata entry lists."""
    for entry in page.get("metadata") or []:
        if entry.get("label") == label:
            return PAGE_NOTE.findall(str(entry.get("value", "")))
    return []


def read_fits(run_dir: Path, row: dict[str, str]) -> list[Fit]:
    """What the run did with each of an item's images, from its published IIIF files."""
    fits: list[Fit] = []
    for column, status in (("main", "placed"), ("keymap", "key map")):
        if not row.get(column):
            continue
        page = json.loads((run_dir / "iiif/chronoscope" / row[column]).read_text())
        fits += [Fit(page_key(annotation), status) for annotation in page["items"]]
        if column == "main":
            fits += [
                Fit(key, "unplaced") for key, _ in page_notes(page, "unplaced pages")
            ]
            fits += [
                Fit(key, "withheld", note) for key, note in page_notes(page, "withheld")
            ]
    # A key map is withheld from the main file because the key-map file has it.
    placed = {fit.image for fit in fits if fit.status != "withheld"}
    return [fit for fit in fits if fit.status != "withheld" or fit.image not in placed]


def fits_by_sheet(fits: list[Fit]) -> dict[str, list[Fit]]:
    """Group fits by the sheet they belong to (p9__2 belongs to p9)."""
    grouped: dict[str, list[Fit]] = {}
    for fit in fits:
        grouped.setdefault(fit.image.split("__")[0], []).append(fit)
    for entries in grouped.values():
        entries.sort(key=lambda fit: fit.image)
    return grouped


def catalogue_fields(record: dict) -> dict:
    """The loc.gov catalogue fields the item page shows, from a metadata.jsonl record."""
    location = (record.get("Location") or [{}])[0]
    coordinates = location.get("Coordinates") or []
    return {
        "title": record.get("Title") or "",
        "date": record.get("Date") or "",
        "city": ", ".join(record.get("City_text") or []),
        "county": ", ".join(record.get("County_text") or []),
        "state": ", ".join(record.get("State_text") or []),
        "coordinates": (
            {"lat": coordinates[0], "lon": coordinates[1]}
            if len(coordinates) == 2
            else None
        ),
        "notes": record.get("Notes") or [],
        "repository": record.get("Repository") or "",
        "part_of": record.get("Part_of") or [],
        "loc_manifest": record.get("IIIF_manifest") or "",
    }


def item_record(
    item: str, catalogue: dict, mirror: dict, run: "RunInfo | None"
) -> dict:
    """An item's full record: catalogue fields, sheets and the run's results."""
    fits = fits_by_sheet(run.fits if run else [])
    row = run.row if run else {}
    sheets = sorted(mirror.get("sheets") or [], key=lambda sheet: sheet["seq"])
    fields = catalogue or catalogue_fields({})
    # Items outside the Sanborn collection's records have only the run's row.
    for name in ("title", "date", "city"):
        fields[name] = fields[name] or row.get(name, "")
    record = {
        "id": item,
        "name": display_title(row) if row else fields["title"] or item,
        **fields,
        "year": row.get("year") or mirror.get("year", ""),
        "volume": row.get("volume", ""),
        "state_code": row.get("state", ""),
        "scans": int(row["scans"]) if row.get("scans") else None,
        "loc_item": f"{LOC_ITEM}{item}/",
        "manifest": f"{DATA_URL}/manifests/{item}.json",
        "sheets": [sheet_record(sheet, fits.get(sheet["key"], [])) for sheet in sheets],
        "broken": mirror.get("broken") or [],
    }
    if run:
        record["run"] = run.summary()
    return record


@dataclass
class RunInfo:
    """One item's row in a run's items.tsv, its fits and where its files are served."""

    version: str
    row: dict[str, str]
    fits: list[Fit]

    def file_urls(self, name: str) -> dict[str, str]:
        """A published file's URL for each image source, and the Allmaps viewer on each."""
        base = f"{DATA_URL}/runs/{self.version}/iiif"
        urls = {
            "loc": f"{base}/loc/{name}",
            "chronoscope": f"{base}/chronoscope/{name}",
        }
        return {
            **urls,
            "allmaps_loc": allmaps_url(urls["loc"]),
            "allmaps_chronoscope": allmaps_url(urls["chronoscope"]),
        }

    def summary(self) -> dict:
        """The run's part of the record: its counts, status and files."""
        row = self.row
        counts = ("sheets", "images", "placed", "published", "sheets_placed")
        summary: dict = {"version": self.version, "status": row["status"]}
        summary |= {name: int(row[name] or 0) for name in counts}
        for column in ("main", "keymap"):
            if row.get(column):
                summary[column] = self.file_urls(row[column])
        return summary


def manifest(record: dict) -> dict:
    """A IIIF Presentation 3 manifest of the item's Chronoscope images."""
    url = record["manifest"]
    metadata = [
        ("Date", record["date"]),
        (
            "Location",
            ", ".join(
                filter(None, [record["city"], record["county"], record["state"]])
            ),
        ),
        ("Notes", "; ".join(record["notes"])),
        ("Repository", record["repository"]),
    ]
    canvases = []
    for sheet in record["sheets"]:
        canvas = f"{url}/canvas/{sheet['seq']}"
        size = {"width": sheet["width"], "height": sheet["height"]}
        canvases.append(
            {
                "id": canvas,
                "type": "Canvas",
                "label": {"none": [sheet["key"]]},
                **size,
                "thumbnail": [
                    {
                        "id": sheet["thumbnail"]["url"],
                        "type": "Image",
                        "format": "image/jpeg",
                        "width": sheet["thumbnail"]["width"],
                        "height": sheet["thumbnail"]["height"],
                    }
                ],
                "items": [
                    {
                        "id": f"{canvas}/page",
                        "type": "AnnotationPage",
                        "items": [
                            {
                                "id": f"{canvas}/page/image",
                                "type": "Annotation",
                                "motivation": "painting",
                                "target": canvas,
                                "body": {
                                    "id": f"{sheet['image']}/full/max/0/default.jpg",
                                    "type": "Image",
                                    "format": "image/jpeg",
                                    **size,
                                    "service": [
                                        {
                                            "id": sheet["image"],
                                            "type": "ImageService3",
                                            "profile": "level0",
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        )
    doc = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": url,
        "type": "Manifest",
        "label": {"en": [record["title"] or record["name"]]},
        "metadata": [
            {"label": {"en": [label]}, "value": {"en": [value]}}
            for label, value in metadata
            if value
        ],
        "requiredStatement": {
            "label": {"en": ["Images"]},
            "value": {"en": [IMAGE_CREDIT]},
        },
        "homepage": [
            {
                "id": f"{ITEM_PAGE}?id={record['id']}",
                "type": "Text",
                "label": {"en": [record["name"]]},
                "format": "text/html",
            }
        ],
        "seeAlso": [
            {
                "id": record["loc_item"],
                "type": "Text",
                "label": {"en": ["Library of Congress item page"]},
                "format": "text/html",
            }
        ],
        "items": canvases,
    }
    if canvases:
        doc["thumbnail"] = canvases[0]["thumbnail"]
    return doc


def index_row(record: dict) -> list:
    """The item's row in items/index.json, in INDEX_COLUMNS order."""
    run = record.get("run") or {}
    return [
        record["id"],
        record["city"],
        record["state_code"] or record["state"],
        record["year"],
        record["date"],
        record["volume"],
        len(record["sheets"]),
        run.get("status", ""),
    ]


def read_catalogue(path: Path, items: set[str]) -> dict[str, dict]:
    """Item -> its catalogue fields, from metadata.jsonl (loc.gov's records)."""
    catalogue: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            item = urlparse(str(record.get("Id", ""))).path.rstrip("/").split("/")[-1]
            if item in items:
                catalogue[item] = catalogue_fields(record)
    return catalogue


def read_s3_paths(path: Path) -> dict[str, str]:
    """Item -> its prefix in the mirror bucket, from loc-sanborn-volumes.tsv."""
    with path.open(newline="") as handle:
        return {
            row["item"]: row["s3_path"]
            for row in csv.DictReader(handle, delimiter="\t")
        }


def fetch_mirror_metadata(s3_paths: dict[str, str], cache: Path) -> None:
    """Copy each item's metadata.json from the mirror bucket into ``cache``, unless it is there."""
    missing = {
        item: s3_path
        for item, s3_path in s3_paths.items()
        if not (cache / f"{item}.json").exists()
    }
    if not missing:
        return
    import boto3  # only needed for a fetch

    client = boto3.client("s3")
    cache.mkdir(parents=True, exist_ok=True)
    for item, s3_path in missing.items():
        dest = cache / f"{item}.json"
        parsed = urlparse(s3_path)
        key = f"{parsed.path.lstrip('/')}/metadata.json"
        try:
            body = client.get_object(Bucket=parsed.netloc, Key=key)["Body"].read()
        except client.exceptions.NoSuchKey:
            print(f"{item}: no metadata.json at {s3_path}", file=sys.stderr)
            continue
        dest.write_bytes(body)


def read_item_list(path: Path) -> list[str]:
    """Item ids from a list file: the first word of each non-blank, non-comment line."""
    return [
        line.split()[0]
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="publish_run.py's output."
    )
    parser.add_argument(
        "--version", required=True, help="The run's version, e.g. v1.3."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="data.mapsnap.org's root."
    )
    parser.add_argument("--site-dir", type=Path, help="Copy item.html into this site.")
    parser.add_argument("--items", type=Path, help="Only these items (one per line).")
    parser.add_argument(
        "--volumes", type=Path, default=DOCUMENTS / "loc-sanborn-volumes.tsv"
    )
    parser.add_argument("--catalogue", type=Path, default=DOCUMENTS / "metadata.jsonl")
    parser.add_argument(
        "--sheets-cache", type=Path, default=DOCUMENTS / "mirror-metadata"
    )
    args = parser.parse_args()

    s3_paths = read_s3_paths(args.volumes)
    items = read_item_list(args.items) if args.items else sorted(s3_paths)
    fetch_mirror_metadata({item: s3_paths[item] for item in items}, args.sheets_cache)
    rows = {row["item"]: row for row in read_items(args.run_dir / "items.tsv")}
    catalogue = read_catalogue(args.catalogue, set(items))

    (args.out / "items").mkdir(parents=True, exist_ok=True)
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    index = []
    for item in items:
        mirror_path = args.sheets_cache / f"{item}.json"
        if not mirror_path.exists():
            continue
        mirror = json.loads(mirror_path.read_text())
        row = rows.get(item)
        run = RunInfo(args.version, row, read_fits(args.run_dir, row)) if row else None
        record = item_record(item, catalogue.get(item) or {}, mirror, run)
        write_min(args.out / "items" / f"{item}.json", record)
        write_min(args.out / "manifests" / f"{item}.json", manifest(record))
        index.append(index_row(record))
    write_min(args.out / "items/index.json", {"columns": INDEX_COLUMNS, "rows": index})
    if args.site_dir:
        shutil.copy(Path(__file__).with_name("item.html"), args.site_dir / "item.html")
    print(f"wrote {len(index)} items to {args.out}")


if __name__ == "__main__":
    main()
