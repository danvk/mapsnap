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

The sheet table comes from the mirror's ``metadata.json`` for each item, and a
split sheet's panel outlines from the run's ``pN.panels.json``, both fetched from
S3 once into local caches. The run's fit is baked in, so the records are
rebuilt with each published run. With ``--site-dir``, item.html is copied there.
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from build_run_site import allmaps_url, display_title, read_items
from publish_run import CDN_BASE, page_key, write_min
from shapely.geometry import Polygon

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
# Parallel S3 reads when filling the caches.
FETCH_WORKERS = 32
# The fit statuses that put an image on the map.
PLACED = ("placed", "key map")
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
    # For a published image: its annotation's id, which Allmaps uses as the
    # map's id, and the run's file holding it ("main" or "keymap").
    map: str = ""
    file: str = ""

    def as_json(self) -> dict[str, str]:
        """The record's form, leaving out empty fields."""
        doc = {"image": self.image, "status": self.status}
        for name in ("note", "map", "file"):
            if getattr(self, name):
                doc[name] = getattr(self, name)
        return doc


class ObjectReader(Protocol):
    """Something that reads an S3 object's bytes (None if missing): S3Reader, or a test's fake."""

    def connect(self) -> None: ...

    def read(self, s3_url: str) -> bytes | None: ...


class S3Reader:
    """Reads objects from the mirror bucket, connecting only when first asked."""

    def __init__(self) -> None:
        self.client = None

    def connect(self) -> None:
        """Make the S3 client now: making one isn't thread-safe, using one is."""
        if self.client is None:
            import boto3  # only needed for a fetch
            from botocore.config import Config

            self.client = boto3.client(
                "s3", config=Config(max_pool_connections=FETCH_WORKERS)
            )

    def read(self, s3_url: str) -> bytes | None:
        """An object's bytes, or None if it doesn't exist."""
        self.connect()
        assert self.client is not None
        parsed = urlparse(s3_url)
        try:
            response = self.client.get_object(
                Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
            )
        except self.client.exceptions.NoSuchKey:
            return None
        return response["Body"].read()


def cached_read(reader: ObjectReader, s3_url: str, cache_path: Path) -> bytes | None:
    """An S3 object via a local copy; a missing object is cached as an empty file."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Written aside and renamed, so an interrupted run leaves no partial copy.
        partial = cache_path.with_name(f"{cache_path.name}.partial")
        partial.write_bytes(reader.read(s3_url) or b"")
        partial.replace(cache_path)
    return cache_path.read_bytes() or None


def panel_shapes(panels_doc: dict) -> list[dict]:
    """A split sheet's panels, in order (panel 1 first), as outlines with a label point inside.

    The outlines are in the mirror's 25% pixels, the Chronoscope image's frame.
    """
    shapes = []
    for ring in panels_doc.get("panels") or []:
        polygon = Polygon(ring).buffer(0)
        point = polygon.representative_point()
        shapes.append(
            {
                "points": [[round(x, 1), round(y, 1)] for x, y in ring],
                "label": [round(point.x, 1), round(point.y, 1)],
            }
        )
    return shapes


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


def sheet_record(
    sheet: dict, fits: list[Fit], panels: list[dict] | None = None
) -> dict:
    """One sheet's entry: its place in LoC's order, its images and what the run did with it."""
    service = service_id(sheet["storage_dir"], sheet["stem"])
    cdn = f"{CDN_BASE}/{service}"
    thumb_width, thumb_height = thumbnail_size(sheet["width"], sheet["height"])
    record = {
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
    if panels:
        record["panels"] = panels
    return record


def page_notes(page: dict, label: str) -> list[tuple[str, str]]:
    """The (page key, note) pairs an annotation page's metadata entry lists."""
    for entry in page.get("metadata") or []:
        if entry.get("label") == label:
            return PAGE_NOTE.findall(str(entry.get("value", "")))
    return []


def read_pages(run_dir: Path, row: dict[str, str]) -> dict[str, dict]:
    """The item's published annotation pages ("main", "keymap"), Chronoscope flavor."""
    return {
        column: json.loads((run_dir / "iiif/chronoscope" / row[column]).read_text())
        for column in ("main", "keymap")
        if row.get(column)
    }


def read_unpublished(run_iiif_dir: Path | None, item: str) -> dict | None:
    """The run's own main annotation page for an item, from publish_run.py's input, if there."""
    path = run_iiif_dir / f"{item}.iiif.json" if run_iiif_dir else None
    return json.loads(path.read_text()) if path and path.exists() else None


def run_tag(pages: dict[str, dict]) -> str:
    """The run that made these pages (corpus-v1), from their "run" metadata; "" if none says."""
    for page in pages.values():
        for entry in page.get("metadata") or []:
            if entry.get("label") == "run":
                return str(entry.get("value", ""))
    return ""


def read_fits(pages: dict[str, dict], unpublished: dict | None = None) -> list[Fit]:
    """What the run did with each of an item's images, from its published annotation pages.

    An item the run placed nothing of has no published main file, and so no
    list of the images it could not place. ``unpublished`` is the run's own
    main annotation page for such an item (publish_run.py's input), which
    has that list. Any page it places was dropped when the run was published,
    so it counts as withheld.
    """
    fits: list[Fit] = []
    if unpublished is not None and "main" not in pages:
        fits += [
            Fit(key, "unplaced") for key, _ in page_notes(unpublished, "unplaced pages")
        ]
        fits += [
            Fit(page_key(annotation), "withheld", "dropped when the run was published")
            for annotation in unpublished.get("items") or []
        ]
    for column, status in (("main", "placed"), ("keymap", "key map")):
        if column not in pages:
            continue
        page = pages[column]
        fits += [
            Fit(page_key(annotation), status, map=annotation["id"], file=column)
            for annotation in page["items"]
        ]
        if column == "main":
            fits += [
                Fit(key, "unplaced") for key, _ in page_notes(page, "unplaced pages")
            ]
            fits += [
                Fit(key, "withheld", note) for key, note in page_notes(page, "withheld")
            ]
    # An image one file places drops out of the other's misses: a key map is
    # withheld from the main file because the key-map file has it, and the main
    # fit may also list it as unplaced, having tried it as a street sheet.
    placed = {fit.image for fit in fits if fit.status in PLACED}
    return [fit for fit in fits if fit.status in PLACED or fit.image not in placed]


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
        "sheets": [
            sheet_record(
                sheet,
                fits.get(sheet["key"], []),
                run.panels.get(sheet["key"]) if run else None,
            )
            for sheet in sheets
        ],
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
    panels: dict[str, list[dict]] = field(default_factory=dict)  # split sheet -> panels

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


def mirror_metadata_source(s3_path: str, cache: Path) -> tuple[str, Path]:
    """Where an item's mirror metadata.json is in S3, and where its local copy goes."""
    return f"{s3_path}/metadata.json", cache / f"{Path(s3_path).name}.json"


def read_mirror_metadata(
    reader: ObjectReader, s3_path: str, cache: Path
) -> dict | None:
    """An item's mirror metadata.json (its sheet table), via ``cache``; None if it has none."""
    body = cached_read(reader, *mirror_metadata_source(s3_path, cache))
    return json.loads(body) if body else None


def split_sheets(fits: list[Fit]) -> list[str]:
    """The sheets a run cut into panels: those with a p9__2-style image."""
    return sorted({fit.image.split("__")[0] for fit in fits if "__" in fit.image})


def panels_source(s3_path: str, tag: str, key: str, cache: Path) -> tuple[str, Path]:
    """Where a run's pN.panels.json for a sheet is in S3, and where its local copy goes."""
    name = f"{key}.panels.json"
    return f"{s3_path}/runs/{tag}/{name}", cache / tag / Path(s3_path).name / name


def read_run_panels(
    reader: ObjectReader, s3_path: str, tag: str, fits: list[Fit], cache: Path
) -> dict[str, list[dict]]:
    """Sheet key -> panel outlines, for each sheet the run split, from its pN.panels.json."""
    panels: dict[str, list[dict]] = {}
    for key in split_sheets(fits):
        body = cached_read(reader, *panels_source(s3_path, tag, key, cache))
        if body:
            panels[key] = panel_shapes(json.loads(body))
    return panels


def prefetch(reader: ObjectReader, sources: list[tuple[str, Path]]) -> None:
    """Fill the local copies of these (S3 url, path) pairs that are missing, in parallel."""
    missing = [(url, path) for url, path in sources if not path.exists()]
    if not missing:
        return
    reader.connect()
    print(f"fetching {len(missing):,} files from S3", file=sys.stderr)
    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        jobs = [pool.submit(cached_read, reader, url, path) for url, path in missing]
        for done, job in enumerate(as_completed(jobs), start=1):
            job.result()
            if done % 5000 == 0:
                print(f"  {done:,} of {len(missing):,}", file=sys.stderr)


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
    parser.add_argument(
        "--run-iiif-dir",
        type=Path,
        help=(
            "publish_run.py's --iiif-dir: the run's own annotations, which list the "
            "images of an item the run placed nothing of"
        ),
    )
    parser.add_argument(
        "--panels-cache",
        type=Path,
        default=DOCUMENTS / "run-panels",
        help="Local copies of the runs' pN.panels.json, by run and item.",
    )
    args = parser.parse_args()

    s3_paths = read_s3_paths(args.volumes)
    items = read_item_list(args.items) if args.items else sorted(s3_paths)
    rows = {row["item"]: row for row in read_items(args.run_dir / "items.tsv")}
    catalogue = read_catalogue(args.catalogue, set(items))
    reader = S3Reader()

    # What each item's run did, and so which split sheets' panels are wanted;
    # then every file from S3 at once, before the records are built.
    runs: dict[str, tuple[list[Fit], str]] = {}
    sources = []
    for item in items:
        sources.append(mirror_metadata_source(s3_paths[item], args.sheets_cache))
        if item in rows:
            pages = read_pages(args.run_dir, rows[item])
            unpublished = (
                None if "main" in pages else read_unpublished(args.run_iiif_dir, item)
            )
            fits = read_fits(pages, unpublished)
            tag = run_tag(pages) or run_tag({"main": unpublished or {}})
            runs[item] = (fits, tag)
            sources += [
                panels_source(s3_paths[item], tag, key, args.panels_cache)
                for key in split_sheets(fits)
            ]
    prefetch(reader, sources)

    (args.out / "items").mkdir(parents=True, exist_ok=True)
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    index = []
    for item in items:
        mirror = read_mirror_metadata(reader, s3_paths[item], args.sheets_cache)
        if mirror is None:
            print(f"{item}: no metadata.json in the mirror", file=sys.stderr)
            continue
        run = None
        if item in runs:
            fits, tag = runs[item]
            run = RunInfo(args.version, rows[item], fits)
            run.panels = read_run_panels(
                reader, s3_paths[item], tag, fits, args.panels_cache
            )
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
