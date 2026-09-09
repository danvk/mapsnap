"""Build the mapsnap Sanborn mirror from full-resolution LoC JP2s.

Source of truth is the Library of Congress `storage-services` tree as served by
the torrent's HTTP mirror: every sheet is a JP2 whose filename carries the
sheet's own identity (``06246_1914-0051``), so no sequence-number mapping is
involved. The plan comes from a mapping table (one row per sheet: item, state,
year, city, sequence, stem, page key, source, bytes, storage dir) built from the
torrent listing and the loc.gov catalog.

Per item (one LoC catalog record, normally one volume):

  * download each kept sheet's JP2 to ``--jp2-dir`` mirroring the torrent tree
    (``storage-services/service/<dir>/<stem>.jp2``), verified against the
    listed byte count, so the copy stays a valid torrent payload;
  * decode it at the JPEG 2000 quarter-resolution level, which IS the pipeline's
    25% working scale, to ``<out>/by-state/<state>/<year>/<item>/p<key>.jpg``
    (JPEG quality 95, as ``mapsnap scale`` writes);
  * for key-map candidates (page 0 and page 1 families, letter pages -- see
    ``keymap.identify.candidate_keys``) also decode at full resolution to
    ``raw/p<key>.jpg``, as the ``data/`` volumes keep them;
  * write ``metadata.json`` (the catalog fields and the sheet table) and a
    ``.done`` marker; optionally ``aws s3 sync`` the item directory to the
    bucket and mark ``.uploaded``.

Sheets whose page key does not start with a digit (covr, ind1, cbd, titl, note)
are skipped: nothing in the pipeline reads them. A sheet whose download or
decode fails after retries is appended to ``broken.log`` (tab-separated: item,
stem, source, reason) and left out of the item; the item still completes.

Resumable and parallel: items with a ``.done`` marker are skipped, a JP2 on
disk at the listed size is not re-fetched, an existing output is not re-decoded;
``--workers`` items run at once, each fetching ``--streams`` sheets at a time.
The HTTP mirror sustains about 20 MB/s in aggregate, so the run is bound by it:
about two days for the whole collection.

    mapsnap loc-mirror ~/Downloads/loc-sanborn-maps.mapping.tsv \\
        --jp2-dir /Volumes/fivetera/loc-sanborn-maps/jp2 \\
        --out-dir /Volumes/fivetera/mapsnap-sanborn \\
        --bucket s3://mapsnap-sanborn --workers 4 --streams 2
    mapsnap loc-mirror MAPPING.tsv ... --upload-only     # sync finished items later
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from mapsnap.keymap.fit_keymap import page_number
from mapsnap.keymap.identify import CANDIDATE_PAGE_NUMBERS, is_letter_page

DEFAULT_MIRROR = "http://50.35.157.188:27182"
LOC_IIIF = "https://tile.loc.gov/image-services/iiif"
JPEG_QUALITY = 95  # what mapsnap scale writes; the pipeline is tuned on it
QUARTER_REDUCE = 2  # JPEG 2000 resolution levels to drop: 1/4 linear = 25%
RETRIES = 4
DONE, UPLOADED = ".done", ".uploaded"


@dataclass
class Sheet:
    """One sheet of an item, as the mapping table describes it."""

    seq: int
    stem: str
    key: str  # mapsnap page key, e.g. p101s
    source: str  # torrent-jp2 | torrent-master-tif | torrent-master-jp2 | loc-iiif
    bytes: int
    storage_dir: (
        str  # gmd/.../g0..., the torrent directory under storage-services/service
    )


@dataclass
class ItemPlan:
    """An item's catalog fields and its kept sheets."""

    item: str
    state: str
    year: str
    city: str
    sheets: list[Sheet] = field(default_factory=list)


def keep_sheet(key: str) -> bool:
    """Whether the pipeline can use this sheet: numbered pages and letter pages only."""
    return re.match(r"p\d", key) is not None or is_letter_page(key)


def is_candidate(key: str) -> bool:
    """Whether the sheet is a key-map candidate, mirroring keymap.identify.candidate_keys."""
    if is_letter_page(key):
        return True
    base = re.sub(r"[a-j]$", "", key) if re.match(r"p\d+[a-j]$", key) else key
    number = page_number(base)
    return number is not None and number in CANDIDATE_PAGE_NUMBERS


def load_mapping(path: Path) -> dict[str, ItemPlan]:
    """Items keyed by id from the mapping TSV, with only the kept sheets."""
    plans: dict[str, ItemPlan] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row["stem"] or not keep_sheet(row["page_key"]):
                continue
            plan = plans.setdefault(
                row["item"],
                ItemPlan(row["item"], row["state"], row["year"], row["city"]),
            )
            plan.sheets.append(
                Sheet(
                    int(row["seq"]),
                    row["stem"],
                    row["page_key"],
                    row["source"],
                    int(row["bytes"] or 0),
                    row["storage_dir"],
                )
            )
    return plans


def item_dir(out_dir: Path, plan: ItemPlan) -> Path:
    """``<out>/by-state/<state>/<year>/<item>``, the same shape as the S3 prefix."""
    return out_dir / "by-state" / plan.state / plan.year / plan.item


def s3_prefix(bucket: str, plan: ItemPlan) -> str:
    """The item's destination prefix, e.g. ``s3://mapsnap-sanborn/by-state/alabama/1922/sanborn00081_001``."""
    return f"{bucket.rstrip('/')}/by-state/{plan.state}/{plan.year}/{plan.item}"


def jp2_path(jp2_dir: Path, sheet: Sheet) -> Path:
    """Where the sheet's JP2 lives locally, mirroring the torrent tree."""
    return (
        jp2_dir
        / "storage-services"
        / "service"
        / sheet.storage_dir
        / f"{sheet.stem}.jp2"
    )


def source_url(mirror: str, sheet: Sheet, full: bool = False) -> str:
    """The URL to fetch a sheet from: the mirror's JP2/TIFF, or LoC's IIIF for the rest."""
    if sheet.source == "torrent-jp2":
        return f"{mirror}/storage-services/service/{sheet.storage_dir}/{sheet.stem}.jp2"
    if sheet.source == "torrent-master-tif":
        return f"{mirror}/storage-services/master/{sheet.storage_dir}/{sheet.stem}.tif"
    if sheet.source == "torrent-master-jp2":
        return f"{mirror}/storage-services/master/{sheet.storage_dir}/{sheet.stem}.jp2"
    service = "service:" + sheet.storage_dir.replace("/", ":")
    size = "full" if full else "pct:25"
    return f"{LOC_IIIF}/{service}:{sheet.stem}/full/{size}/0/default.jpg"


def fetch(url: str, dest: Path, expected_bytes: int = 0) -> None:
    """Download ``url`` to ``dest`` with retries; verify the byte count when known."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            partial = dest.with_suffix(dest.suffix + ".part")
            with (
                urllib.request.urlopen(url, timeout=300) as response,
                open(partial, "wb") as out,
            ):
                shutil.copyfileobj(response, out, 1 << 20)
            size = partial.stat().st_size
            if expected_bytes and size != expected_bytes:
                raise OSError(f"size {size} != listed {expected_bytes}")
            partial.replace(dest)
            return
        except (OSError, urllib.error.URLError) as error:
            last = error
            time.sleep(5 * (attempt + 1))
    raise OSError(f"{url}: {last}")


def opj_available() -> bool:
    """Whether the OpenJPEG decoder CLI is on the PATH."""
    return shutil.which("opj_decompress") is not None


def decode_jp2(jp2: Path, out_jpg: Path, reduce: int) -> tuple[int, int]:
    """Decode a JP2 at ``reduce`` resolution levels down and write a JPEG; returns its size.

    Uses ``opj_decompress`` (fast, multithreaded) when present, else Pillow.
    """
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    if opj_available():
        with tempfile.TemporaryDirectory() as tmp:
            ppm = Path(tmp) / "decoded.ppm"
            subprocess.run(
                [
                    "opj_decompress",
                    "-threads",
                    "4",
                    "-i",
                    str(jp2),
                    "-r",
                    str(reduce),
                    "-o",
                    str(ppm),
                ],
                check=True,
                capture_output=True,
            )
            image = Image.open(ppm)
            image.load()
    else:
        image = Image.open(jp2)
        image.reduce = reduce  # type: ignore[attr-defined]
        image.load()
    image.convert("RGB").save(out_jpg, "JPEG", quality=JPEG_QUALITY)
    return image.size


def scale_to_quarter(src: Path, out_jpg: Path) -> tuple[int, int]:
    """Write a 25% JPEG of a full-resolution TIFF or JPEG (the non-JP2 sources)."""
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(src)
    image.load()
    small = image.convert("RGB").resize(
        (max(1, image.width // 4), max(1, image.height // 4)), Image.Resampling.LANCZOS
    )
    small.save(out_jpg, "JPEG", quality=JPEG_QUALITY)
    return small.size


@dataclass
class Settings:
    """Everything a worker needs; picklable for the process pool."""

    jp2_dir: Path
    out_dir: Path
    mirror: str = DEFAULT_MIRROR
    bucket: str | None = None
    streams: int = 2
    upload: bool = False
    dry_run: bool = False


def broken_log_path(out_dir: Path) -> Path:
    return out_dir / "broken.log"


def log_broken(out_dir: Path, plan: ItemPlan, sheet: Sheet, reason: str) -> None:
    """Append one tab-separated line: item, stem, source, reason."""
    with open(broken_log_path(out_dir), "a") as handle:
        handle.write(f"{plan.item}\t{sheet.stem}\t{sheet.source}\t{reason}\n")


def sheet_outputs(dest: Path, sheet: Sheet) -> tuple[Path, Path | None]:
    """(25% JPEG path, raw full-resolution path or None) for a sheet."""
    quarter = dest / f"{sheet.key}.jpg"
    raw = dest / "raw" / f"{sheet.key}.jpg" if is_candidate(sheet.key) else None
    return quarter, raw


def process_sheet(plan: ItemPlan, sheet: Sheet, settings: Settings) -> dict | None:
    """Fetch and decode one sheet; returns its metadata row, or None if broken."""
    dest = item_dir(settings.out_dir, plan)
    quarter, raw = sheet_outputs(dest, sheet)
    row = asdict(sheet)
    try:
        if sheet.source.startswith("torrent"):
            local = jp2_path(settings.jp2_dir, sheet)
            if sheet.source == "torrent-master-tif":
                local = local.with_suffix(".tif")
            if not (
                local.exists()
                and (not sheet.bytes or local.stat().st_size == sheet.bytes)
            ):
                fetch(source_url(settings.mirror, sheet), local, sheet.bytes)
            if not quarter.exists():
                if local.suffix == ".jp2":
                    decode_jp2(local, quarter, QUARTER_REDUCE)
                else:
                    scale_to_quarter(local, quarter)
            if raw and not raw.exists():
                if local.suffix == ".jp2":
                    decode_jp2(local, raw, 0)
                else:
                    Image.MAX_IMAGE_PIXELS = None
                    Image.open(local).convert("RGB").save(
                        raw, "JPEG", quality=JPEG_QUALITY
                    )
        else:  # loc-iiif: the sheet has no file in the torrent
            if not quarter.exists():
                fetch(source_url(settings.mirror, sheet), quarter)
            if raw and not raw.exists():
                fetch(source_url(settings.mirror, sheet, full=True), raw)
        with Image.open(quarter) as image:
            row["width"], row["height"] = image.size
        row["raw"] = raw is not None
        return row
    except Exception as error:  # noqa: BLE001 -- any failure is a broken sheet, logged and skipped
        log_broken(
            settings.out_dir,
            plan,
            sheet,
            f"{error.__class__.__name__}: {str(error)[:200]}",
        )
        for path in (quarter, raw):
            if path and path.exists() and path.stat().st_size == 0:
                path.unlink()
        return None


def write_metadata(
    dest: Path, plan: ItemPlan, rows: list[dict], broken: list[str]
) -> None:
    """The item's metadata.json: catalog fields plus the sheet table."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metadata.json").write_text(
        json.dumps(
            {
                "item": plan.item,
                "loc_url": f"https://www.loc.gov/item/{plan.item}/",
                "state": plan.state,
                "year": plan.year,
                "city": plan.city,
                "storage_dir": plan.sheets[0].storage_dir if plan.sheets else None,
                "jpeg_quality": JPEG_QUALITY,
                "scale_percent": 25,
                "sheets": rows,
                "broken": broken,
            },
            indent=1,
        )
    )


def upload_item(dest: Path, plan: ItemPlan, bucket: str) -> None:
    """``aws s3 sync`` the item directory to its prefix, then mark it uploaded."""
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            str(dest),
            s3_prefix(bucket, plan),
            "--exclude",
            ".*",
            "--only-show-errors",
        ],
        check=True,
    )
    (dest / UPLOADED).touch()


def process_item(args: tuple[ItemPlan, Settings]) -> dict:
    """Do one item end to end; returns a progress record."""
    plan, settings = args
    dest = item_dir(settings.out_dir, plan)
    started = time.time()
    if settings.dry_run:
        return {"item": plan.item, "sheets": len(plan.sheets), "dry_run": True}
    if not (dest / DONE).exists():
        rows: list[dict] = []
        broken: list[str] = []
        with ThreadPoolExecutor(max_workers=settings.streams) as pool:
            for sheet, row in zip(
                plan.sheets,
                pool.map(lambda s: process_sheet(plan, s, settings), plan.sheets),
            ):
                if row is None:
                    broken.append(sheet.stem)
                else:
                    rows.append(row)
        write_metadata(dest, plan, rows, broken)
        (dest / DONE).touch()
    if settings.upload and settings.bucket and not (dest / UPLOADED).exists():
        upload_item(dest, plan, settings.bucket)
    return {
        "item": plan.item,
        "sheets": len(plan.sheets),
        "seconds": round(time.time() - started, 1),
        "uploaded": (dest / UPLOADED).exists(),
    }


def select_items(
    plans: dict[str, ItemPlan], args: argparse.Namespace
) -> list[ItemPlan]:
    """The items to run, filtered by --states / --items, in a stable order."""
    states = set(args.states.split(",")) if args.states else None
    wanted = set(args.items.split(",")) if args.items else None
    chosen = [
        plan
        for plan in plans.values()
        if (states is None or plan.state in states)
        and (wanted is None or plan.item in wanted)
    ]
    chosen.sort(key=lambda plan: (plan.state, plan.year, plan.item))
    return chosen[: args.limit] if args.limit else chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mapping",
        type=Path,
        help="Mapping TSV (item, state, year, ..., stem, page_key, source, bytes, storage_dir).",
    )
    parser.add_argument(
        "--jp2-dir",
        type=Path,
        required=True,
        help="Local mirror of the torrent's storage-services tree.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Staging root; by-state/<state>/<year>/<item>/ goes under it.",
    )
    parser.add_argument(
        "--mirror", default=DEFAULT_MIRROR, help="HTTP root serving the torrent's tree."
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="s3://bucket to sync finished items to (with --upload).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="aws s3 sync each item after it completes.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only sync items already marked done.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Items processed at once."
    )
    parser.add_argument(
        "--streams", type=int, default=2, help="Concurrent sheet downloads per item."
    )
    parser.add_argument(
        "--states",
        default=None,
        help="Comma-separated catalog state folders to include.",
    )
    parser.add_argument(
        "--items", default=None, help="Comma-separated item ids to include."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after this many items (0 = all)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the items and sheet counts; touch nothing.",
    )
    args = parser.parse_args()

    if not opj_available():
        print(
            "warning: opj_decompress not found; decoding with Pillow (several times slower)",
            file=sys.stderr,
        )
    plans = load_mapping(args.mapping)
    chosen = select_items(plans, args)
    settings = Settings(
        jp2_dir=args.jp2_dir,
        out_dir=args.out_dir,
        mirror=args.mirror,
        bucket=args.bucket,
        streams=args.streams,
        upload=args.upload or args.upload_only,
        dry_run=args.dry_run,
    )
    if args.upload_only:
        chosen = [
            plan for plan in chosen if (item_dir(args.out_dir, plan) / DONE).exists()
        ]
    total_sheets = sum(len(plan.sheets) for plan in chosen)
    total_bytes = sum(sheet.bytes for plan in chosen for sheet in plan.sheets)
    print(
        f"{len(chosen)} items, {total_sheets:,} sheets, {total_bytes / 1e9:,.1f} GB of JP2",
        file=sys.stderr,
    )
    if args.dry_run:
        for plan in chosen[:20]:
            print(
                f"  {plan.item} {plan.state}/{plan.year} {len(plan.sheets)} sheets -> {item_dir(args.out_dir, plan)}"
            )
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    started = time.time()
    with (
        open(args.out_dir / "progress.jsonl", "a") as progress,
        Pool(args.workers) as pool,
    ):
        for record in pool.imap_unordered(
            process_item, [(plan, settings) for plan in chosen]
        ):
            progress.write(json.dumps(record) + "\n")
            progress.flush()
            done += 1
            if done % 25 == 0 or done == len(chosen):
                elapsed = time.time() - started
                remaining = (len(chosen) - done) * elapsed / done / 3600
                print(
                    f"{done}/{len(chosen)} items, {elapsed / 3600:.1f} h elapsed, "
                    f"{remaining:.1f} h remaining at this rate",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
