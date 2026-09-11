"""Build the mapsnap Sanborn mirror from full-resolution LoC JP2s.

Source of truth is the Library of Congress ``storage-services`` tree as served
by the torrent's HTTP mirror: every sheet is a JP2 whose filename carries the
sheet's own identity (``06246_1914-0051``), so no sequence-number mapping is
involved. The plan is a mapping table (one row per sheet: item, state, year,
city, sequence, stem, page key, source, bytes, storage dir) built from the
torrent listing and the loc.gov catalog.

Three stages run concurrently, each bounded by a different resource:

  1. download (network, ``--streams`` connections): a map sheet whose 25%
     JPEG the mirror has already rendered (beside its source, with a ``.jpg``
     suffix: ``storage-services/service/<dir>/<stem>.jpg``, or under
     ``master/`` for the TIFF-only sheets) is fetched as that JPEG, straight
     into staging; otherwise its JP2 goes to ``--jp2-dir`` mirroring the
     torrent tree (``storage-services/service/<dir>/<stem>.jp2``), verified
     against the listed byte count, so the copy stays a valid torrent payload
     and is kept. The key-map candidates below always take the JP2 route,
     since their raw copy needs the full resolution;
  2. decode (CPU, ``--decode-workers`` processes): a JP2 is decoded at the
     JPEG 2000 quarter-resolution level -- the pipeline's 25% working scale --
     to ``<staging>/by-state/<state>/<year>/<item>/p<key>.jpg`` (JPEG quality
     95, what ``mapsnap scale`` writes; the mirror's pre-rendered JPEGs are
     the same decode); page-0 sheets (``p0``, ``p0b``, ``p0L``) and letter
     pages, the key-map candidates, are also decoded at full resolution to
     ``raw/p<key>.jpg``;
  3. upload (your uplink, ``--upload-workers`` ``aws s3 sync`` at a time):
     the item directory goes to ``<bucket>/by-state/<state>/<year>/<item>/``
     and its staging copy is deleted (``--keep-staging`` to retain it).

State lives only on local disk, under ``--out-dir``: per item a copy of
``metadata.json`` (catalog fields plus the sheet table) with ``.done`` and
``.uploaded`` markers, plus ``broken.log`` (tab-separated item, stem, source,
reason) and ``progress.jsonl``. Each sheet row of ``metadata.json`` records
``prerendered`` (its 25% JPEG arrived ready-made) and ``source_on_disk`` (its
JP2 or TIFF is in ``--jp2-dir``), so the sheets with no local full-resolution
copy can be listed later. Resuming never lists S3: an item with its marker is
skipped, a JP2 on disk at the listed size is not re-fetched (and is used in
preference to the mirror's JPEG), and an output already in staging is not
re-decoded. ``--retry-broken`` clears the markers of finished items whose
metadata lists broken sheets, so those run again (the sync re-uploads only
what changed).

The progress bar counts sheets through the decode stage; its postfix shows
what actually gates the finish once the mirror's JPEGs carry most sheets:
the uplink, as ``up_rate`` (averaged over the last ten minutes) and ``up_eta``
(the staging backlog plus the undecoded remainder at that rate).

Items are processed in a seeded random order, so however far the run has
got, the finished subset is a uniform sample of the collection
(``--sequential`` for state, year, item order). Sheets whose page key does not start with a digit (covr, ind1, cbd, titl,
note) are skipped: nothing in the pipeline reads them. A sheet that fails to
download or decode after retries is logged as broken and left out of its item.

    mapsnap loc-mirror ~/Downloads/loc-sanborn-maps.mapping.tsv \\
        --mirror http://<host>:<port> \\
        --jp2-dir /Volumes/fivetera/loc-sanborn-maps/jp2 \\
        --out-dir /Volumes/fivetera/mapsnap-sanborn \\
        --staging-dir /Volumes/fivetera/mapsnap-sanborn/staging \\
        --bucket s3://mapsnap-sanborn --upload
"""

import argparse
import csv
import http.client
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from PIL import Image
from tqdm import tqdm

from mapsnap.keymap.fit_keymap import page_number
from mapsnap.keymap.identify import is_letter_page

LOC_IIIF = "https://tile.loc.gov/image-services/iiif"
# tile.loc.gov answers 403 to urllib's default User-Agent; an honest one is fine.
USER_AGENT = "mapsnap loc-mirror/1.0 (+https://github.com/danvk/mapsnap)"
JPEG_QUALITY = 95  # what mapsnap scale writes; the pipeline is tuned on it
QUARTER_REDUCE = 2  # JPEG 2000 resolution levels to drop: 1/4 linear = 25%
RETRIES = 4
RETRY_DELAY = 5.0  # seconds, times the attempt number
# A failed upload is retried by the running process, first after this many
# seconds and then with doubling delays up to UPLOAD_RETRY_MAX: an expired
# `aws login` session (12 h) fails every sync until it is renewed, and the
# renewal should not need a restart.
UPLOAD_RETRY_DELAY = 60.0
UPLOAD_RETRY_MAX = 600.0
UPLOAD_RATE_WINDOW = 600.0  # seconds of completed uploads the bar's rate averages over
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


@dataclass
class Settings:
    """Everything the stages need; picklable for the process pool."""

    jp2_dir: Path
    out_dir: Path
    staging_dir: Path
    mirror: str  # HTTP root serving the torrent's storage-services tree
    bucket: str | None = None
    upload: bool = False
    keep_staging: bool = False


@dataclass(frozen=True)
class Fetched:
    """What the download stage brought to disk for one sheet."""

    prerendered: bool  # the 25% JPEG arrived ready-made (mirror JPEG or LoC IIIF)
    bytes: int  # downloaded by this call


class NotFound(OSError):
    """The URL does not exist (HTTP 404, a missing file): an answer, not retried."""


def keep_sheet(key: str) -> bool:
    """Whether the pipeline can use this sheet: numbered pages and letter pages only."""
    return re.match(r"p\d", key) is not None or is_letter_page(key)


def is_candidate(key: str) -> bool:
    """Whether a raw copy is kept: the page-0 family (p0, p0b, p0L) and letter pages.

    keymap.identify also nominates the page-1 family, but page 0 is the key
    map wherever one exists, and raw copies of every volume's sheet 1 would
    cost 34,000 full-resolution decodes for candidates that are almost all
    ordinary map pages.
    """
    if is_letter_page(key):
        return True
    base = re.sub(r"[a-j]$", "", key) if re.match(r"p\d+[a-j]$", key) else key
    return page_number(base) == 0


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


def item_relative(plan: ItemPlan) -> Path:
    """``by-state/<state>/<year>/<item>``: the item's path under staging, state, and the bucket."""
    return Path("by-state") / plan.state / plan.year / plan.item


def s3_prefix(bucket: str, plan: ItemPlan) -> str:
    """The item's destination, e.g. ``s3://mapsnap-sanborn/by-state/alabama/1922/sanborn00081_001``."""
    return f"{bucket.rstrip('/')}/{item_relative(plan).as_posix()}"


def source_relative(sheet: Sheet, suffix: str | None = None) -> str:
    """A torrent sheet's path under ``storage-services``: its branch, directory, stem, suffix.

    JP2s live in the ``service`` tree, the TIFF-only sheets in ``master``;
    ``suffix`` overrides the source's own (``.jpg`` for the pre-rendered copy).
    """
    branch = "master" if sheet.source.startswith("torrent-master") else "service"
    if suffix is None:
        suffix = ".tif" if sheet.source == "torrent-master-tif" else ".jp2"
    return f"storage-services/{branch}/{sheet.storage_dir}/{sheet.stem}{suffix}"


def jp2_path(jp2_dir: Path, sheet: Sheet) -> Path:
    """Where the sheet's JP2 (or TIFF master) lives locally, mirroring the torrent tree."""
    return jp2_dir / source_relative(sheet)


def source_url(mirror: str, sheet: Sheet, full: bool = False) -> str:
    """The URL to fetch a sheet from: the mirror's JP2/TIFF, or LoC's IIIF for the rest."""
    if sheet.source.startswith("torrent"):
        return f"{mirror}/{source_relative(sheet)}"
    service = "service:" + sheet.storage_dir.replace("/", ":")
    return f"{LOC_IIIF}/{service}:{sheet.stem}/full/{'full' if full else 'pct:25'}/0/default.jpg"


def quarter_url(mirror: str, sheet: Sheet) -> str:
    """The mirror's pre-rendered 25% JPEG of a torrent sheet: beside its source, ``.jpg`` suffix.

    The render job wrote each JPEG next to the file it came from, so a JP2's
    copy is in the ``service`` tree and a TIFF master's in ``master``.
    """
    return f"{mirror}/{source_relative(sheet, '.jpg')}"


def sheet_outputs(staging_item: Path, sheet: Sheet) -> tuple[Path, Path | None]:
    """(25% JPEG path, raw full-resolution path or None) for a sheet."""
    quarter = staging_item / f"{sheet.key}.jpg"
    raw = staging_item / "raw" / f"{sheet.key}.jpg" if is_candidate(sheet.key) else None
    return quarter, raw


_connections = threading.local()


def mirror_connection(host: str, port: int | None) -> http.client.HTTPConnection:
    """This thread's persistent HTTP/1.1 connection to the mirror, opened on first use.

    One connection per thread, reused across files: the mirror's uplink is
    fast, but each new TCP connection pays a handshake and slow start that an
    8 MB file never gets past, which cost two thirds of the throughput when
    every request opened its own.
    """
    key = (host, port)
    conn = getattr(_connections, "conn", None)
    if conn is None or getattr(_connections, "key", None) != key:
        if conn is not None:
            conn.close()
        conn = http.client.HTTPConnection(host, port, timeout=300)
        _connections.conn, _connections.key = conn, key
    return conn


def drop_connection() -> None:
    """Close this thread's mirror connection after an error, so the next fetch reconnects."""
    conn = getattr(_connections, "conn", None)
    if conn is not None:
        conn.close()
    _connections.conn = None


class Body(Protocol):
    """A response body read in chunks (http.client's and urllib's both are)."""

    def read(self, amt: int, /) -> bytes: ...


def stream_body(response: Body, partial: Path, content_length: str | None) -> int:
    """Copy a response body to ``partial``; returns its size.

    A body shorter than its Content-Length raises OSError: http.client ends a
    truncated body silently, and a short file would otherwise pass as an image.
    """
    written = 0
    with open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            written += len(chunk)
    if content_length and written != int(content_length):
        raise OSError(f"short body: {written} of {content_length} bytes")
    return written


def fetch_http(url: str, partial: Path) -> int:
    """GET ``url`` over the thread's keep-alive connection into ``partial``; returns its size.

    A 404 raises NotFound, its body drained first so the connection stays reusable.
    """
    parts = urlsplit(url)
    conn = mirror_connection(parts.hostname or "", parts.port)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    conn.request("GET", path, headers={"Connection": "keep-alive"})
    response = conn.getresponse()
    try:
        if response.status != 200:
            response.read()
            if response.status == 404:
                raise NotFound(f"HTTP 404 {url}")
            raise OSError(f"HTTP {response.status}")
        return stream_body(response, partial, response.getheader("Content-Length"))
    finally:
        response.close()


def fetch_urllib(url: str, partial: Path) -> int:
    """GET ``url`` through urllib (https, file://) into ``partial``; returns its size.

    A 404, or a file:// path that does not exist, raises NotFound.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return stream_body(
                response, partial, response.headers.get("Content-Length")
            )
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise NotFound(f"HTTP 404 {url}") from error
        raise
    except urllib.error.URLError as error:
        if isinstance(error.reason, FileNotFoundError):
            raise NotFound(f"missing {url}") from error
        raise


def fetch(url: str, dest: Path, expected_bytes: int = 0) -> int:
    """Download ``url`` to ``dest`` with retries; returns its size, checked when listed.

    Plain-http URLs (the mirror) reuse a per-thread keep-alive connection;
    anything else (LoC's IIIF over https, file:// in tests) goes through
    urllib. NotFound is raised at once, without retries: a 404 is the mirror's
    answer (no such rendering), not a failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(RETRIES):
        try:
            if urlsplit(url).scheme == "http":
                size = fetch_http(url, partial)
            else:
                size = fetch_urllib(url, partial)
            if expected_bytes and size != expected_bytes:
                raise OSError(f"size {size} != listed {expected_bytes}")
            partial.replace(dest)
            return size
        except NotFound:
            partial.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            last = error
            drop_connection()
            time.sleep(RETRY_DELAY * (attempt + 1))
    partial.unlink(missing_ok=True)
    raise OSError(f"{url}: {last}")


def broken_log_path(out_dir: Path) -> Path:
    return out_dir / "broken.log"


def log_error(out_dir: Path, stage: str, item: str, error: BaseException) -> None:
    """Append a stage failure to errors.log; the item is left for the next resume."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "errors.log", "a") as handle:
        handle.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{stage}\t{item}\t"
            f"{error.__class__.__name__}: {str(error)[:300]}\n"
        )


def log_broken(out_dir: Path, plan: ItemPlan, sheet: Sheet, reason: str) -> None:
    """Append one tab-separated line: item, stem, source, reason."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(broken_log_path(out_dir), "a") as handle:
        handle.write(f"{plan.item}\t{sheet.stem}\t{sheet.source}\t{reason}\n")


# ---------------------------------------------------------------- stage 1: download


def fetch_sheet(plan: ItemPlan, sheet: Sheet, settings: Settings) -> Fetched | None:
    """Bring the sheet's inputs to disk; None (and a broken-log line) on failure.

    A torrent sheet whose full-resolution source is already in the JP2 mirror
    needs nothing more. Otherwise a plain map sheet tries the mirror's
    pre-rendered 25% JPEG first, straight into staging, and only when the
    mirror has none does its JP2 come down. Key-map candidates need the full
    resolution for their raw copy, so they always take the JP2. The few
    LoC-only sheets are fetched already rendered from LoC's IIIF.
    """
    try:
        if not sheet.source.startswith("torrent"):
            return fetch_loc_sheet(plan, sheet, settings)
        quarter, raw = sheet_outputs(settings.staging_dir / item_relative(plan), sheet)
        local = jp2_path(settings.jp2_dir, sheet)
        if local.exists() and (not sheet.bytes or local.stat().st_size == sheet.bytes):
            return Fetched(prerendered=False, bytes=0)
        if raw is None:
            if quarter.exists():  # staged ready-made on an earlier run
                return Fetched(prerendered=True, bytes=0)
            try:
                size = fetch(quarter_url(settings.mirror, sheet), quarter)
                return Fetched(prerendered=True, bytes=size)
            except NotFound:
                pass
        elif quarter.exists() and raw.exists():
            return Fetched(prerendered=False, bytes=0)
        size = fetch(source_url(settings.mirror, sheet), local, sheet.bytes)
        return Fetched(prerendered=False, bytes=size)
    except Exception as error:  # noqa: BLE001 -- any failure is a broken sheet
        log_broken(
            settings.out_dir,
            plan,
            sheet,
            f"{error.__class__.__name__}: {str(error)[:200]}",
        )
        return None


def fetch_loc_sheet(plan: ItemPlan, sheet: Sheet, settings: Settings) -> Fetched:
    """A LoC-only sheet: its 25% rendering (and raw copy, for a candidate) straight into staging."""
    quarter, raw = sheet_outputs(settings.staging_dir / item_relative(plan), sheet)
    size = 0
    if not quarter.exists():
        size += fetch(source_url(settings.mirror, sheet), quarter)
    if raw and not raw.exists():
        size += fetch(source_url(settings.mirror, sheet, full=True), raw)
    return Fetched(prerendered=True, bytes=size)


# ---------------------------------------------------------------- stage 2: decode


def opj_available() -> bool:
    """Whether the OpenJPEG decoder CLI is on the PATH."""
    return shutil.which("opj_decompress") is not None


def save_jpeg(image: Image.Image, out_jpg: Path) -> None:
    """Write ``image`` as a quality-95 JPEG, atomically.

    Written under a ``.part`` name and renamed, so a run killed mid-write
    leaves no half file that a resume would take for a finished output.
    """
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    partial = out_jpg.with_suffix(out_jpg.suffix + ".part")
    image.convert("RGB").save(partial, "JPEG", quality=JPEG_QUALITY)
    partial.replace(out_jpg)


def decode_jp2(jp2: Path, out_jpg: Path, reduce: int) -> tuple[int, int]:
    """Decode a JP2 ``reduce`` resolution levels down and write a JPEG; returns its size.

    Uses ``opj_decompress`` (fast, multithreaded) when present, else Pillow.
    """
    if opj_available():
        with tempfile.TemporaryDirectory() as tmp:
            ppm = Path(tmp) / "decoded.ppm"
            subprocess.run(
                [
                    "opj_decompress",
                    "-threads",
                    "2",
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
    save_jpeg(image, out_jpg)
    return image.size


def scale_to_quarter(src: Path, out_jpg: Path) -> tuple[int, int]:
    """Write a 25% JPEG of a full-resolution TIFF (the master-only sheets)."""
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(src)
    image.load()
    small = image.convert("RGB").resize(
        (max(1, image.width // 4), max(1, image.height // 4)), Image.Resampling.LANCZOS
    )
    save_jpeg(small, out_jpg)
    return small.size


def decode_sheet(
    plan: ItemPlan, sheet: Sheet, settings: Settings, fetched: Fetched
) -> dict | None:
    """Produce the sheet's outputs in staging; returns its metadata row, None if broken.

    Whatever the download stage staged ready-made (a pre-rendered 25% JPEG) is
    kept as is; the rest is decoded from the local JP2 or TIFF.
    """
    quarter, raw = sheet_outputs(settings.staging_dir / item_relative(plan), sheet)
    row = asdict(sheet)
    local = (
        jp2_path(settings.jp2_dir, sheet)
        if sheet.source.startswith("torrent")
        else None
    )
    try:
        if local is not None:
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
                    save_jpeg(Image.open(local), raw)
        with Image.open(quarter) as image:
            row["width"], row["height"] = image.size
        row["raw"] = raw is not None
        row["prerendered"] = fetched.prerendered
        row["source_on_disk"] = local is not None and local.exists()
        return row
    except Exception as error:  # noqa: BLE001 -- any failure is a broken sheet
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


def metadata_document(plan: ItemPlan, rows: list[dict], broken: list[str]) -> dict:
    """The item's metadata.json content: catalog fields plus the sheet table."""
    return {
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
    }


def decode_item(
    plan: ItemPlan, fetched: list[Fetched | None], settings: Settings
) -> tuple[int, int, int]:
    """Decode every fetched sheet, write metadata to staging and state, mark done.

    Returns (sheets decoded, sheets broken, bytes now waiting in staging).
    Runs in a worker process.
    """
    rows: list[dict] = []
    broken: list[str] = []
    for sheet, result in zip(plan.sheets, fetched):
        row = decode_sheet(plan, sheet, settings, result) if result else None
        if row is None:
            broken.append(sheet.stem)
        else:
            rows.append(row)
    document = json.dumps(metadata_document(plan, rows, broken), indent=1)
    for root in (settings.staging_dir, settings.out_dir):
        directory = root / item_relative(plan)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(document)
    (settings.out_dir / item_relative(plan) / DONE).touch()
    return (
        len(rows),
        len(broken),
        staged_size(settings.staging_dir / item_relative(plan)),
    )


# ---------------------------------------------------------------- stage 3: upload


def upload_item(plan: ItemPlan, settings: Settings) -> None:
    """``aws s3 sync`` the staging directory to its prefix, mark uploaded, drop the staging copy."""
    assert settings.bucket
    staging = settings.staging_dir / item_relative(plan)
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            str(staging),
            s3_prefix(settings.bucket, plan),
            "--only-show-errors",
        ],
        check=True,
    )
    (settings.out_dir / item_relative(plan) / UPLOADED).touch()
    if not settings.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
        prune_empty_parents(staging.parent, settings.staging_dir)


def prune_empty_parents(directory: Path, root: Path) -> None:
    """Remove ``directory`` and its parents up to ``root`` while they are empty.

    Other threads prune and create siblings concurrently (two uploads finishing
    under one state directory; a decode writing a new year directory), so a
    directory can vanish or fill between the check and the rmdir. Either way
    the right response is to stop.
    """
    while directory != root:
        try:
            directory.rmdir()
        except OSError:
            return
        directory = directory.parent


# ---------------------------------------------------------------- orchestration


def staged_size(directory: Path) -> int:
    """Bytes of files under a staging item directory (0 if absent)."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def awaiting_upload(plan: ItemPlan, settings: Settings) -> bool:
    """Decoded on a previous run with its staging copy intact: only the upload is left."""
    state = settings.out_dir / item_relative(plan)
    staging = settings.staging_dir / item_relative(plan)
    return (state / DONE).exists() and (staging / "metadata.json").exists()


def item_complete(plan: ItemPlan, settings: Settings) -> bool:
    """Whether resume can skip the item outright, from local markers only."""
    state = settings.out_dir / item_relative(plan)
    if settings.upload:
        return (state / UPLOADED).exists()
    return (state / DONE).exists()


def broken_sheets(plan: ItemPlan, settings: Settings) -> list[str]:
    """Stems the item's recorded metadata lists as broken (empty if never decoded)."""
    metadata = settings.out_dir / item_relative(plan) / "metadata.json"
    if not metadata.exists():
        return []
    return list(json.loads(metadata.read_text()).get("broken", []))


def retry_broken(plan: ItemPlan, settings: Settings) -> list[str]:
    """Clear a finished item's markers when it has broken sheets; returns their stems.

    The item then runs again from the download stage. A broken sheet starts
    over: whatever it left in staging is removed (a half-written file from a
    killed run must not pass for a finished output), and so is its local JP2
    or TIFF, since a source that failed to decode is suspect and the mirror's
    pre-rendered JPEG may now exist for it. Items with no broken sheets are
    untouched.
    """
    broken = broken_sheets(plan, settings)
    if not broken:
        return []
    staging_item = settings.staging_dir / item_relative(plan)
    for sheet in plan.sheets:
        if sheet.stem in broken:
            for path in sheet_outputs(staging_item, sheet):
                if path is not None:
                    path.unlink(missing_ok=True)
            if sheet.source.startswith("torrent"):
                jp2_path(settings.jp2_dir, sheet).unlink(missing_ok=True)
    state = settings.out_dir / item_relative(plan)
    for marker in (DONE, UPLOADED):
        (state / marker).unlink(missing_ok=True)
    return broken


def select_items(
    plans: dict[str, ItemPlan],
    *,
    states: str | None = None,
    items: str | None = None,
    limit: int = 0,
    seed: int | None = 0,
) -> list[ItemPlan]:
    """The items to run, filtered by state folders and ids.

    The order is a random permutation seeded by ``seed`` (so a restart walks the
    same sequence, and any prefix of the run is a uniform sample of the
    collection, which the state-by-state order would not be); ``seed=None``
    keeps the state, year, item order. ``limit`` truncates after ordering.
    """
    wanted_states = set(states.split(",")) if states else None
    wanted_items = set(items.split(",")) if items else None
    chosen = [
        plan
        for plan in plans.values()
        if (wanted_states is None or plan.state in wanted_states)
        and (wanted_items is None or plan.item in wanted_items)
    ]
    chosen.sort(key=lambda plan: (plan.state, plan.year, plan.item))
    if seed is not None:
        random.Random(seed).shuffle(chosen)
    return chosen[:limit] if limit else chosen


class UploadMeter:
    """Rolling upload throughput from completed items, for the bar's upload ETA.

    Bytes are credited when an item's sync finishes and averaged over the
    last ``window`` seconds (or since ``start`` while the window is still
    filling), so the rate tracks the uplink as it is now, not the run's
    history.
    """

    def __init__(self, start: float, window: float = UPLOAD_RATE_WINDOW) -> None:
        self.start = start
        self.window = window
        self.completed: deque[tuple[float, int]] = deque()
        self.any_completed = False

    def add(self, now: float, nbytes: int) -> None:
        self.completed.append((now, nbytes))
        self.any_completed = True

    def rate(self, now: float) -> float | None:
        """Bytes per second over the window: None before the first completed upload, 0.0 when stalled."""
        while self.completed and self.completed[0][0] < now - self.window:
            self.completed.popleft()
        span = min(self.window, now - self.start)
        if not self.any_completed or span <= 0:
            return None
        return sum(nbytes for _, nbytes in self.completed) / span


def format_hours(seconds: float) -> str:
    """A duration as ``h:mm``."""
    minutes = int(seconds // 60)
    return f"{minutes // 60}:{minutes % 60:02d}"


def run_pipeline(
    items: list[ItemPlan],
    settings: Settings,
    *,
    streams: int = 12,
    decode_workers: int = 4,
    prefetch_items: int = 16,
    upload_workers: int = 2,
    max_staging_bytes: float = 250e9,
    progress: bool = True,
) -> dict[str, int]:
    """Drive the three stages concurrently over ``items``; returns totals.

    Downloads are submitted sheet by sheet for up to ``prefetch_items`` items
    ahead of decoding, so small items do not starve the connections; an item
    moves to decoding when all its sheets are on disk, and to upload when its
    metadata is written. The upload stage is decoupled: downloads and decodes
    never wait for the uplink. New items are held back only while more than
    ``max_staging_bytes`` of decoded output is waiting to upload, so a slow
    uplink costs staging disk rather than download throughput. A stage failure
    is logged to errors.log and never ends the run: a failed upload (an expired
    AWS session, say) is retried by this process with backoff, its staging copy
    kept; a failed decode is left for the next resume. One tqdm bar counts
    sheets through decoding, with downloaded gigabytes, uploaded items, the
    staging backlog, broken sheets, stage errors and, when uploading, the
    rolling upload rate and the ETA for the backlog plus the undecoded
    remainder in the postfix.
    """
    todo = deque(item for item in items if not item_complete(item, settings))
    totals = {
        "items": 0,
        "sheets": 0,
        "broken": 0,
        "uploaded": 0,
        "bytes": 0,
        "errors": 0,
        "retries": 0,
    }
    bar = tqdm(
        total=sum(len(item.sheets) for item in todo),
        unit="sheet",
        smoothing=0,
        disable=not progress,
        dynamic_ncols=True,
    )
    # Closed on every exit path: a live bar's __del__ during interpreter
    # teardown after an exception segfaults CPython 3.13 in tqdm's formatter.
    try:
        run_stages(
            todo,
            settings,
            totals,
            bar,
            streams=streams,
            decode_workers=decode_workers,
            prefetch_items=prefetch_items,
            upload_workers=upload_workers,
            max_staging_bytes=max_staging_bytes,
        )
    finally:
        bar.close()
    return totals


def run_stages(
    todo: deque[ItemPlan],
    settings: Settings,
    totals: dict[str, int],
    bar: tqdm,
    *,
    streams: int,
    decode_workers: int,
    prefetch_items: int,
    upload_workers: int,
    max_staging_bytes: float,
) -> None:
    """The coordinator loop behind run_pipeline; mutates ``totals`` and drives ``bar``."""
    downloads: dict[Future, tuple[ItemPlan, int]] = {}
    item_fetch: dict[str, list[Fetched | None]] = {}
    item_pending: dict[str, int] = {}  # downloads still out, per item in item_fetch
    decodes: dict[Future, ItemPlan] = {}
    uploads: dict[Future, tuple[ItemPlan, int]] = {}
    item_staged: dict[str, int] = {}
    staged_bytes = 0
    # Failed uploads wait here as (not-before time, attempt, item) until retried.
    retry_queue: deque[tuple[float, int, ItemPlan]] = deque()
    meter = UploadMeter(time.time())
    total_sheets = sum(len(plan.sheets) for plan in todo)
    handled_sheets = 0  # through decoding (or straight to upload on a resume)
    decoded_sheets = decoded_bytes = 0  # this run's decode output: bytes per sheet
    settings.out_dir.mkdir(parents=True, exist_ok=True)
    with (
        open(settings.out_dir / "progress.jsonl", "a") as progress_log,
        ThreadPoolExecutor(max_workers=streams) as fetch_pool,
        ProcessPoolExecutor(max_workers=decode_workers) as decode_pool,
        ThreadPoolExecutor(max_workers=upload_workers) as upload_pool,
    ):

        def start_decode(plan: ItemPlan) -> None:
            fetched = item_fetch.pop(plan.item)
            item_pending.pop(plan.item, None)
            decodes[decode_pool.submit(decode_item, plan, fetched, settings)] = plan

        while todo or downloads or decodes or uploads or retry_queue:
            now = time.time()
            for _ in range(len(retry_queue)):
                not_before, attempt, plan = retry_queue.popleft()
                if now >= not_before:
                    totals["retries"] += 1
                    uploads[upload_pool.submit(upload_item, plan, settings)] = (
                        plan,
                        attempt,
                    )
                else:
                    retry_queue.append((not_before, attempt, plan))
            while (
                todo
                and len(item_fetch) + len(decodes) < prefetch_items
                and staged_bytes < max_staging_bytes
            ):
                plan = todo.popleft()
                if (
                    settings.upload
                    and settings.bucket
                    and awaiting_upload(plan, settings)
                ):
                    staged = staged_size(settings.staging_dir / item_relative(plan))
                    item_staged[plan.item] = staged
                    staged_bytes += staged
                    uploads[upload_pool.submit(upload_item, plan, settings)] = (plan, 0)
                    bar.update(len(plan.sheets))
                    handled_sheets += len(plan.sheets)
                    continue
                item_fetch[plan.item] = [None] * len(plan.sheets)
                item_pending[plan.item] = len(plan.sheets)
                if not plan.sheets:
                    start_decode(plan)
                for index, sheet in enumerate(plan.sheets):
                    future = fetch_pool.submit(fetch_sheet, plan, sheet, settings)
                    downloads[future] = (plan, index)
            pending = list(downloads) + list(decodes) + list(uploads)
            if not pending:  # only the retry queue is left: wait for its clock
                time.sleep(min(1.0, max(0.0, retry_queue[0][0] - time.time())))
                continue
            done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                if future in downloads:
                    plan, index = downloads.pop(future)
                    try:
                        result: Fetched | None = future.result()
                    except Exception as error:  # noqa: BLE001 -- logged; the sheet counts as broken
                        log_error(
                            settings.out_dir, "download", plan.sheets[index].stem, error
                        )
                        totals["errors"] += 1
                        result = None
                    item_fetch[plan.item][index] = result
                    item_pending[plan.item] -= 1
                    if result:
                        totals["bytes"] += result.bytes
                    if item_pending[plan.item] == 0:
                        start_decode(plan)
                elif future in decodes:
                    plan = decodes.pop(future)
                    try:
                        decoded, broken, staged = future.result()
                    except Exception as error:  # noqa: BLE001 -- logged; retried on resume
                        log_error(settings.out_dir, "decode", plan.item, error)
                        totals["errors"] += 1
                        bar.update(len(plan.sheets))
                        handled_sheets += len(plan.sheets)
                        continue
                    totals["items"] += 1
                    totals["sheets"] += decoded
                    totals["broken"] += broken
                    bar.update(len(plan.sheets))
                    handled_sheets += len(plan.sheets)
                    decoded_sheets += decoded
                    decoded_bytes += staged
                    progress_log.write(
                        json.dumps(
                            {"item": plan.item, "sheets": decoded, "broken": broken}
                        )
                        + "\n"
                    )
                    progress_log.flush()
                    if settings.upload and settings.bucket:
                        item_staged[plan.item] = staged
                        staged_bytes += staged
                        uploads[upload_pool.submit(upload_item, plan, settings)] = (
                            plan,
                            0,
                        )
                else:
                    plan, attempt = uploads.pop(future)
                    try:
                        future.result()
                    except Exception as error:  # noqa: BLE001 -- staging kept; retried with backoff
                        log_error(settings.out_dir, "upload", plan.item, error)
                        totals["errors"] += 1
                        delay = min(UPLOAD_RETRY_DELAY * 2**attempt, UPLOAD_RETRY_MAX)
                        retry_queue.append((time.time() + delay, attempt + 1, plan))
                        continue
                    totals["uploaded"] += 1
                    item_bytes = item_staged.pop(plan.item, 0)
                    staged_bytes -= item_bytes
                    meter.add(time.time(), item_bytes)
            # Redraw only when something finished: tqdm repaints on update(),
            # and once decoding is done nothing updates the bar, so a postfix
            # set without a refresh would freeze at the last decoded sheet
            # while the upload backlog drains for a day.
            finished_something = bool(done)
            postfix: dict[str, object] = {
                "dl": f"{totals['bytes'] / 1e9:.1f}GB",
                "up": totals["uploaded"],
                "staged": f"{staged_bytes / 1e9:.1f}GB",
                "broken": totals["broken"],
                "errors": totals["errors"],
                "retry": len(retry_queue),
            }
            if settings.upload:
                rate = meter.rate(time.time())
                per_sheet = decoded_bytes / decoded_sheets if decoded_sheets else 0.0
                pending_bytes = (
                    staged_bytes + (total_sheets - handled_sheets) * per_sheet
                )
                postfix["up_rate"] = (
                    f"{rate / 1e6:.2f}MB/s" if rate is not None else "?"
                )
                postfix["up_eta"] = format_hours(pending_bytes / rate) if rate else "?"
            bar.set_postfix(postfix, refresh=finished_something)


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
        help="Local mirror of the torrent's storage-services tree (kept).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="State root: per-item metadata.json + markers, broken.log, progress.jsonl.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Where decoded images wait for upload (default <out-dir>/staging).",
    )
    parser.add_argument(
        "--mirror",
        required=True,
        metavar="URL",
        help="HTTP root serving the torrent's storage-services tree, e.g. "
        "http://host:port (no default: the mirror is somebody's machine, not "
        "a property of this program).",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="s3://bucket to sync finished items to (with --upload).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="aws s3 sync each item after it is decoded, then drop its staging copy.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the staging copy after upload.",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=12,
        help="Concurrent keep-alive downloads from the mirror.",
    )
    parser.add_argument(
        "--decode-workers", type=int, default=4, help="Decode processes."
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=2,
        help="Concurrent `aws s3 sync` processes (the uplink is the bottleneck "
        "once the mirror's pre-rendered JPEGs carry most sheets).",
    )
    parser.add_argument(
        "--prefetch-items",
        type=int,
        default=16,
        help="Items downloading or decoding at once (the upload queue is unbounded).",
    )
    parser.add_argument(
        "--max-staging-gb",
        type=float,
        default=250.0,
        help="Pause downloads while this much decoded output waits for upload.",
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
        "--seed",
        type=int,
        default=0,
        help="Seed of the random item order (the default order), so a restart "
        "walks the same sequence and any prefix is a uniform sample.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process items in state, year, item order instead of at random.",
    )
    parser.add_argument(
        "--retry-broken",
        action="store_true",
        help="Run finished items whose metadata lists broken sheets again: their "
        "markers are cleared, so they take the whole pipeline (the sync "
        "re-uploads only what changed).",
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
    settings = Settings(
        jp2_dir=args.jp2_dir,
        out_dir=args.out_dir,
        staging_dir=args.staging_dir or args.out_dir / "staging",
        mirror=args.mirror,
        bucket=args.bucket,
        upload=args.upload,
        keep_staging=args.keep_staging,
    )
    if settings.upload and not settings.bucket:
        sys.exit("--upload needs --bucket")
    chosen = select_items(
        load_mapping(args.mapping),
        states=args.states,
        items=args.items,
        limit=args.limit,
        seed=None if args.sequential else args.seed,
    )
    if args.retry_broken:
        lookup = broken_sheets if args.dry_run else retry_broken
        retried = {
            plan.item: stems for plan in chosen if (stems := lookup(plan, settings))
        }
        print(
            f"--retry-broken: {sum(len(v) for v in retried.values())} broken sheets "
            f"in {len(retried)} items {'would be' if args.dry_run else 'will be'} "
            "retried",
            file=sys.stderr,
        )
    pending = [plan for plan in chosen if not item_complete(plan, settings)]
    print(
        f"{len(chosen)} items selected, {len(pending)} to do: "
        f"{sum(len(p.sheets) for p in pending):,} sheets, up to "
        f"{sum(s.bytes for p in pending for s in p.sheets) / 1e9:,.1f} GB of JP2 "
        "(a sheet the mirror has pre-rendered downloads its ~1 MB JPEG instead)",
        file=sys.stderr,
    )
    if args.dry_run:
        for plan in pending[:20]:
            print(f"  {plan.item} {plan.state}/{plan.year} {len(plan.sheets)} sheets")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings.staging_dir.mkdir(parents=True, exist_ok=True)
    totals = run_pipeline(
        chosen,
        settings,
        streams=args.streams,
        decode_workers=args.decode_workers,
        prefetch_items=args.prefetch_items,
        upload_workers=args.upload_workers,
        max_staging_bytes=args.max_staging_gb * 1e9,
    )
    print(
        f"done: {totals['items']} items, {totals['sheets']:,} sheets, {totals['broken']} broken, "
        f"{totals['uploaded']} uploaded; broken sheets in {broken_log_path(args.out_dir)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
