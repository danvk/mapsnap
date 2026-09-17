"""Run the CPU half of the pipeline over one shard or queue of the mirror (#354).

``loc-craft`` leaves every sheet with its CRAFT boxes and its P(road) map, which
is everything the GPU is needed for. What remains -- split, adjacency, keymap,
ocr and fit -- is CPU work, and all of it is scoped to the *volume*: the key map
is confirmed against the volume's page set, adjacency needs every sheet, georef
needs the volume's reference scale, and reconcile is explicitly one joint
decision per volume. An LoC item is a volume, so one queue message is one unit
of work for the whole chain, and running the stages as separate passes would
only re-download the same item three times.

    mapsnap loc-fit --queue "$QUEUE" --counties items.tsv

Where the outputs go
--------------------

The item's stable half -- its images, CRAFT boxes and P(road) maps -- is written
once at the item root and shared by every run. Everything this chain produces
goes under ``<item>/runs/<tag>/`` instead, because ocr and fit outputs change
from run to run while craft's do not. At 133 KB a page, a corpus pass of run
outputs is about 55 GB, so keeping runs apart costs roughly a dollar a month and
buys a great deal: a pilot cannot collide with the full pass, two runs at the
same commit can be compared to see how deterministic the chain is, and a bad run
is one ``aws s3 rm --recursive`` rather than an unpickable mixture.

The tag comes from the queue message, not from this worker's flags, so the
prefix the outputs land in and the run recorded inside them are the same string.
Use one queue per run: SQS has no selective receive, so a worker cannot decline
a message meant for another run, and merely passing one over counts against
``maxReceiveCount`` until it dead-letters.

What is uploaded, and what is not
---------------------------------

Split's panel *images* stay on the worker. ``make_iiif_georef`` builds every
page's image URL from its parent (split pages share the parent's canvas), so
nothing downstream ever reads ``p209__1.jpg``; it reads the parent plus
``p209.panels.json``. The panel image, its derived boxes and its cropped P(road)
map are all deterministic functions of the parent and the rings, and re-cutting
them costs 0.35 vCPU-seconds a page, against about 50 GB of storage that every
later pass would re-download. So the chain re-runs ``split`` locally each time
and uploads only what cannot be recomputed cheaply: the cut itself, the reads,
the poses and the annotation page.

Ordering against loc-craft
--------------------------

This chain needs ``<stem>.boxes.json``, which is loc-craft's output. An item
whose boxes are not all present yet is *not ready* rather than failed: its
message goes back to the queue rather than being retired, so it is picked up
again once the GPU pass has been through it.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mapsnap import experiments, work_queue
from mapsnap.aws_cli import run_aws
from mapsnap.keymap.records import recorded_keymap_keys
from mapsnap.loc_craft import (
    Item,
    QueueSource,
    format_duration,
    item_prefix,
    list_prefix,
    read_manifest,
    resolve_manifest,
    select_shard,
    sync,
)
from mapsnap.utils import list_pages, source_images

# The county extracts, one per FIPS, cut by `mapsnap osm-counties`.
COUNTY_PREFIX = "osm-by-county"
# What `default_centerlines` looks for beside the pages; load_centerlines reads
# the .pbf directly, so the county extract needs no conversion (#408).
CENTERLINES_NAME = "centerlines.osm.pbf"
# The mirror's key-map record, written by `loc-keymaps` before anything is
# split. This chain re-derives it after the split instead (see run_chain).
KEYMAPS_NAME = "keymaps.json"
# `fit`'s own archive name and the annotation filename. NOT the run tag: that
# names a whole corpus pass and comes from the queue message.
ARCHIVE_TAG = "mapsnap"
# Every run's outputs live under this directory inside the item, one tag deep,
# so the stable half (images, CRAFT boxes, P(road)) is written once and shared,
# while ocr and fit outputs -- 133 KB a page, about 55 GB a corpus pass -- stay
# separate per run. A single `--exclude "runs/*"` keeps a download from pulling
# every previous run, which a bare tag directory beside `raw/` could not.
RUNS_DIRNAME = "runs"
# Uploaded on its own AFTER everything else, so its presence means the whole
# chain ran for this item. It used to ride in the same sync as the sidecars,
# where it sorts before `p*.streets.json` and so landed first: an interrupted
# upload left a done marker over a partial item, which is then skipped forever.
DONE_MARKER = f"{ARCHIVE_TAG}.iiif.json"
# The reads `--ocr-from` brings forward from an earlier run. `ocr --resume`
# decides what to keep: it re-reads any page whose recognizer weights differ,
# and any page the previous run never had (a new panel, say).
OCR_REUSE_GLOBS = ("p*.streets.json", "p*.txt")
# What the upload is expected to carry, for the tests to assert against: the
# sync itself works by exclusion, so this list documents the intent and the
# excludes below enforce it.
UPLOAD_GLOBS = (
    "p*.panels.json",
    "p*.streets.json",
    "p*.georef*.json",
    "p*.contradiction.json",
    "p*.provenance.json",
    "artifacts/osm_snap/candidates.jsonl",
    "artifacts/street_solve/candidates.jsonl",
    f"artifacts/{ARCHIVE_TAG}/manifest.json",
    "adjacency.json",
    "keymaps.json",
    f"{ARCHIVE_TAG}.iiif.json",
    # The key map's own annotation page: it is georeferenced like any other
    # sheet and is the one sheet that shows how the volume is laid out.
    f"{ARCHIVE_TAG}.keymap.iiif.json",
    "raw/*.keymap.json",
    "raw/*.keymap-raw.json",
    "raw/*.keymap.txt",
    "raw/*.regions.panels.json",
    "raw/*.inset.panels.json",
    "raw/*.cartouche.json",
    "raw/*.georef.json",
    "raw/*.panels.json",
)
# The panel images and the two sidecars derived from the parent's: a later run
# re-cuts them from the parent in 0.35 vCPU-s a page.
#
# The candidate files are NOT excluded, though they are nominally a cache.
# They are the only record of what snap and street-solve considered and
# rejected -- the oracle-of-eight analysis ran on exactly this -- and
# regenerating them means re-running the search, which is the expensive part.
# Madison p20__3 made the case concretely: its provenance says snap offered no
# hypothesis, and the reason why was in a candidates file that had not been
# kept. Measured at 10.6 KB a page, about 4.4 GB over the corpus.
UPLOAD_EXCLUDES = (
    # Every image and every CRAFT box file: loc-craft already wrote these once
    # at the item root, where all runs share them. Re-uploading them per run
    # would multiply 64-232 MB a volume by the number of runs, against the
    # 12-30 MB of output a run actually produces. This also covers the panel
    # images and their derived boxes, which are re-cut locally in 0.35
    # vCPU-seconds a page and never uploaded at all.
    "*.jpg",
    "*.boxes.json",
    "metadata.json",
    "artifacts/reconcile/*",
    # `fit` archives the whole run, which is a second copy of every sidecar in
    # the item. Its manifest is the part worth keeping -- the git SHA, the
    # model hashes, the stage timings and the fit-state counts -- so that is
    # re-included below.
    f"artifacts/{ARCHIVE_TAG}/*",
)
# Applied after the excludes, so it wins: aws s3 sync takes the last matching
# filter.
UPLOAD_INCLUDES = (f"artifacts/{ARCHIVE_TAG}/manifest.json",)
QUEUE_DEPTH_EVERY = 25


def run_prefix(bucket: str, item: Item, run_tag: str) -> str:
    """Where one run's outputs live for one item."""
    return f"{item_prefix(bucket, item)}/{RUNS_DIRNAME}/{run_tag}"


def resolve_run_tag(message_tag: str | None, flag_tag: str | None) -> str:
    """The run this item belongs to, from exactly one source.

    The message is the authority: it travels with the work, so the S3 prefix and
    the provenance recorded inside the outputs cannot name different runs. A
    ``--run-tag`` on the worker is an assertion against it, not an override --
    a mismatch means the fleet was pointed at the wrong queue, which is worth
    stopping for rather than quietly writing into another run's directory.
    """
    if message_tag and flag_tag and message_tag != flag_tag:
        raise ValueError(
            f"queue says run {message_tag!r}, this worker was launched for "
            f"{flag_tag!r}. Point it at the right queue, or drop --run-tag."
        )
    tag = message_tag or flag_tag
    if not tag:
        raise ValueError(
            "no run tag: fill the queue with `work-queue fill --run-tag TAG`, "
            "or pass --run-tag to this worker."
        )
    return tag


@dataclass(frozen=True)
class County:
    """The OSM extract an item's streets come from."""

    fips: str

    @property
    def key(self) -> str:
        return f"{COUNTY_PREFIX}/{self.fips}.osm.pbf"


def resolve_counties(sources: list[str], work_dir: Path) -> list[Path]:
    """The county mappings as local files, downloading any given as ``s3://`` URLs.

    The fleet needs them on every instance, and the bucket is what every
    instance already has. Each lands under ``work_dir`` by its own basename, so
    items.tsv and city-items.tsv do not collide the way a fixed name would.
    """
    paths: list[Path] = []
    for source in sources:
        if not source.startswith("s3://"):
            paths.append(Path(source))
            continue
        local = work_dir / source.rsplit("/", 1)[-1]
        if not local.exists():
            work_dir.mkdir(parents=True, exist_ok=True)
            run_aws(["aws", "s3", "cp", source, str(local), "--only-show-errors"])
        paths.append(local)
    return paths


def read_counties(paths: list[Path]) -> dict[str, County]:
    """Item -> county, from `loc-counties`' items.tsv and `loc-cities`' city-items.tsv.

    Both files carry one row per item with a ``fips`` column; the cities file
    covers the 428 items whose LoC county is an independent city and which have
    no Natural Earth county at all.
    """
    counties: dict[str, County] = {}
    for path in paths:
        with path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                fips = (row.get("fips") or "").strip()
                if row.get("item") and fips:
                    counties[row["item"]] = County(fips)
    return counties


@dataclass
class FitWork:
    """One item's state: what it has, what it still needs, and whether it can run."""

    item: Item
    pages: list[str]
    county: County | None
    ready: bool
    done: bool
    reason: str = ""
    run_tag: str = ""


def plan_fit(
    item: Item, present: list[str], county: County | None, run_tag: str
) -> FitWork:
    """Decide whether this item can run the CPU chain, and whether it already has.

    Not ready is not failure: an item whose CRAFT boxes are missing is waiting on
    the GPU pass, and an item with no county extract cannot read street names at
    all. Both go back to the queue rather than being retired.
    """
    keys = set(present)
    images = sorted(key for key in keys if key.endswith(".jpg") and key.count(".") == 1)
    pages = [key for key in images if "/" not in key and "__" not in key]
    # Done means done FOR THIS RUN: another tag's marker says nothing about this
    # one, which is the point of keeping runs apart.
    if f"{RUNS_DIRNAME}/{run_tag}/{DONE_MARKER}" in keys:
        return FitWork(item, pages, county, True, True, run_tag=run_tag)
    if not pages:
        return FitWork(
            item, pages, county, False, False, "no pages in the mirror", run_tag
        )
    unboxed = [
        page for page in pages if f"{page[: -len('.jpg')]}.boxes.json" not in keys
    ]
    if unboxed:
        return FitWork(
            item,
            pages,
            county,
            False,
            False,
            f"{len(unboxed)} page(s) await craft",
            run_tag,
        )
    if county is None:
        return FitWork(
            item, pages, county, False, False, "no county extract known", run_tag
        )
    return FitWork(item, pages, county, True, False, run_tag=run_tag)


def fetch_item(
    work: FitWork, bucket: str, work_dir: Path, ocr_from: str | None = None
) -> Path:
    """Sync the item down and put its county extract beside the pages.

    Three layers, in the order they must land: the stable half every run shares,
    then the reads an earlier run is lending (``--ocr-from``), then this run's
    own outputs, which win over both so an interrupted item resumes rather than
    restarting.
    """
    local = work_dir / work.item.item
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    sync(item_prefix(bucket, work.item), str(local), "--exclude", f"{RUNS_DIRNAME}/*")
    assert work.county is not None
    run_aws(
        [
            "aws",
            "s3",
            "cp",
            f"{bucket.rstrip('/')}/{work.county.key}",
            str(local / CENTERLINES_NAME),
            "--only-show-errors",
        ]
    )
    if ocr_from:
        borrow_reads(local, bucket, work.item, ocr_from)
    sync(run_prefix(bucket, work.item, work.run_tag), str(local))
    return local


def manifest_key(bucket: str, item: Item, run_tag: str) -> str:
    """Where one run archived its manifest for one item."""
    return f"{run_prefix(bucket, item, run_tag)}/artifacts/{ARCHIVE_TAG}/manifest.json"


def reused_reads_are_valid(local: Path, bucket: str, item: Item, source: str) -> bool:
    """Whether an earlier run's reads were made against the streets we now have.

    A read is a match between a page's text and a county extract, so it is only
    reusable if the extract has not changed underneath it -- and it has, for
    every county: the 0-buffer re-cut removed 187,439 foreign ways. The source
    run's manifest records the sha it used, which is the cheapest honest check.
    `ocr --resume` covers the other half by re-reading any page whose recognizer
    weights differ.
    """
    try:
        manifest = run_aws(
            [
                "aws",
                "s3",
                "cp",
                manifest_key(bucket, item, source),
                "-",
            ],
            capture=True,
        ).stdout
    except OSError:
        manifest = ""
    if not manifest.strip():
        print(
            f"{item.item}: run {source} has no manifest; not reusing its reads.",
            file=sys.stderr,
        )
        return False
    recorded = (json.loads(manifest).get("inputs") or {}).get("centerlines_sha")
    current = experiments.file_sha256(local / CENTERLINES_NAME)
    if recorded != current:
        print(
            f"{item.item}: run {source} read against {recorded}, this run has "
            f"{current}; not reusing its reads.",
            file=sys.stderr,
        )
        return False
    return True


def borrow_reads(local: Path, bucket: str, item: Item, source: str) -> None:
    """Bring an earlier run's OCR output forward, when its inputs still match."""
    if not reused_reads_are_valid(local, bucket, item, source):
        return
    filters = ["--exclude", "*"]
    for glob in OCR_REUSE_GLOBS:
        filters += ["--include", glob]
    sync(run_prefix(bucket, item, source), str(local), *filters)


def stage(command: list[str], local: Path) -> None:
    """Run one pipeline command, failing loudly.

    Deliberately does NOT set ``cwd`` to the item's directory. Several stages
    load their weights from a path relative to the repo -- keymap wants
    ``models/number_detector.pt`` -- so running from the scratch directory makes
    them fail on a missing model. Every command here is given absolute paths,
    so the working directory only ever needs to be the repo.
    """
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-6:]
        raise OSError(f"{' '.join(command[:2])} failed: {' | '.join(tail)}")


def raw_images(local: Path) -> list[str]:
    """Every full-resolution scan under raw/: key-map sheets and their panels."""
    return [str(path) for path in source_images(local / "raw")]


def keymap_sheets(local: Path) -> list[str]:
    """The key-map scans to run the keymap chain on: the keys loc-keymaps recorded.

    The record, not a glob: a split key-map sheet is recorded by its panel
    (``p0__1``), and the chain must run on that panel rather than on the whole
    sheet, exactly as `run-loc` does.
    """
    return [
        str(local / "raw" / f"{key}.jpg") for key in sorted(recorded_keymap_keys(local))
    ]


def run_chain(local: Path, work: FitWork, run_tag: str | None = None) -> None:
    """split, adjacency, keymap, ocr, fit -- the order `run-loc` uses.

    adjacency runs before keymap so its mutual edges can repair the key map's
    page-number assignments; both run before ocr so the vocabulary can be
    narrowed to each page's key-map neighbourhood.
    """
    pages = [str(local / name) for name in work.pages]
    stage(["mapsnap", "split", *pages], local)
    # Identify the key map HERE, after the split, rather than trusting the
    # keymaps.json the mirror carries. `loc-keymaps` runs before anything is
    # split, so it names the whole sheet -- p0a -- while a local run, splitting
    # first, names the panel that is actually the key map -- pa__2. The keymap
    # pipeline then ran on the whole sheet, which for Los Angeles 1949 vol 14
    # meant fitting one transform across the key map AND its p1499 inset.
    # `mapsnap split` already mirrors its cut onto the full-resolution copy, so
    # by this point raw/<parent>__N.jpg exists and can be named.
    #
    # Costs about 15 seconds a volume, near enough independent of page count
    # (only the low-numbered candidates are tested), against roughly 10,140
    # vCPU-hours for the corpus: about 1%.
    # Drop the mirror's copy first. keymap-detect writes a record either way,
    # but an item whose identification fails partway would otherwise fall back
    # to the stale whole-sheet answer, which is the bug being fixed.
    (local / KEYMAPS_NAME).unlink(missing_ok=True)
    stage(["mapsnap", "keymap-detect", str(local)], local)
    # The *effective* pages: a panel supersedes its parent, so a split sheet is
    # read panel by panel and the whole sheet is not read at all.
    effective = [str(path) for path in list_pages(local)]
    # split gives a panel its image and its P(road) crop but not its boxes:
    # those are derived by craft from the parent's (#361). Every parent here is
    # already crafted, so this pass detects nothing and never loads the model
    # (the reader is built lazily, inside detect); it only writes each
    # <parent>__N.boxes.json, raw key-map panels included.
    stage(["mapsnap", "craft", "--resume", *effective, *raw_images(local)], local)
    stage(["mapsnap", "adjacency", str(local)], local)
    sheets = keymap_sheets(local)
    if sheets:
        stage(["mapsnap", "keymap", *sheets], local)
    stage(
        [
            "mapsnap",
            "ocr",
            "--resume",
            "--centerlines",
            str(local / CENTERLINES_NAME),
            *effective,
        ],
        local,
    )
    # The previous run's archive comes down with the sync, and `fit` refuses to
    # overwrite one: re-fitting an item after a code change failed outright
    # until this removed it. Dropping it is right here -- its manifest is
    # re-written by the run about to happen, and the rest of it duplicates
    # sidecars this chain regenerates anyway.
    shutil.rmtree(local / "artifacts" / ARCHIVE_TAG, ignore_errors=True)
    # No --image-base-url: fit finds the item's metadata.json and builds the
    # canvases against LoC's own image servers, so the annotation is usable
    # without anything being hosted (#354).
    stage(
        [
            "mapsnap",
            "fit",
            str(local),
            "--tag",
            ARCHIVE_TAG,
            *(["--run-tag", run_tag] if run_tag else []),
        ],
        local,
    )


def upload(local: Path, bucket: str, item: Item, run_tag: str) -> None:
    """Sync this run's sidecars up, then the done marker, in that order.

    Two calls, not one: the marker is what `plan_fit` reads to decide an item is
    finished, so it must not exist until everything it vouches for does. In a
    single sync it sorts before `p*.streets.json` and landed first, and a spot
    interruption in between retired a half-uploaded item for good.
    """
    excludes: list[str] = []
    for pattern in UPLOAD_EXCLUDES:
        excludes += ["--exclude", pattern]
    for pattern in UPLOAD_INCLUDES:
        excludes += ["--include", pattern]
    destination = run_prefix(bucket, item, run_tag)
    run_aws(
        [
            "aws",
            "s3",
            "sync",
            str(local),
            destination,
            "--exclude",
            CENTERLINES_NAME,
            "--exclude",
            DONE_MARKER,
            *excludes,
            "--only-show-errors",
        ]
    )
    marker = local / DONE_MARKER
    if marker.exists():
        run_aws(
            [
                "aws",
                "s3",
                "cp",
                str(marker),
                f"{destination}/{DONE_MARKER}",
                "--only-show-errors",
            ]
        )


def process_item(work: FitWork, local: Path, bucket: str) -> int:
    """Run the chain over a downloaded item and sync its sidecars up."""
    try:
        run_chain(local, work, work.run_tag)
        upload(local, bucket, work.item, work.run_tag)
        return len(work.pages)
    finally:
        shutil.rmtree(local, ignore_errors=True)


@dataclass
class Prepared:
    """The next item to run, plus what the scan passed over finding it."""

    work: FitWork | None
    local: Path | None
    index: int
    skipped: int = 0
    waiting: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def prepare_next(
    items: Iterator[tuple[int, Item]],
    bucket: str,
    work_dir: Path,
    *,
    counties: dict[str, County],
    tag_for: Callable[[Item], str],
    fetch: bool = True,
    ocr_from: str | None = None,
    retire: Callable[[Item], None] | None = None,
    release: Callable[[Item], None] | None = None,
) -> Prepared:
    """List forward until an item can run, download it, and return it.

    ``retire`` settles an item that is finished; ``release`` puts back one that
    is merely not ready yet, so the GPU pass can catch up and a later worker can
    take it. Getting those two the wrong way round would either lose items or
    spin on them.
    """
    skipped = waiting = 0
    failures: list[tuple[str, str]] = []
    for index, item in items:
        try:
            present = list_prefix(bucket, item.prefix)
            work = plan_fit(item, present, counties.get(item.item), tag_for(item))
            if work.done:
                skipped += 1
                if retire is not None:
                    retire(item)
                continue
            if not work.ready:
                waiting += 1
                print(
                    f"{item.item}: not ready ({work.reason})",
                    file=sys.stderr,
                    flush=True,
                )
                if release is not None:
                    release(item)
                continue
            local = (
                fetch_item(work, bucket, work_dir, ocr_from)
                if fetch
                else work_dir / item.item
            )
        except OSError as error:
            failures.append((item.item, str(error)))
            continue
        return Prepared(work, local, index, skipped, waiting, failures)
    return Prepared(None, None, 0, skipped, waiting, failures)


def build_parser() -> argparse.ArgumentParser:
    """The command line, separately so the fleet's flags can be checked in-process."""
    parser = argparse.ArgumentParser(
        description="Run split, adjacency, keymap, ocr and fit over the mirror."
    )
    parser.add_argument("--bucket", default="s3://mapsnap-sanborn")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--queue", help="SQS queue URL to take items from, instead of --shard/--shards."
    )
    parser.add_argument("--manifest")
    parser.add_argument(
        "--counties",
        nargs="+",
        required=True,
        help="items.tsv and city-items.tsv, local paths or s3:// URLs: "
        "the item -> county FIPS mapping.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Accepted so the fleet bootstrap can pass it; the chain is CPU work.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/loc-fit"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--run-tag",
        metavar="TAG",
        help=(
            "Name for this corpus pass -- a cut release, say. Outputs go to "
            "<item>/runs/TAG/ and the tag is recorded in every manifest and "
            "published annotation page. On a queue whose messages carry a tag "
            "this is an ASSERTION against them, not an override: a mismatch "
            "means this worker was pointed at the wrong queue and it stops. "
            "Use one queue per run -- a worker cannot decline a message it has "
            "received, and passing one over counts toward maxReceiveCount "
            f"(default {work_queue.DEFAULT_MAX_RECEIVES}) until it dead-letters."
        ),
    )
    parser.add_argument(
        "--ocr-from",
        metavar="TAG",
        help=(
            "Reuse an earlier run's reads instead of re-running OCR, which is "
            "two thirds of both the output bytes and the CPU. Refused when that "
            "run read against a different county extract; `ocr --resume` then "
            "re-reads any page whose recognizer weights differ, and any page "
            "the earlier run did not have."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-args",
        action="store_true",
        help=(
            "Parse the arguments and exit 0. A worker's flags are otherwise "
            "first checked on the instance, after boot: a missing required one "
            "then costs a whole fleet and leaves the queue untouched, which is "
            "how the first test-200 launch died. Let a launcher check here."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check_args:
        # Nothing is fetched and no queue is touched: reaching here is the
        # whole answer, because argparse has already rejected what is invalid.
        print("loc-fit arguments OK")
        return

    manifest = resolve_manifest(args.manifest, args.bucket, args.work_dir)
    all_items = read_manifest(manifest)
    counties = read_counties(resolve_counties(args.counties, args.work_dir))
    print(f"{len(counties):,} items mapped to a county extract", file=sys.stderr)

    source: QueueSource | None = None
    if args.queue:
        source = QueueSource(args.queue, {item.item: item for item in all_items})
        total = work_queue.depth(args.queue).total
        label = "queue"
        pending: Iterator[tuple[int, Item]] = iter(source)
    else:
        items = select_shard(all_items, args.shard, args.shards)
        total = len(items)
        label = f"s{args.shard}"
        pending = iter(list(enumerate(items, start=1)))
    print(f"{label}: {total:,} items", file=sys.stderr, flush=True)
    retire = source.retire if source is not None else None
    release = source.release if source is not None else None

    def tag_for(item: Item) -> str:
        message_tag = source.tag_for(item) if source is not None else None
        return resolve_run_tag(message_tag, args.run_tag)

    started = time.perf_counter()
    done = skipped = waiting = failed = pages = 0
    broken = args.work_dir / f"broken-{args.shard}.log"

    def record(prepared: Prepared) -> None:
        nonlocal skipped, waiting, failed
        skipped += prepared.skipped
        waiting += prepared.waiting
        for name, error in prepared.failures:
            print(f"{name}: {error}", file=sys.stderr, flush=True)
            with broken.open("a") as handle:
                handle.write(f"{name}\t{error}\n")
            failed += 1

    with ThreadPoolExecutor(1) as fetcher:
        future = fetcher.submit(
            prepare_next,
            pending,
            args.bucket,
            args.work_dir,
            counties=counties,
            tag_for=tag_for,
            fetch=not args.dry_run,
            ocr_from=args.ocr_from,
            retire=retire,
            release=release,
        )
        while True:
            prepared = future.result()
            record(prepared)
            if prepared.work is None or (args.limit and done >= args.limit):
                if prepared.local is not None and prepared.work is not None:
                    shutil.rmtree(prepared.local, ignore_errors=True)
                break
            work, local, index = prepared.work, prepared.local, prepared.index
            assert local is not None
            future = fetcher.submit(
                prepare_next,
                pending,
                args.bucket,
                args.work_dir,
                counties=counties,
                tag_for=tag_for,
                fetch=not args.dry_run,
                ocr_from=args.ocr_from,
                retire=retire,
                release=release,
            )
            if args.dry_run:
                print(
                    f"{work.item.item}: {len(work.pages)} pages, county {work.county}"
                )
                done += 1
                continue
            try:
                with work_queue.lease(
                    args.queue, source.handle_for(work.item) if source else None
                ):
                    pages += process_item(work, local, args.bucket)
            except OSError as error:
                print(f"{work.item.item}: FAILED: {error}", file=sys.stderr, flush=True)
                with broken.open("a") as handle:
                    handle.write(f"{work.item.item}\t{error}\n")
                failed += 1
                if source is not None:
                    source.release(work.item)
                continue
            if retire is not None:
                retire(work.item)
            done += 1
            if source is not None and done % QUEUE_DEPTH_EVERY == 0:
                total = work_queue.depth(args.queue).total
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            left = total - index if source is None else total
            print(
                f"{datetime.now(UTC):%H:%M:%S} {label} [{index}/{total}] "
                f"{work.item.item}: {len(work.pages)} pages"
                f" | {rate * 3600:.0f} items/h, "
                f"eta {format_duration(left / rate if rate else 0.0)}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(
        f"{label}: {done} items fitted, {skipped} already done, {waiting} awaiting "
        f"craft, {failed} failed; {pages} pages in {format_duration(elapsed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
