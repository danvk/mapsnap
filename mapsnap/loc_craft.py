"""Run the corpus's two image-only passes over the Sanborn mirror (#354).

``mapsnap craft`` and ``mapsnap roadprob`` are the only stages whose output
depends on nothing but the image and a model checksum: CRAFT's boxes and the
road UNet's P(road) map. Everything downstream (split, ocr, keymap, fit) varies
from run to run, so those two are computed once, on GPU instances, and their
sidecars live beside the images in the mirror for good.

One process handles one **shard** of the mirror: the items whose id hashes to
its number. Shards are static, so a worker needs no coordinator and no queue --
it can die and be replaced by an identical one, which is what makes spot
instances usable. Per item it lists the item's prefix, skips the item outright
when every expected sidecar is already there, and otherwise syncs the item down,
runs both passes in process (the models are loaded once per worker, not once per
item), syncs the new sidecars back up and deletes the local copy.

Sidecars land beside their image, so an item that started as::

    by-state/alabama/1924/sanborn00001_003/{metadata.json,p1.jpg,raw/p1.jpg}

gains ``p1.boxes.json``, ``p1.roadprob.jpg`` and ``raw/p1.boxes.json``. Raw
key-map sheets get CRAFT only: their P(road) comes from the colour model in
``mapsnap.keymap.road_prob``, which needs a georeference this pass does not have.

Each item logs a timestamped, shard-tagged line, and the closing summary
reports pages per hour -- the figure to compare when two configurations process
items of different sizes.

    mapsnap loc-craft --bucket s3://mapsnap-sanborn --shard 0 --shards 8
"""

import argparse
import hashlib
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Columns of the mirror's manifest (loc-sanborn-maps.mapping.tsv at the bucket root).
MANIFEST_NAME = "loc-sanborn-maps.mapping.tsv"
ITEM_COLUMNS = ("item", "state", "year")
# What this pass writes for one page, and for one raw key-map sheet.
# Fixed so a resumed shard, and a --limit sample, are reproducible.
SHUFFLE_SEED = 0
PAGE_OUTPUTS = ("boxes.json", "roadprob.jpg")
RAW_OUTPUTS = ("boxes.json",)


@dataclass(frozen=True)
class Item:
    """One LoC item (a volume) and where it lives in the mirror."""

    item: str
    state: str
    year: str

    @property
    def prefix(self) -> str:
        """The item's key prefix in the bucket, without a trailing slash."""
        return f"by-state/{self.state}/{self.year}/{self.item}"


@dataclass
class ItemWork:
    """What one item still needs: its images, and whether anything is missing."""

    item: Item
    pages: list[str]
    raw_sheets: list[str]
    missing: list[str]

    @property
    def complete(self) -> bool:
        return not self.missing


def read_manifest(path: Path) -> list[Item]:
    """Every item in the mirror's manifest, in file order, one row per item.

    The manifest is per *sheet*, so the first row of each item wins; sheets add
    nothing here because the item's own prefix listing is what enumerates its
    images (and is authoritative about what actually mirrored).
    """
    items: dict[str, Item] = {}
    with path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        missing = [name for name in ITEM_COLUMNS if name not in header]
        if missing:
            sys.exit(f"{path}: manifest has no {missing[0]!r} column")
        index = {name: header.index(name) for name in ITEM_COLUMNS}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < len(header):
                continue
            item = fields[index["item"]]
            if item not in items:
                items[item] = Item(
                    item=item,
                    state=fields[index["state"]],
                    year=fields[index["year"]],
                )
    return list(items.values())


def shard_of(item: str, shards: int) -> int:
    """Which shard owns ``item``.

    Hashed with sha1 rather than ``hash()``, whose salt changes per process:
    a restarted worker has to claim exactly the same items as the one it
    replaces, or interrupted work is never picked up.
    """
    digest = hashlib.sha1(item.encode()).digest()
    return int.from_bytes(digest[:8], "big") % max(1, shards)


def select_shard(
    items: list[Item], shard: int, shards: int, seed: int = SHUFFLE_SEED
) -> list[Item]:
    """The items this worker owns, in a deterministic shuffled order.

    Manifest order is item-id order, and id correlates with era and format: the
    corpus's first item is an 1867 Boston atlas of unsplit two-page spreads at an
    unusual aspect ratio, which tiles into four and takes twelve minutes. A
    ``--limit`` sample has to see a representative mix rather than the oldest
    volumes, so the shard is shuffled -- with a fixed seed, so every worker and
    every restart walks the same order.
    """
    chosen = [item for item in items if shard_of(item.item, shards) == shard]
    random.Random(seed).shuffle(chosen)
    return chosen


# A transient S3 failure must not cost an item: the first call a freshly booted
# instance makes can beat its instance-profile credentials out of the metadata
# service (the first pilot lost one item that way, non-zero exit and empty
# stderr, seconds into the run), and a multi-day pass also meets throttling.
AWS_ATTEMPTS = 4
AWS_BACKOFF_SECONDS = 3.0


def run_aws(
    command: list[str], *, capture: bool = False
) -> subprocess.CompletedProcess:
    """Run an aws CLI command, retrying transient failures with a growing delay.

    Raises OSError with whatever the CLI said (and its exit status, since a
    credential race reports nothing at all) once the attempts are spent.
    """
    last = ""
    status = 0
    for attempt in range(AWS_ATTEMPTS):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result
        status = result.returncode
        last = (result.stderr or result.stdout or "").strip()
        if attempt + 1 < AWS_ATTEMPTS:
            delay = AWS_BACKOFF_SECONDS * 2**attempt
            print(
                f"  aws {command[1]} {command[2]} failed (exit {status}), "
                f"retrying in {delay:.0f}s: {last[:120]}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise OSError(
        f"{' '.join(command[:4])} failed after {AWS_ATTEMPTS} attempts "
        f"(exit {status}): {last or 'no output'}"
    )


def bucket_name(bucket: str) -> str:
    """The bucket itself, from an ``s3://bucket[/path]`` URL."""
    return bucket.rstrip("/").removeprefix("s3://").partition("/")[0]


def key_prefix(bucket: str, prefix: str) -> str:
    """The full S3 key prefix of an item, honouring any path in the bucket URL.

    ``aws s3 ls --recursive`` prints keys relative to the *bucket*, not to the URL
    it was given, so a ``--bucket`` with a path (``s3://mapsnap-sanborn/_craft/x``)
    yields keys that do not start with the item prefix. Matching on the item prefix
    alone found nothing there and reported every item complete -- silently doing no
    work, which is the one failure a corpus pass must not have.
    """
    root = bucket.rstrip("/").removeprefix("s3://").partition("/")[2]
    return f"{root}/{prefix}" if root else prefix


def list_prefix(bucket: str, prefix: str) -> list[str]:
    """Keys under an item's prefix, relative to it.

    Uses ``s3api list-objects-v2`` rather than ``s3 ls``, which cannot tell an
    empty prefix from a failure: both exit 1 with nothing on either stream. An
    item the mirror never produced (45 of the manifest's 35,159) therefore looked
    like a transient error, spent four retries on it and was counted a failure.
    s3api answers the empty case with exit 0 and "None", and a real failure with
    a non-zero status and a message, so the two stop being the same event.
    """
    full = key_prefix(bucket, prefix)
    result = run_aws(
        [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket_name(bucket),
            "--prefix",
            f"{full}/",
            "--query",
            "Contents[].Key",
            "--output",
            "text",
        ],
        capture=True,
    )
    text = result.stdout.strip()
    if not text or text == "None":
        return []
    return [key[len(full) + 1 :] for key in text.split() if key.startswith(f"{full}/")]


def plan_item(item: Item, present: list[str]) -> ItemWork:
    """Split an item's existing keys into images to process and outputs still missing.

    Pages are the item's top-level ``p*.jpg``; raw key-map sheets are ``raw/p*.jpg``.
    A page whose sidecars are all present contributes nothing, so an item that ran
    before is skipped without downloading a byte.
    """
    keys = set(present)
    # A sidecar is <stem>.<something>.jpg, so a source image is the one-dot name.
    images = sorted(key for key in keys if key.endswith(".jpg") and key.count(".") == 1)
    pages = [key for key in images if "/" not in key]
    raw_sheets = [key for key in images if key.startswith("raw/")]
    missing = []
    for image, outputs in ((pages, PAGE_OUTPUTS), (raw_sheets, RAW_OUTPUTS)):
        for key in image:
            stem = key[: -len(".jpg")]
            missing += [
                f"{stem}.{suffix}"
                for suffix in outputs
                if f"{stem}.{suffix}" not in keys
            ]
    return ItemWork(item=item, pages=pages, raw_sheets=raw_sheets, missing=missing)


def sync(source: str, destination: str) -> None:
    """``aws s3 sync`` one direction, quietly, failing loudly.

    Downward it brings the images plus any sidecars an interrupted run already
    wrote, so this one resumes inside the item rather than redoing it; upward it
    copies only what is new, because the CLI gives a downloaded file its object's
    last-modified time and so does not consider it changed.
    """
    run_aws(["aws", "s3", "sync", source, destination, "--only-show-errors"])


class Worker:
    """Holds the two models for the life of the process and runs items through them."""

    def __init__(self, gpu: bool = False) -> None:
        import torch

        from mapsnap.detect_text import build_reader
        from mapsnap.keymap.number_model import select_device
        from mapsnap.road_model import ROAD_MODEL_PATH, load_model

        self.device = select_device() if gpu else torch.device("cpu")
        print(f"device: {self.device}", file=sys.stderr, flush=True)
        self.reader = build_reader(gpu)
        self.road_model = load_model(ROAD_MODEL_PATH, self.device)

    def craft(self, images: list[str]) -> int:
        """Write ``<stem>.boxes.json`` for each image that has none; return the count."""
        from mapsnap.craft import pending_images
        from mapsnap.detect_text import write_craft_boxes

        todo = pending_images(images, resume=True)
        for image in todo:
            write_craft_boxes(image, self.reader)
        return len(todo)

    def roadprob(self, images: list[str]) -> int:
        """Write ``<stem>.roadprob.jpg`` for each image that has none; return the count."""
        import cv2

        from mapsnap.road_model import predict_page
        from mapsnap.roadprob import pending_images, roadprob_path, save_roadprob

        todo = pending_images(images, resume=True)
        written = 0
        for image in todo:
            gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  unreadable: {image}", file=sys.stderr)
                continue
            save_roadprob(
                roadprob_path(image), predict_page(self.road_model, gray, self.device)
            )
            written += 1
        return written


def item_prefix(bucket: str, item: Item) -> str:
    """The item's directory URL in the bucket."""
    return f"{bucket.rstrip('/')}/{item.prefix}"


def fetch_item(work: ItemWork, bucket: str, work_dir: Path) -> Path:
    """Sync one item's images into its own directory under ``work_dir``."""
    local = work_dir / work.item.item
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    sync(item_prefix(bucket, work.item), str(local))
    return local


def process_item(
    work: ItemWork, local: Path, bucket: str, worker: Worker
) -> tuple[int, int]:
    """Run both passes over an already-downloaded item and sync the sidecars up."""
    try:
        pages = [str(local / name) for name in work.pages]
        raw_sheets = [str(local / name) for name in work.raw_sheets]
        detected = worker.craft(pages + raw_sheets)
        predicted = worker.roadprob(pages)
        sync(str(local), item_prefix(bucket, work.item))
        return detected, predicted
    finally:
        shutil.rmtree(local, ignore_errors=True)


@dataclass
class Prepared:
    """The next item to compute, plus everything passed over while finding it.

    ``work`` is None when the shard is exhausted; ``skipped`` (already finished),
    ``absent`` (in the manifest but never mirrored) and ``failures`` still
    describe what the scan saw on the way, so the caller counts them once.
    """

    work: ItemWork | None
    local: Path | None
    index: int
    skipped: int = 0
    absent: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def prepare_next(
    items: Iterator[tuple[int, Item]],
    bucket: str,
    work_dir: Path,
    *,
    fetch: bool = True,
) -> Prepared:
    """List forward until an item needs work, download it, and return it.

    Runs on a background thread so one item's S3 round trip overlaps the
    previous one's inference: about a third of a single worker's time was the
    download, with the GPU idle. Items already complete, and items whose listing
    or download fails, are reported in the result rather than raised, so the
    caller keeps the counting and the logging in one place.
    """
    skipped = absent = 0
    failures: list[tuple[str, str]] = []
    for index, item in items:
        try:
            present = list_prefix(bucket, item.prefix)
            if not present:
                # In the manifest but not in the mirror: 45 of 35,159 items.
                absent += 1
                continue
            work = plan_item(item, present)
            if work.complete:
                skipped += 1
                continue
            local = (
                fetch_item(work, bucket, work_dir) if fetch else work_dir / item.item
            )
        except OSError as error:
            failures.append((item.item, str(error)))
            continue
        return Prepared(work, local, index, skipped, absent, failures)
    return Prepared(None, None, 0, skipped, absent, failures)


def format_duration(seconds: float) -> str:
    """``h:mm`` for a duration, for progress lines in a multi-day log."""
    minutes = int(seconds // 60)
    return f"{minutes // 60}:{minutes % 60:02d}"


def resolve_manifest(manifest: str | None, bucket: str, work_dir: Path) -> Path:
    """The manifest path, downloading the bucket's copy when none is given."""
    if manifest and not manifest.startswith("s3://"):
        return Path(manifest)
    source = manifest or f"{bucket.rstrip('/')}/{MANIFEST_NAME}"
    local = work_dir / MANIFEST_NAME
    if not local.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        run_aws(["aws", "s3", "cp", source, str(local), "--only-show-errors"])
    return local


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache CRAFT boxes and P(road) maps for one shard of the mirror."
    )
    parser.add_argument(
        "--bucket",
        default="s3://mapsnap-sanborn",
        help="Mirror bucket (default: %(default)s).",
    )
    parser.add_argument(
        "--shard", type=int, default=0, help="This worker's shard number."
    )
    parser.add_argument("--shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument(
        "--manifest",
        help=f"Sheet manifest (default: the bucket's {MANIFEST_NAME}).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/loc-craft"),
        help="Scratch directory for one item at a time (default: %(default)s).",
    )
    parser.add_argument(
        "--limit", type=int, help="Stop after this many items (a pilot)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SHUFFLE_SEED,
        help="Seed for the shard's item order (default: %(default)s).",
    )
    parser.add_argument("--gpu", action="store_true", help="Run the models on the GPU.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what each item needs without downloading or computing.",
    )
    args = parser.parse_args()

    if not 0 <= args.shard < args.shards:
        sys.exit(f"--shard must be in [0, {args.shards})")

    manifest = resolve_manifest(args.manifest, args.bucket, args.work_dir)
    items = select_shard(read_manifest(manifest), args.shard, args.shards, args.seed)
    print(
        f"shard {args.shard}/{args.shards}: {len(items)} items",
        file=sys.stderr,
        flush=True,
    )

    worker = None
    started = time.perf_counter()
    done = skipped = absent = failed = pages_detected = pages_predicted = 0
    broken = args.work_dir / f"broken-{args.shard}.log"

    def record(prepared: Prepared) -> None:
        """Count what the scan passed over on its way to this item."""
        nonlocal skipped, absent, failed
        skipped += prepared.skipped
        absent += prepared.absent
        for name, error in prepared.failures:
            print(f"{name}: listing/fetch failed: {error}", file=sys.stderr, flush=True)
            with broken.open("a") as handle:
                handle.write(f"{name}\t{error}\n")
            failed += 1

    # One thread runs a step ahead, so the next item is on local disk by the
    # time this one finishes computing.
    pending = iter(list(enumerate(items, start=1)))
    with ThreadPoolExecutor(1) as fetcher:
        future = fetcher.submit(
            prepare_next, pending, args.bucket, args.work_dir, fetch=not args.dry_run
        )
        while True:
            prepared = future.result()
            record(prepared)
            if prepared.work is None or (args.limit and done >= args.limit):
                # --limit stops one item after the prefetcher ran ahead; drop
                # what it downloaded rather than leaving it in the scratch dir.
                if prepared.local is not None and prepared.work is not None:
                    shutil.rmtree(prepared.local, ignore_errors=True)
                break
            work, local, index = prepared.work, prepared.local, prepared.index
            assert local is not None
            # Start the next download before computing this one.
            future = fetcher.submit(
                prepare_next,
                pending,
                args.bucket,
                args.work_dir,
                fetch=not args.dry_run,
            )
            if args.dry_run:
                print(
                    f"{work.item.item}: {len(work.pages)} pages, "
                    f"{len(work.raw_sheets)} raw, {len(work.missing)} sidecars missing"
                )
                done += 1
                continue
            if worker is None:
                worker = Worker(gpu=args.gpu)
            try:
                detected, predicted = process_item(work, local, args.bucket, worker)
            except OSError as error:
                print(f"{work.item.item}: FAILED: {error}", file=sys.stderr, flush=True)
                with broken.open("a") as handle:
                    handle.write(f"{work.item.item}\t{error}\n")
                failed += 1
                continue
            done += 1
            pages_detected += detected
            pages_predicted += predicted
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            remaining = (len(items) - index) / rate if rate else 0.0
            print(
                f"{datetime.now(UTC):%H:%M:%S} s{args.shard} [{index}/{len(items)}] "
                f"{work.item.item}: {detected} craft, {predicted} P(road)"
                f" | {rate * 3600:.0f} items/h, eta {format_duration(remaining)}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(
        f"shard {args.shard}: {done} items processed, {skipped} already complete, "
        f"{absent} not in the mirror, {failed} failed; {pages_detected} pages crafted, {pages_predicted} P(road) maps"
        f" in {format_duration(elapsed)} ({elapsed:.0f}s, "
        f"{pages_detected / elapsed * 3600 if elapsed else 0:.0f} pages/h)",
        file=sys.stderr,
        flush=True,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
