"""Identify every volume's key map(s) across the Sanborn mirror (#354).

``mapsnap keymap-detect`` decides which pages of one volume are key maps, using
only the 25%-scale images the mirror already holds. This runs it over a shard of
the corpus and writes each volume's ``keymaps.json`` back beside its pages, so
the later ``mapsnap keymap`` pass knows what to work on without re-deciding.

It is far cheaper than it sounds, because most items never load a model:

  * an item with fewer pages than the coverage floor (20,753 of 35,158) cannot
    have a detectable key map at all, and is recorded as having none;
  * an item with an unsplit page-0 sheet (3,220) is a key map by convention -- a
    census found page 0 is always the key map -- so its record is written from
    the file names alone, with nothing downloaded;
  * only the remainder (about 11,185) download their one to four candidate pages
    and run the CNN localizer and CRNN reader over them.

Which makes this the cheapest way to answer the question that matters before any
full-resolution fetching: the mirror kept raw copies of the page-0 family and
lettered sheets only, so a volume whose key map is in the page-1 family has no
raw sheet yet, and this pass names them.

The candidate stage reads file names rather than pixels, so a shard's items are
materialized as empty placeholder files and only the pages actually tested are
downloaded.

    mapsnap loc-keymaps --bucket s3://mapsnap-sanborn --shard 0 --shards 32
"""

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mapsnap.loc_craft import (
    Item,
    format_duration,
    item_prefix,
    list_prefix,
    read_manifest,
    resolve_manifest,
    run_aws,
    select_shard,
)

KEYMAPS_NAME = "keymaps.json"
# Heartbeat interval, in items recorded.
PROGRESS_EVERY = 25


@dataclass
class KeymapWork:
    """One item's plan: the pages it has, and which of them need the model."""

    item: Item
    page_keys: list[str]
    assumed: list[str]
    to_test: list[str]

    @property
    def needs_model(self) -> bool:
        return bool(self.to_test)


def page_keys_of(present: list[str]) -> list[str]:
    """Stems of the item's top-level page images, e.g. ["p1", "p2"].

    A sidecar is ``<stem>.<something>.jpg``, so a page is the one-dot name; the
    corpus pass writes ``p1.roadprob.jpg`` beside these and must not be counted.
    """
    return sorted(
        key[: -len(".jpg")]
        for key in present
        if key.endswith(".jpg") and key.count(".") == 1 and "/" not in key
    )


def placeholder_volume(work_dir: Path, item: Item, page_keys: list[str]) -> Path:
    """An empty stand-in for the volume: one zero-byte file per page.

    Candidate generation (``detection_plan``) reads file names only, so the shard
    can plan every item without downloading anything; the pages that actually get
    read are fetched afterwards, over the top of their placeholder.
    """
    volume = work_dir / item.item
    shutil.rmtree(volume, ignore_errors=True)
    volume.mkdir(parents=True)
    for key in page_keys:
        (volume / f"{key}.jpg").touch()
    return volume


def plan_keymaps(item: Item, present: list[str], volume: Path) -> KeymapWork:
    """Split an item's pages into key maps by convention and candidates to confirm."""
    from mapsnap.keymap.identify import detection_plan

    page_keys = page_keys_of(present)
    assumed, to_test = detection_plan(volume)
    return KeymapWork(item, page_keys, assumed, to_test)


def fetch_pages(item: Item, bucket: str, volume: Path, keys: list[str]) -> None:
    """Download just the pages about to be read, over their placeholders."""
    prefix = item_prefix(bucket, item)
    for key in keys:
        run_aws(
            [
                "aws",
                "s3",
                "cp",
                f"{prefix}/{key}.jpg",
                str(volume / f"{key}.jpg"),
                "--only-show-errors",
            ]
        )


def confirm_keymaps(
    volume: Path,
    to_test: list[str],
    models: tuple,
    *,
    min_coverage: float,
    min_distinct: int,
) -> list[str]:
    """Which candidates read back enough of the volume's own page set to be key maps."""
    from mapsnap.keymap.identify import is_keymap, read_valid_pages, volume_valid_pages
    from mapsnap.keymap.records import page_key_sort

    cnn, crnn, device = models
    valid_pages = volume_valid_pages(volume)
    confirmed = []
    for key in to_test:
        image = volume / f"{key}.jpg"
        if not image.exists() or not image.stat().st_size:
            continue
        found = read_valid_pages(str(image), valid_pages, cnn, crnn, device)
        if is_keymap(
            len(found),
            len(valid_pages),
            min_coverage=min_coverage,
            min_distinct=min_distinct,
        ):
            confirmed.append(key)
    return sorted(confirmed, key=page_key_sort)


def upload_record(item: Item, bucket: str, volume: Path) -> None:
    """Copy the volume's keymaps.json up to its prefix."""
    run_aws(
        [
            "aws",
            "s3",
            "cp",
            str(volume / KEYMAPS_NAME),
            f"{item_prefix(bucket, item)}/{KEYMAPS_NAME}",
            "--only-show-errors",
        ]
    )


@dataclass
class Tally:
    """What a shard did, for the closing summary."""

    absent: int = 0
    already: int = 0
    too_small: int = 0
    by_convention: int = 0
    tested: int = 0
    found: int = 0
    unmirrored: int = 0
    failed: int = 0


def main() -> None:
    from mapsnap.keymap.identify import (
        DEFAULT_CNN_WEIGHTS,
        DEFAULT_CRNN_WEIGHTS,
        MIN_COVERAGE,
        MIN_DISTINCT,
    )

    parser = argparse.ArgumentParser(
        description="Identify each volume's key map(s) over one shard of the mirror."
    )
    parser.add_argument("--bucket", default="s3://mapsnap-sanborn")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--manifest")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/loc-keymaps"))
    parser.add_argument(
        "--limit", type=int, help="Stop after this many items (a pilot)."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    parser.add_argument("--min-distinct", type=int, default=MIN_DISTINCT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo items that already have a keymaps.json (after new weights, say).",
    )
    parser.add_argument("--gpu", action="store_true", help="Run the models on the GPU.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan without downloading or writing."
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

    models = None
    tally = Tally()
    started = time.perf_counter()
    done = 0
    for index, item in enumerate(items, start=1):
        if args.limit and done >= args.limit:
            break
        try:
            present = list_prefix(args.bucket, item.prefix)
        except OSError as error:
            print(f"{item.item}: listing failed: {error}", file=sys.stderr, flush=True)
            tally.failed += 1
            continue
        if not present:
            tally.absent += 1
            continue
        if KEYMAPS_NAME in present and not args.force:
            tally.already += 1
            continue

        page_keys = page_keys_of(present)
        volume = placeholder_volume(args.work_dir, item, page_keys)
        try:
            keys: list[str] = []
            if len(page_keys) < args.min_distinct:
                # Too few pages for any candidate to clear the coverage floor
                # (#405): recorded as having no key map rather than re-examined.
                tally.too_small += 1
            else:
                work = plan_keymaps(item, present, volume)
                if work.assumed:
                    keys = work.assumed
                    tally.by_convention += 1
                elif work.to_test:
                    tally.tested += 1
                    if not args.dry_run:
                        if models is None:
                            import torch

                            from mapsnap.keymap.identify import load_models
                            from mapsnap.keymap.number_model import select_device

                            device = (
                                select_device() if args.gpu else torch.device("cpu")
                            )
                            models = load_models(
                                DEFAULT_CNN_WEIGHTS, DEFAULT_CRNN_WEIGHTS, device
                            )
                            print(f"models on {device}", file=sys.stderr, flush=True)
                        fetch_pages(item, args.bucket, volume, work.to_test)
                        keys = confirm_keymaps(
                            volume,
                            work.to_test,
                            models,
                            min_coverage=args.min_coverage,
                            min_distinct=args.min_distinct,
                        )
            if keys:
                tally.found += 1
                # A key map outside the page-0 family and the lettered sheets has
                # no raw copy in the mirror, so it still needs fetching.
                from mapsnap.loc_mirror import is_candidate

                if not all(is_candidate(key.split("__")[0]) for key in keys):
                    tally.unmirrored += 1
            if not args.dry_run:
                from mapsnap.keymap.records import write_keymaps_record

                write_keymaps_record(volume, keys)
                upload_record(item, args.bucket, volume)
            done += 1
            # Every key map, and a heartbeat besides: a shard is ~1,100 items and
            # most have none, so keying the log on finds alone shows almost nothing.
            if keys or done % PROGRESS_EVERY == 0:
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(
                    f"{datetime.now(UTC):%H:%M:%S} s{args.shard} [{index}/{len(items)}] "
                    f"{item.item}: {' '.join(keys) if keys else 'no key map'}"
                    f" | {rate * 3600:.0f} items/h",
                    flush=True,
                )
        except OSError as error:
            print(f"{item.item}: FAILED: {error}", file=sys.stderr, flush=True)
            tally.failed += 1
        finally:
            shutil.rmtree(volume, ignore_errors=True)

    elapsed = time.perf_counter() - started
    print(
        f"shard {args.shard}: {done} items recorded ({tally.found} with a key map, "
        f"{tally.unmirrored} of them NOT in the mirror's raw set); "
        f"{tally.by_convention} by convention, {tally.tested} model-tested, "
        f"{tally.too_small} too small; {tally.already} already done, "
        f"{tally.absent} not in the mirror, {tally.failed} failed "
        f"in {format_duration(elapsed)} ({elapsed:.0f}s)",
        file=sys.stderr,
        flush=True,
    )
    if tally.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
