"""Fetch the full-resolution sheets the mirror skipped, for key maps (#354).

``mapsnap loc-mirror`` kept a raw full-resolution JPEG only for the page-0 family
and lettered sheets, on the reasoning that page 0 is the key map wherever one
exists. Most volumes have no page 0 at all: their key map is page 1, and
``mapsnap loc-keymaps`` names those. This fetches them.

The key-map pipeline needs the raw sheet rather than the 25% page, because the
sheet is about four times the linear resolution and that is what makes a key
map's small printed numbers legible to the tiled detector.

Run it on an instance, not a laptop. The original mirror took two days because a
home uplink caps around 2.5 MB/s and it had 367 GB to push; in-region the upload
is free and effectively instant, which leaves the source mirror's own rate as the
only limit.

Two modes:

  * ``--build-list FILE`` sweeps the bucket's ``keymaps.json`` records and writes
    the (item, key) pairs that have no raw copy yet -- the work list;
  * the default takes that list, and for its shard downloads each sheet's JP2
    from the mirror, decodes it at full resolution, and uploads the JPEG to the
    item's ``raw/`` prefix.

Keep the shard count low. The mirror is somebody's machine on a home connection,
measured at about 20 MB/s in total, and at roughly 7 MB a sheet that ceiling is
already near 10,000 sheets an hour -- more shards would only crowd each other and
the person hosting it.

    mapsnap loc-raw --build-list keymaps.tsv
    mapsnap loc-raw --list keymaps.tsv --mirror http://host:port --shard 0 --shards 4
"""

import argparse
import csv
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
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
    shard_of,
)

LIST_COLUMNS = ("item", "key")


@dataclass(frozen=True)
class Wanted:
    """One sheet to fetch: which item, and which page key of it."""

    item: str
    key: str


def read_list(path: Path) -> list[Wanted]:
    """The work list: one (item, key) row per sheet to fetch."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [
            name for name in LIST_COLUMNS if name not in (reader.fieldnames or [])
        ]
        if missing:
            sys.exit(f"{path}: list has no {missing[0]!r} column")
        return [Wanted(row["item"], row["key"]) for row in reader if row["item"]]


def write_list(path: Path, wanted: list[Wanted]) -> None:
    """Write the work list as a TSV, sorted so reruns produce the same file."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(LIST_COLUMNS)
        for want in sorted(wanted, key=lambda w: (w.item, w.key)):
            writer.writerow([want.item, want.key])


def missing_raw(present: list[str], keys: list[str]) -> list[str]:
    """Which of an item's key-map keys still have no ``raw/<key>.jpg``.

    A split panel (``p1__2``) is cut locally from its parent sheet, so the parent
    is what gets fetched.
    """
    have = {key for key in present if key.startswith("raw/")}
    parents = {key.split("__")[0] for key in keys}
    return sorted(key for key in parents if f"raw/{key}.jpg" not in have)


def keymap_keys(present: list[str], body: str) -> list[str]:
    """Key-map page keys from an item's ``keymaps.json`` body, [] when unreadable."""
    import json

    if "keymaps.json" not in present:
        return []
    try:
        return [str(key) for key in json.loads(body).get("keys", [])]
    except ValueError:
        return []


def build_list(bucket: str, items: list[Item], workers: int) -> list[Wanted]:
    """Sweep every item's key-map record and collect the sheets with no raw copy."""

    def one(item: Item) -> list[Wanted]:
        try:
            present = list_prefix(bucket, item.prefix)
        except OSError as error:
            print(f"{item.item}: listing failed: {error}", file=sys.stderr)
            return []
        if "keymaps.json" not in present:
            return []
        try:
            body = run_aws(
                ["aws", "s3", "cp", f"{item_prefix(bucket, item)}/keymaps.json", "-"],
                capture=True,
            ).stdout
        except OSError as error:
            print(f"{item.item}: keymaps.json unreadable: {error}", file=sys.stderr)
            return []
        keys = keymap_keys(present, body)
        return [Wanted(item.item, key) for key in missing_raw(present, keys)]

    wanted: list[Wanted] = []
    with ThreadPoolExecutor(workers) as pool:
        for index, rows in enumerate(pool.map(one, items), start=1):
            wanted.extend(rows)
            if index % 2000 == 0:
                print(
                    f"  swept {index}/{len(items)} items, {len(wanted)} sheets wanted",
                    file=sys.stderr,
                    flush=True,
                )
    return wanted


def fetch_one(want: Wanted, plan, bucket: str, mirror: str, work_dir: Path) -> int:
    """Download one sheet's JP2, decode it at full resolution, upload the JPEG.

    Returns the JPEG's size in bytes. The JP2 is deleted either way: at roughly
    8 MB a sheet, keeping them would be 80 GB of scratch for nothing.
    """
    from mapsnap.loc_mirror import decode_jp2, fetch, source_url

    sheet = next((s for s in plan.sheets if s.key == want.key), None)
    if sheet is None:
        raise OSError(f"{want.item}: no sheet {want.key} in the mapping")
    item = Item(item=plan.item, state=plan.state, year=plan.year)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        scratch = Path(tmp)
        source = scratch / f"{want.key}{Path(source_url(mirror, sheet)).suffix}"
        fetch(source_url(mirror, sheet), source, sheet.bytes)
        out_jpg = scratch / f"{want.key}.jpg"
        decode_jp2(source, out_jpg, 0)
        run_aws(
            [
                "aws",
                "s3",
                "cp",
                str(out_jpg),
                f"{item_prefix(bucket, item)}/raw/{want.key}.jpg",
                "--only-show-errors",
            ]
        )
        return out_jpg.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the full-resolution key-map sheets the mirror skipped."
    )
    parser.add_argument("--bucket", default="s3://mapsnap-sanborn")
    parser.add_argument(
        "--mirror",
        help="HTTP root serving the torrent's storage-services tree, e.g. http://host:port.",
    )
    parser.add_argument("--list", type=Path, help="Work list TSV (item, key).")
    parser.add_argument(
        "--build-list",
        type=Path,
        metavar="FILE",
        help="Sweep the bucket's keymaps.json records and write the work list here.",
    )
    parser.add_argument("--manifest", help="Sheet mapping TSV (default: the bucket's).")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/loc-raw"))
    parser.add_argument("--limit", type=int, help="Stop after this many sheets.")
    parser.add_argument(
        "--sweep-workers",
        type=int,
        default=32,
        metavar="N",
        help="Concurrent listings while building the list (default: %(default)s).",
    )
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        sys.exit(f"--shard must be in [0, {args.shards})")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = resolve_manifest(args.manifest, args.bucket, args.work_dir)

    if args.build_list:
        items = read_manifest(manifest)
        print(
            f"sweeping {len(items)} items for key-map records",
            file=sys.stderr,
            flush=True,
        )
        wanted = build_list(args.bucket, items, args.sweep_workers)
        write_list(args.build_list, wanted)
        print(
            f"wrote {args.build_list}: {len(wanted)} sheets to fetch", file=sys.stderr
        )
        return

    if not args.list:
        sys.exit("give --list FILE to fetch, or --build-list FILE to make one")
    if not args.mirror:
        sys.exit(
            "--mirror is required: the mirror is somebody's machine, not a default"
        )

    from mapsnap.loc_mirror import load_mapping

    plans = load_mapping(manifest)
    wanted = [
        want
        for want in read_list(args.list)
        if shard_of(want.item, args.shards) == args.shard
    ]
    print(
        f"shard {args.shard}/{args.shards}: {len(wanted)} sheets",
        file=sys.stderr,
        flush=True,
    )

    started = time.perf_counter()
    done = failed = skipped = 0
    written = 0
    for index, want in enumerate(wanted, start=1):
        if args.limit and done >= args.limit:
            break
        plan = plans.get(want.item)
        if plan is None:
            print(f"{want.item}: not in the mapping", file=sys.stderr, flush=True)
            failed += 1
            continue
        item = Item(item=plan.item, state=plan.state, year=plan.year)
        try:
            if f"raw/{want.key}.jpg" in list_prefix(args.bucket, item.prefix):
                skipped += 1
                continue
            written += fetch_one(want, plan, args.bucket, args.mirror, args.work_dir)
        except OSError as error:
            print(
                f"{want.item} {want.key}: FAILED: {error}", file=sys.stderr, flush=True
            )
            failed += 1
            continue
        done += 1
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        print(
            f"{datetime.now(UTC):%H:%M:%S} s{args.shard} [{index}/{len(wanted)}] "
            f"{want.item} {want.key}: {written / max(done, 1) / 2**20:.1f} MB avg"
            f" | {rate * 3600:.0f} sheets/h, eta "
            f"{format_duration((len(wanted) - index) / rate if rate else 0)}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    print(
        f"shard {args.shard}: {done} sheets fetched ({written / 2**30:.1f} GB), "
        f"{skipped} already there, {failed} failed in {format_duration(elapsed)} "
        f"({elapsed:.0f}s)",
        file=sys.stderr,
        flush=True,
    )
    shutil.rmtree(args.work_dir / "tmp", ignore_errors=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
