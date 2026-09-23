#!/usr/bin/env python
"""Which items a run has not published, by asking S3 rather than the job states.

    scripts/batch/missing-items.py corpus-v1 items.txt --out corpus-v1

A child of an array job carries several items and publishes each as it
finishes, so a FAILED child is not a set of lost items: a child reclaimed by
spot halfway through has already published everything it got to, on both of its
attempts. Counting failures therefore overstates the work left, and
resubmitting failed children re-runs items that are already done.

The done marker is the thing to trust. ``loc-fit`` writes
``runs/<tag>/mapsnap.iiif.json`` last, after every sidecar it vouches for, so
its presence means the item is finished and its absence means it is not --
whatever the job state says.

Writes ``<prefix>-done.txt`` and ``<prefix>-missing.txt``; the latter is what
``submit.sh`` wants. One HEAD per item, 64 at a time: 33,000 items take about a
minute, against hours of listing a bucket that holds millions of keys. boto3
rather than the aws CLI for the same reason -- 33,000 process spawns would cost
more than the requests do.
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

DEFAULT_MAPPING = Path.home() / "Downloads/loc-sanborn-maps.mapping.tsv"
MARKER = "mapsnap.iiif.json"


def mirror_prefixes(mapping: Path) -> dict[str, tuple[str, str]]:
    """Item -> (state, year), the mirror's own layout for it."""
    prefixes: dict[str, tuple[str, str]] = {}
    with open(mapping, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            prefixes.setdefault(row["item"], (row["state"], row["year"]))
    return prefixes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_tag")
    parser.add_argument("items", type=Path, help="One item id per line")
    parser.add_argument("--bucket", default="mapsnap-sanborn")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--profile", default="mapsnap")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--out", default="run", help="Prefix for the two lists")
    args = parser.parse_args()

    prefixes = mirror_prefixes(args.mapping)
    items = [
        line.strip() for line in args.items.read_text().splitlines() if line.strip()
    ]
    unknown = [item for item in items if item not in prefixes]
    known = [item for item in items if item in prefixes]
    print(
        f"{len(items):,} items; {len(unknown):,} not in the mirror mapping",
        file=sys.stderr,
    )

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    # Retries matter: 33,000 HEADs will draw some throttling, and a throttled
    # request read as "absent" would put a finished item on the resubmit list.
    client = session.client(
        "s3",
        config=Config(
            max_pool_connections=args.workers * 2,
            retries={"max_attempts": 8, "mode": "adaptive"},
        ),
    )

    def published(item: str) -> tuple[str, bool | None]:
        state, year = prefixes[item]
        key = f"by-state/{state}/{year}/{item}/runs/{args.run_tag}/{MARKER}"
        try:
            client.head_object(Bucket=args.bucket, Key=key)
            return item, True
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return item, False
            # Anything else (403, throttling that outlived its retries) is not
            # evidence of absence, and must not be reported as such.
            return item, None

    done: list[str] = []
    missing: list[str] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (item, ok) in enumerate(pool.map(published, known), start=1):
            (done if ok else missing if ok is False else errors).append(item)
            if n % 5000 == 0:
                print(f"  {n:,}/{len(known):,} checked", file=sys.stderr)

    Path(f"{args.out}-done.txt").write_text("\n".join(sorted(done)) + "\n")
    Path(f"{args.out}-missing.txt").write_text(
        "\n".join(sorted(missing + unknown)) + "\n"
    )
    print(
        f"\npublished {len(done):,}  missing {len(missing):,}"
        f"  unmapped {len(unknown):,}  indeterminate {len(errors):,}"
    )
    if errors:
        print(f"  NOT counted as missing (S3 would not say): {', '.join(errors[:10])}")
    print(f"\n  {args.out}-missing.txt is the resubmission list")


if __name__ == "__main__":
    main()
