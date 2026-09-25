#!/usr/bin/env python
"""Sync a corpus item's mirror prefix into data/, under a readable name.

    uv run python scripts/pull_item.py sanborn08035_001
    uv run python scripts/pull_item.py sanborn08035_001 sanborn06116_049 --dry-run

Looks the item up in the volumes TSV (item, state, city, year, pages, title,
s3_path) and runs ``aws s3 sync`` from its prefix into
``data/<city>_<state>_<year>/``, e.g. ``data/wernersville_pa_1914/``. When the
TSV has more than one item for that town and year, the item id is appended
(``data/chicago_il_1950_sanborn01790_085/``), since the TSV carries no volume
number to tell them apart.

The layout is the mirror's own, kept as it is: page scans and their CRAFT
boxes and P(road) maps at the top, each run's outputs under ``runs/<tag>/``.
The volume viewer reads a run's annotation, sidecars and adjacency from there.

``aws s3 sync`` downloads ten files at a time by default and skips any file
already present with the same size and an up-to-date copy, so re-running it
after a new run fetches only that run. For more parallelism, raise the
profile's limit once:

    aws configure set s3.max_concurrent_requests 32 --profile mapsnap
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from mapsnap.loc_counties import STATE_POSTAL

DEFAULT_TSV = Path.home() / "Documents/mapsnap/loc-sanborn-volumes.tsv"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def read_volumes(tsv: Path) -> list[dict[str, str]]:
    """The volumes TSV's rows."""
    with open(tsv, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def state_abbreviation(state_slug: str) -> str:
    """The lowercase postal code for a state slug, or the slug if it has none.

    "pennsylvania" is "pa" and "new-york" is "ny"; "multiple-states-cuba" stays as it is.
    """
    postal = STATE_POSTAL.get(state_slug.replace("-", " "))
    return postal.lower() if postal else state_slug


def slug(text: str) -> str:
    """Lowercase, with runs of anything but letters and digits as one underscore."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def directory_name(row: dict[str, str], rows: list[dict[str, str]]) -> str:
    """The data/ directory for a volume: ``<city>_<state>_<year>``, plus the item when shared.

    Falls back to the item id for a row with no city, like the 1868 Toledo
    atlas the TSV files under no town.
    """
    if not row["city"]:
        return row["item"]
    base = f"{slug(row['city'])}_{state_abbreviation(row['state'])}_{row['year']}"
    siblings = [
        other
        for other in rows
        if (other["state"], other["city"], other["year"])
        == (row["state"], row["city"], row["year"])
    ]
    return base if len(siblings) == 1 else f"{base}_{row['item']}"


def sync_command(s3_path: str, destination: Path) -> list[str]:
    """The ``aws s3 sync`` that mirrors one item's prefix into ``destination``."""
    return [
        "aws",
        "s3",
        "sync",
        s3_path.rstrip("/") + "/",
        str(destination),
        "--exclude",
        "*.DS_Store",
        "--only-show-errors",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("items", nargs="+", help="Item ids, e.g. sanborn08035_001")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--name", help="Directory name under data/, instead of the derived one"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be synced, and stop"
    )
    args = parser.parse_args()
    if args.name and len(args.items) > 1:
        parser.error("--name names one directory, so it takes one item")

    rows = read_volumes(args.tsv)
    by_item = {row["item"]: row for row in rows}
    missing = [item for item in args.items if item not in by_item]
    if missing:
        sys.exit(f"not in {args.tsv}: {', '.join(missing)}")

    environment = {**os.environ}
    environment.setdefault("AWS_PROFILE", "mapsnap")
    environment.setdefault("AWS_REGION", "us-west-2")
    for item in args.items:
        row = by_item[item]
        destination = args.data_dir / (args.name or directory_name(row, rows))
        command = sync_command(row["s3_path"], destination)
        print(f"{item}: {row['title']} ({row['year']}) -> {destination}")
        if args.dry_run:
            print("  " + " ".join(command))
            continue
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, env=environment)
        files = [path for path in destination.rglob("*") if path.is_file()]
        size_mb = sum(path.stat().st_size for path in files) / 1e6
        runs = sorted(path.name for path in (destination / "runs").glob("*"))
        print(
            f"  {len(files)} files, {size_mb:.1f} MB; runs: {', '.join(runs) or 'none'}"
        )


if __name__ == "__main__":
    main()
