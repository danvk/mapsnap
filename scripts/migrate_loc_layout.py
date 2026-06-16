"""Migrate Library of Congress data directories to the suffix-free layout.

LoC pages were downloaded at pct:25 into ``<page_key>.raw.jpg``; that file is already at
25% scale, so the new canonical layout just drops the ``.raw`` qualifier:
``<page_key>.raw.jpg`` → ``<page_key>.jpg``. Sidecars (``.boxes.json``, ``.streets.json``,
``.georef.json``) are keyed by the bare stem and need no renaming, and there is no
full-resolution ``raw/`` directory for migrated LoC volumes.

Recurses into each given directory (covering multivolume ``vol*/`` subdirectories). Skips a
rename when the destination already exists. Run ``mapsnap split`` afterwards to generate
panels for any pages that split.

Usage:
    uv run scripts/migrate_loc_layout.py DIR [DIR ...] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

RAW_SUFFIX = ".raw.jpg"


def migrate_dir(dir_path: Path, dry_run: bool) -> int:
    """Rename every *.raw.jpg under dir_path to *.jpg. Returns the number renamed."""
    renamed = 0
    for src in sorted(dir_path.rglob(f"*{RAW_SUFFIX}")):
        dest = src.with_name(src.name[: -len(RAW_SUFFIX)] + ".jpg")
        if dest.exists():
            print(f"  skip (exists): {dest}", file=sys.stderr)
            continue
        print(f"  {src.name} → {dest.name}  ({src.parent})")
        if not dry_run:
            src.rename(dest)
        renamed += 1
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs", nargs="+", type=Path, metavar="DIR", help="LoC data directories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the renames without performing them.",
    )
    args = parser.parse_args()

    total = 0
    for dir_path in args.dirs:
        if not dir_path.is_dir():
            sys.exit(f"Not a directory: {dir_path}")
        total += migrate_dir(dir_path, args.dry_run)

    verb = "would rename" if args.dry_run else "renamed"
    print(f"\nDone: {verb} {total} file(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
