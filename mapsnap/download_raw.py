"""Download the full-resolution version of a single already-downloaded LOC page image.

Given a path to a 25%-scaled JPG like data/<volume>/p0.jpg, looks up the matching
canvas in the sibling manifest.json and downloads the full-resolution image to
data/<volume>/raw/p0.jpg.

Usage:
    python download_raw.py <jpg_file> [<jpg_file> ...] [--dry-run]
"""

import argparse
from dataclasses import replace
from pathlib import Path

from mapsnap.download_loc_iiif import (
    CanvasTarget,
    disambiguate_keys,
    load_targets,
    process_canvas,
)


def find_target(targets: list[CanvasTarget], page_key: str) -> CanvasTarget:
    """The canvas written as ``<page_key>.jpg``, or raise.

    Matched on the disambiguated key, so a scaled p0a.jpg finds the first of the
    two page-0 canvases a multi-volume manifest carries rather than failing.
    """
    for target in targets:
        if target.page_key == page_key:
            return target
    raise ValueError(f"No canvas found for page key {page_key!r}")


def download_raw(jpg_path: Path, dry_run: bool = False) -> Path:
    """Download the full-resolution counterpart of a scaled-down LOC page JPG."""
    output_dir = jpg_path.parent
    targets = load_targets(output_dir / "manifest.json")
    disambiguate_keys(targets)
    target = find_target(targets, jpg_path.stem)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    return process_canvas(replace(target, output_dir=raw_dir), "full", dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the full-resolution LOC image for one or more scaled JPGs."
        )
    )
    parser.add_argument(
        "jpg_files",
        nargs="+",
        metavar="FILE",
        help="Scaled-down page JPG(s), e.g. data/ellenville_ny_1910/p0.jpg",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    for jpg_file in args.jpg_files:
        out_path = download_raw(Path(jpg_file), dry_run=args.dry_run)
        print(out_path)


if __name__ == "__main__":
    main()
