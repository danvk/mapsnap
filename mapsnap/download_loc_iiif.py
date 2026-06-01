"""Download full-resolution images from a Library of Congress IIIF Presentation manifest.

Reads a local LOC IIIF v2 manifest JSON file, extracts a page key from each
canvas @id (e.g. "p451" from "...04720_04_1951-0451"), constructs a
full-resolution IIIF image URL, and downloads it.

Usage:
    python download_loc_iiif.py <iiif_file> [--dry-run]
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from mapsnap.compare_iiif_georef import source_id_to_page_key
from mapsnap.download_oim_iiif import download_with_retry
from mapsnap.utils import jpeg_dimensions


def canvas_to_page_key(canvas_id: str, label: str) -> str:
    """Extract a short page key from a LOC IIIF canvas @id and label.

    Delegates to source_id_to_page_key for numeric page IDs (e.g. p451).
    Falls back to the last colon-separated segment for non-numeric ones
    (e.g. "covr", "titl", "ind1").
    """
    last_segment = canvas_id.split(":")[-1]
    if re.search(r"-\d", last_segment):
        return source_id_to_page_key(canvas_id, label)
    return last_segment


def process_canvas(
    canvas: dict,
    output_dir: Path,
    dry_run: bool,
) -> Path:
    """Download the full-resolution image for one canvas.

    Returns a Path to the output file, or raises on failure.
    """
    canvas_id: str = canvas.get("@id", "")
    label: str = canvas.get("label", "unknown")

    page_key = canvas_to_page_key(canvas_id, label)
    assert page_key != "iiif", f"Could not extract valid page key from {canvas_id}"
    # size = "pct:25"
    size = "full"
    image_url = f"{canvas_id}/full/{size}/0/default.jpg"
    image_path = output_dir / f"{page_key}.raw.jpg"

    if image_path.exists():
        print(f"  Already done: {image_path.name}", file=sys.stderr)
        return image_path

    print(f"  {label} ({page_key}) {image_path} {image_url}", file=sys.stderr)

    if dry_run:
        return image_path

    print(f"    Downloading → {image_path.name} ...", file=sys.stderr)
    download_with_retry(image_url, image_path, initial_delay=15.0)
    # unconditional delay to avoid making too many requests if they all succeed.
    time.sleep(15.0)
    dl_width, dl_height = jpeg_dimensions(image_path)
    print(f"    Downloaded: {dl_width}×{dl_height}", file=sys.stderr)

    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download LOC images from a local IIIF Presentation manifest. "
            "Saves one .raw.jpg per canvas alongside the IIIF file."
        )
    )
    parser.add_argument(
        "iiif_files",
        nargs="+",
        metavar="FILE",
        help="Local LOC IIIF Presentation manifest JSON file(s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    num_total = 0
    out_paths = Counter[Path]()
    for iiif_file in args.iiif_files:
        iiif_path = Path(iiif_file)
        output_dir = iiif_path.parent
        data: dict = json.load(iiif_path.open())

        sequences: list[dict] = data.get("sequences", [])
        if not sequences:
            print("No sequences found in manifest.", file=sys.stderr)
            sys.exit(1)
        canvases: list[dict] = sequences[0].get("canvases", [])
        print(f"Found {len(canvases)} canvases in {iiif_path.name}.", file=sys.stderr)

        for i, canvas in enumerate(canvases, 1):
            print(f"[{i}/{len(canvases)}] ", file=sys.stderr, end="")
            out_path = process_canvas(canvas, output_dir, args.dry_run)
            out_paths[out_path] += 1
            num_total += 1

    print(
        f"\nDone: {num_total} canvases {'would be ' if args.dry_run else ''}processed.",
        file=sys.stderr,
    )
    if len(out_paths) < num_total:
        print(f"Unique paths: {len(out_paths)}; there are collisions.", file=sys.stderr)
        print(
            [*((path, count) for path, count in out_paths.most_common() if count > 1)],
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
