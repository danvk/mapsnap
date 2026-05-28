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
import urllib.error
import urllib.request
from pathlib import Path

from mapsnap.compare_iiif_georef import source_id_to_page_key
from mapsnap.utils import jpeg_dimensions


def download_with_retry(
    url: str,
    dest: Path,
    max_attempts: int = 5,
    initial_delay: float = 15.0,
) -> None:
    """Download a URL to dest, retrying on transient errors with exponential backoff."""
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mapsnap/0.1"})
            with urllib.request.urlopen(req) as resp:
                dest.write_bytes(resp.read())
            return
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < max_attempts:
                print(
                    f"  HTTP {exc.code}; retrying in {delay:.0f}s "
                    f"(attempt {attempt}/{max_attempts})",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            else:
                raise
        except Exception:
            if attempt < max_attempts:
                print(
                    f"  Error on attempt {attempt}; retrying in {delay:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            else:
                raise


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
) -> bool:
    """Download the full-resolution image for one canvas.

    Returns True on success, False if the canvas was skipped or failed.
    """
    canvas_id: str = canvas.get("@id", "")
    label: str = canvas.get("label", "unknown")

    page_key = canvas_to_page_key(canvas_id, label)
    image_url = f"{canvas_id}/full/full/0/default.jpg"
    image_path = output_dir / f"{page_key}.raw.jpg"

    if image_path.exists():
        print(f"  Already done: {image_path.name}", file=sys.stderr)
        return True

    print(f"  {label} ({page_key})", file=sys.stderr)
    print(f"    URL: {image_url}", file=sys.stderr)

    if dry_run:
        return True

    print(f"    Downloading → {image_path.name} ...", file=sys.stderr)
    download_with_retry(image_url, image_path)
    dl_width, dl_height = jpeg_dimensions(image_path)
    print(f"    Downloaded: {dl_width}×{dl_height}", file=sys.stderr)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download LOC images from a local IIIF Presentation manifest. "
            "Saves one .raw.jpg per canvas alongside the IIIF file."
        )
    )
    parser.add_argument(
        "iiif_file",
        metavar="FILE",
        help="Local LOC IIIF Presentation manifest JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    iiif_path = Path(args.iiif_file)
    output_dir = iiif_path.parent
    data: dict = json.load(iiif_path.open())

    sequences: list[dict] = data.get("sequences", [])
    if not sequences:
        print("No sequences found in manifest.", file=sys.stderr)
        sys.exit(1)
    canvases: list[dict] = sequences[0].get("canvases", [])
    print(f"Found {len(canvases)} canvases in {iiif_path.name}.", file=sys.stderr)

    success = 0
    for i, canvas in enumerate(canvases, 1):
        print(f"[{i}/{len(canvases)}]", file=sys.stderr)
        ok = process_canvas(canvas, output_dir, args.dry_run)
        if ok:
            success += 1

    print(
        f"\nDone: {success}/{len(canvases)} canvases {'would be ' if args.dry_run else ''}processed.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
