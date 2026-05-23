"""Download images for an OIM IIIF AnnotationPage from the OIM S3 bucket.

Reads a local IIIF AnnotationPage JSON file, extracts a page key from each
annotation label (e.g. "p156" from "New Orleans, La. | 1896 | Vol. 2 p156"),
constructs an OIM S3 image URL using a caller-supplied prefix, and downloads it.

Usage:
    python download_oim_iiif.py <iiif_file> --oim-url-prefix URL_PREFIX [--dry-run]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from mapsnap.utils import jpeg_dimensions, label_to_page_key


def download_with_retry(
    url: str,
    dest: Path,
    max_attempts: int = 5,
    initial_delay: float = 1.0,
) -> None:
    """Download a URL to dest, retrying on transient errors with exponential backoff."""
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(url) as resp:
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


def process_annotation(
    item: dict,
    output_dir: Path,
    oim_url_prefix: str,
    dry_run: bool,
) -> bool:
    """Download the image for one annotation item.

    Returns True on success, False if the item was skipped or failed.
    """
    label: str = item.get("label", "unknown")
    page_key = label_to_page_key(label)
    if page_key is None:
        print(f"  Skipping '{label}': could not extract page key.", file=sys.stderr)
        return False

    target = item.get("target", {})
    source = target.get("source", {})
    full_width: int = source.get("width", 0)
    full_height: int = source.get("height", 0)

    if not full_width or not full_height:
        print(f"  Skipping '{label}': missing source dimensions.", file=sys.stderr)
        return False

    image_url = f"{oim_url_prefix}{page_key}.jpg"
    image_path = output_dir / f"{page_key}.raw.jpg"

    if image_path.exists():
        print(f"  Already done: {image_path.name}", file=sys.stderr)
        return True

    print(f"  {label}", file=sys.stderr)
    print(f"    Image: {image_url}", file=sys.stderr)
    print(f"    Source: {full_width}×{full_height}", file=sys.stderr)

    if dry_run:
        return True

    actual_url = image_url
    if not image_path.exists():
        print(f"    Downloading → {image_path.name} ...", file=sys.stderr)
        try:
            download_with_retry(image_url, image_path)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and "documents" in image_url and "__" in image_url:
                actual_url = image_url.replace("/documents/", "/regions/")
                print(f"    404; retrying with: {actual_url}", file=sys.stderr)
                download_with_retry(actual_url, image_path)
            else:
                raise

    dl_width, dl_height = jpeg_dimensions(image_path)
    print(f"    Downloaded: {dl_width}×{dl_height}", file=sys.stderr)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download OIM images from a local IIIF AnnotationPage. "
            "Saves one .raw.jpg per annotation item alongside the IIIF file."
        )
    )
    parser.add_argument(
        "iiif_file",
        metavar="FILE",
        help="Local OIM IIIF AnnotationPage JSON file",
    )
    parser.add_argument(
        "--oim-url-prefix",
        required=True,
        metavar="URL_PREFIX",
        help=(
            "OIM S3 URL prefix for images. The page key extracted from each annotation "
            "label is appended to this prefix followed by '.jpg'. "
            "Example: https://s3.us-central-1.wasabisys.com/oldinsurancemaps/uploaded/"
            "regions/new_orleans_la_1896_vol_2_"
        ),
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

    items: list[dict] = data.get("items", [])
    print(f"Found {len(items)} annotation items in {iiif_path.name}.", file=sys.stderr)

    success = 0
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}]", file=sys.stderr)
        ok = process_annotation(item, output_dir, args.oim_url_prefix, args.dry_run)
        if ok:
            success += 1

    print(
        f"\nDone: {success}/{len(items)} items {'would be ' if args.dry_run else ''}processed.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
