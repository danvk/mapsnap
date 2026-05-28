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
            req = urllib.request.Request(url, headers={"User-Agent": "mapsnap/0.1"})
            with urllib.request.urlopen(req) as resp:
                dest.write_bytes(resp.read())
            return
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503, 520) and attempt < max_attempts:
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


def _download_unsplit_image(
    base_key: str,
    output_dir: Path,
    oim_url_prefix: str,
    dry_run: bool,
) -> None:
    """Download the un-split version of a split page (e.g. p4 for p4__2).

    Saves to <base_key>.unsplit.jpg. Skips if the file already exists.
    """
    unsplit_path = output_dir / f"{base_key}.unsplit.jpg"
    if unsplit_path.exists():
        print(f"    Unsplit already done: {unsplit_path.name}", file=sys.stderr)
        return

    unsplit_url = f"{oim_url_prefix}{base_key}.jpg"
    print(f"    Unsplit URL: {unsplit_url}", file=sys.stderr)

    if dry_run:
        return

    print(f"    Downloading unsplit → {unsplit_path.name} ...", file=sys.stderr)
    download_with_retry(unsplit_url, unsplit_path)
    dl_width, dl_height = jpeg_dimensions(unsplit_path)
    print(f"    Unsplit downloaded: {dl_width}×{dl_height}", file=sys.stderr)


def process_annotation(
    item: dict,
    output_dir: Path,
    oim_url_prefix: str,
    dry_run: bool,
) -> bool:
    """Download the image for one annotation item.

    Returns True on success, False if the item was skipped or failed.
    For split page keys (ending in __N), also downloads the un-split image
    as <base_key>.unsplit.jpg so it can serve as a reference for sub-image bounds.
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
    else:
        print(f"  {label}", file=sys.stderr)
        print(f"    Image: {image_url}", file=sys.stderr)
        print(f"    Source: {full_width}×{full_height}", file=sys.stderr)

        if not dry_run:
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

    # For split pages, also fetch the un-split original.
    if "__" in page_key:
        base_key = page_key.split("__")[0]
        _download_unsplit_image(base_key, output_dir, oim_url_prefix, dry_run)

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
