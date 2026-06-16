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
            if (exc.code == 429 or (500 <= exc.code < 600)) and attempt < max_attempts:
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


def download_oim_image(url: str, dest: Path) -> None:
    """Download an OIM image, retrying with a documents↔regions URL swap on 404.

    Split pages (dest filename contains '__') swap /documents/ → /regions/ on 404.
    Unsplit pages swap /regions/ → /documents/ on 404.
    """
    is_split = "__" in dest.name
    try:
        download_with_retry(url, dest)
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and "/documents/" in url and is_split:
            actual_url = url.replace("/documents/", "/regions/")
        elif exc.code == 404 and "/regions/" in url and not is_split:
            actual_url = url.replace("/regions/", "/documents/")
        else:
            raise
        print(f"    404; retrying with: {actual_url}", file=sys.stderr)
        download_with_retry(actual_url, dest)


def _download_if_missing(url: str, dest: Path, dry_run: bool) -> None:
    """Download url to dest unless dest already exists. Creates parent dirs."""
    if dest.exists():
        print(f"    Already done: {dest.name}", file=sys.stderr)
        return
    print(f"    {url} → {dest}", file=sys.stderr)
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_oim_image(url, dest)
    dl_width, dl_height = jpeg_dimensions(dest)
    print(f"    Downloaded: {dl_width}×{dl_height}", file=sys.stderr)


def process_annotation(
    item: dict,
    output_dir: Path,
    oim_url_prefix: str,
    dry_run: bool,
) -> bool:
    """Download the images for one annotation item into the canonical layout.

    The full-resolution unsplit page goes to raw/<base_key>.jpg (the pipeline input).
    For split annotations (page key ending in __N), OIM's manually-generated split
    region also goes to oim/<page_key>.jpg as comparison ground truth; the unsplit
    raw/<base_key>.jpg serves as the template-matching reference.

    Returns True on success, False if the item was skipped.
    """
    label: str = item.get("label", "unknown")
    page_key = label_to_page_key(label)
    if page_key is None:
        print(f"  Skipping '{label}': could not extract page key.", file=sys.stderr)
        return False

    print(f"  {label}", file=sys.stderr)
    base_key = page_key.split("__")[0]

    # Full-resolution unsplit page → raw/ (downloaded once per page).
    _download_if_missing(
        f"{oim_url_prefix}{base_key}.jpg",
        output_dir / "raw" / f"{base_key}.jpg",
        dry_run,
    )

    # OIM's manual split region → oim/ (ground truth for comparison).
    if "__" in page_key:
        _download_if_missing(
            f"{oim_url_prefix}{page_key}.jpg",
            output_dir / "oim" / f"{page_key}.jpg",
            dry_run,
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download OIM images from a local IIIF AnnotationPage. "
            "Saves full-resolution pages to raw/ and OIM split regions to oim/."
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
