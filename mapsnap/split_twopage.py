"""Split double-page Sanborn scans (vol7-twopage) into single-page images.

Uses the filename numbering scheme — no OCR needed.

Filename format:
  sb000NNN0.raw.jpg  →  left page = NNN  (e.g. sb000060 → page 6, sb000570 → page 57)
  sb00001d.raw.jpg   →  left page = 1    (special first-page naming)

Content type is determined by the difference between consecutive page numbers:
  diff = 1 (file increment  10):  both halves show the same page number —
    one color version and one skeleton. Color content identifies which is which.
  diff = 2 (file increment  20):  two separate map sheets — left = page N,
    right = page N+1, each named with an 's' suffix if it is a skeleton.

Single-page images (taller than wide) are copied with an appropriate name.
Files whose names don't match the expected pattern (e.g. sb00001a/b/c) are skipped.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Skeleton detection: fraction of pixels with max(R,G,B)-min(R,G,B) > SPREAD_THRESHOLD
# that must be exceeded for a half to be considered a color (non-skeleton) page.
_COLOR_SPREAD = 50
_SKELETON_THRESHOLD = 0.10


def is_skeleton(img: Image.Image) -> bool:
    """Return True if the image has too few colored pixels to be a color map."""
    arr = np.array(img.convert("RGB"), dtype=np.int16)
    spread = arr.max(axis=2) - arr.min(axis=2)
    return float((spread > _COLOR_SPREAD).mean()) < _SKELETON_THRESHOLD


def parse_page_number(filename: str) -> int | None:
    """Extract the left-page number from a two-page filename.

    sb00001d.raw.jpg → 1   (special first-page case)
    sb000060.raw.jpg → 6
    sb000570.raw.jpg → 57
    Returns None for unrecognized patterns (e.g. sb00001a/b/c).
    """
    stem = filename.removesuffix(".raw.jpg")
    if re.fullmatch(r"sb0+1d", stem):
        return 1
    m = re.fullmatch(r"sb(\d+)", stem)
    if m:
        n = int(m.group(1))
        if n % 10 == 0:
            return n // 10
    return None


def page_filename(page_num: int, skeleton: bool) -> str:
    return f"p{page_num}{'s' if skeleton else ''}.raw.jpg"


def save_half(
    half: Image.Image,
    page_num: int,
    out_dir: Path,
    source_name: str,
    side: str,
    dry_run: bool,
) -> bool:
    """Write a half-page (or single-page) image. Returns True on success."""
    skeleton = is_skeleton(half)
    name = page_filename(page_num, skeleton)
    kind = "skeleton" if skeleton else "color"
    out_path = out_dir / name
    print(f"  {source_name} {side} → {name} ({kind})")
    if dry_run:
        return True
    if out_path.exists():
        print(f"    WARNING: {name} already exists, skipping")
        return False
    half.save(out_path, quality=92)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="",
        help="Directory of raw JPEGs",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="",
        help="Output directory for single-page JPEGs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing files",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    all_files = sorted(in_dir.glob("*.raw.jpg"))
    if not all_files:
        print(f"No .raw.jpg files in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse page numbers; skip files with unrecognised names.
    file_pages: list[tuple[Path, int | None]] = [
        (f, parse_page_number(f.name)) for f in all_files
    ]
    n_skipped = sum(1 for _, p in file_pages if p is None)
    if n_skipped:
        for f, p in file_pages:
            if p is None:
                print(f"Skipping {f.name} (unrecognised filename pattern)")

    valid = [(f, p) for f, p in file_pages if p is not None]
    if not valid:
        print("No files with parseable page numbers.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_fail = 0

    for i, (img_path, page_num) in enumerate(valid):
        # Determine the next valid page number for the diff calculation.
        next_page = valid[i + 1][1] if i + 1 < len(valid) else None

        with Image.open(img_path) as img:
            w, h = img.size

        if w <= h:
            # Single-page image — copy directly.
            with Image.open(img_path) as img:
                full = img.copy()
            ok = save_half(
                full, page_num, out_dir, img_path.name, "single", args.dry_run
            )
            n_success += ok
            n_fail += not ok
            continue

        # Double-page: split at horizontal midpoint.
        with Image.open(img_path) as img:
            mid = w // 2
            left_half = img.crop((0, 0, mid, h)).copy()
            right_half = img.crop((mid, 0, w, h)).copy()

        diff = (next_page - page_num) if next_page is not None else 2

        if diff == 1:
            # Color/skeleton pair: both halves share page_num.
            # is_skeleton() assigns the 's' suffix to whichever half has less color.
            for half, side in [(left_half, "left"), (right_half, "right")]:
                ok = save_half(
                    half, page_num, out_dir, img_path.name, side, args.dry_run
                )
                n_success += ok
                n_fail += not ok

        elif diff == 2:
            # Two separate sheets: left=page_num, right=page_num+1.
            for half, side, pnum in [
                (left_half, "left", page_num),
                (right_half, "right", page_num + 1),
            ]:
                ok = save_half(half, pnum, out_dir, img_path.name, side, args.dry_run)
                n_success += ok
                n_fail += not ok

        else:
            print(f"WARNING: {img_path.name} has unexpected page diff {diff}, skipping")
            n_fail += 2

    mode = " (dry run)" if args.dry_run else ""
    print(f"\nDone{mode}: {n_success} pages written, {n_fail} issues.")


if __name__ == "__main__":
    main()
