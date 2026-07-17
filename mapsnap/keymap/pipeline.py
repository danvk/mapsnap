"""Prepare a full-resolution Sanborn key map for use by the main pipeline.

A key map is the volume's index sheet: one saturated colored block per page, its page number
printed inside, drawn over an overview of the city's major streets. Downstream, ``mapsnap ocr``
and ``mapsnap georef`` take ``--keymap`` files to restrict each page's vocabulary/matching to
its key-map neighborhood; that needs three sidecars next to the (raw) key-map image:

  * ``<stem>.keymap.json`` — the pixel location of every page number, from the CNN localizer +
    CRNN recognizer (``mapsnap.keymap.detect_numbers_crnn``). ``--pages`` is derived from the
    volume's page images so decodes snap to valid page numbers and the narrow-detection re-read
    (which recovers a squished multi-digit number) is enabled.
  * ``<stem>.georef.json`` — the key map georeferenced in the world, so a page number's pixel
    location maps to a world location. Produced by OCR'ing the key map's own street labels and
    fitting a transform, exactly like a regular page (``mapsnap ocr`` then ``mapsnap georef``).
    Running it on its own here lets the key map use a page-appropriate ``--min-short-side``: the
    raw sheet is ~4x the linear resolution of the 25%-scale volume pages, so its text is ~4x
    larger and a larger detector floor is right. OCR tiles the oversized sheet at native
    resolution by default, which is what makes the key map's small labels detectable.
  * ``<stem>.regions.panels.json`` — the colored block polygon around each page number
    (``mapsnap.keymap.page_regions``), so a page's key-map neighborhood is its own block.

They are built in that order for a reason: ``<stem>.keymap.json`` is what identifies a page as a
key map, and the georef step reads it to decide whether to refit the corners with a full 6-DOF
affine. Detecting the page numbers after georeferencing would leave a first run with the plain
4-parameter similarity.

    uv run mapsnap keymap data/chicago_il_1950_vol_1/raw/p0b.jpg
"""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from mapsnap.keymap.fit_keymap import volume_page_numbers
from mapsnap.keymap.records import keymap_path
from mapsnap.utils import default_centerlines, run_cmd


def format_page_spec(numbers: Iterable[int]) -> str:
    """Collapse page numbers into a compact spec like ``"1-3,5,7-9"`` (a parse_page_spec input)."""
    ordered = sorted(set(numbers))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for number in ordered[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def keymap_volume_dir(image_path: Path) -> Path:
    """Volume directory holding a key map's page images.

    Full-resolution key maps live in a ``raw/`` subdirectory (``<volume>/raw/p0b.jpg``); the
    scaled per-page images used to derive the valid page set live in ``<volume>``. So the volume
    is the image's parent, or its grandparent when the image sits under ``raw/``.
    """
    parent = image_path.parent
    return parent.parent if parent.name == "raw" else parent


def valid_page_spec(keymap_images: list[Path]) -> str:
    """A ``--pages`` spec of the volume's real page numbers, for the CRNN reader.

    Unions the page numbers found across each key map's volume (from its ``p*.jpg`` page images),
    drops any non-positive number (a ``p0b`` key map's own "page 0"), and formats the result as a
    compact range spec. Returns "" when no page numbers are found.
    """
    numbers: set[int] = set()
    for volume in {keymap_volume_dir(image) for image in keymap_images}:
        numbers |= volume_page_numbers(volume)
    return format_page_spec(number for number in numbers if number >= 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare full-resolution key map(s): OCR+georef, page numbers, and regions."
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="Full-resolution key-map image(s), e.g. data/<volume>/raw/p0b.jpg.",
    )
    parser.add_argument(
        "--min-short-side",
        type=int,
        default=60,
        metavar="PX",
        help=(
            "Detector floor for the key-map OCR pass (default: %(default)s, ~4x the 25%%-scale "
            "page default of 15 to match the full-resolution key map)."
        ),
    )
    parser.add_argument(
        "--pages",
        metavar="SPEC",
        help=(
            "Valid page-number set for the CRNN reader (default: derived from the volume's page "
            "images, e.g. '1-112')."
        ),
    )
    parser.add_argument(
        "--centerlines",
        metavar="GEOJSON",
        help=(
            "Centerlines GeoJSON for the OCR/georef passes (default: a centerlines.geojson found "
            "next to the key map(s) or in the volume directory)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip the key-map OCR pass for images that already have a .streets.json output.",
    )
    parser.add_argument(
        "--reuse-boxes",
        action="store_true",
        help=(
            "Pass --reuse-boxes --allow-missing-boxes to the key-map OCR pass: reuse existing "
            "<stem>.boxes.json CRAFT boxes and run detection only for key maps without one."
        ),
    )
    args = parser.parse_args()

    images = [Path(image) for image in args.images]

    centerlines = args.centerlines
    if centerlines is None:
        found = default_centerlines(images[0].parent)
        if found is None:
            sys.exit(
                "No --centerlines given and no centerlines.geojson found next to the key map(s)."
            )
        centerlines = str(found)
        print(f"Using centerlines: {centerlines}", file=sys.stderr)

    pages = args.pages or valid_page_spec(images)
    if not pages:
        sys.exit(
            "Could not derive a --pages spec from the volume's page images; pass --pages."
        )

    image_args = [str(image) for image in images]

    # 1. Locate every page number with the CNN localizer + CRNN recognizer. Passing --pages both
    #    snaps decodes to valid page numbers and enables the narrow-detection re-read.
    #
    #    This runs first, before the georef, even though it needs nothing the georef produces:
    #    <stem>.keymap.json is what marks a page as a key map, and georef refits a key map's
    #    corners with a full 6-DOF affine. Detecting the numbers afterwards would leave the file
    #    absent at georef time, so a first run would silently get the 4-parameter similarity and
    #    only a *second* run would pick up the affine.
    run_cmd(
        [
            sys.executable,
            "-m",
            "mapsnap.keymap.detect_numbers_crnn",
            "--pages",
            pages,
            *image_args,
        ]
    )

    # 2. Georeference each key map from its own street labels, exactly like a regular page, so the
    #    downstream --keymap flag has a <stem>.georef.json to read. OCR runs at a key-map-
    #    appropriate detector floor and (by default) tiles the oversized sheet at native
    #    resolution; georef must be told to geocode key maps, which it skips by default for a page
    #    with a <stem>.keymap.json sibling. Both pass --ignore-keymap so they do not auto-discover
    #    the key map's own .keymap.json and try to locate it against itself.
    ocr_cmd = [
        "mapsnap",
        "ocr",
        "--ignore-keymap",
        "--centerlines",
        centerlines,
        "--min-short-side",
        str(args.min_short_side),
    ]
    if args.resume:
        ocr_cmd.append("--resume")
    if args.reuse_boxes:
        ocr_cmd.extend(["--reuse-boxes", "--allow-missing-boxes"])
    run_cmd([*ocr_cmd, *image_args])
    run_cmd(
        [
            "mapsnap",
            "georef",
            "--ignore-keymap",
            "--centerlines",
            centerlines,
            "--geocode_keymaps",
            *image_args,
        ]
    )

    # 3. Segment the colored block around each page number (one key map at a time).
    for image in images:
        run_cmd(
            [
                sys.executable,
                "-m",
                "mapsnap.keymap.page_regions",
                keymap_path(str(image)),
            ]
        )


if __name__ == "__main__":
    main()
