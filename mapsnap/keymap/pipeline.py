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

Between the first and the rest, the page-number assignments can be repaired against the volume's
printed adjacency graph (``mapsnap.keymap.adjacency_assign``, issue #213): a number misread as a
shorter one and a page whose number never read at all are both settled by which pages cite it in
their margins. With ``--repair-assignments`` the repaired detections BECOME
``<stem>.keymap.json`` (the original is kept as ``<stem>.keymap-raw.json``), so the corrections
flow into the region segmentation below and into every downstream consumer of the key map.

**Off by default** (issue #239). The repair's gap-fills invent a key-map location for a page
whose number was never read, and every consumer treats that guess as if it had been detected:
``ocr`` restricts the page's vocabulary to streets near it, and ``page_regions`` seeds a block
around it for ``snap`` to target. Measured end to end, turning the repairs off was worth
**+7.6 points on Grand Rapids and +6.9 on Nashville**, both landing above their pre-feature
baselines -- the vocabulary restriction alone discarded 794 correct street labels on Grand
Rapids. The pass still runs and reports what it would have changed, so its accuracy stays
visible while it is being improved.

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

from mapsnap.keymap.adjacency_assign import repair_volume
from mapsnap.keymap.fit_keymap import collapse_skeleton_keys, volume_page_keys
from mapsnap.keymap.records import keymap_path, page_key_sort
from mapsnap.utils import default_centerlines, run_cmd


def format_page_spec(keys: Iterable[int | str]) -> str:
    """Collapse page keys into a compact spec like ``"1-3,5,7-9,33A"`` (a parse_page_spec input).

    Bare numbers collapse into ranges; letter-suffixed keys are emitted as
    individual tokens after the ranges.
    """
    numbers: set[int] = set()
    lettered: set[str] = set()
    for key in keys:
        text = str(key)
        if text.isdigit():
            numbers.add(int(text))
        else:
            lettered.add(text.upper())
    ordered = sorted(numbers)
    ranges: list[str] = []
    if ordered:
        start = previous = ordered[0]
        for number in ordered[1:]:
            if number == previous + 1:
                previous = number
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = number
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
    ranges.extend(sorted(lettered, key=page_key_sort))
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
    """A ``--pages`` spec of the volume's real page keys, for the CRNN reader.

    Unions the page keys found across each key map's volume (from its ``p*.jpg`` page
    images, letter suffixes included), drops page 0 and its variants (a ``p0b`` key
    map's own sheet), and formats the result as a compact spec. Returns "" when no
    page keys are found.
    """
    keys: set[str] = set()
    for volume in {keymap_volume_dir(image) for image in keymap_images}:
        keys |= collapse_skeleton_keys(volume_page_keys(volume))
    return format_page_spec(key for key in keys if page_key_sort(key)[0] >= 1)


def build_parser() -> argparse.ArgumentParser:
    """The `mapsnap keymap` argument parser, split out so flags are testable."""
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
        "--repair-assignments",
        action="store_true",
        help=(
            "Apply the adjacency-driven page-number repairs (#213) to the "
            "detected numbers. Off by default: the repairs invent key-map "
            "locations for pages that were never read, and those guesses then "
            "drive OCR vocabulary restriction and region seeding, which cost "
            "more than the repairs gain (see #239). The pass is still planned "
            "and reported so its quality can be judged."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report the adjacency-driven page-number repairs and stop, writing "
            "nothing (the detection pass still runs)."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    images = [Path(image) for image in args.images]

    centerlines = args.centerlines
    if centerlines is None:
        # Search from the VOLUME directory, not the key map's own (``raw/``)
        # one: default_centerlines checks a directory and its parent, so
        # starting at raw/ reaches only the volume, and a centerlines.geojson
        # shared by several volumes of one set (brooklyn_1904-1908/vol13/raw ->
        # brooklyn_1904-1908) would be missed.
        found = default_centerlines(keymap_volume_dir(images[0]))
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

    # 2. Repair the page-number assignments against the printed adjacency graph
    #    (#213): a number misread as a shorter one (Detroit's 22 read as "2"),
    #    and a page whose number never read at all, are both settled by which
    #    pages cite it in their margins. Runs here, before the regions are
    #    segmented, so corrected numbers SEED the segmentation -- which is also
    #    why a wrong one is expensive, and why applying them is opt-in (#239).
    #    It reads only <stem>.keymap.json, so a poor segmentation can never
    #    corrupt a page number. A volume with no adjacency.json is a no-op.
    volume = keymap_volume_dir(images[0])
    # Planned either way, applied only when asked. Reporting the repairs a run
    # would have made keeps them visible -- and comparable against the detected
    # numbers -- while they are being improved.
    applying = args.repair_assignments and not args.dry_run
    repairs = repair_volume(volume, dry_run=not applying)
    for repair in repairs:
        print(f"  assignment repair: {repair.describe()}", file=sys.stderr)
    if applying:
        note = ""
    elif args.dry_run:
        note = " (dry run, nothing written)"
    else:
        note = " NOT applied (pass --repair-assignments to apply; see #239)"
    print(f"{len(repairs)} assignment repair(s){note}", file=sys.stderr)
    if args.dry_run:
        return

    # 3. Georeference each key map from its own street labels, exactly like a regular page, so the
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

    # 4. Segment the colored block around each page number (one key map at a time).
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
