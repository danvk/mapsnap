"""Identify which page(s) of a volume are key maps, from the 25%-scale page images.

A key map is, by definition, an index of the whole volume: one colored block per page with its
page number printed inside. So its printed numbers, read back, reconstruct the volume's own page
set — and that is what tells a key map apart from a regular map page. Detection *count* does not:
a dense downtown page can carry more number-like glyphs (lot and dimension numbers) than the key
map. But *coverage* of the volume's page set separates them cleanly (measured across four volumes:
key maps recover 88-96% of their pages; regular pages <=4%).

Two stages:

  * candidates (cheap, filenames only) — the key map is always page 0 or page 1, including
    lettered/split variants (p0, p0b, p0L/p0R, p1a-d). candidate_keys nominates the page-0 and
    page-1 families, no model needed.
  * confirmation (content) — read each candidate's numbers with the CNN localizer + CRNN
    recognizer, snap to the valid page set, and keep candidates whose distinct valid reads cover a
    large fraction of the volume (see is_keymap). Handles split key maps (each half still covers a
    large share) and rejects number-dense regular pages.

Everything runs on the always-present ``<volume>/p*.jpg``: 25% scale is the CNN's native working
resolution (keymap_patches.working_scale uses it as-is), so no full-resolution download is needed
to decide which pages to fetch and process as key maps.

    uv run mapsnap keymap-detect data/chicago_il_1950_vol_1
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapsnap.keymap.crnn_model import build_crnn, central_group, ctc_greedy_decode
from mapsnap.keymap.detect_numbers_cnn import (
    DEFAULT_NMS_DIST,
    DEFAULT_STRIDE,
    DEFAULT_THRESHOLD,
    detect_candidate_centers,
)
from mapsnap.keymap.detect_numbers_crnn import read_candidates, snap_to_pages
from mapsnap.keymap.fit_keymap import page_number, volume_page_numbers
from mapsnap.keymap.number_model import build_model, select_device
from mapsnap.utils import image_stem

DEFAULT_CNN_WEIGHTS = Path("models/number_detector.pt")
DEFAULT_CRNN_WEIGHTS = Path("models/number_crnn.pt")

# A candidate is confirmed a key map when its distinct valid page reads cover at least
# MIN_COVERAGE of the volume's pages and number at least MIN_DISTINCT. The observed gap is huge
# (key maps >=0.88, regular pages <=0.04), so 0.3 leaves room for a split key map (~0.45 each half)
# while never admitting a regular page; MIN_DISTINCT guards tiny volumes where the ratio is coarse.
MIN_COVERAGE = 0.3
MIN_DISTINCT = 6

# The key map is always page 0 or page 1 (its lettered/split siblings share that number); the
# real map pages may begin far higher (e.g. p125), so we nominate by absolute number, not rank.
CANDIDATE_PAGE_NUMBERS = (0, 1)


def volume_valid_pages(volume: Path) -> list[str]:
    """The volume's real page numbers as strings (positive, from its ``p*.jpg`` images)."""
    return [
        str(number) for number in sorted(volume_page_numbers(volume)) if number >= 1
    ]


def candidate_keys(volume: Path) -> list[str]:
    """Page keys to test as key maps: every page numbered 0 or 1 (CANDIDATE_PAGE_NUMBERS).

    The key map is always page 0 or page 1, and shares its number with any lettered/split siblings
    (p0/p0b/p0L/p0R, p1a-d); nominating both families covers a page-0 cover sitting in front of a
    page-1 key map while staying a handful of pages. Numbering by absolute value (not rank) avoids
    dragging in the first real map page when it starts high, e.g. p125. Split panels (stems with
    ``__``) are never key maps and are skipped.
    """
    by_number: dict[int, list[str]] = defaultdict(list)
    for image in sorted(volume.glob("p*.jpg")):
        stem = image_stem(str(image))
        if "__" in stem:
            continue
        number = page_number(stem)
        if number is not None and number in CANDIDATE_PAGE_NUMBERS:
            by_number[number].append(stem)
    keys: list[str] = []
    for number in sorted(by_number):
        keys.extend(by_number[number])
    return keys


def is_keymap(
    distinct_valid: int,
    volume_pages: int,
    *,
    min_coverage: float = MIN_COVERAGE,
    min_distinct: int = MIN_DISTINCT,
) -> bool:
    """True when ``distinct_valid`` page reads cover enough of a ``volume_pages``-page volume.

    A key map's numbers reconstruct the volume's page set (high coverage); a regular page yields
    only a few coincidental valid reads. Requires both a coverage fraction and an absolute floor so
    a tiny volume cannot pass on one lucky read.
    """
    if volume_pages <= 0:
        return False
    coverage = distinct_valid / volume_pages
    return distinct_valid >= min_distinct and coverage >= min_coverage


@torch.no_grad()
def read_valid_pages(
    image_path: str,
    valid_pages: list[str],
    cnn: torch.nn.Module,
    crnn: torch.nn.Module,
    device,
) -> set[str]:
    """Distinct read numbers on ``image_path`` that snap to a valid volume page.

    Runs the same CNN-localize then CRNN-read as the key-map detector, but in memory (no sidecar
    written) and keeping only the central number of each crop snapped to ``valid_pages``.
    """
    image = np.asarray(Image.open(image_path).convert("RGB"))
    centers, factor = detect_candidate_centers(
        image,
        cnn,
        device,
        stride=DEFAULT_STRIDE,
        threshold=DEFAULT_THRESHOLD,
        nms_dist=DEFAULT_NMS_DIST,
    )
    reads, _ = read_candidates(image, centers, factor, crnn, device)
    valid = set(valid_pages)
    found: set[str] = set()
    for _, path in reads:
        group = central_group(path)
        if group is None:
            continue
        text = snap_to_pages(
            ctc_greedy_decode(path[group[0] : group[1] + 1]), valid_pages
        )
        if text in valid:
            found.add(text)
    return found


def load_models(
    cnn_weights: Path, crnn_weights: Path
) -> tuple[torch.nn.Module, torch.nn.Module, object]:
    """Load the CNN localizer and CRNN recognizer onto the selected device."""
    device = select_device()
    cnn = build_model(pretrained=False)
    cnn.load_state_dict(torch.load(cnn_weights, map_location=device))
    cnn.to(device)
    crnn = build_crnn()
    crnn.load_state_dict(torch.load(crnn_weights, map_location=device))
    crnn.to(device)
    return cnn, crnn, device


def identify_keymaps(
    volume: Path,
    *,
    scan_all: bool = False,
    min_coverage: float = MIN_COVERAGE,
    min_distinct: int = MIN_DISTINCT,
    cnn_weights: Path = DEFAULT_CNN_WEIGHTS,
    crnn_weights: Path = DEFAULT_CRNN_WEIGHTS,
) -> list[str]:
    """Page keys of ``volume`` that are key maps (candidate generation + coverage confirmation).

    By default only the low-numbered candidate pages are tested (candidate_keys); ``scan_all``
    tests every non-split page image, a slower fallback for a volume whose key map is not in the
    front matter. Returns the confirmed keys sorted by page number.
    """
    valid_pages = volume_valid_pages(volume)
    if scan_all:
        keys = [
            image_stem(str(image))
            for image in sorted(volume.glob("p*.jpg"))
            if "__" not in image_stem(str(image))
        ]
    else:
        keys = candidate_keys(volume)

    cnn, crnn, device = load_models(cnn_weights, crnn_weights)
    confirmed: list[str] = []
    for key in keys:
        image = volume / f"{key}.jpg"
        if not image.exists():
            continue
        found = read_valid_pages(str(image), valid_pages, cnn, crnn, device)
        coverage = len(found) / len(valid_pages) if valid_pages else 0.0
        keymap = is_keymap(
            len(found),
            len(valid_pages),
            min_coverage=min_coverage,
            min_distinct=min_distinct,
        )
        print(
            f"  {key:8s} distinct-valid={len(found):4d}/{len(valid_pages)} "
            f"coverage={coverage:5.2f} {'KEY MAP' if keymap else ''}",
            file=sys.stderr,
        )
        if keymap:
            confirmed.append(key)
    return sorted(confirmed, key=lambda key: (page_number(key) or 0, key))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify which page(s) of a volume are key maps (from 25%-scale images)."
    )
    parser.add_argument(
        "volume", type=Path, help="Volume directory holding p*.jpg page images."
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Test every page, not just the low-numbered candidates (slower fallback).",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=MIN_COVERAGE, metavar="FRAC"
    )
    parser.add_argument("--min-distinct", type=int, default=MIN_DISTINCT, metavar="N")
    parser.add_argument("--cnn-weights", type=Path, default=DEFAULT_CNN_WEIGHTS)
    parser.add_argument("--crnn-weights", type=Path, default=DEFAULT_CRNN_WEIGHTS)
    args = parser.parse_args()

    keys = identify_keymaps(
        args.volume,
        scan_all=args.scan_all,
        min_coverage=args.min_coverage,
        min_distinct=args.min_distinct,
        cnn_weights=args.cnn_weights,
        crnn_weights=args.crnn_weights,
    )
    if not keys:
        print(f"No key map identified in {args.volume}.", file=sys.stderr)
        sys.exit(1)
    print(" ".join(keys))


if __name__ == "__main__":
    main()
