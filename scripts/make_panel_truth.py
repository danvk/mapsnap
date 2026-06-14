#!/usr/bin/env python3
"""Generate ground-truth panel polygons for the split-detection test set.

For each image in the splits directory (e.g. champaign-p20.jpg) this finds the
corresponding source files in data/<city>/ and derives the true panel polygons:

  pNNN__M.raw.jpg  - one file per split panel. A panel is stored either full-page-sized
                     with the OTHER panels masked to pure white, or as a tight rectangular
                     crop. Either way the panel's shape is its non-white region, placed at
                     the offset found by template-matching it into pNNN.unsplit.jpg.
  pNNN.raw.jpg     - a page with no __M files is an unsplit single panel (whole page).

Panels are written to <splits>/<name>.panels.json as polygons in the splits-image pixel
coordinate frame (the frame of <name>.jpg). For non-rectangular panels the polygon follows
the precise dividing boundary, since the masked-out region outside a panel is pure white
(255) while the panel's own scan paper is below WHITE_THRESHOLD.

Run from the project root:
  uv run python scripts/make_panel_truth.py [SPLITS_DIR] [--data-root DIR] [--visualize]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mapsnap.make_iiif_georef import locate_split_in_unsplit

# Maps a splits-image name prefix to its source directory under the data root.
PREFIX_TO_DIR = {
    "champaign": "champaign_ill_1915",
    "detroit": "detroit_mich_1929_vol_11",
    "nola-1896": "new_orleans_la_1896_vol_2",
    "nola": "new_orleans_la_1951_vol_5",
    "washington": "washington_dc_1916_vol_2",
}

WHITE_THRESHOLD = 250  # pixels at or above this are masked-out (not part of the panel)
CLOSE_KERNEL_PX = 25  # close small holes/noise in the panel mask before contouring
APPROX_EPS_FRAC = 0.003  # Douglas-Peucker tolerance as a fraction of contour perimeter
MIN_PANEL_AREA_FRAC = 0.03  # ignore content/masked components smaller than this
FALLBACK_MIN_SCORE = 0.70  # template-match threshold for pages with no full-page raw


def parse_splits_name(stem: str) -> tuple[str, str] | None:
    """Split a name like 'nola-1896-p101' into ('nola-1896', 'p101').

    Returns None if the stem doesn't match the '<prefix>-p<number>' pattern.
    """
    match = re.match(r"^(?P<prefix>.+)-(?P<page>p\d+[a-z]*)$", stem)
    if match is None:
        return None
    return match.group("prefix"), match.group("page")


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """Largest external contour of a binary mask as an (N, 2) [x, y] polygon, or None."""
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL_PX, CLOSE_KERNEL_PX), np.uint8)
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    eps = APPROX_EPS_FRAC * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2).astype(float)


def decompose_full_page(raw_path: Path) -> list[np.ndarray]:
    """Derive all panels from one full-page-sized masked raw, in unsplit coordinates.

    Such a raw shows one panel as scan content (below WHITE_THRESHOLD) with the other
    panels masked to pure white. Every large connected component — whether content or
    masked — is a panel, so both sides of the divide come from this single file without
    any template matching. Returns one [x, y] polygon per panel.
    """
    arr = np.array(Image.open(raw_path).convert("L"))
    total = arr.shape[0] * arr.shape[1]
    polygons = []
    for component_mask in ((arr < WHITE_THRESHOLD), (arr >= WHITE_THRESHOLD)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            component_mask.astype(np.uint8), connectivity=8
        )
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < MIN_PANEL_AREA_FRAC * total:
                continue
            poly = mask_to_polygon((labels == label).astype(np.uint8))
            if poly is not None:
                polygons.append(poly)
    return polygons


def extract_page_panels(
    panel_raws: list[Path], unsplit_path: Path, unsplit_hw: tuple[int, int]
) -> list[np.ndarray]:
    """Return all panel polygons for a page in unsplit-image coordinates.

    Prefers decomposing a full-page-sized raw (content + masked components — robust and
    template-match-free), which captures panels with no raw of their own. But the masked
    regions of two+ other panels can merge into one component and undercount, so this only
    trusts decomposition when it yields at least as many panels as there are raw files;
    otherwise it falls back to per-raw extraction (each raw's non-white region at its located
    offset), which also covers pages with no full-page raw (e.g. overlapping strips).
    """
    uh, uw = unsplit_hw
    full_page = [
        r for r in panel_raws if np.array(Image.open(r).convert("L")).shape == (uh, uw)
    ]
    if full_page:
        decomposed = decompose_full_page(full_page[0])
        if len(decomposed) >= len(panel_raws):
            return decomposed

    polygons = []
    for raw_path in panel_raws:
        if np.array(Image.open(raw_path).convert("L")).shape == (uh, uw):
            offset_x, offset_y = 0, 0
        else:
            offset_x, offset_y = locate_split_in_unsplit(
                raw_path, unsplit_path, min_score=FALLBACK_MIN_SCORE
            )
        raw = np.array(Image.open(raw_path).convert("L"))
        poly = mask_to_polygon((raw < WHITE_THRESHOLD).astype(np.uint8))
        if poly is None:
            raise ValueError(f"No panel content found in {raw_path.name}")
        poly[:, 0] += offset_x
        poly[:, 1] += offset_y
        polygons.append(poly)
    return polygons


def build_truth(splits_path: Path, data_root: Path) -> dict | None:
    """Build the panels.json content for one splits image, or None if unresolvable.

    Polygons are scaled from unsplit pixels to the splits image's pixel frame.
    """
    parsed = parse_splits_name(splits_path.stem)
    if parsed is None:
        print(f"  {splits_path.name}: unrecognized name pattern, skipping")
        return None
    prefix, page = parsed
    source_dir = data_root / PREFIX_TO_DIR.get(prefix, "")
    if prefix not in PREFIX_TO_DIR or not source_dir.is_dir():
        print(
            f"  {splits_path.name}: no source directory for prefix '{prefix}', skipping"
        )
        return None

    with Image.open(splits_path) as img:
        scaled_w, scaled_h = img.size

    panel_raws = sorted(source_dir.glob(f"{page}__*.raw.jpg"))
    unsplit_path = source_dir / f"{page}.unsplit.jpg"

    if not panel_raws:
        # No split files: treat as a single full-page panel.
        print(
            f"  {splits_path.name}: no split files → single panel (review if unexpected)"
        )
        polygons = [
            [[0.0, 0.0], [scaled_w, 0.0], [scaled_w, scaled_h], [0.0, scaled_h]]
        ]
        return {
            "image": splits_path.name,
            "width": scaled_w,
            "height": scaled_h,
            "panels": polygons,
        }

    if not unsplit_path.exists():
        print(f"  {splits_path.name}: split files but no {unsplit_path.name}, skipping")
        return None

    with Image.open(unsplit_path) as img:
        unsplit_w, unsplit_h = img.size
    scale = scaled_w / unsplit_w
    if abs(scale - scaled_h / unsplit_h) > 1e-3:
        print(f"  {splits_path.name}: non-uniform scale, skipping")
        return None

    try:
        raw_polys = extract_page_panels(
            panel_raws, unsplit_path, (unsplit_h, unsplit_w)
        )
    except ValueError as err:
        print(f"  {splits_path.name}: {err} — skipping page")
        return None
    polygons = [
        [[round(x * scale, 1), round(y * scale, 1)] for x, y in poly]
        for poly in raw_polys
    ]

    print(f"  {splits_path.name}: {len(polygons)} panel(s)")
    return {
        "image": splits_path.name,
        "width": scaled_w,
        "height": scaled_h,
        "panels": polygons,
    }


def save_visualization(splits_path: Path, truth: dict, out_dir: Path) -> None:
    """Save the splits image with truth panels overlaid for visual checking."""
    rgb = np.array(Image.open(splits_path).convert("RGB"))
    colors = [(255, 60, 60), (60, 200, 60), (60, 60, 255), (255, 200, 60)]
    for i, poly in enumerate(truth["panels"]):
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(rgb, [pts], True, colors[i % len(colors)], 4)
    out_path = out_dir / f"{splits_path.stem}.truth.png"
    Image.fromarray(rgb).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth panel polygons (panels.json) for split images."
    )
    parser.add_argument(
        "splits_dir",
        nargs="?",
        default="data/splits",
        type=Path,
        help="Directory of split test images (default: data/splits).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root holding the per-city source directories (default: data).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also save <name>.truth.png overlays in <splits_dir>_output.",
    )
    args = parser.parse_args()

    image_paths = sorted(args.splits_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {args.splits_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.splits_dir.parent / f"{args.splits_dir.name}_output"
    if args.visualize:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in image_paths:
        truth = build_truth(path, args.data_root)
        if truth is None:
            continue
        (path.parent / f"{path.stem}.panels.json").write_text(
            json.dumps(truth, indent=2)
        )
        if args.visualize:
            save_visualization(path, truth, out_dir)
        written += 1

    print(f"\nWrote {written} panels.json files.")


if __name__ == "__main__":
    main()
