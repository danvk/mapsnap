#!/usr/bin/env python3
"""Run the key-map corner-box detector over every key-map candidate sheet.

For each unsplit page-0 / page-1 / letter sheet under data/ (the sheets
split.is_keymap_sheet nominates), run the splitter's front-end, hand the
uncapped box-thick candidates to mapsnap.corner_boxes, and print what it finds:
the corner, share of the sheet, and how the box was closed. Pass --draw DIR to
write a thumbnail per sheet with a box, candidates in blue and boxes in red.

The precision census for #276 step 5 (2026-09-02): 9 boxes on 70 sheets, all
real boxed regions -- Detroit's volume-index inset, Ellenville's Napanoch,
Kingston's Port Ewen, Los Angeles pa's Vol-13 continuation, the KEY legends of
Schenectady (both editions) and New York 1905, and the Kansas City and New
Orleans 1951 insets the divider pipeline already cuts. Rerun after touching the
detector's rules; a new box on any other sheet needs looking at.

    uv run python scripts/corner_box_sweep.py [--draw DIR] [data/<volume>/p0.jpg ...]
"""

import argparse
import glob
from pathlib import Path

from PIL import Image, ImageDraw

from mapsnap import corner_boxes as cb
from mapsnap import split as sp


def candidate_sheets() -> list[Path]:
    """Every unsplit key-map candidate sheet under data/."""
    return sorted(
        Path(path)
        for path in glob.glob("data/*/p*.jpg")
        if "__" not in Path(path).stem and sp.is_keymap_sheet(Path(path).stem)
    )


def sweep_sheet(image_path: Path, draw_dir: Path | None) -> list[cb.BoxCandidate]:
    """Detect corner boxes on one sheet, printing and optionally drawing them."""
    rgb = sp.crop_border(sp.load_rgb(image_path))
    gray = sp.crop_border(sp.load_gray(image_path))
    h, w = gray.shape
    binary = sp.binarize(rgb, gray)
    candidates = sp.box_candidates(sp.detect_lines(binary), h, w, binary)
    boxes = cb.corner_boxes(candidates, binary, h, w)
    key = f"{image_path.parent.name}/{image_path.stem}"
    print(f"{key:45s} candidates={len(candidates):3d} boxes={len(boxes)}")
    for candidate in boxes:
        share = candidate.area / (h * w)
        print(f"    {'-'.join(candidate.corner)} {share:.1%}: {candidate.detail}")
    if draw_dir is not None and boxes:
        image = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(image)
        for x0, y0, x1, y1 in candidates:
            draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 255), width=4)
        for candidate in boxes:
            ring = [(x, y) for x, y in candidate.polygon.exterior.coords]
            draw.polygon(ring, outline=(255, 0, 0), width=8)
        image.thumbnail((900, 900))
        draw_dir.mkdir(parents=True, exist_ok=True)
        image.save(draw_dir / f"{key.replace('/', '__')}.jpg", quality=75)
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the key-map corner-box detector over key-map candidate sheets."
    )
    parser.add_argument("images", nargs="*", type=Path, help="sheets (default: all)")
    parser.add_argument("--draw", type=Path, help="write thumbnails with boxes here")
    args = parser.parse_args()
    sheets = args.images or candidate_sheets()
    total = 0
    for image_path in sheets:
        total += len(sweep_sheet(image_path, args.draw))
    print(f"{total} box(es) on {len(sheets)} sheet(s)")


if __name__ == "__main__":
    main()
