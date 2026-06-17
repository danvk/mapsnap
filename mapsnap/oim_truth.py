"""Locate OIM's manually-generated split regions on their parent canvas (ground truth).

OIM publishes split map pages as separate IIIF annotations whose SVG selectors fuse
splitting and content masking, so they do not expose a clean split rectangle. This command
template-matches each downloaded OIM split region (oim/pN__i.jpg) against the full-resolution
unsplit page (raw/pN.jpg) to recover the split's rectangular placement on the parent canvas,
then writes oim/pN.panels.json with the regions as rings in canvas (full-resolution) pixel
coordinates. The mapsnap compare command uses these regions to associate OIM's splits with
our own, whose numbering need not agree.

Usage:
    mapsnap oim-split-truth <main.iiif.json>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mapsnap.split import PanelsJson, panels_json_path
from mapsnap.utils import label_to_page_key


WHITE_THRESHOLD = 250  # pixels at or above this are masked-out (not part of the panel)


def _masked_match(
    image: np.ndarray, template: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Masked normalized cross-correlation of template (with mask) over image.

    Pure-white masked-out pixels are excluded from the correlation entirely, rather
    than merely zeroed, so a non-rectangular panel matches its true location even when
    the unsplit page has unrelated content in the masked area. Returns the score map
    with any NaN/inf (from degenerate, fully-masked windows) replaced by 0.
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED, mask=mask)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def locate_split_in_unsplit(
    split_path: Path,
    unsplit_path: Path,
    min_score: float = 0.90,
    coarse_downsample: int = 4,
    refine_margin: int = 20,
) -> tuple[int, int]:
    """Find the pixel offset of a split sub-image within its unsplit original.

    Uses a two-stage search: coarse masked normalized cross-correlation at downsampled
    resolution to find the rough location, then full-resolution refinement in a small
    window around that estimate. Pure-white pixels in the split are masked-out regions
    that share no content with the unsplit page, so they are excluded from the match via
    an OpenCV mask (without this, the unsplit content under those pixels suppresses the
    correlation score of an otherwise correct match).

    Raises ValueError if the refined match score is below min_score (uncertain
    placement) or if the located region overflows the unsplit image bounds.

    Returns (offset_x, offset_y) in unsplit image pixel coordinates (top-left corner).
    """
    split_arr = np.array(Image.open(split_path).convert("L"), dtype=np.uint8)
    unsplit_arr = np.array(Image.open(unsplit_path).convert("L"), dtype=np.uint8)
    mask = ((split_arr < WHITE_THRESHOLD).astype(np.uint8)) * 255

    ds = coarse_downsample
    split_small = split_arr[::ds, ::ds]
    mask_small = mask[::ds, ::ds]
    unsplit_small = unsplit_arr[::ds, ::ds]

    if (
        split_small.shape[0] > unsplit_small.shape[0]
        or split_small.shape[1] > unsplit_small.shape[1]
    ):
        raise ValueError(
            f"Split ({split_arr.shape[1]}×{split_arr.shape[0]}) is larger than "
            f"unsplit ({unsplit_arr.shape[1]}×{unsplit_arr.shape[0]}) at coarse resolution"
        )

    coarse_result = _masked_match(unsplit_small, split_small, mask_small)
    _, _, _, coarse_loc = cv2.minMaxLoc(coarse_result)
    coarse_x = coarse_loc[0] * ds
    coarse_y = coarse_loc[1] * ds

    # Refine at full resolution within a small window around the coarse estimate.
    sh, sw = split_arr.shape
    uh, uw = unsplit_arr.shape
    x1 = max(0, coarse_x - refine_margin)
    x2 = min(uw, coarse_x + sw + refine_margin)
    y1 = max(0, coarse_y - refine_margin)
    y2 = min(uh, coarse_y + sh + refine_margin)

    if x2 - x1 < sw or y2 - y1 < sh:
        raise ValueError(
            f"Refinement window too small ({x2 - x1}×{y2 - y1}) for template ({sw}×{sh})"
        )

    region = unsplit_arr[y1:y2, x1:x2]
    result = _masked_match(region, split_arr, mask)
    _, max_score, _, max_loc = cv2.minMaxLoc(result)

    if max_score < min_score:
        raise ValueError(
            f"Template match score {max_score:.3f} < threshold {min_score} "
            f"({split_path.name} in {unsplit_path.name})"
        )

    offset_x = x1 + max_loc[0]
    offset_y = y1 + max_loc[1]

    if offset_x + sw > uw or offset_y + sh > uh:
        raise ValueError(
            f"Located position ({offset_x}, {offset_y}) + {sw}×{sh} overflows "
            f"unsplit bounds {uw}×{uh}"
        )

    return offset_x, offset_y


def locate_oim_splits(
    parent_key: str,
    split_indices: list[int],
    raw_dir: Path,
    oim_dir: Path,
) -> PanelsJson:
    """Build a canvas-coordinate panels.json for one page's OIM split regions.

    Template-matches each OIM split region oim/<parent>__<i>.jpg against the unsplit
    full-resolution page raw/<parent>.jpg, recording the located rectangle as a polygon
    ring in canvas (full-resolution) pixel coordinates. Panels are ordered by OIM split
    index. Raises ValueError if a required image is missing or a match is too uncertain.
    """
    unsplit_path = raw_dir / f"{parent_key}.jpg"
    if not unsplit_path.exists():
        raise ValueError(f"unsplit page not found: {unsplit_path}")
    unsplit_arr = np.array(Image.open(unsplit_path).convert("L"))
    canvas_height, canvas_width = unsplit_arr.shape

    rings: list[list[list[float]]] = []
    for i in sorted(split_indices):
        split_path = oim_dir / f"{parent_key}__{i}.jpg"
        if not split_path.exists():
            raise ValueError(f"OIM split region not found: {split_path}")
        offset_x, offset_y = locate_split_in_unsplit(split_path, unsplit_path)
        split_w, split_h = Image.open(split_path).size
        x0, y0 = float(offset_x), float(offset_y)
        x1, y1 = float(offset_x + split_w), float(offset_y + split_h)
        rings.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])

    return {
        "image": f"{parent_key}.jpg",
        "width": canvas_width,
        "height": canvas_height,
        "panels": rings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Locate OIM's manually-generated split regions on their parent canvas, "
            "writing oim/pN.panels.json (rings in full-resolution canvas coordinates)."
        )
    )
    parser.add_argument(
        "iiif_file", metavar="FILE", help="Local OIM IIIF AnnotationPage JSON file"
    )
    args = parser.parse_args()

    iiif_path = Path(args.iiif_file)
    base_dir = iiif_path.parent
    raw_dir = base_dir / "raw"
    oim_dir = base_dir / "oim"
    data: dict = json.load(iiif_path.open())

    # Group split indices by parent page key.
    splits_by_parent: dict[str, list[int]] = defaultdict(list)
    for item in data.get("items", []):
        page_key = label_to_page_key(item.get("label", ""))
        if page_key is None or "__" not in page_key:
            continue
        parent_key, split_str = page_key.split("__")
        splits_by_parent[parent_key].append(int(split_str))

    if not splits_by_parent:
        print("No split pages found in IIIF file.", file=sys.stderr)
        return

    written = 0
    for parent_key, split_indices in sorted(splits_by_parent.items()):
        try:
            panels = locate_oim_splits(parent_key, split_indices, raw_dir, oim_dir)
        except ValueError as exc:
            print(f"Warning: skipping {parent_key}: {exc}", file=sys.stderr)
            continue
        out_path = panels_json_path(oim_dir / f"{parent_key}.jpg")
        out_path.write_text(json.dumps(panels, indent=2))
        print(
            f"{parent_key}: located {len(panels['panels'])} OIM split(s) → {out_path.name}"
        )
        written += 1

    print(
        f"\nWrote {written}/{len(splits_by_parent)} OIM truth panels.json files.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
