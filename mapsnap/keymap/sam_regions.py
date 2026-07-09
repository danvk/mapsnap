"""Segment key-map page regions with SAM, prompted by the detected page numbers.

The colour-cluster segmentation (mapsnap.keymap.page_regions) depends on block fills being
distinguishable from the paper; on pale, aged key maps (Hudson County) it under-segments
badly — regions come out at a median 0.58x their true scale with a long degenerate tail, and
some blocks vanish entirely. SAM needs no colour separation: each CNN-detected page number
is a positive point prompt sitting on its block, and every *other* page number is a natural
negative prompt — a mask that would swallow a neighbouring block must cross that block's
seed. The model's boundary prior does the rest (the heavy black block outlines are exactly
the edges it was trained to respect).

Writes the same ``<stem>.regions.panels.json`` schema as page_regions, so the debugger,
KeymapLocator, and the scale-prior machinery consume it unchanged.

Requires the SAM ViT-B checkpoint (not committed; ~375 MB):

    curl -L -o models/sam_vit_b_01ec64.pth \\
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    uv run python -m mapsnap.keymap.sam_regions data/hudson_co_nj_1950_vol_9/raw/p0.keymap.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from mapsnap.keymap.page_regions import (
    box_center,
    load_seeds,
    mask_to_polygon,
    nearest_neighbor_distance,
    regions_panels_doc,
    render_overlay,
)

Point = tuple[float, float]

DEFAULT_CHECKPOINT = Path("models/sam_vit_b_01ec64.pth")

# The segmentation runs on a copy downscaled to this long side (SAM resizes to 1024
# internally; 2048 keeps the returned masks crisp enough for clean polygons).
TARGET_LONG_SIDE = 2048

# Negative prompts per seed: enough nearby blocks to fence the mask in without drowning the
# positive point.
NUM_NEGATIVES = 8

# A mask is rejected as degenerate/runaway when its area falls outside these multiples of
# the median-spacing-squared (a proxy for the typical block area). The upper bound leaves
# room for giant waterfront sheets (~6x linear = ~36x area) while rejecting paper floods.
MIN_AREA_FACTOR = 0.05
MAX_AREA_FACTOR = 60.0


def nearest_negatives(
    center: Point, others: list[Point], count: int = NUM_NEGATIVES
) -> list[Point]:
    """The ``count`` seed centers nearest to ``center`` (excluding itself)."""
    ordered = sorted(
        (p for p in others if p != center),
        key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2,
    )
    return ordered[:count]


def segment_seed(
    predictor,
    center: Point,
    negatives: list[Point],
    area_bounds: tuple[float, float],
) -> np.ndarray | None:
    """The best SAM mask for one page-number seed, or None if all candidates fail sanity.

    Prompts with the seed as positive and its nearest neighbours as negatives; of SAM's three
    candidate masks, keeps the highest-scoring one whose area falls inside ``area_bounds``.
    """
    points = np.array([center, *negatives], dtype=np.float32)
    labels = np.array([1] + [0] * len(negatives), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels, multimask_output=True
    )
    best: np.ndarray | None = None
    best_score = -np.inf
    for mask, score in zip(masks, scores):
        area = float(mask.sum())
        if not (area_bounds[0] <= area <= area_bounds[1]):
            continue
        if score > best_score:
            best, best_score = mask, float(score)
    return best


def segment_page_regions_sam(
    image: np.ndarray,
    seeds: list[tuple[float, float, float, float]],
    predictor,
    scale: float,
) -> dict[int, list[Point]]:
    """Polygon (full-res pixels) of the block around each seed, via point-prompted SAM.

    ``image`` is the downscaled RGB array the predictor was set with; ``seeds`` are the
    page-number boxes in full-resolution coordinates and ``scale`` the downscale factor
    applied to them. Returns seed index -> polygon in full-resolution coordinates, omitting
    seeds whose masks fail the area sanity bounds.
    """
    centers = [
        (cx * scale, cy * scale) for cx, cy in (box_center(box) for box in seeds)
    ]
    spacing = nearest_neighbor_distance(centers)
    if spacing == 0.0:
        spacing = max(image.shape[:2]) / 10
    area_bounds = (
        MIN_AREA_FACTOR * spacing**2,
        MAX_AREA_FACTOR * spacing**2,
    )
    polygons: dict[int, list[Point]] = {}
    for index, center in enumerate(centers):
        mask = segment_seed(
            predictor, center, nearest_negatives(center, centers), area_bounds
        )
        if mask is None:
            continue
        polygon = mask_to_polygon(mask, simplify_tolerance=2.0)
        if len(polygon) >= 3:
            polygons[index] = [(x / scale, y / scale) for x, y in polygon]
    return polygons


def keymap_image_path(keymap_path: Path) -> Path:
    """Sibling JPEG of a ``<stem>.keymap.json``."""
    name = keymap_path.name
    suffix = ".keymap.json"
    stem = name[: -len(suffix)] if name.endswith(suffix) else keymap_path.stem
    return keymap_path.with_name(stem + ".jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "keymap", type=Path, help="<stem>.keymap.json with page-number detections."
    )
    parser.add_argument(
        "--image", type=Path, help="Key-map JPEG (default: sibling of the JSON)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Panels JSON output (default: <stem>.regions.panels.json).",
    )
    parser.add_argument(
        "--overlay", type=Path, help="Also write a labelled overlay PNG to this path."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="SAM ViT-B checkpoint path (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "torch device (default: cpu — one image encode plus a mask decode per page "
            "number is fast enough without a GPU)."
        ),
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(
            f"SAM checkpoint not found at {args.checkpoint}; download it with:\n"
            "  curl -L -o models/sam_vit_b_01ec64.pth "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )
    from segment_anything import SamPredictor, sam_model_registry

    image_path = args.image or keymap_image_path(args.keymap)
    output = args.output or args.keymap.with_name(
        image_path.stem + ".regions.panels.json"
    )
    seeds, texts = load_seeds(args.keymap)

    full = Image.open(image_path).convert("RGB")
    scale = min(1.0, TARGET_LONG_SIDE / max(full.size))
    small = full.resize(
        (round(full.width * scale), round(full.height * scale)),
        Image.Resampling.LANCZOS,
    )
    print(
        f"SAM on {image_path.name} at {small.size[0]}x{small.size[1]} "
        f"({len(seeds)} seeds, device={args.device})...",
        file=sys.stderr,
    )
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    sam.to(args.device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.asarray(small))

    polygons = segment_page_regions_sam(np.asarray(small), seeds, predictor, scale)
    doc = regions_panels_doc(image_path.name, full.size, polygons, texts)
    output.write_text(json.dumps(doc))
    print(f"Wrote {output}: {len(polygons)}/{len(seeds)} regions found.")
    if args.overlay:
        render_overlay(image_path, polygons, texts, args.overlay)
        print(f"Wrote {args.overlay}")


if __name__ == "__main__":
    main()
