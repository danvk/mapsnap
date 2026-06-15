#!/usr/bin/env python3
"""Score split detection against ground-truth panels.json via matched IoU.

For each <name>.panels.json in the splits directory, runs explore_splits.compute_panels on
<name>.jpg and compares the detected panels to the truth with a matched
intersection-over-union score:

  Each detected panel is matched to a truth panel by an optimal assignment that maximizes
  total intersection (one-to-one). The image score is sum(intersection) / sum(union) over
  the matched pairs, with unmatched panels (count mismatch) adding their full area to the
  union. 1.0 is a perfect split; missing or spurious large panels are penalized more than
  small ones.

Both detected and truth panels are in the full (uncropped) scaled-image frame.

Run from the project root:
  uv run python scripts/score_splits.py [SPLITS_DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from mapsnap import split as es


def matched_iou(truth: list[Polygon], gen: list[Polygon]) -> float:
    """Matched intersection-over-union score for one image (see module docstring)."""
    if not truth and not gen:
        return 1.0
    if not truth or not gen:
        return 0.0
    inter = np.zeros((len(truth), len(gen)))
    for i, t in enumerate(truth):
        for j, g in enumerate(gen):
            inter[i, j] = t.intersection(g).area
    rows, cols = linear_sum_assignment(inter, maximize=True)
    total_int = 0.0
    total_union = 0.0
    matched_t, matched_g = set(), set()
    for i, j in zip(rows, cols):
        if inter[i, j] <= 0:
            continue
        total_int += inter[i, j]
        total_union += truth[i].area + gen[j].area - inter[i, j]
        matched_t.add(i)
        matched_g.add(j)
    for i, t in enumerate(truth):
        if i not in matched_t:
            total_union += t.area
    for j, g in enumerate(gen):
        if j not in matched_g:
            total_union += g.area
    return total_int / total_union if total_union > 0 else 1.0


def make_valid(polygon: Polygon) -> Polygon:
    """Repair a possibly self-intersecting polygon with a zero-width buffer."""
    return polygon if polygon.is_valid else polygon.buffer(0)


def load_truth(json_path: Path, image_path: Path) -> list[Polygon]:
    """Load truth panel polygons, scaling to the actual image size if needed.

    panels.json records the frame it was generated in via width/height. If the test image
    is a different size (e.g. truth was produced from a higher-res version), coordinates
    are scaled proportionally so the metric runs in the test image's pixel frame.
    """
    data = json.loads(json_path.read_text())
    truth_w, truth_h = data["width"], data["height"]
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    sx = img_w / truth_w
    sy = img_h / truth_h
    polys = []
    for ring in data["panels"]:
        scaled = [[x * sx, y * sy] for x, y in ring]
        polys.append(make_valid(Polygon(scaled)))
    return polys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score split detection against panels.json ground truth."
    )
    parser.add_argument(
        "splits_dir",
        nargs="?",
        default="data/splits",
        type=Path,
        help="Directory of split images and their .panels.json truth (default: data/splits).",
    )
    args = parser.parse_args()

    truth_files = sorted(args.splits_dir.glob("*.panels.json"))
    if not truth_files:
        print(f"No .panels.json files in {args.splits_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{'image':18s}{'truth':>7s}{'detected':>10s}{'IoU':>8s}")
    scores = []
    for truth_path in truth_files:
        name = truth_path.name.removesuffix(".panels.json")
        image_path = args.splits_dir / f"{name}.jpg"
        if not image_path.exists():
            print(f"{name:18s}  (no image {image_path.name})")
            continue
        truth = load_truth(truth_path, image_path)
        gen = [make_valid(p) for p in es.compute_panels(image_path)]
        score = matched_iou(truth, gen)
        scores.append(score)
        print(f"{name:18s}{len(truth):>7d}{len(gen):>10d}{score:>8.3f}")

    if scores:
        print(f"\nmean IoU over {len(scores)} images: {np.mean(scores):.3f}")


if __name__ == "__main__":
    main()
