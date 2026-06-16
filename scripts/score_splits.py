#!/usr/bin/env python3
"""Score split detection against ground-truth panels.json via matched IoU.

Runs mapsnap.split.compute_panels on each given image and compares the detected panels to
ground truth with a matched intersection-over-union score:

  Each detected panel is matched to a truth panel by an optimal assignment that maximizes
  total intersection (one-to-one). The image score is sum(intersection) / sum(union) over
  the matched pairs, with unmatched panels (count mismatch) adding their full area to the
  union. 1.0 is a perfect split; missing or spurious large panels are penalized more than
  small ones.

Images and truth files are paired by file-name stem before the first dot, so p217.raw.jpg
pairs with p217.scaled.panels.json. An image with no truth file is scored as a single panel.
Truth records its own width/height, so it is scaled to the image's pixel frame as needed.

Run from the project root, e.g.:
  uv run python scripts/score_splits.py --truth_dir testdata/splits/<volume> <images-glob>
"""

import argparse
import json
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
    return float(total_int / total_union) if total_union > 0 else 1.0


def make_valid(polygon: Polygon) -> Polygon:
    """Repair a possibly self-intersecting polygon with a zero-width buffer."""
    return polygon if polygon.is_valid else polygon.buffer(0)


def stem_key(path: Path) -> str:
    """Pairing key: the file name up to the first dot (p217.raw.jpg → p217)."""
    return path.name.split(".", 1)[0]


def single_panel_truth(image_path: Path) -> list[Polygon]:
    """Truth for a page with no truth file: one panel covering the whole image."""
    with Image.open(image_path) as img:
        w, h = img.size
    return [Polygon([(0, 0), (w, 0), (w, h), (0, h)])]


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
        "images",
        nargs="+",
        type=Path,
        metavar="IMAGE",
        help="Image files to split and score (e.g. a shell glob).",
    )
    parser.add_argument(
        "--truth_dir",
        required=True,
        type=Path,
        help="Directory of <stem>.panels.json truth files, paired with images by the "
        "file-name stem before the first dot. An image with no truth file scores as "
        "a single panel.",
    )
    args = parser.parse_args()

    truth_by_key = {
        stem_key(f): f for f in sorted(args.truth_dir.glob("*.panels.json"))
    }

    print(f"{'image':18s}{'truth':>7s}{'detected':>10s}{'IoU':>8s}")
    scores = []
    for image_path in args.images:
        truth_path = truth_by_key.get(stem_key(image_path))
        truth = (
            load_truth(truth_path, image_path)
            if truth_path is not None
            else single_panel_truth(image_path)
        )
        gen = [make_valid(p) for p in es.compute_panels(image_path)]
        score = matched_iou(truth, gen)
        scores.append(score)
        print(f"{stem_key(image_path):18s}{len(truth):>7d}{len(gen):>10d}{score:>8.3f}")

    if scores:
        print(f"\nmean IoU over {len(scores)} images: {np.mean(scores):.3f}")


if __name__ == "__main__":
    main()
