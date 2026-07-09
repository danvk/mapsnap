"""Score a key-map page-region segmentation against projected-truth footprints (IoU).

Compares a ``<stem>.regions.panels.json`` (from page_regions' k-means or sam_regions) with
the ``<stem>.truth.regions.panels.json`` written by mapsnap.keymap.truth_regions. Panels are
matched by page-number label; when a page has several footprints/blocks (splits), pairs are
associated greedily by greatest intersection. Truth footprints with no matching prediction
score 0, so the summary reflects misses as well as shape quality.

Also reports how much the *predicted* regions overlap one another — page blocks tile the key
map, so predicted overlap is impossible ink and a direct measure of segmentation nonsense.

    uv run python -m mapsnap.keymap.score_regions data/<vol>/raw/p0.regions.panels.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


def default_truth_path(regions: Path) -> Path:
    """The truth file for a predictions file: ``<page-stem>.truth.regions.panels.json``.

    Derived from the page stem (the file name up to the first dot), so it works for any
    sidecar naming (``p0.regions.panels.json``, ``p0.kmeans2.panels.json``, ...) — a naive
    suffix replace silently yields the predictions path itself for unconventional names,
    and scoring a file against itself reports a perfect 1.000.
    """
    stem = regions.name.split(".")[0]
    return regions.with_name(stem + ".truth.regions.panels.json")


def panel_polygons(doc: dict) -> list[tuple[str, BaseGeometry]]:
    """(label, polygon) pairs from a panels.json document, repaired and non-degenerate.

    Segmentation output can be spiky to the point of self-intersection; ``make_valid``
    repairs such rings so areas and intersections are meaningful.
    """
    out: list[tuple[str, BaseGeometry]] = []
    for ring, label in zip(doc["panels"], doc["labels"]):
        if len(ring) < 4 or not str(label).isdigit():
            continue
        polygon = make_valid(Polygon([(x, y) for x, y in ring]))
        if polygon.area > 0:
            out.append((str(label), polygon))
    return out


def scale_to(doc: dict, target: dict) -> None:
    """Rescale ``doc``'s panel coordinates in place to ``target``'s pixel space."""
    scale_x = target["width"] / doc["width"]
    scale_y = target["height"] / doc["height"]
    if scale_x == 1.0 and scale_y == 1.0:
        return
    doc["panels"] = [
        [[x * scale_x, y * scale_y] for x, y in ring] for ring in doc["panels"]
    ]


@dataclass
class RegionScore:
    """Per-truth-footprint IoUs plus the spurious/overlap diagnostics."""

    ious: list[tuple[str, float]] = field(default_factory=list)  # (label, IoU)
    spurious: int = 0  # predicted panels no truth footprint claimed
    predicted_overlap: float = 0.0  # fraction of predicted area overlapped by peers


def greedy_match(
    truths: list[BaseGeometry], predictions: list[BaseGeometry]
) -> list[tuple[int, int]]:
    """Pair truth and predicted polygons of one label by greatest intersection, greedily."""
    pairs = sorted(
        (
            (truth.intersection(prediction).area, i, j)
            for i, truth in enumerate(truths)
            for j, prediction in enumerate(predictions)
        ),
        reverse=True,
    )
    matched: list[tuple[int, int]] = []
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    for area, i, j in pairs:
        if area <= 0 or i in used_truth or j in used_prediction:
            continue
        matched.append((i, j))
        used_truth.add(i)
        used_prediction.add(j)
    return matched


def score_regions(truth_doc: dict, predicted_doc: dict) -> RegionScore:
    """IoU of every truth footprint against its greedily-matched prediction."""
    truth_panels = panel_polygons(truth_doc)
    predicted_panels = panel_polygons(predicted_doc)
    truth_by_label: dict[str, list[BaseGeometry]] = {}
    for label, polygon in truth_panels:
        truth_by_label.setdefault(label, []).append(polygon)
    predicted_by_label: dict[str, list[BaseGeometry]] = {}
    for label, polygon in predicted_panels:
        predicted_by_label.setdefault(label, []).append(polygon)

    score = RegionScore()
    for label, truths in sorted(truth_by_label.items(), key=lambda kv: int(kv[0])):
        predictions = predicted_by_label.get(label, [])
        matched = greedy_match(truths, predictions)
        matched_truths = set()
        for i, j in matched:
            union = truths[i].union(predictions[j]).area
            iou = truths[i].intersection(predictions[j]).area / union if union else 0.0
            score.ious.append((label, iou))
            matched_truths.add(i)
        for i in range(len(truths)):
            if i not in matched_truths:
                score.ious.append((label, 0.0))
        score.spurious += len(predictions) - len(matched)
    for label in predicted_by_label:
        if label not in truth_by_label:
            score.spurious += len(predicted_by_label[label])

    polygons = [polygon for _, polygon in predicted_panels]
    total = sum(polygon.area for polygon in polygons)
    if total > 0:
        overlap = sum(
            polygons[i].intersection(polygons[j]).area
            for i in range(len(polygons))
            for j in range(i + 1, len(polygons))
        )
        score.predicted_overlap = overlap / total
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "regions",
        type=Path,
        help="Predicted <stem>.regions.panels.json (k-means or SAM output).",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        help=(
            "Truth panels file (default: <stem>.truth.regions.panels.json next to the "
            "predictions, from mapsnap.keymap.truth_regions)."
        ),
    )
    parser.add_argument(
        "--per-page", action="store_true", help="Also print every footprint's IoU."
    )
    args = parser.parse_args()

    truth_path = args.truth or default_truth_path(args.regions)
    if truth_path.resolve() == args.regions.resolve():
        sys.exit(
            f"Truth file {truth_path} IS the predictions file; scoring a file against "
            "itself is always a perfect 1.000. Pass the predictions sidecar, not the truth."
        )
    if not truth_path.exists():
        sys.exit(f"Not found: {truth_path} (run mapsnap.keymap.truth_regions first).")
    truth_doc = json.load(open(truth_path))
    predicted_doc = json.load(open(args.regions))
    scale_to(truth_doc, predicted_doc)
    score = score_regions(truth_doc, predicted_doc)

    values = sorted(iou for _, iou in score.ious)
    n = len(values)
    if not n:
        sys.exit("No truth footprints to score.")
    mean = sum(values) / n
    print(f"truth footprints: {n}")
    print(f"IoU: mean {mean:.3f}  median {values[n // 2]:.3f}")
    print(
        f"     >=0.5: {sum(1 for v in values if v >= 0.5)}/{n}   "
        f"missed (IoU 0): {sum(1 for v in values if v == 0.0)}"
    )
    missed_pages = sorted({int(label) for label, iou in score.ious if iou == 0.0})
    if missed_pages:
        print(f"missed pages: {', '.join(str(page) for page in missed_pages)}")
    print(f"spurious predicted regions: {score.spurious}")
    print(f"predicted-region self-overlap: {score.predicted_overlap:.1%} of area")
    if args.per_page:
        for label, iou in sorted(score.ious, key=lambda li: li[1]):
            print(f"  p{label:4s} {iou:.3f}")


if __name__ == "__main__":
    main()
