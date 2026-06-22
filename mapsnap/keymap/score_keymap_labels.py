"""Score generated key-map page-number detections against hand-labeled truth points.

The generated detections (``<stem>.streets.json`` from mapsnap.keymap.records)
carry a bounding polygon, a text, a confidence, and short/long side lengths. The truth
data (``<stem>.labels.json`` from the labeler tool) is just points: ``{x, y, text}``.

A detection and a truth point match when the point falls inside the detection's polygon.
Detections are first filtered by ``--min-confidence`` and ``--min-short-side``; matching
is greedy and one-to-one (highest-confidence detections claim a point first), so:

  precision = matched / kept detections   (how many kept detections are real)
  recall    = matched / truth points      (how many truth points were found)

Among matched pairs it also counts how many disagree on the recognized text.
"""

import argparse
import json
import sys
from collections.abc import Sequence


def point_in_polygon(
    point: tuple[float, float], polygon: Sequence[Sequence[float]]
) -> bool:
    """Whether ``point`` (x, y) lies inside ``polygon`` (list of [x, y] vertices).

    Ray-casting (even-odd rule). Points exactly on an edge are reported inconsistently,
    which is fine for scoring against interior label points.
    """
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def filter_detections(
    detections: list[dict], min_confidence: float, min_short_side: float
) -> list[dict]:
    """Keep detections meeting the confidence and short-side thresholds."""
    return [
        d
        for d in detections
        if d.get("confidence", 0.0) >= min_confidence
        and d.get("short_side", 0.0) >= min_short_side
    ]


def match_detections(
    detections: list[dict], labels: list[dict]
) -> list[tuple[int, int]]:
    """Greedy one-to-one matches as (detection_index, label_index) pairs.

    Detections are considered highest-confidence first; each claims the first
    not-yet-matched truth point that falls inside its polygon. Every detection and every
    truth point is used at most once.
    """
    order = sorted(
        range(len(detections)),
        key=lambda i: detections[i].get("confidence", 0.0),
        reverse=True,
    )
    matched_labels: set[int] = set()
    matches: list[tuple[int, int]] = []
    for di in order:
        polygon = detections[di]["polygon"]
        for li, label in enumerate(labels):
            if li in matched_labels:
                continue
            if point_in_polygon((label["x"], label["y"]), polygon):
                matches.append((di, li))
                matched_labels.add(li)
                break
    return matches


def normalize_text(text: object) -> str:
    """Normalize a label/detection text for comparison (strip surrounding whitespace)."""
    return str(text).strip()


def score(detections: list[dict], labels: list[dict]) -> dict:
    """Score detections against truth labels; return summary statistics.

    Keys: true_positives, false_positives, false_negatives, precision, recall,
    text_disagreements (count), and disagreement_examples (list of (detected, truth)).
    """
    matches = match_detections(detections, labels)
    true_positives = len(matches)
    false_positives = len(detections) - true_positives
    false_negatives = len(labels) - true_positives

    disagreements = [
        (detections[di]["text"], labels[li]["text"])
        for di, li in matches
        if normalize_text(detections[di]["text"]) != normalize_text(labels[li]["text"])
    ]

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / len(detections) if detections else 0.0,
        "recall": true_positives / len(labels) if labels else 0.0,
        "text_disagreements": len(disagreements),
        "disagreement_examples": disagreements,
    }


def load_detections(path: str) -> list[dict]:
    """Load the detection list from a .streets.json file (wrapped object or bare list)."""
    doc = json.load(open(path))
    return doc["streets"] if isinstance(doc, dict) else doc


def load_labels(path: str) -> list[dict]:
    """Load the truth points from a .labels.json file (wrapped object or bare list)."""
    doc = json.load(open(path))
    return doc["labels"] if isinstance(doc, dict) else doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score key-map detections (.streets.json) against truth (.labels.json)."
    )
    parser.add_argument("detections", help="Generated <stem>.streets.json file.")
    parser.add_argument("truth", help="Truth <stem>.labels.json file.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        metavar="C",
        help="Drop detections with confidence below this (default: %(default)s).",
    )
    parser.add_argument(
        "--min-short-side",
        type=float,
        default=0.0,
        metavar="PX",
        help="Drop detections whose short side is below this (default: %(default)s).",
    )
    args = parser.parse_args()

    all_detections = load_detections(args.detections)
    labels = load_labels(args.truth)
    detections = filter_detections(
        all_detections, args.min_confidence, args.min_short_side
    )

    result = score(detections, labels)

    print(
        f"Detections: {len(all_detections)} total, {len(detections)} kept "
        f"(confidence>={args.min_confidence}, short_side>={args.min_short_side})"
    )
    print(f"Truth points: {len(labels)}")
    print(
        f"Matched (true positives): {result['true_positives']}  "
        f"unmatched detections (false positives): {result['false_positives']}  "
        f"unmatched truth (false negatives): {result['false_negatives']}"
    )
    print(f"Precision: {result['precision']:.1%}")
    print(f"Recall:    {result['recall']:.1%}")
    print(f"Text disagreements among matched: {result['text_disagreements']}")
    for detected, truth in result["disagreement_examples"]:
        print(f"  detected {detected!r} vs truth {truth!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
