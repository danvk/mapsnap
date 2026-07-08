"""Score a volume's adjacency.json against OIM truth footprints.

Two pages count as truth-adjacent when their human-georeferenced footprints (from
``main.iiif.json``) come within ``--max-gap`` metres of each other; OIM's manual splits clip
pages to tile cleanly, so genuinely adjacent pages touch or nearly touch rather than overlap.

Reports the mutual-edge precision (fraction of edges whose pages are truth-adjacent), recall
against all truth-adjacent pairs, and page coverage, and lists the edges that contradict truth
for inspection. Two caveats when reading the numbers: recall's denominator includes
corner-to-corner contacts that sheets print no reference for, so it reads low; and a
truth-consistent edge can still rest on a junk read that reciprocated by coincidence (page
numbering correlates with adjacency), so precision reads high — inspect surviving edges in the
debugger's adjacency mode before trusting a new filter.

    uv run python -m mapsnap.score_adjacency data/hudson_co_nj_1950_vol_9
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from mapsnap.compare_iiif_georef import truth_polygons_by_page
from mapsnap.keymap.locate import M_PER_DEG_LAT, M_PER_DEG_LON_EQUATOR


def truth_shapes(
    truth_by_page: dict[int, list[list[list[float]]]],
) -> dict[int, object]:
    """One shapely geometry per page number, in a local metre frame.

    Each page's truth footprints (several for a split page) are unioned. Coordinates are
    projected equirectangularly about the volume's mean latitude so distances are metres.
    """
    points = [
        point
        for polygons in truth_by_page.values()
        for polygon in polygons
        for point in polygon
    ]
    if not points:
        return {}
    lat0 = sum(point[1] for point in points) / len(points)
    scale_x = M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat0))
    return {
        number: unary_union(
            [
                Polygon([(x * scale_x, y * M_PER_DEG_LAT) for x, y in polygon])
                for polygon in polygons
            ]
        )
        for number, polygons in truth_by_page.items()
    }


def truth_adjacent_pairs(
    shapes: dict[int, object], max_gap_m: float
) -> set[frozenset[int]]:
    """Unordered page-number pairs whose truth footprints come within ``max_gap_m`` metres."""
    numbers = sorted(shapes)
    pairs: set[frozenset[int]] = set()
    for i, a in enumerate(numbers):
        for b in numbers[i + 1 :]:
            if shapes[a].distance(shapes[b]) < max_gap_m:  # type: ignore[attr-defined]
                pairs.add(frozenset((a, b)))
    return pairs


@dataclass
class AdjacencyScore:
    """Mutual-edge accuracy against truth adjacency (see score_adjacency)."""

    edges: int = 0
    known: int = 0  # edges where both pages have truth footprints
    correct: int = 0
    wrong: list[tuple[str, str]] = field(default_factory=list)
    unknown: list[tuple[str, str]] = field(default_factory=list)
    truth_pairs: int = 0  # truth-adjacent pairs among scanned pages
    recovered: int = 0
    pages: int = 0
    pages_covered: int = 0
    single_digit_known: int = 0
    single_digit_correct: int = 0


def score_adjacency(
    doc: dict, truth_pairs: set[frozenset[int]], truth_numbers: set[int]
) -> AdjacencyScore:
    """Score an adjacency.json document's mutual edges against truth-adjacent pairs.

    Edges between pages sharing a number (split variants) are skipped; edges where either
    page lacks a truth footprint are counted as ``unknown`` rather than wrong.
    """
    pages = doc["pages"]
    score = AdjacencyScore(pages=len(pages))
    covered: set[str] = set()
    recovered_pairs: set[frozenset[int]] = set()
    for stem_a, stem_b in doc["adjacency"]:
        number_a, number_b = pages[stem_a]["number"], pages[stem_b]["number"]
        score.edges += 1
        covered.update((stem_a, stem_b))
        if number_a == number_b:
            continue
        if number_a not in truth_numbers or number_b not in truth_numbers:
            score.unknown.append((stem_a, stem_b))
            continue
        score.known += 1
        pair = frozenset((number_a, number_b))
        is_correct = pair in truth_pairs
        single = min(number_a, number_b) < 10
        if single:
            score.single_digit_known += 1
        if is_correct:
            score.correct += 1
            recovered_pairs.add(pair)
            if single:
                score.single_digit_correct += 1
        else:
            score.wrong.append((stem_a, stem_b))
    scanned_numbers = {page["number"] for page in pages.values()}
    relevant = {pair for pair in truth_pairs if all(n in scanned_numbers for n in pair)}
    score.truth_pairs = len(relevant)
    score.recovered = len(recovered_pairs & relevant)
    score.pages_covered = len(covered)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score adjacency.json mutual edges against OIM truth footprints."
    )
    parser.add_argument(
        "volume",
        type=Path,
        help="Volume directory holding adjacency.json and main.iiif.json.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=30.0,
        metavar="M",
        help=(
            "Footprints within this many metres count as truth-adjacent "
            "(default: %(default)s; OIM clips pages to tile, so neighbors nearly touch)."
        ),
    )
    args = parser.parse_args()

    adjacency_path = args.volume / "adjacency.json"
    truth_path = args.volume / "main.iiif.json"
    for path in (adjacency_path, truth_path):
        if not path.exists():
            sys.exit(f"Not found: {path}")
    doc = json.load(open(adjacency_path))
    shapes = truth_shapes(truth_polygons_by_page(truth_path))
    if not shapes:
        sys.exit(f"No usable truth footprints in {truth_path}.")
    truth_pairs = truth_adjacent_pairs(shapes, args.max_gap)
    score = score_adjacency(doc, truth_pairs, set(shapes))

    print(f"pages scanned:  {score.pages} ({score.pages_covered} in >=1 mutual edge)")
    print(f"mutual edges:   {score.edges}")
    print(
        f"precision:      {score.correct}/{score.known} truth-consistent"
        + (f" ({score.correct / score.known:.0%})" if score.known else "")
    )
    if score.single_digit_known:
        print(
            f"  single-digit: {score.single_digit_correct}/{score.single_digit_known}"
        )
    print(
        f"recall:         {score.recovered}/{score.truth_pairs} truth-adjacent pairs "
        "(denominator includes corner contacts sheets don't print)"
    )
    if score.wrong:
        print("edges contradicting truth:")
        for stem_a, stem_b in score.wrong:
            print(f"  {stem_a} <-> {stem_b}")
    if score.unknown:
        print(f"edges without truth coverage: {len(score.unknown)}")


if __name__ == "__main__":
    main()
