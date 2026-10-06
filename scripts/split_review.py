#!/usr/bin/env python
"""Build a split review: two splitter arms' panels beside OIM truth, for the browser.

The split review page (``app/split-review.html``) steps through pages showing
arm A's panels and arm B's side by side, each scored against the truth panels
OIM's volunteers drew. This writes what it reads: ``<out>/review.json`` and a
copy of each page image under ``<out>/images/``. Put ``<out>`` under ``data/``
so the dev server serves it, then open

    http://localhost:5173/mapsnap/split-review.html?review=data/<out under data>/review.json

Each arm is a directory of ``<image stem>.panels.json`` files, as
``score_splits_oim.py --write-panels`` or ``fetch_run_panels.py`` writes them; a
page with no file there was left whole. Pages come from a cutline benchmark
manifest (see score_splits_oim.py). Run from the project root:

    MAPSNAP_SPLITTER=classical uv run python scripts/score_splits_oim.py \\
        --manifest ~/Documents/mapsnap/cutline-training/manifest.tsv --fold test \\
        --write-panels /tmp/classical
    uv run python scripts/score_splits_oim.py \\
        --manifest ~/Documents/mapsnap/cutline-training/manifest.tsv --fold test \\
        --write-panels /tmp/model
    uv run python scripts/split_review.py \\
        --manifest ~/Documents/mapsnap/cutline-training/manifest.tsv --fold test \\
        --arm classical=/tmp/classical --arm "cutline model=/tmp/model" \\
        --changed --sample 100 --out data/split-review/test-100
"""

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_splits_oim import (
    Case,
    manifest_cases,
    panels_in_frame,
    run_panels,
    score_case,
    unfinished_names,
    whole_page,
)

# Two arms' panels "agree" at or above this matched IoU (and equal counts); a
# page below it has a change worth reviewing.
CHANGE_IOU = 0.98


@dataclass
class Arm:
    """One side of the comparison: a label and its panels directory."""

    label: str
    panels_dir: Path


def parse_arm(spec: str) -> Arm:
    """An --arm value, LABEL=DIR."""
    label, sep, directory = spec.partition("=")
    if not sep or not label or not directory:
        raise argparse.ArgumentTypeError(f"expected LABEL=DIR, got {spec!r}")
    return Arm(label, Path(directory))


def rings(panels: list[Polygon]) -> list[list[list[float]]]:
    """Polygons as open [x, y] rings, rounded to a tenth of a pixel."""
    return [
        [[round(x, 1), round(y, 1)] for x, y in list(p.exterior.coords)[:-1]]
        for p in panels
    ]


def truth_panels(case: Case, size: tuple[int, int]) -> list[Polygon]:
    """OIM's panels for a page, scaled to its image; the whole page if unsplit."""
    if case.truth is None:
        return whole_page(size)
    return panels_in_frame(json.loads(case.truth.read_text()), size)


def changed(a: list[Polygon], b: list[Polygon]) -> bool:
    """Whether two arms split a page differently: other counts, or panels that moved."""
    if len(a) != len(b):
        return True
    iou, _ = score_case(a, b)
    return iou < CHANGE_IOU


def page_record(
    case: Case, titles: dict[str, str], arms: list[Arm], size: tuple[int, int]
) -> dict:
    """One page of review.json: its truth, and each arm's panels scored against it."""
    truth = truth_panels(case, size)
    arm_records = []
    for arm in arms:
        panels = run_panels(arm.panels_dir, case, size)
        iou, _ = score_case(truth, panels)
        arm_records.append({"panels": rings(panels), "iou": round(iou, 4)})
    return {
        "name": case.image.stem,
        "image": f"images/{case.image.name}",
        "title": titles.get(case.image.stem, ""),
        "item": case.volume,
        "page": case.page,
        "label": "split" if case.truth else "unsplit",
        "width": size[0],
        "height": size[1],
        "truth": rings(truth),
        "arms": arm_records,
    }


def read_titles(manifest: Path) -> dict[str, str]:
    """Volume title per image stem, from the manifest's optional title column."""
    with manifest.open() as handle:
        return {
            Path(row["image"]).stem: row.get("title") or ""
            for row in csv.DictReader(handle, delimiter="\t")
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--arm",
        type=parse_arm,
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="An arm's label and panels directory; give exactly two (A, then B).",
    )
    parser.add_argument("--fold", default=None, help="Only pages of this fold.")
    parser.add_argument(
        "--changed",
        action="store_true",
        help="Only pages the two arms split differently.",
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Draw this many pages (0 = all)."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=None, help="Heading for the review.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.arm) != 2:
        parser.error("give exactly two --arm values")

    skipped = set().union(*(unfinished_names(arm.panels_dir) for arm in args.arm))
    cases = [
        case
        for case in manifest_cases(args.manifest)
        if (args.fold is None or case.fold == args.fold)
        and case.image.stem not in skipped
    ]
    titles = read_titles(args.manifest)
    pages = []
    for case in cases:
        with Image.open(case.image) as img:
            size = img.size
        if args.changed and not changed(
            *(run_panels(arm.panels_dir, case, size) for arm in args.arm)
        ):
            continue
        pages.append((case, size))
    print(f"{len(pages)} of {len(cases)} pages qualify", file=sys.stderr)
    if args.sample and len(pages) > args.sample:
        pages = random.Random(args.seed).sample(pages, args.sample)
    pages.sort(key=lambda page: page[0].image.stem)

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for case, size in pages:
        shutil.copy(case.image, args.out / "images" / case.image.name)
        records.append(page_record(case, titles, args.arm, size))
    review = {
        "title": args.title or args.out.name,
        "arms": [arm.label for arm in args.arm],
        "pages": records,
    }
    (args.out / "review.json").write_text(json.dumps(review))
    print(f"wrote {args.out / 'review.json'} ({len(records)} pages)", file=sys.stderr)


if __name__ == "__main__":
    main()
