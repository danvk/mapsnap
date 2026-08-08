#!/usr/bin/env python3
"""Score the splitter against OIM's hand-drawn panel truth, corpus-wide (#83 Phase 0).

Where PR #70's harness (score_splits.py) scores against a small hand-made
testdata set, this one scores against ``data/<vol>/oim/pN.panels.json`` — the
panel polygons OIM's volunteers drew for every truth-split sheet, built by
``mapsnap oim-split-truth``. Cases:

* **positive** — a page whose OIM panels.json has >= 2 panels: the splitter
  should reproduce those panels (matched IoU, per-panel recall).
* **negative** — every other page with a local image: the splitter should
  leave it whole. Over-splitting was PR #70's dominant failure mode, so the
  guards are regression-tested here, never assumed.

San Francisco is excluded unconditionally: its Sanborn streets are not drawn
to scale and OIM's truth puts every block in its own split. Those are not
dividing-line splits and must never enter this metric (or any training set).

Headline metrics:

* mean matched IoU over positives (PR #70's metric, unchanged);
* negative accuracy — the fraction of negatives left unsplit;
* small-panel recall — truth panels under SMALL_PANEL_FRAC of their page
  matched at >= RECALL_IOU. This is the number the #83 work exists to move:
  the small insets that MIN_PANEL_FRAC glue-away discards today.

Run from the project root:

  uv run python scripts/score_splits_oim.py                       # baseline
  uv run python scripts/score_splits_oim.py --min-panel-frac 0.01
  uv run python scripts/score_splits_oim.py --small-face verified
  uv run python scripts/score_splits_oim.py --negatives 200 --out arm.json
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapsnap.split import compute_panels

EXCLUDED_VOLUMES = ("san_francisco",)
SMALL_PANEL_FRAC = 0.05  # a truth panel below this fraction of the page is "small"
RECALL_IOU = 0.5  # per-panel matched IoU at or above this counts as recalled


def make_valid(polygon: Polygon) -> Polygon:
    """Repair a possibly self-intersecting polygon with a zero-width buffer."""
    return polygon if polygon.is_valid else polygon.buffer(0)


def load_oim_truth(json_path: Path, image_path: Path) -> list[Polygon]:
    """OIM truth panels scaled from their canvas frame to the local image frame."""
    data = json.loads(json_path.read_text())
    with Image.open(image_path) as img:
        img_w, img_h = img.size
    sx = img_w / data["width"]
    sy = img_h / data["height"]
    return [
        make_valid(Polygon([[x * sx, y * sy] for x, y in ring]))
        for ring in data["panels"]
    ]


def score_case(truth: list[Polygon], gen: list[Polygon]) -> tuple[float, list[float]]:
    """(matched IoU for the page, per-truth-panel IoU under the same assignment)."""
    inter = np.zeros((len(truth), len(gen)))
    for i, t in enumerate(truth):
        for j, g in enumerate(gen):
            inter[i, j] = t.intersection(g).area
    rows, cols = linear_sum_assignment(inter, maximize=True)
    per_truth = [0.0] * len(truth)
    total_int = 0.0
    total_union = 0.0
    matched_t, matched_g = set(), set()
    for i, j in zip(rows, cols):
        if inter[i, j] <= 0:
            continue
        union = truth[i].area + gen[j].area - inter[i, j]
        per_truth[i] = inter[i, j] / union if union > 0 else 0.0
        total_int += inter[i, j]
        total_union += union
        matched_t.add(i)
        matched_g.add(j)
    for i, t in enumerate(truth):
        if i not in matched_t:
            total_union += t.area
    for j, g in enumerate(gen):
        if j not in matched_g:
            total_union += g.area
    return (float(total_int / total_union) if total_union > 0 else 1.0), per_truth


def gather_cases(
    data_dir: Path, negatives: int, seed: int
) -> tuple[list[tuple[str, Path, Path]], list[tuple[str, Path]]]:
    """(positives, negatives): positives pair a page image with its OIM panels.json."""
    positives = []
    negative_pool = []
    for volume in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if any(volume.name.startswith(x) for x in EXCLUDED_VOLUMES):
            continue
        oim = volume / "oim"
        if not oim.is_dir():
            continue
        split_stems = set()
        for panels_path in sorted(oim.glob("p*.panels.json")):
            stem = panels_path.name.split(".")[0]
            image = volume / f"{stem}.jpg"
            if not image.exists():
                continue
            n = len(json.loads(panels_path.read_text()).get("panels", []))
            if n >= 2:
                positives.append((volume.name, image, panels_path))
                split_stems.add(stem)
        for image in sorted(volume.glob("p*.jpg")):
            stem = image.name.split(".")[0]
            if "__" in stem or stem in split_stems:
                continue
            negative_pool.append((volume.name, image))
    if 0 <= negatives < len(negative_pool):
        random.Random(seed).shuffle(negative_pool)
        negative_pool = negative_pool[:negatives]
    return positives, sorted(negative_pool)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--negatives",
        type=int,
        default=-1,
        metavar="N",
        help="Sample N negative pages (-1 = all; 0 = skip negatives).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--min-panel-frac",
        type=float,
        default=None,
        help="Override split.MIN_PANEL_FRAC for this run.",
    )
    parser.add_argument(
        "--small-face",
        choices=("glue", "verified"),
        default="glue",
        help="Small-face policy: glue (PR #70 default) or divider-verified keep.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write per-case JSON.")
    args = parser.parse_args()

    positives, negative_cases = gather_cases(args.data_dir, args.negatives, args.seed)
    print(
        f"{len(positives)} positive sheets, {len(negative_cases)} negatives "
        f"(san_francisco excluded)",
        file=sys.stderr,
    )

    records = []
    by_volume: dict[str, list[float]] = defaultdict(list)
    small_total = small_recalled = 0
    for volume, image, panels_path in positives:
        truth = load_oim_truth(panels_path, image)
        gen = [
            make_valid(p)
            for p in compute_panels(
                image,
                min_panel_frac=args.min_panel_frac,
                small_face_policy=args.small_face,
            )
        ]
        iou, per_truth = score_case(truth, gen)
        page_area = float(np.prod(Image.open(image).size))
        smalls = []
        for t, panel_iou in zip(truth, per_truth):
            if t.area < SMALL_PANEL_FRAC * page_area:
                small_total += 1
                recalled = panel_iou >= RECALL_IOU
                small_recalled += recalled
                smalls.append(round(panel_iou, 3))
        by_volume[volume].append(iou)
        records.append(
            {
                "kind": "positive",
                "volume": volume,
                "page": image.name.split(".")[0],
                "iou": round(iou, 4),
                "n_truth": len(truth),
                "n_gen": len(gen),
                "small_panel_ious": smalls,
            }
        )
    neg_ok = 0
    for volume, image in negative_cases:
        gen = compute_panels(
            image,
            min_panel_frac=args.min_panel_frac,
            small_face_policy=args.small_face,
        )
        ok = len(gen) == 1
        neg_ok += ok
        records.append(
            {
                "kind": "negative",
                "volume": volume,
                "page": image.name.split(".")[0],
                "n_gen": len(gen),
                "ok": ok,
            }
        )

    print(f"\n{'volume':28s} {'sheets':>6s} {'mean IoU':>9s}")
    for volume in sorted(by_volume):
        vals = by_volume[volume]
        print(f"{volume:28s} {len(vals):6d} {sum(vals) / len(vals):9.3f}")
    pos_ious = [r["iou"] for r in records if r["kind"] == "positive"]
    print(
        f"\npositives mean IoU: {sum(pos_ious) / len(pos_ious):.3f} (n={len(pos_ious)})"
    )
    if small_total:
        print(
            f"small-panel recall (<{SMALL_PANEL_FRAC:.0%} of page, IoU>={RECALL_IOU}): "
            f"{small_recalled}/{small_total} = {small_recalled / small_total:.1%}"
        )
    if negative_cases:
        print(
            f"negative accuracy (left unsplit): {neg_ok}/{len(negative_cases)} "
            f"= {neg_ok / len(negative_cases):.1%}"
        )
    worst = sorted(
        (r for r in records if r["kind"] == "positive"), key=lambda r: r["iou"]
    )[:15]
    print("\nworst positives:")
    for r in worst:
        print(
            f"  {r['volume']:26s} {r['page']:8s} IoU {r['iou']:.3f} "
            f"(truth {r['n_truth']}, got {r['n_gen']})"
        )
    if args.out:
        args.out.write_text(json.dumps(records, indent=1))
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
