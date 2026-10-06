#!/usr/bin/env python3
"""Score the splitter against OIM's hand-drawn panel truth (#83).

Where PR #70's harness (score_splits.py) scores against a small hand-made
testdata set, this one scores against panel polygons OIM's volunteers drew.
Cases come from one of two places:

* ``--data-dir`` (default ``data/``): ``data/<vol>/oim/pN.panels.json`` for
  every truth-split sheet of the local volumes, and every other local page as
  a negative.
* ``--manifest``: a benchmark like ``~/Documents/mapsnap/cutline-training``,
  whose manifest.tsv lists images (relative to it), a ``label`` of split or
  unsplit, and optionally ``fold``, ``size_band``, a sampling ``weight`` and
  ``volume_has_p0`` (true/false) and ``volume_sheets`` for the splitter's key-map
  rules;
  split pages' truth is ``labels/<image stem>.panels.json``.

Cases:

* **positive** — a page OIM cut into >= 2 panels: the splitter should
  reproduce those panels (matched IoU, per-panel recall).
* **negative** — a page OIM left whole: the splitter should leave it whole.
  Over-splitting was PR #70's dominant failure mode, so the guards are
  regression-tested here, never assumed.

San Francisco is excluded from ``--data-dir`` runs: its Sanborn streets are
not drawn to scale and OIM's truth puts every block in its own split. Those are
not dividing-line splits and must never enter this metric (or any training set).

The splitter's panels are computed here unless ``--panels-dir`` names a
directory of ``<image stem>.panels.json`` files a run already wrote (see
fetch_run_panels.py); a page with none there was left whole, and pages listed
in its ``unfinished.json`` are skipped.

Headline metrics:

* mean matched IoU over positives (PR #70's metric, unchanged), and the same
  mean weighted by 1 / sampling weight, which undoes a benchmark's oversampling;
* cut at all / right panel count — matched IoU is lenient on a miss: a sheet
  left whole scores A / (2P - A) for its largest panel's area A and page area P,
  so one whose only cut sets off a 10% inset still scores 0.82;
* negative accuracy — the fraction of negatives left unsplit;
* small-panel recall — truth panels under SMALL_PANEL_FRAC of their page
  matched at >= RECALL_IOU: the small insets MIN_PANEL_FRAC glue-away discards.

Run from the project root:

  uv run python scripts/score_splits_oim.py                       # baseline
  uv run python scripts/score_splits_oim.py --min-panel-frac 0.01
  uv run python scripts/score_splits_oim.py --small-face verified
  uv run python scripts/score_splits_oim.py --negatives 200 --out arm.json
  uv run python scripts/score_splits_oim.py \\
      --manifest ~/Documents/mapsnap/cutline-training/manifest.tsv \\
      --panels-dir corpus-v1-panels
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapsnap.split import SheetContext, compute_panels

EXCLUDED_VOLUMES = ("san_francisco",)
SMALL_PANEL_FRAC = 0.05  # a truth panel below this fraction of the page is "small"
RECALL_IOU = 0.5  # per-panel matched IoU at or above this counts as recalled
GOOD_IOU = 0.9  # a page matched at or above this counts as split well
SIZE_BANDS = ("1-5", "6-15", "16-40", "41+")


@dataclass
class Case:
    """One benchmark page: its image, and its truth panels.json if OIM split it."""

    name: str  # unique within the run: <vol>/<page>, or the image stem
    volume: str
    page: str
    image: Path
    truth: Path | None = None  # None: a negative, truth is the whole page
    fold: str = ""
    size_band: str = ""
    weight: float = 1.0  # sampling weight; the corrected mean divides it out
    # For the splitter's key-map rules; None: judge by the images beside it.
    volume_has_page_zero: bool | None = None
    volume_sheets: int | None = None

    def sheet(self) -> SheetContext | None:
        """The SheetContext a manifest gives, or None to read the image's directory."""
        if self.volume_has_page_zero is None or self.volume_sheets is None:
            return None
        return SheetContext(self.page, self.volume_sheets, self.volume_has_page_zero)


def make_valid(polygon: Polygon) -> Polygon:
    """Repair a possibly self-intersecting polygon with a zero-width buffer."""
    return polygon if polygon.is_valid else polygon.buffer(0)


def panels_in_frame(data: dict, size: tuple[int, int]) -> list[Polygon]:
    """A panels.json's rings scaled from its recorded frame to an image's size."""
    sx = size[0] / data["width"]
    sy = size[1] / data["height"]
    return [
        make_valid(Polygon([[x * sx, y * sy] for x, y in ring]))
        for ring in data["panels"]
    ]


def load_oim_truth(json_path: Path, image_path: Path) -> list[Polygon]:
    """OIM truth panels scaled from their canvas frame to the local image frame."""
    with Image.open(image_path) as img:
        size = img.size
    return panels_in_frame(json.loads(json_path.read_text()), size)


def whole_page(size: tuple[int, int]) -> list[Polygon]:
    """A single panel covering a page of this size."""
    w, h = size
    return [Polygon([(0, 0), (w, 0), (w, h), (0, h)])]


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


def gather_cases(data_dir: Path, negatives: int, seed: int) -> list[Case]:
    """Positives (pages with an OIM panels.json of >= 2 panels), then negatives."""
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
                positives.append(
                    Case(f"{volume.name}/{stem}", volume.name, stem, image, panels_path)
                )
                split_stems.add(stem)
        for image in sorted(volume.glob("p*.jpg")):
            stem = image.name.split(".")[0]
            if "__" in stem or stem in split_stems:
                continue
            negative_pool.append(
                Case(f"{volume.name}/{stem}", volume.name, stem, image)
            )
    if 0 <= negatives < len(negative_pool):
        random.Random(seed).shuffle(negative_pool)
        negative_pool = negative_pool[:negatives]
    return positives + sorted(negative_pool, key=lambda case: case.name)


def page_zero_flag(value: str | None) -> bool | None:
    """A manifest's volume_has_p0 cell as a flag; None when the column is absent."""
    return None if value in (None, "") else value == "true"


def manifest_cases(manifest: Path) -> list[Case]:
    """Cases from a benchmark manifest.tsv (see the module docstring)."""
    root = manifest.parent
    cases = []
    with manifest.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            image = root / row["image"]
            truth = (
                root / "labels" / f"{image.stem}.panels.json"
                if row["label"] == "split"
                else None
            )
            cases.append(
                Case(
                    image.stem,
                    row.get("item") or "",
                    row.get("page") or image.stem,
                    image,
                    truth,
                    fold=row.get("fold") or "",
                    size_band=row.get("size_band") or "",
                    weight=float(row.get("weight") or 1),
                    volume_has_page_zero=page_zero_flag(row.get("volume_has_p0")),
                    volume_sheets=int(row["volume_sheets"])
                    if row.get("volume_sheets")
                    else None,
                )
            )
    return cases


def run_panels(panels_dir: Path, case: Case, size: tuple[int, int]) -> list[Polygon]:
    """The panels a run wrote for this case's image, or the whole page if none."""
    path = panels_dir / f"{case.image.stem}.panels.json"
    if not path.exists():
        return whole_page(size)
    return panels_in_frame(json.loads(path.read_text()), size)


def unfinished_names(panels_dir: Path) -> set[str]:
    """Image stems whose item the run never finished (fetch_run_panels.py's list)."""
    path = panels_dir / "unfinished.json"
    return set(json.loads(path.read_text())) if path.exists() else set()


def score_record(case: Case, gen: list[Polygon], size: tuple[int, int]) -> dict:
    """One case's record: its matched IoU, panel counts and small-panel IoUs."""
    record = {
        "kind": "positive" if case.truth else "negative",
        "volume": case.volume,
        "page": case.page,
        "fold": case.fold,
        "size_band": case.size_band,
        "weight": case.weight,
        "n_gen": len(gen),
    }
    if case.truth is None:
        iou, _ = score_case(whole_page(size), gen)
        return record | {"iou": round(iou, 4), "n_truth": 1, "ok": len(gen) == 1}
    truth = panels_in_frame(json.loads(case.truth.read_text()), size)
    iou, per_truth = score_case(truth, gen)
    page_area = size[0] * size[1]
    smalls = [
        round(panel_iou, 3)
        for t, panel_iou in zip(truth, per_truth)
        if t.area < SMALL_PANEL_FRAC * page_area
    ]
    return record | {
        "iou": round(iou, 4),
        "n_truth": len(truth),
        "small_panel_ious": smalls,
    }


def outcome(n_truth: int, n_gen: int) -> str:
    """How a positive's panel count compares: whole, too few, right or too many."""
    if n_gen == 1:
        return "whole"
    if n_gen < n_truth:
        return "too few"
    return "right" if n_gen == n_truth else "too many"


def share(flags: list[bool]) -> str:
    """A percentage of true flags, or '-' for none."""
    return f"{sum(flags) / len(flags):.1%}" if flags else "-"


def summary_line(records: list[dict]) -> str:
    """Headline numbers for a set of records, positives then negatives."""
    pos = [r for r in records if r["kind"] == "positive"]
    neg = [r for r in records if r["kind"] == "negative"]
    parts = []
    if pos:
        ious = np.array([r["iou"] for r in pos])
        corrected = np.average(ious, weights=[1 / r["weight"] for r in pos])
        smalls = [s for r in pos for s in r["small_panel_ious"]]
        recalled = sum(s >= RECALL_IOU for s in smalls)
        parts.append(
            f"positives {len(pos)}: IoU {ious.mean():.3f} (corrected {corrected:.3f}), "
            f"cut {share([r['n_gen'] > 1 for r in pos])}, "
            f"right count {share([r['n_gen'] == r['n_truth'] for r in pos])}, "
            f"IoU>={GOOD_IOU} {share(list(ious >= GOOD_IOU))}, "
            f"small panels {recalled}/{len(smalls)}"
        )
    if neg:
        parts.append(
            f"negatives {len(neg)}: left whole {share([r['ok'] for r in neg])}"
        )
    return "; ".join(parts)


def summarize(records: list[dict]) -> list[str]:
    """The report: headline, by fold, by volume size, and outcomes by panel count."""
    lines = [summary_line(records)]
    for key, values in (("fold", ("train", "test")), ("size_band", SIZE_BANDS)):
        present = [v for v in values if any(r[key] == v for r in records)]
        if present:
            lines.append(f"\nby {key}:")
            for value in present:
                group = [r for r in records if r[key] == value]
                lines.append(f"  {value:6s} {summary_line(group)}")
    pos = [r for r in records if r["kind"] == "positive"]
    if pos:
        lines.append("\npositives by truth panel count:")
        lines.append(
            f"  {'panels':>6s} {'pages':>6s} {'IoU':>6s} "
            + " ".join(f"{o:>9s}" for o in ("whole", "too few", "right", "too many"))
        )
        for n in (2, 3, 4, 5):
            group = [
                r for r in pos if r["n_truth"] == n or (n == 5 and r["n_truth"] > 5)
            ]
            if group:
                outcomes = [outcome(r["n_truth"], r["n_gen"]) for r in group]
                lines.append(
                    f"  {str(n) + ('+' if n == 5 else ''):>6s} {len(group):6d} "
                    f"{np.mean([r['iou'] for r in group]):6.3f} "
                    + " ".join(
                        f"{share([o == name for o in outcomes]):>9s}"
                        for name in ("whole", "too few", "right", "too many")
                    )
                )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Score a benchmark manifest.tsv instead of the volumes in --data-dir.",
    )
    parser.add_argument(
        "--panels-dir",
        type=Path,
        default=None,
        help="Read a run's <image stem>.panels.json from here instead of splitting.",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=-1,
        metavar="N",
        help="Sample N negative pages from --data-dir (-1 = all; 0 = skip negatives).",
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

    cases = (
        manifest_cases(args.manifest)
        if args.manifest
        else gather_cases(args.data_dir, args.negatives, args.seed)
    )
    skipped = unfinished_names(args.panels_dir) if args.panels_dir else set()
    cases = [case for case in cases if case.image.stem not in skipped]
    print(
        f"{sum(c.truth is not None for c in cases)} positive sheets, "
        f"{sum(c.truth is None for c in cases)} negatives"
        + (f" ({len(skipped)} unfinished skipped)" if skipped else ""),
        file=sys.stderr,
    )

    records = []
    for case in cases:
        with Image.open(case.image) as img:
            size = img.size
        if args.panels_dir:
            gen = run_panels(args.panels_dir, case, size)
        else:
            gen = [
                make_valid(p)
                for p in compute_panels(
                    case.image,
                    min_panel_frac=args.min_panel_frac,
                    small_face_policy=args.small_face,
                    sheet=case.sheet(),
                )
            ]
        records.append(score_record(case, gen, size))

    if not args.manifest:
        by_volume: dict[str, list[float]] = defaultdict(list)
        for r in records:
            if r["kind"] == "positive":
                by_volume[r["volume"]].append(r["iou"])
        print(f"\n{'volume':28s} {'sheets':>6s} {'mean IoU':>9s}")
        for volume in sorted(by_volume):
            vals = by_volume[volume]
            print(f"{volume:28s} {len(vals):6d} {sum(vals) / len(vals):9.3f}")
    print()
    print("\n".join(summarize(records)))
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
