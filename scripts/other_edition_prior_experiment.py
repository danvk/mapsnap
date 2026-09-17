#!/usr/bin/env python3
"""Rescue pages from another edition's placement instead of the key map's.

Sanborn re-issued a volume under the same number for decades and kept its sheet
numbers, so a placed edition tells you where every sheet of another edition sits.
Across the pairings under `data/`, the same sheet number lands a few metres
apart (median) and always within 300 m across editions, against a random-page
null of over a kilometre.

This experiment runs snap's own candidate generation and decision trace
(osm_snap_experiment.page_record, the production rescue rules) on every page of
VOLUME the other edition places, treating each as UNPLACED so the rescue path
decides, with the search center taken from one of two priors:

  --arm keymap    the page's recorded key-map location and the volume radius
                  (what snap uses today)
  --arm edition   the other edition's center, --radius-m wide (default 50)

Chicago 1950 vol 1 against its 1906 issue (74 sheets): the key-map prior sits a
median 1,064 m from truth and rescues 9 (6 good, 2 disasters); the edition prior
sits 6 m from truth and rescues 56 (54 within 25 ft, 0 disasters). The three
pages the full pipeline could not place are all rescued at 7 ft. At a 100 m
window a one-block alias enters the search and the margin rule refuses them, so
the window must stay inside the pairing's measured agreement.

Nothing is written into VOLUME; page_record only reads. The road-UNet maps must
already be cached (artifacts/edge_join/roadprob), or a GPU is needed.

    uv run python scripts/other_edition_prior_experiment.py data/chicago_il_1950_vol_1 \\
        --other-edition data/chicago_il_1906_vol_1/main.iiif.json --arm edition \\
        --out edition.jsonl
    uv run python scripts/other_edition_prior_experiment.py data/chicago_il_1950_vol_1 \\
        --other-edition-centers centers.json --arm keymap --out keymap.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
from pathlib import Path

from mapsnap import osm_snap_experiment as ose
from mapsnap.other_edition_prior import annotation_centers, section_key
from mapsnap.utils import haversine_m


def load_centers(path: Path) -> dict[str, tuple[float, float]]:
    """Section key -> (lon, lat) from a JSON object {key: [lon, lat]}.

    Keys that do not name a numbered sheet are dropped, as annotation_centers
    drops them, rather than collapsing onto one empty key.
    """
    doc = json.loads(path.read_text())
    return {
        section_key(key): (float(value[0]), float(value[1]))
        for key, value in doc.items()
        if section_key(key)
    }


def keymap_centers(
    volume: Path, unit: ose.PageUnit, vctx: ose.VolumeContext
) -> tuple[list[tuple[float, float]], list[list[list[float]]] | None]:
    """(centers, regions) the run's own key-map prior gave this page."""
    georef_path = volume / f"{unit.stem}.georef.json"
    if georef_path.exists():
        keymap = json.loads(georef_path.read_text()).get("keymap")
        if keymap and keymap.get("centers"):
            centers = [(float(c[0]), float(c[1])) for c in keymap["centers"]]
            return centers, keymap.get("regions")
    if unit.keymap_centers:
        return list(unit.keymap_centers), unit.keymap_regions
    if vctx.locator is not None:
        entry = vctx.locator.page_keymap(unit.number)
        if entry:
            return [tuple(c) for c in entry["centers"]], entry.get("regions")
    return [], None


def truth_center(unit: ose.PageUnit) -> tuple[float, float] | None:
    """(lon, lat) of the page center under the truth pose, if the page has truth."""
    if unit.truth is None:
        return None
    affine = unit.truth.affine_local
    center_x, center_y = unit.width / 2.0, unit.height / 2.0
    return (
        float(affine[0, 0] * center_x + affine[0, 1] * center_y + affine[0, 2]),
        float(affine[1, 0] * center_x + affine[1, 1] * center_y + affine[1, 2]),
    )


def rescue_row(
    vctx: ose.VolumeContext,
    unit: ose.PageUnit,
    centers: list[tuple[float, float]],
    regions: list[list[list[float]]] | None,
    *,
    arm: str,
) -> dict:
    """Run page_record on the page as if unplaced, seeded from `centers`; summarize."""
    as_unplaced = dataclasses.replace(
        unit,
        fit_state="nofit",
        gen_affine=None,
        demoted_affine=None,
        gcp_hints=[],
        runner_up_affines=[],
        keymap_centers=centers,
        keymap_regions=regions,
    )
    record = ose.page_record(vctx, as_unplaced)
    candidates = record.get("candidates") or []
    top = candidates[0] if candidates else None
    decision = record.get("decision") or {}
    truth = truth_center(unit)
    center_to_truth = None
    if truth is not None and centers:
        center_to_truth = min(
            haversine_m(truth[1], truth[0], lat, lon) for lon, lat in centers
        )
    graded = [c["rmse_ft"] for c in candidates if c.get("rmse_ft") is not None]
    return {
        "stem": unit.stem,
        "arm": arm,
        "original_fit_state": unit.fit_state,
        "original_rmse_ft": unit.rmse_ft,
        "status": record.get("status"),
        "center_to_truth_m": None
        if center_to_truth is None
        else round(center_to_truth, 1),
        "n_candidates": len(candidates),
        "top_rmse_ft": top.get("rmse_ft") if top else None,
        "top_select_score": top.get("select_score") if top else None,
        "best_rmse_ft": min(graded) if graded else None,
        "margin": record.get("margin"),
        "verdict": decision.get("page_verdict"),
        "failed_rules": [
            b["rule"] for b in decision.get("bars", []) if b.get("verdict") == "fail"
        ],
    }


def summarize(rows: list[dict]) -> str:
    """One line: prior accuracy, rank-1 quality, and what the rescue rules accept."""
    rescued = [r for r in rows if r["verdict"] == "rescue"]
    distances = [
        r["center_to_truth_m"] for r in rows if r["center_to_truth_m"] is not None
    ]

    def within(values: list, limit: float) -> int:
        return sum(1 for v in values if v is not None and v <= limit)

    median = statistics.median(distances) if distances else math.nan
    return (
        f"pages {len(rows)}; center->truth median {median:.0f} m; "
        f"rank-1 <=25 ft {within([r['top_rmse_ft'] for r in rows], 25)}; "
        f"rescued {len(rescued)} (good {within([r['top_rmse_ft'] for r in rescued], 25)}, "
        f"disaster {sum(1 for r in rescued if (r['top_rmse_ft'] or 0) >= 200)}); "
        f"abstain {sum(1 for r in rows if r['verdict'] == 'abstain')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("volume", type=Path, help="mapsnap volume directory to rescue")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--other-edition", type=Path, help="a IIIF annotation placing another edition"
    )
    source.add_argument(
        "--other-edition-centers", type=Path, help="JSON {sheet key: [lon, lat]}"
    )
    parser.add_argument("--arm", choices=["keymap", "edition"], default="edition")
    parser.add_argument(
        "--radius-m", type=float, default=50.0, help="other-edition search radius"
    )
    parser.add_argument(
        "--stems", nargs="*", default=[], help="restrict to these page stems"
    )
    parser.add_argument("--out", type=Path, help="write one JSON row per page here")
    args = parser.parse_args()

    edition_centers = (
        annotation_centers(args.other_edition)
        if args.other_edition is not None
        else load_centers(args.other_edition_centers)
    )
    # Both arms are decided by what this harness injects, so the prior a
    # `snap --other-edition` run may have left in the volume is dropped: with one on
    # disk, other_edition_plan replaces the keymap arm's centers and overrules
    # --radius-m.
    vctx = dataclasses.replace(ose.load_volume_context(args.volume), other_edition=None)
    if args.arm == "edition":
        vctx = dataclasses.replace(
            vctx, radius_m=args.radius_m, radius_source="other-edition"
        )
    wanted = set(args.stems)
    rows: list[dict] = []
    for unit in vctx.units:
        key = section_key(unit.stem)
        if key not in edition_centers or (wanted and unit.stem not in wanted):
            continue
        if args.arm == "edition":
            centers, regions = [edition_centers[key]], None
        else:
            centers, regions = keymap_centers(args.volume, unit, vctx)
        row = rescue_row(vctx, unit, centers, regions, arm=args.arm)
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(summarize(rows))
    if args.out is not None:
        args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
