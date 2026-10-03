"""Research harness for the OSM snap channel: truth-graded reports and gate sweeps.

The production engine (volume loading, candidate search and caching, the
rescue/arbitration/refinement gates) lives in snap_volume.py and runs as
`mapsnap snap`. This module keeps the commands that need truth or exist to
tune that engine -- reports against main.iiif.json and the gate sweeps -- and
imports everything else from snap_volume.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from mapsnap.page_units import grid_rmse_ft_between, load_page_units
from mapsnap.snap_volume import (
    PRODUCTION_ARBITRATE_GATE,
    PRODUCTION_GATE_MARGIN,
    PRODUCTION_GATE_SCORE,
    RESCUE_STATES,
    arbitrate_challenge,
    artifacts_dir,
    attach_missing_truth,
    cmd_candidates,
    cmd_materialize,
    cmd_select,
    distinct_margin,
    load_candidates,
    load_panel_units,
    refine_adoption,
)
from mapsnap.utils import default_centerlines

"""Search radius when challenging a defensible incumbent: covers refinement
(<100 ft agreement) and rung flips (co-located) with margin, nothing more."""


"""Run snap's half-sheet pass: re-rescue one half of a two-scan sheet from its other.

Some volumes scan each sheet as two halves, pNL and pNR, cut at the gutter. The
labels split with them -- "WEST 32nd" on the left, "STREET" on the right -- so
one half often fits and the other has nothing to fit with. The two are one rigid
sheet: across the 1,481 corpus-v1 sheets where both halves were placed, their
scales agree to 2% and their rotations to about 1 degree (10th-90th percentile).
"""


"""How much of a half's width the two scans share at the gutter.

The median over those 1,207 both-placed pairs whose halves sit side by side
(interquartile range 0.040-0.088); the tops of the two scans line up
(median offset 0.1% of the height).
"""


"""Search radius around a half-sheet seed: half a Manhattan street block.

The seed lands a median 13 m from where its half was independently placed. A
wider search only admits the grid's aliases: at 150 m, Manhattan 1899 vol 5's
right halves found candidates one and two streets away (80 m, 160 m) that
verified within 0.003 of the seed's, and several won. Refinement may slide a
candidate at most REFINE_SHIFT_MAX_M further, so the next street is out of reach.
"""


"""Rotation-prior sigma for a half-sheet seed: the halves agree to ~1 degree."""


"""A seeded candidate turned further than this from its sibling is not the same sheet."""


"""A panel smaller than this share of its half is an inset or a fragment, not the map.

An inset draws somewhere else entirely (Denver p66R__2, a 12% box of a sheet
outside the volume, was placed on the main grid from its seed), and a sliver
the splitter cut off along a diagonal street (Buffalo p104R__3, 12%) slid 40 m
along it. Measured on four volumes' split halves: insets and slivers are
0.04-0.12 of the half, map panels 0.27 and up.
"""


def cmd_report(volume: Path) -> None:
    """Recall / ranking diagnostics for the cached candidates, against truth."""
    records = [
        r for r in load_candidates(volume) if r.get("fit_state") in RESCUE_STATES
    ]
    by_status: dict[str, int] = {}
    for record in records:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
    print(f"== {volume.name}: {len(records)} pages ==")
    print("  status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))

    scored = [
        r
        for r in records
        if r["status"] == "ok" and r.get("has_truth") and r.get("candidates")
    ]
    if not scored:
        print("  no truth-scored pages")
        return
    recall50 = rank1_50 = rank1_25 = 0
    best_rmses: list[float] = []
    print(
        f"  {'page':<9}{'state':<9}{'cands':>6}{'best set':>10}{'rank-1':>9}"
        f"{'score':>8}{'margin':>8}  theta_src"
    )
    for record in scored:
        candidates = record["candidates"]
        rmses = [c["rmse_ft"] for c in candidates if "rmse_ft" in c]
        if not rmses:
            continue
        best_in_set = min(rmses)
        top = candidates[0]
        top_rmse = top.get("rmse_ft")
        if best_in_set <= 50.0:
            recall50 += 1
        if top_rmse is not None and top_rmse <= 50.0:
            rank1_50 += 1
        if top_rmse is not None and top_rmse <= 25.0:
            rank1_25 += 1
        if top_rmse is not None:
            best_rmses.append(top_rmse)
        print(
            f"  {record['target']:<9}{record['fit_state']:<9}{len(candidates):>6}"
            f"{best_in_set:>9.0f}f{top_rmse:>8.0f}f"
            f"{top.get('select_score') if top.get('select_score') is not None else float('nan'):>8.2f}"
            f"{record.get('margin') if record.get('margin') is not None else float('nan'):>8.2f}"
            f"  {top['theta_source']}"
        )
    n = len(scored)
    print(
        f"  truth-in-top-K (<=50ft): {recall50}/{n} ({recall50 / n:.0%})   "
        f"rank-1 <=50ft: {rank1_50}/{n} ({rank1_50 / n:.0%})   "
        f"rank-1 <=25ft: {rank1_25}/{n} ({rank1_25 / n:.0%})"
    )
    if best_rmses:
        best_rmses.sort()
        median = best_rmses[len(best_rmses) // 2]
        print(f"  rank-1 rmse: median {median:.0f}ft, max {best_rmses[-1]:.0f}ft")


def truth_land_weights(volume: Path) -> tuple[dict[str, float], float]:
    """(land m² per unsplit truth page key, total land over ALL truth items).

    Approximates the `mapsnap score` land weighting closely enough to tune
    gates on cached candidates without re-running iiif+score per setting; the
    final numbers always come from the real pipeline.
    """
    from shapely.geometry import Polygon

    from mapsnap.score import (
        LocalFrame,
        land_fraction,
        street_tree,
        truth_footprint_ring,
    )
    from mapsnap.utils import source_id_to_page_key

    items = json.loads((volume / "main.iiif.json").read_text()).get("items", [])
    centerlines = default_centerlines(volume)
    weights: dict[str, float] = {}
    total = 0.0
    frame: LocalFrame | None = None
    tree = None
    for item in items:
        ring = truth_footprint_ring(item)
        if not ring:
            continue
        if frame is None:
            frame = LocalFrame(ring[0][0], ring[0][1])
            assert centerlines is not None
            tree = street_tree(centerlines, frame)
        polygon = Polygon([frame.to_xy(lon, lat) for lon, lat in ring]).buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            continue
        assert tree is not None
        land = polygon.area * land_fraction(polygon, tree)
        total += land
        key = source_id_to_page_key(
            item.get("target", {}).get("source", {}).get("id"), item.get("label", "")
        )
        if "__" not in key:
            weights[key] = weights.get(key, 0.0) + land
    return weights, total


def simulate_delta_net(
    records: list[dict],
    weights: dict[str, float],
    total_land: float,
    gate_score: float,
    gate_margin: float,
) -> tuple[float, int, int, int]:
    """(simulated Δnet, accepted, good adds, disaster adds) at one gate setting."""
    accepted = good = disaster = 0
    delta = 0.0
    for record in records:
        if record.get("status") != "ok" or not record.get("candidates"):
            continue
        top = record["candidates"][0]
        score = top.get("select_score")
        margin = distinct_margin(record)
        if score is None or score < gate_score:
            continue
        if margin is None or margin < gate_margin:
            continue
        accepted += 1
        rmse = top.get("rmse_ft")
        weight = weights.get(record["target"])
        if rmse is None or weight is None:
            continue
        if rmse <= 25.0:
            good += 1
            delta += weight
        elif rmse >= 200.0:
            disaster += 1
            delta -= weight
    return (delta / total_land if total_land else 0.0), accepted, good, disaster


def cmd_sweep(volume: Path) -> None:
    """Grid the abstention gates and print the simulated Δnet for each."""
    records = [
        r for r in load_candidates(volume) if r.get("fit_state") in RESCUE_STATES
    ]
    weights, total_land = truth_land_weights(volume)
    print(f"== {volume.name}: simulated Δnet over gate grid ==")
    print(f"  {'gate':>6} " + "".join(f"m>={m:<4.1f}" + " " * 14 for m in GATE_MARGINS))
    for gate in GATE_SCORES:
        cells = []
        for margin in GATE_MARGINS:
            delta, accepted, good, disaster = simulate_delta_net(
                records, weights, total_land, gate, margin
            )
            cells.append(
                f"{delta * 100:+5.1f}% ({accepted:>2}a {good:>2}g {disaster}d)"
            )
        print(f"  {gate:>6.2f} " + "  ".join(cells))
    print("  (a=accepted, g=good <=25ft, d=disaster >=200ft; Δnet is land-weighted)")


GATE_SCORES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


GATE_MARGINS = [0.0, 0.1, 0.25, 0.5]


ARBITRATE_GATES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75]


def cmd_sweep_arbitrate(volume: Path) -> None:
    """Grid the arbitration gate; print the simulated Δnet from challenges."""
    records = load_candidates(volume)
    weights, total_land = truth_land_weights(volume)

    def bucket_value(rmse: float | None) -> int:
        if rmse is None:
            return 0
        if rmse <= 25.0:
            return 1
        if rmse >= 200.0:
            return -1
        return 0

    print(f"== {volume.name}: simulated arbitration Δnet ==")
    for gate in ARBITRATE_GATES:
        delta = 0.0
        challenged = improved = worsened = unweighted = 0
        details = []
        for record in records:
            if record.get("fit_state") != "fitted":
                continue
            challenge = arbitrate_challenge(record, gate)
            if challenge is None:
                continue
            challenged += 1
            old_rmse = (record.get("incumbent") or {}).get("rmse_ft")
            new_rmse = record["candidates"][0].get("rmse_ft")
            if new_rmse is not None and old_rmse is not None:
                if new_rmse < old_rmse:
                    improved += 1
                elif new_rmse > old_rmse:
                    worsened += 1
                details.append(f"{record['target']}:{old_rmse:.0f}->{new_rmse:.0f}")
            weight = weights.get(record["target"])
            if weight is None or old_rmse is None or new_rmse is None:
                unweighted += 1
                continue
            delta += weight * (bucket_value(new_rmse) - bucket_value(old_rmse))
        net = delta / total_land if total_land else 0.0
        print(
            f"  gate {gate:>5.2f}: {net * 100:+5.1f}%  {challenged} challenged"
            f" ({improved} better, {worsened} worse, {unweighted} unweighted)"
        )
        if details:
            print("      " + "  ".join(details[:10]))


# Volume-level train/holdout split for the refinement-margin sweep (#153).
# The dev-4 volumes tuned every other constant, so they stay on the train
# side; NO-1896's truth is known-noisy, so it trains rather than judges;
# chicago balances the split at 6/6.
REFINE_SWEEP_TRAIN = {
    "chicago_il_1950_vol_1",
    "detroit_mich_1929_vol_11",
    "hudson_co_nj_1950_vol_9",
    "los_angeles_ca_1949_vol_14",
    "new_orleans_la_1896_vol_2",
    "washington_dc_1916_vol_2",
}


REFINE_SWEEP_MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]


# Band-aware margins: below the incumbent-verification edge the incumbent is
# weakly supported (a low margin is safe); above it the incumbent is already
# well-verified and a higher bar protects it from churn. inf = never refine.
REFINE_SWEEP_BAND_EDGES = [0.25, 0.5, 0.75, 1.0]


REFINE_SWEEP_BAND_MARGINS = [0.0, 0.05, 0.1, 0.2, 0.4, math.inf]


def truth_item_land_weights(volume: Path) -> dict[str, float]:
    """Land-weighted area (m^2) for every truth item, split panels included.

    Unlike truth_land_weights (whole pages only, for the rescue sweeps), this
    keys every truth item by its own page key — the region-graded scorer
    grades split panels individually, so the refinement sweep needs their
    individual weights.
    """
    from shapely.geometry import Polygon

    from mapsnap.score import (
        LocalFrame,
        land_fraction,
        street_tree,
        truth_footprint_ring,
    )
    from mapsnap.utils import source_id_to_page_key

    items = json.loads((volume / "main.iiif.json").read_text()).get("items", [])
    centerlines = default_centerlines(volume)
    weights: dict[str, float] = {}
    frame: LocalFrame | None = None
    tree = None
    for item in items:
        ring = truth_footprint_ring(item)
        if not ring:
            continue
        if frame is None:
            frame = LocalFrame(ring[0][0], ring[0][1])
            assert centerlines is not None
            tree = street_tree(centerlines, frame)
        polygon = Polygon([frame.to_xy(lon, lat) for lon, lat in ring]).buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            continue
        assert tree is not None
        key = source_id_to_page_key(
            item.get("target", {}).get("source", {}).get("id"), item.get("label", "")
        )
        land = polygon.area * land_fraction(polygon, tree)
        weights[key] = weights.get(key, 0.0) + land
    return weights


def refine_eligible_features(records: list[dict]) -> dict[str, dict]:
    """Per-target features for every fitted page refinement could ever adopt.

    Eligibility mirrors cmd_select's arbitrate branch with the margin removed:
    fitted, not claimed by arbitration, and carrying an agreeing top
    challenger with a verification head-to-head available. The sweep applies
    margin rules to these features in-process.
    """
    eligible: dict[str, dict] = {}
    for record in records:
        if record.get("fit_state") != "fitted":
            continue
        if arbitrate_challenge(record, PRODUCTION_ARBITRATE_GATE) is not None:
            continue
        adoption = refine_adoption(record, margin=-math.inf)
        if adoption is None:
            continue
        incumbent = record["incumbent"]
        top = record["candidates"][0]
        eligible[record["target"]] = {
            "incumbent_verification": incumbent["verification"],
            "challenger_verification": top["verification"],
            "incumbent_name": (incumbent.get("name") or {}).get("score") or 0.0,
            "challenger_name": (top.get("name") or {}).get("score") or 0.0,
            "disagreement_ft": adoption["disagreement_ft"],
            "incumbent_rmse_ft": incumbent.get("rmse_ft"),
            "challenger_rmse_ft": top.get("rmse_ft"),
        }
    return eligible


def build_hybrid_iiif(volume: Path, output: Path) -> None:
    """Build the osm-first hybrid IIIF for the volume's current sidecars."""
    import subprocess

    georef_glob = f"{volume / '*.georef-snap.json'},{volume / '*.georef.json'}"
    subprocess.run(
        [
            "mapsnap",
            "iiif",
            str(volume / "main.iiif.json"),
            georef_glob,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def grade_refine_variants(volume: Path, recompute: bool = False) -> dict:
    """Grade every truth item with refinement fully off and fully on.

    Materializes two sidecar variants (refine_margin +inf / -inf), builds the
    hybrid IIIF for each, and grades both through the real region-graded
    scorer (compare_pages) so the sweep is exactly faithful to `mapsnap
    score`. Restores the production selection afterwards. Cached in
    artifacts/osm_snap/refine_sweep.json.
    """
    from mapsnap.compare_iiif_georef import compare_pages

    cache_path = artifacts_dir(volume) / "refine_sweep.json"
    candidates_path = artifacts_dir(volume) / "candidates.jsonl"
    if (
        cache_path.exists()
        and not recompute
        and cache_path.stat().st_mtime >= candidates_path.stat().st_mtime
    ):
        return canonicalize_refine_keys(json.loads(cache_path.read_text()))

    records = load_candidates(volume)
    eligible = refine_eligible_features(records)
    graded: dict[str, dict[str, dict]] = {}
    try:
        for variant, margin in (("none", math.inf), ("all", -math.inf)):
            cmd_select(
                volume,
                "arbitrate",
                PRODUCTION_GATE_SCORE,
                PRODUCTION_GATE_MARGIN,
                PRODUCTION_ARBITRATE_GATE,
                refine_margin=margin,
            )
            cmd_materialize(volume, "arbitrate")
            variant_iiif = volume / f"refine-sweep-{variant}.iiif.json"
            build_hybrid_iiif(volume, variant_iiif)
            rows, missing = compare_pages(volume / "main.iiif.json", variant_iiif)
            variant_iiif.unlink()
            by_key: dict[str, dict] = {}
            for row in rows:
                by_key.setdefault(
                    row["page_key"],
                    {"gen_key": row["gen_page_key"], "rmse_ft": row["rmse_ft"]},
                )
            for row in missing:
                by_key.setdefault(row["page_key"], {"gen_key": None, "rmse_ft": None})
            graded[variant] = by_key
    finally:
        # Leave the volume's sidecars and selection in the production state.
        cmd_select(
            volume,
            "arbitrate",
            PRODUCTION_GATE_SCORE,
            PRODUCTION_GATE_MARGIN,
            PRODUCTION_ARBITRATE_GATE,
        )
        cmd_materialize(volume, "arbitrate")

    weights = truth_item_land_weights(volume)
    none_rows, all_rows = graded["none"], graded["all"]
    items = []
    total_land = 0.0
    for key in sorted(none_rows):
        weight = weights.get(key)
        if weight is None:
            continue
        total_land += weight
        none_row = none_rows[key]
        all_row = all_rows.get(key, none_row)
        items.append(
            {
                "key": key,
                "gen_key": none_row["gen_key"],
                "land_m2": weight,
                "rmse_none": none_row["rmse_ft"],
                "rmse_all": all_row["rmse_ft"],
            }
        )
    result = canonicalize_refine_keys(
        {
            "volume": volume.name,
            "eligible": eligible,
            "total_land_m2": total_land,
            "items": items,
        }
    )
    cache_path.write_text(json.dumps(result))
    return result


def canonicalize_refine_keys(result: dict) -> dict:
    """Join sidecar-cased targets with truth-cased gen keys, in place.

    Sidecar stems are lowercase for some volumes (chicago p101w) while the
    truth annotations carry the case (p101W); compare's gen_page_key uses the
    truth's casing, so without this remap those pages' adoptions silently
    fall out of every margin rule's outcome. Idempotent.
    """
    canonical = {target.lower(): target for target in result["eligible"]}
    for item in result["items"]:
        gen_key = item.get("gen_key")
        if gen_key is not None:
            item["gen_key"] = canonical.get(gen_key.lower(), gen_key)
    return result


def refine_bucket(rmse_ft: float | None) -> int:
    """Score-bucket value of one truth item: +1 good, -1 disaster, else 0."""
    if rmse_ft is None:
        return 0
    if rmse_ft <= 25.0:
        return 1
    if rmse_ft >= 200.0:
        return -1
    return 0


def refine_adopt_set(
    eligible: dict[str, dict],
    margin: float,
    name_parity: bool = False,
    band: tuple[float, float, float] | None = None,
) -> set[str]:
    """Targets a margin rule adopts.

    band=(edge, low, high) overrides margin: low applies below the
    incumbent-verification edge, high above it. name_parity additionally
    requires the challenger's name score to be at least the incumbent's.
    """
    adopted = set()
    for target, features in eligible.items():
        rule_margin = margin
        if band is not None:
            edge, low, high = band
            rule_margin = low if features["incumbent_verification"] < edge else high
        head_to_head = (
            features["challenger_verification"] - features["incumbent_verification"]
        )
        if head_to_head <= rule_margin:
            continue
        if name_parity and features["challenger_name"] < features["incumbent_name"]:
            continue
        adopted.add(target)
    return adopted


def refine_rule_outcome(
    volume_data: dict, adopted: set[str]
) -> tuple[float, float, int, int]:
    """(Δland_m2 vs no refinement, total land_m2, bucket gains, losses)."""
    delta = 0.0
    gains = losses = 0
    for item in volume_data["items"]:
        if item["gen_key"] not in adopted:
            continue
        before = refine_bucket(item["rmse_none"])
        after = refine_bucket(item["rmse_all"])
        if after == before:
            continue
        delta += (after - before) * item["land_m2"]
        if after > before:
            gains += 1
        else:
            losses += 1
    return delta, volume_data["total_land_m2"], gains, losses


def cmd_sweep_refine(volumes: list[Path], recompute: bool = False) -> None:
    """Sweep the refinement margin (#153) against the region-graded scorer."""
    data = [grade_refine_variants(volume, recompute) for volume in volumes]
    mismatched = sum(
        1
        for d in data
        for item in d["items"]
        if item["gen_key"] not in d["eligible"]
        and item["rmse_none"] != item["rmse_all"]
    )
    if mismatched:
        print(f"WARNING: {mismatched} non-eligible item(s) changed between variants")
    train = [d for d in data if d["volume"] in REFINE_SWEEP_TRAIN]
    holdout = [d for d in data if d["volume"] not in REFINE_SWEEP_TRAIN]
    n_eligible = sum(len(d["eligible"]) for d in data)
    print(
        f"{n_eligible} refinement-eligible pages across {len(data)} volume(s)"
        f" ({len(train)} train, {len(holdout)} holdout)"
    )

    def evaluate(
        subset: list[dict],
        margin: float,
        name_parity: bool = False,
        band: tuple[float, float, float] | None = None,
    ) -> tuple[float, int, int, int]:
        delta = land = 0.0
        adopted_total = gains = losses = 0
        for volume_data in subset:
            adopted = refine_adopt_set(
                volume_data["eligible"], margin, name_parity, band
            )
            adopted_total += len(adopted)
            vol_delta, vol_land, vol_gains, vol_losses = refine_rule_outcome(
                volume_data, adopted
            )
            delta += vol_delta
            land += vol_land
            gains += vol_gains
            losses += vol_losses
        return (delta / land if land else 0.0), adopted_total, gains, losses

    def print_row(
        label: str,
        margin: float,
        name_parity: bool = False,
        band: tuple[float, float, float] | None = None,
    ) -> None:
        cells = [f"  {label:<24}"]
        for subset, name in ((train, "train"), (holdout, "hold"), (data, "all")):
            net, adopted, gains, losses = evaluate(subset, margin, name_parity, band)
            cells.append(
                f"{name} {net * 100:+5.2f}% ({adopted:>3}a {gains:>3}g {losses:>2}l)"
            )
        print("  ".join(cells))

    print("== global margin sweep (a=adopted pages, g/l=bucket gains/losses) ==")
    for margin in REFINE_SWEEP_MARGINS:
        print_row(f"margin {margin:.2f}", margin)
    print("== + name parity (challenger name score >= incumbent's) ==")
    for margin in REFINE_SWEEP_MARGINS:
        print_row(f"margin {margin:.2f} +name", margin, name_parity=True)
    print("== band-aware margins, top 10 by train Δnet ==")
    combos = []
    for edge in REFINE_SWEEP_BAND_EDGES:
        for low in REFINE_SWEEP_BAND_MARGINS:
            for high in REFINE_SWEEP_BAND_MARGINS:
                band = (edge, low, high)
                train_net, _, _, train_losses = evaluate(train, 0.0, band=band)
                combos.append((train_net, -train_losses, band))
    combos.sort(reverse=True)
    for train_net, _, band in combos[:10]:
        edge, low, high = band
        print_row(f"edge {edge:.2f} lo {low:g} hi {high:g}", 0.0, band=band)


"""Relaxed rescue bar for stamp-corroborated candidates.

A contradiction-demoted page's rescue candidate that lands its printed claim
back on the hinting neighbor's stamp carries external evidence the select
score cannot see — the neighbor's printed testimony pins the pose at the seam.
Measured true poses refused by the 1.25 bar: KC p551 at 0.77 (24 ft), GR p828
at 0.93 (23 ft). Stamp-INconsistent candidates are already implausible, so the
relaxed bar only ever admits poses the neighbors vouch for; the margin is
computed against the best non-corroborated rival (NO-1896 p125's true pose at
1.82 was margin-blocked by its own 16-degree twin)."""


"""Candidate/incumbent scale ratios treated as an up-rung challenge.

UP only, never down: verification carries a small-footprint bias. A half-scale
candidate explains 1/4 the ground and matches dense OSM cheaply — calibrating
over every rung-band candidate pair in the twelve truth volumes, every
would-be DOWN flip that passed any margin was a break (23-98 ft incumbents sent
to 340-11000 ft: NO-1951 p433/p434/p436, Chicago p61w, Detroit p21, Nashville
p44, LA p1499o/r), while every fix but one was an up flip. A doubled candidate
must explain 4x the ground, so verification preferring it is hard-won evidence.
"""


"""How closely a scale must match the page's printed note to claim its authority."""


OVERLAP_HARD = 0.5  # and where it becomes prohibitive


def cmd_reannotate(volume: Path) -> None:
    """Refresh the rmse_ft annotations on cached candidates from current truth.

    Cheap (no matching): recomputes every candidate's grid rmse against the
    unit's truth affine, including pages that attach_missing_truth
    now covers (case-mismatched keys and split-only truth). Rewrites candidates.jsonl in place.
    """
    unit_list = load_page_units(volume) + load_panel_units(volume)
    units = {u.stem: u for u in unit_list}
    newly = attach_missing_truth(volume, unit_list)
    records = load_candidates(volume)
    changed = 0
    for record in records:
        unit = units.get(record["target"])
        if unit is None:
            continue
        has_truth = unit.truth is not None
        if record.get("has_truth") != has_truth:
            record["has_truth"] = has_truth
            changed += 1
        for candidate in record.get("candidates") or []:
            if unit.truth is None:
                candidate.pop("rmse_ft", None)
                continue
            candidate["rmse_ft"] = round(
                grid_rmse_ft_between(
                    unit.truth.affine_local,
                    np.array(candidate["world_affine"]),
                    unit.width,
                    unit.height,
                ),
                1,
            )
        incumbent = record.get("incumbent")
        if incumbent is not None:
            if unit.truth is not None and unit.gen_affine is not None:
                incumbent["rmse_ft"] = round(
                    grid_rmse_ft_between(
                        unit.truth.affine_local,
                        unit.gen_affine,
                        unit.width,
                        unit.height,
                    ),
                    1,
                )
            else:
                incumbent.pop("rmse_ft", None)
    out_path = artifacts_dir(volume) / "candidates.jsonl"
    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(
        f"{volume.name}: {newly} split-truth pages attached, "
        f"{changed} records flipped has_truth"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cand = sub.add_parser("candidates", help="generate snap candidates")
    p_cand.add_argument("volume", type=Path)
    p_cand.add_argument("--pages", type=str, default=None, help="comma-separated stems")
    p_cand.add_argument(
        "--all-pages",
        action="store_true",
        help="include fitted pages (arbitration study), not just rescue targets",
    )
    p_cand.add_argument("--limit", type=int, default=None)
    p_cand.add_argument("--recompute", action="store_true")
    p_cand.add_argument("--no-vis", action="store_true", help="skip contact sheets")
    p_cand.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help="worker processes for the per-page matching pass (default: %(default)s)",
    )

    p_rep = sub.add_parser("report", help="ranking diagnostics vs truth")
    p_rep.add_argument("volume", type=Path, nargs="+")
    p_rep.add_argument(
        "--sweep", action="store_true", help="grid the gates, print simulated Δnet"
    )
    p_rep.add_argument(
        "--sweep-arbitrate",
        action="store_true",
        help="grid the arbitration gate over fitted-page challenges",
    )
    p_rep.add_argument(
        "--sweep-refine",
        action="store_true",
        help=(
            "sweep the refinement margin against the region-graded scorer "
            "(#153); accepts multiple volumes for the train/holdout split"
        ),
    )
    p_rep.add_argument(
        "--recompute",
        action="store_true",
        help="rebuild the cached refine-sweep gradings",
    )

    p_sel = sub.add_parser("select", help="pick candidates / abstain per page")
    p_sel.add_argument("volume", type=Path)
    p_sel.add_argument(
        "--mode",
        choices=["argmax", "volume", "union", "arbitrate"],
        default="argmax",
        help=(
            "argmax/volume: single rescue committee; union: both committees; "
            "arbitrate: union PLUS challenges and refinements of placed fits"
        ),
    )
    p_sel.add_argument("--gate-score", type=float, default=PRODUCTION_GATE_SCORE)
    p_sel.add_argument("--gate-margin", type=float, default=PRODUCTION_GATE_MARGIN)
    p_sel.add_argument(
        "--arbitrate-gate", type=float, default=PRODUCTION_ARBITRATE_GATE
    )

    p_mat = sub.add_parser("materialize", help="write pN.georef-snap.json sidecars")
    p_mat.add_argument("volume", type=Path)
    p_mat.add_argument(
        "--mode",
        choices=["argmax", "volume", "union", "arbitrate"],
        default="argmax",
    )

    p_re = sub.add_parser("reannotate", help="refresh cached rmse annotations")
    p_re.add_argument("volume", type=Path)

    args = parser.parse_args()
    if args.command == "candidates":
        cmd_candidates(
            args.volume,
            args.pages.split(",") if args.pages else None,
            args.all_pages,
            args.limit,
            args.recompute,
            vis=not args.no_vis,
            num_workers=args.num_workers,
        )
    elif args.command == "report":
        if args.sweep_refine:
            cmd_sweep_refine(args.volume, args.recompute)
        elif args.sweep_arbitrate:
            cmd_sweep_arbitrate(args.volume[0])
        elif args.sweep:
            cmd_sweep(args.volume[0])
        else:
            cmd_report(args.volume[0])
    elif args.command == "select":
        cmd_select(
            args.volume,
            args.mode,
            args.gate_score,
            args.gate_margin,
            args.arbitrate_gate,
        )
    elif args.command == "materialize":
        cmd_materialize(args.volume, args.mode)
    elif args.command == "reannotate":
        cmd_reannotate(args.volume)


if __name__ == "__main__":
    main()
