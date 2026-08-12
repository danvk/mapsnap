"""Report-only joint arbitration of every candidate pose per volume (#270 v1).

The fit pipeline publishes by stage precedence: each stage gates its
predecessors, publication is first-glob-wins over channel sidecars, and no
step ever compares two candidate poses for the same page (fargo p55:
``georef-keymap-outlier.json`` correct, ``georef-snap.json`` published, never
weighed against each other). Every regression autopsied on the 2026-08-10 run
was an ordering artifact of exactly this shape — displaced rescues enjoying
incumbency (washington_dc), a rank-2 pose at 11 ft losing to name score
(philadelphia p213), disasters published because snap's abstention cannot
unpublish (fargo p63__4).

``mapsnap reconcile`` selects per page among the poses ALREADY ON DISK — the
published pose, every rejected georef variant, the top snap candidates, the
street-solve pose, and explicit UNPLACED — by minimizing a joint energy:

- unary: the snap matcher's channel-neutral evidence (``evaluate_pose``:
  P(road)-vs-OSM verification + street-name alignment), an effective-GCP tier
  bonus, robust keymap-distance and rung/printed-note terms, and a small
  incumbent epsilon;
- pairwise: printed-stamp agreement between the CHOSEN poses of mutually
  claiming neighbors (robust, scale-aware — the adjacency gate's math as a
  factor instead of a demotion) and footprint overlap, whose threshold is
  self-calibrated per volume.

Arbitration-only: no continuous optimization; the optimizer is ICM over
hypothesis indices (generalizing snap's ``select_volume``), deterministic by
construction. Output goes under ``artifacts/reconcile/`` — verdicts.jsonl,
report.md, and (with ``--grade``) a materialized IIIF graded through the real
``compare_pages``. The volume root is never written.

    mapsnap reconcile data/fargo_nd_1958 --grade
    mapsnap reconcile data/washington_dc_1916_vol_2 --sidecars-from artifacts/reconcile-base
"""

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from mapsnap import sidecar
from mapsnap.adjacency_gate import (
    GATE_STAMP_M,
    FittedPage,
    edge_scale_factor,
    stamp_worlds,
)
from mapsnap.edge_join_experiment import (
    PageUnit,
    grid_rmse_ft_between,
)
from mapsnap.osm_snap import W_CONTAIN, W_NAME, evaluate_pose, region_containment_frac
from mapsnap.road_model import effective_gcp_count, page_world_affine
from mapsnap.utils import haversine_m

# The abstention bar: a pose must beat UNPLACED (energy 0) by scoring above
# this on the evaluate_pose evidence. Equal to snap's PRODUCTION_GATE_SCORE so
# reconcile's bar for existence matches the rescue committee's.
# Entry bar: a pose may only ENTER a page the pipeline left unplaced with
# rescue-grade evidence. Equal to snap's PRODUCTION_GATE_SCORE, whose rescue
# committee this mirrors.
ENTRY_PENALTY = 1.25
# KEEP prior: the MEASURED expected land-score value of publishing a pose,
# conditioned on the two features that predict it — how constrained the fit is
# (effective GCPs) and whether the geometric channel has anything to say
# (verification sign). Land-weighted over all 1,323 published poses of the
# 2026-08-10 corpus: E[keep] = good_share - disaster_share, directly
# comparable to E[unplaced] = 0 by construction.
#
# This replaces an invented GCP ladder and a hand-picked keep bar. It is also
# why "keep unless contradicted" is not a rule here: a 4+ GCP pose carries
# 0.57-0.94 of prior, so only real contradicting evidence can move it, while a
# 1-GCP pose with no P(road) support carries ~0 and is genuinely a coin flip
# (that bucket measures 0.224 good land against 0.224 disaster land).
KEEP_PRIOR = {
    ("0-1", False): 0.00,
    ("0-1", True): 0.69,
    ("2-3", False): 0.25,
    ("2-3", True): 0.68,
    ("4+", False): 0.57,
    ("4+", True): 0.94,
}
# Hypotheses closer than this are the same pose; keep one, merge provenance.
DEDUPE_FT = 10.0
# Robust keymap-distance term: (distance/radius)^2 clamped — SOFT prior, per
# the p55 lesson (the keymap location itself can be km wrong).
W_KEYMAP = 0.05
KEYMAP_CLAMP = 3.0
# Rung term: sitting ON any integer rung of the volume family costs nothing —
# second scale families are legitimate. Only a scale BETWEEN rungs pays; a
# printed note contradicted by the pose pays W_NOTE_MISMATCH.
W_RUNG_OFF = 0.15
W_NOTE_MISMATCH = 0.30
# Robust stamp factor between chosen neighbor poses, clamped so one junk edge
# cannot dominate (the fargo p64 lesson).
W_STAMP = 0.30
STAMP_CLAMP = 2.0
# Max-mixture junk model: beyond this multiple of the bar the disagreement is
# more likely a junk claim than a misplacement, so the cost goes flat.
STAMP_OUTLIER_RATIO = 5.0
STAMP_JUNK_COST = 0.15
# Footprint-overlap factor, self-calibrated per volume from the p90 of
# published adjacent pairs (adjacent sheets overlap at their seams).
W_OVERLAP = 3.0
OVERLAP_SOFT = 0.15
OVERLAP_HARD_DELTA = 0.35
# NO sibling factor. Split panels are separate maps that happen to share a
# sheet: no geographic relationship (champaign p4's panels sit 891 m apart at
# +0.2 and +48.9 degrees) and they may legitimately depict OVERLAPPING ground,
# because a small inset often details an area the large panel also covers
# (kansas_city p526: 0.33 overlap, both fits correct at 6.7 and 17.4 ft).
# Snap candidates entering the hypothesis set, by rank, plausible only.
SNAP_TOP_K = 3
# A raw candidate may ENTER an unplaced page only with snap's own admission
# evidence behind it: rank-1, cached select over the production gate, and the
# record margin over the production margin. Reconcile v1 arbitrates across
# channels; it does not relitigate rescues snap's calibrated gates declined
# (fargo p61__1/p16: entries at 1,022-1,375 ft that snap itself had refused).
ENTRY_SELECT_MIN = 1.25  # = PRODUCTION_GATE_SCORE
ENTRY_MARGIN_MIN = 0.25  # = PRODUCTION_GATE_MARGIN
# A pose the adjacency gate demoted carried a hard signal; re-admitting it
# must overcome that recorded evidence.
W_CONTRADICTED = 0.5
# Ambiguity: a NON-published pose whose evidence is within this of a rival at
# a genuinely different pose (>DISTINCT_M center distance) pays W_AMBIG —
# snap's distinct-margin gate as a factor. The smoke run's p2__2 showed why:
# three aliased candidates 10 km apart within 0.14 of each other, one of
# which would otherwise have entered at 10,352 ft. Published poses won a
# margin once already and are exempt.
W_AMBIG = 0.5
AMBIG_MARGIN = 0.25
DISTINCT_M = 100.0
# ICM
ICM_MAX_ROUNDS = 25
EXHAUSTIVE_LIMIT = 20_000
# Published-channel precedence, mirroring fit.py's IIIF glob order.
CHANNEL_ORDER = ("georef-street", "georef-snap", "georef")

UNPLACED = "unplaced"


@dataclass
class Hypothesis:
    """One candidate pose for a page (or the explicit unplaced state)."""

    source: str  # "georef" | "georef-snap" | ... | "snap:0" | "streets" | UNPLACED
    affine: np.ndarray | None  # page px -> (lon, lat); None iff unplaced
    effective_gcps: int = 0
    merged_sources: list[str] = field(default_factory=list)
    status: str = sidecar.VALID  # the writing channel's verdict on this pose
    scores: dict = field(default_factory=dict)
    unary_terms: dict = field(default_factory=dict)
    unary: float = 0.0


@dataclass
class PageNode:
    """A page (base or panel) with its hypothesis set."""

    unit: PageUnit
    is_panel: bool
    base: str | None
    hypotheses: list[Hypothesis]
    published_index: int | None  # index of the published pose, None if unplaced


def sidecar_pose(doc: dict) -> np.ndarray | None:
    """The world affine of a georef-variant doc, or None (e.g. -nofit)."""
    if not doc.get("corners") or not doc.get("width") or not doc.get("height"):
        return None
    return page_world_affine(doc)


def published_channel(sidecar_dir: Path, stem: str) -> str | None:
    """Which channel would publish this stem on its own account, or None.

    A channel sidecar now exists whether or not that channel stands behind the
    pose inside it (a demotion is a recorded verdict, not a rename), so the
    incumbent is the highest-precedence channel whose verdict is ACCEPTED —
    exactly the set the old glob-over-renamed-files arrangement published.
    """
    for channel in CHANNEL_ORDER:
        path = sidecar_dir / f"{stem}.{channel}.json"
        if not path.exists():
            continue
        try:
            if sidecar.internally_valid(json.loads(path.read_text())):
                return channel
        except (OSError, ValueError):
            continue
    return None


def collect_hypotheses(
    sidecar_dir: Path,
    stem: str,
    snap_record: dict | None,
    street_record: dict | None,
    page_size: tuple[int, int] | None = None,
) -> tuple[list[Hypothesis], int | None]:
    """(hypotheses, published index) for one page, deterministic order.

    Order: published channel first, remaining sidecar variants sorted, snap
    candidates by rank, street-solve pose, UNPLACED last. Every
    ``p<stem>.georef*.json`` is globbed directly — the shared GEOREF_VARIANTS
    list omits ``-keymap-outlier``, which is exactly the p55 pose this pass
    exists to weigh.
    """
    channel = published_channel(sidecar_dir, stem)
    variant_paths = sorted(sidecar_dir.glob(f"{stem}.georef*.json"))
    ordered: list[tuple[str, Path]] = []
    for path in variant_paths:
        name = path.name[len(stem) + 1 : -len(".json")]
        ordered.append((name, path))
    if channel is not None:
        ordered.sort(key=lambda item: (item[0] != channel, item[0]))

    hypotheses: list[Hypothesis] = []
    for name, path in ordered:
        doc = json.loads(path.read_text())
        # A channel's own sidecar plus any pose it reached and set aside (the
        # key-map retry keeps both). All of them are hypotheses; the status is
        # what the channel concluded, not whether the pose is worth weighing.
        for variant in (doc, *sidecar.rejected_poses(doc)):
            affine = sidecar_pose(variant)
            if affine is None:
                continue
            status = sidecar.status(variant)
            hypotheses.append(
                Hypothesis(
                    source=name if status == sidecar.VALID else f"{name}:{status}",
                    affine=affine,
                    effective_gcps=effective_gcp_count(variant),
                    status=status,
                )
            )
    if snap_record is not None:
        record_margin = snap_record.get("margin")
        rank = 0
        for candidate in snap_record.get("candidates") or []:
            if rank >= SNAP_TOP_K:
                break
            if not candidate.get("plausible") or candidate.get("world_affine") is None:
                continue
            # Entry credentials: snap's own admission verdict for this pose.
            select = candidate.get("select_score")
            admitted = (
                rank == 0
                and select is not None
                and select >= ENTRY_SELECT_MIN
                and record_margin is not None
                and record_margin >= ENTRY_MARGIN_MIN
            )
            hypotheses.append(
                Hypothesis(
                    source=f"snap:{rank}",
                    affine=np.array(candidate["world_affine"], dtype=float),
                    effective_gcps=0,
                    scores={
                        "cached": {
                            "verification": candidate.get("verification"),
                            "select_score": select,
                            "name": (candidate.get("name") or {}).get("score"),
                        },
                        **({} if admitted else {"entry_barred": True}),
                    },
                )
            )
            rank += 1
    if (
        street_record is not None
        and street_record.get("status") == "posed"
        and street_record.get("corners")
        and page_size is not None
    ):
        from mapsnap.street_solve_experiment import corners_to_affine

        hypotheses.append(
            Hypothesis(
                source="streets-candidate",
                affine=corners_to_affine(street_record["corners"], page_size),
                effective_gcps=0,
            )
        )

    hypotheses = dedupe_hypotheses(hypotheses)
    hypotheses.append(Hypothesis(source=UNPLACED, affine=None))
    published = None
    if channel is not None:
        for i, hypothesis in enumerate(hypotheses):
            if hypothesis.source == channel or channel in hypothesis.merged_sources:
                published = i
                break
    return hypotheses, published


def dedupe_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Merge near-identical poses, keeping earlier precedence and max GCPs."""
    kept: list[Hypothesis] = []
    for hypothesis in hypotheses:
        merged = False
        for existing in kept:
            if existing.affine is None or hypothesis.affine is None:
                continue
            # Width/height cancel out of the comparison frame; a nominal
            # 1000x1000 canvas suffices for "same pose" detection.
            if (
                grid_rmse_ft_between(existing.affine, hypothesis.affine, 1000, 1000)
                < DEDUPE_FT
            ):
                existing.merged_sources.append(hypothesis.source)
                existing.effective_gcps = max(
                    existing.effective_gcps, hypothesis.effective_gcps
                )
                merged = True
                break
        if not merged:
            kept.append(hypothesis)
    return kept


def keep_prior(effective_gcps: int, verification: float | None) -> float:
    """Measured expected value of publishing a pose with these features."""
    tier = "0-1" if effective_gcps <= 1 else ("2-3" if effective_gcps <= 3 else "4+")
    return KEEP_PRIOR[(tier, verification is not None and verification >= 0)]


def carries_gcp_evidence(source: str) -> bool:
    """Whether a hypothesis's GCP count is meaningful (georef-family sidecars)."""
    return source.startswith("georef") and source not in (
        "georef-snap",
        "georef-street",
    )


def pose_center(affine: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """(lon, lat) of the posed page center."""
    return (
        affine[0, 0] * width / 2 + affine[0, 1] * height / 2 + affine[0, 2],
        affine[1, 0] * width / 2 + affine[1, 1] * height / 2 + affine[1, 2],
    )


def pose_scale_log2(affine: np.ndarray) -> float:
    """log2 of the pose's metre-per-pixel scale (y row, like adjacency_gate)."""
    lat = affine[1, 2]
    ky = 110_540.0
    kx = 111_320.0 * math.cos(math.radians(lat))
    sx = math.hypot(affine[0, 0] * kx, affine[1, 0] * ky)
    return math.log2(max(sx, 1e-12))


def keymap_distance_m(
    affine: np.ndarray, width: int, height: int, centers: list[tuple[float, float]]
) -> float | None:
    """Distance from the posed page center to the nearest keymap center."""
    if not centers:
        return None
    lon_c = affine[0, 0] * width / 2 + affine[0, 1] * height / 2 + affine[0, 2]
    lat_c = affine[1, 0] * width / 2 + affine[1, 1] * height / 2 + affine[1, 2]
    return min(haversine_m(lat_c, lon_c, lat, lon) for lon, lat in centers)


def transfer_gcp_tiers(node: PageNode) -> None:
    """Share GCP evidence among near-identical poses (within ~100 ft).

    RANSAC's GCPs corroborate a LOCATION, not an exact pose: a refined channel
    pose 20 ft from the GCP-carrying fit is supported by the same GCPs. Without
    this, the tier bonus re-litigates snap-refine's calibrated verification
    duel with a thumb on RANSAC's side (LA gate run 1: 13 osm poses lost to
    their own pre-refinement fits).
    """
    for h in node.hypotheses:
        if h.affine is None or not carries_gcp_evidence(h.source):
            continue
        for other in node.hypotheses:
            if other is h or other.affine is None:
                continue
            rmse = grid_rmse_ft_between(
                h.affine, other.affine, node.unit.width, node.unit.height
            )
            if rmse < 100.0:
                other.effective_gcps = max(other.effective_gcps, h.effective_gcps)
                other.scores["gcp_transfer_from"] = h.source


def apply_ambiguity_penalty(node: PageNode) -> None:
    """Mark non-published poses whose evidence ties a genuinely distinct rival.

    Snap's distinct-margin gate as a factor: aliased candidates (regular
    grids) can all verify well; a pose that cannot beat a rival at a
    different location by AMBIG_MARGIN is not evidence of place. The
    published pose is exempt — it won a margin at publication time.
    """

    def evidence(h: Hypothesis) -> float | None:
        v = h.scores.get("verification")
        if v is None:
            v = (h.scores.get("cached") or {}).get("verification")
        return v

    for i, h in enumerate(node.hypotheses):
        if h.affine is None or i == node.published_index:
            continue
        ev = evidence(h)
        if ev is None:
            continue
        center = pose_center(h.affine, node.unit.width, node.unit.height)
        for j, rival in enumerate(node.hypotheses):
            if j == i or rival.affine is None:
                continue
            rival_ev = evidence(rival)
            if rival_ev is None or rival_ev < ev - AMBIG_MARGIN:
                continue
            rc = pose_center(rival.affine, node.unit.width, node.unit.height)
            if haversine_m(center[1], center[0], rc[1], rc[0]) > DISTINCT_M:
                h.scores["ambiguous"] = True
                break


def unary_energy(
    hypothesis: Hypothesis,
    is_published: bool,
    keymap_radius_m: float,
    family_log2: float | None,
    note_ratio: float | None,
    page_placed: bool = True,
) -> float:
    """Energy of one hypothesis in isolation; UNPLACED is exactly 0.

    Deliberately absent: the incumbent-defensibility veto, margin gates, and
    select_score (which double-counts the name evidence) — those are the stage
    rules this pass replaces with posterior comparison.
    """
    if hypothesis.affine is None:
        hypothesis.unary_terms = {"unplaced": 0.0}
        hypothesis.unary = 0.0
        return 0.0
    terms: dict[str, float] = {}
    verification = hypothesis.scores.get("verification")
    name = hypothesis.scores.get("name", 0.0) or 0.0
    containment = hypothesis.scores.get("containment", 0.0) or 0.0
    if verification is None:
        # No P(road) for this page: fall back to any cached channel score, and
        # mark the hypothesis so the report shows it was never verified.
        verification = (hypothesis.scores.get("cached") or {}).get("verification")
        hypothesis.scores["unverified"] = True
    if verification is None:
        verification = 0.0
    terms["evidence"] = -(verification + W_NAME * name + W_CONTAIN * containment)
    if not page_placed:
        terms["entry"] = ENTRY_PENALTY
    if is_published:
        # The incumbent's measured prior: what publishing a pose with this
        # much support is worth, against E[unplaced] = 0. Replaces both the
        # invented incumbent epsilon and the flat keep bar.
        terms["keep_prior"] = -keep_prior(
            hypothesis.effective_gcps, hypothesis.scores.get("verification")
        )
    keymap_dist = hypothesis.scores.get("keymap_dist_m")
    if keymap_dist is not None and keymap_radius_m > 0:
        terms["keymap"] = W_KEYMAP * min(
            KEYMAP_CLAMP, (keymap_dist / keymap_radius_m) ** 2
        )
    if family_log2 is not None:
        # Distance to the nearest integer rung of the volume family. Being on
        # ANY rung is free (second families are legitimate); between rungs
        # pays; a printed note overrides in both directions.
        offset = pose_scale_log2(hypothesis.affine) - family_log2
        rung_distance = abs(offset - round(offset))
        if note_ratio is not None and not (0.8 <= note_ratio <= 1.25):
            note_offset = offset - math.log2(note_ratio)
            terms["rung"] = 0.0 if abs(note_offset) < 0.25 else W_NOTE_MISMATCH
        else:
            terms["rung"] = 0.0 if rung_distance < 0.25 else W_RUNG_OFF
    if hypothesis.scores.get("ambiguous"):
        terms["ambiguity"] = W_AMBIG
    if hypothesis.status == sidecar.CONTRADICTED:
        terms["contradicted"] = W_CONTRADICTED
    if not page_placed and hypothesis.scores.get("entry_barred"):
        terms["entry_barred"] = 10.0
    hypothesis.unary_terms = terms
    hypothesis.unary = sum(terms.values())
    return hypothesis.unary


def footprint_metres(affine: np.ndarray, width: int, height: int, origin):
    """Shapely polygon of the posed page in metres about origin (lon0, lat0)."""
    from shapely.geometry import Polygon

    lon0, lat0 = origin
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    pts = []
    for x, y in [(0, 0), (width, 0), (width, height), (0, height)]:
        lon = affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]
        lat = affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]
        pts.append(((lon - lon0) * kx, (lat - lat0) * ky))
    return Polygon(pts).buffer(0)


def overlap_penalty(iou_over_min: float, soft: float, hard: float) -> float:
    """select_volume's overlap shape: free below soft, 1.0 at hard, linear between."""
    if iou_over_min <= soft:
        return 0.0
    if iou_over_min >= hard:
        return 1.0
    return (iou_over_min - soft) / (hard - soft)


def fitted_page_for(hypothesis: Hypothesis, unit: PageUnit) -> FittedPage:
    """Wrap a hypothesis pose in adjacency_gate's FittedPage shape."""
    affine = hypothesis.affine
    assert affine is not None
    return FittedPage(
        stem=unit.stem,
        affine=affine,
        width=unit.width,
        height=unit.height,
        channel_paths=[],
        gcps=hypothesis.effective_gcps,
        theta_deg=0.0,
        log_scale=pose_scale_log2(affine) * math.log(2),
    )


def stamp_energy(
    adjacency: dict,
    node_a: PageNode,
    hyp_a: Hypothesis,
    node_b: PageNode,
    hyp_b: Hypothesis,
    median_log_scale: float,
) -> float:
    """Robust printed-stamp disagreement between two chosen poses."""
    if hyp_a.affine is None or hyp_b.affine is None:
        return 0.0
    page_a = fitted_page_for(hyp_a, node_a.unit)
    page_b = fitted_page_for(hyp_b, node_b.unit)
    stamps = stamp_worlds(adjacency, page_a, node_b.unit.stem) + stamp_worlds(
        adjacency, page_b, node_a.unit.stem
    )
    if not stamps:
        return 0.0
    # Distance between each side's claim of the other and the other's nearest
    # claim back — the gate's min-cross-distance, against hypothesis poses.
    side_a = stamp_worlds(adjacency, page_a, node_b.unit.stem)
    side_b = stamp_worlds(adjacency, page_b, node_a.unit.stem)
    if not side_a or not side_b:
        return 0.0
    distance = min(
        haversine_m(la, lo, lb, ob) for lo, la in side_a for ob, lb in side_b
    )
    bar = GATE_STAMP_M * edge_scale_factor(page_a, page_b, median_log_scale)
    return W_STAMP * min(STAMP_CLAMP, (distance / bar) ** 2)


def pairwise_energy(
    kind: str,
    adjacency: dict,
    node_a: PageNode,
    hyp_a: Hypothesis,
    node_b: PageNode,
    hyp_b: Hypothesis,
    median_log_scale: float,
    origin,
    overlap_soft: float = OVERLAP_SOFT,
) -> float:
    """Energy of a pair of chosen hypotheses; zero when either is UNPLACED."""
    if hyp_a.affine is None or hyp_b.affine is None:
        return 0.0
    if kind == "stamp":
        # Stamp edges carry the relative-pose evidence themselves; overlap
        # between mutually-claiming neighbors is legitimate sheet layout (DC's
        # p153/p154 overlap ~0.6 at their CORRECT poses), so adding an overlap
        # term here is double jeopardy — it evicted stamp-agreeing 10 ft fits
        # in gate run 1.
        return stamp_energy(adjacency, node_a, hyp_a, node_b, hyp_b, median_log_scale)
    # Overlap pair (the only non-stamp kind): adjacent sheets overlap at their
    # seams legitimately, so the threshold is self-calibrated per volume.
    energy = 0.0
    soft, hard = overlap_soft, overlap_soft + OVERLAP_HARD_DELTA
    poly_a = footprint_metres(
        hyp_a.affine, node_a.unit.width, node_a.unit.height, origin
    )
    poly_b = footprint_metres(
        hyp_b.affine, node_b.unit.width, node_b.unit.height, origin
    )
    if poly_a.is_valid and poly_b.is_valid and poly_a.area and poly_b.area:
        inter = poly_a.intersection(poly_b).area
        iou_over_min = inter / min(poly_a.area, poly_b.area)
        energy += W_OVERLAP * overlap_penalty(iou_over_min, soft, hard)
    return energy


def calibrate_overlap_soft(
    nodes: dict[str, PageNode], adjacency: dict, origin
) -> float:
    """p90 IoU-over-min across published adjacent pairs, floored at OVERLAP_SOFT.

    Mirrors select_volume's self-calibration: adjacent Sanborn sheets overlap
    legitimately at their seams, and how much is a property of the volume.
    """
    ious = []
    for a, b in adjacency.get("adjacency", []) if adjacency else []:
        na, nb = nodes.get(a), nodes.get(b)
        if not na or not nb or na.published_index is None or nb.published_index is None:
            continue
        ha = na.hypotheses[na.published_index]
        hb = nb.hypotheses[nb.published_index]
        if ha.affine is None or hb.affine is None:
            continue
        pa = footprint_metres(ha.affine, na.unit.width, na.unit.height, origin)
        pb = footprint_metres(hb.affine, nb.unit.width, nb.unit.height, origin)
        if pa.is_valid and pb.is_valid and pa.area and pb.area:
            ious.append(pa.intersection(pb).area / min(pa.area, pb.area))
    if not ious:
        return OVERLAP_SOFT
    return max(OVERLAP_SOFT, float(np.percentile(ious, 90)))


def panel_base(stem: str) -> str | None:
    """Parent stem of a split panel ('p63__4' -> 'p63'), else None."""
    return stem.split("__")[0] if "__" in stem else None


def solve(
    nodes: dict[str, PageNode],
    edges: list[tuple[str, str, str]],
    adjacency: dict,
    median_log_scale: float,
    origin,
    overlap_soft: float = OVERLAP_SOFT,
) -> dict[str, int]:
    """Choose one hypothesis per page minimizing joint energy (deterministic).

    Connected components over the edge graph; exhaustive when the product of
    hypothesis counts is small, else ICM from three fixed starts
    (published-init, unary-argmax, all-UNPLACED), sorted sweeps, ties to the
    lower index; across starts: lowest energy, then fewest flips-from-
    published, then start order.
    """
    neighbors: dict[str, list[tuple[str, str]]] = {stem: [] for stem in nodes}
    for kind, a, b in edges:
        neighbors[a].append((kind, b))
        neighbors[b].append((kind, a))

    def pair_e(kind: str, a: str, ia: int, b: str, ib: int) -> float:
        return pairwise_energy(
            kind,
            adjacency,
            nodes[a],
            nodes[a].hypotheses[ia],
            nodes[b],
            nodes[b].hypotheses[ib],
            median_log_scale,
            origin,
            overlap_soft=overlap_soft,
        )

    # Union-find components.
    parent = {stem: stem for stem in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, a, b in edges:
        parent[find(a)] = find(b)
    components: dict[str, list[str]] = {}
    for stem in sorted(nodes):
        components.setdefault(find(stem), []).append(stem)

    assignment: dict[str, int] = {}
    for stems in components.values():
        component_edges = [
            (kind, a, b) for kind, a, b in edges if a in stems and b in stems
        ]
        counts = [len(nodes[stem].hypotheses) for stem in stems]
        size = math.prod(counts)
        published_start = {
            stem: (
                nodes[stem].published_index
                if nodes[stem].published_index is not None
                else len(nodes[stem].hypotheses) - 1
            )
            for stem in stems
        }
        if size <= EXHAUSTIVE_LIMIT:
            best = None
            for combo in product(*[range(c) for c in counts]):
                candidate = dict(zip(stems, combo))
                energy = sum(nodes[s].hypotheses[i].unary for s, i in candidate.items())
                for kind, a, b in component_edges:
                    energy += pair_e(kind, a, candidate[a], b, candidate[b])
                flips = sum(1 for s in stems if candidate[s] != published_start[s])
                key = (round(energy, 9), flips, combo)
                if best is None or key < best[0]:
                    best = (key, candidate)
            assert best is not None
            assignment.update(best[1])
            continue
        # ICM from three fixed starts.
        starts = [
            dict(published_start),
            {
                stem: min(
                    range(len(nodes[stem].hypotheses)),
                    key=lambda i: (nodes[stem].hypotheses[i].unary, i),
                )
                for stem in stems
            },
            {stem: len(nodes[stem].hypotheses) - 1 for stem in stems},
        ]
        results = []
        for start in starts:
            assign = dict(start)
            for _ in range(ICM_MAX_ROUNDS):
                changed = False
                for stem in stems:
                    best_index = assign[stem]
                    best_local = None
                    for i in range(len(nodes[stem].hypotheses)):
                        local = nodes[stem].hypotheses[i].unary
                        for kind, other in neighbors[stem]:
                            if other in assign:
                                local += pair_e(kind, stem, i, other, assign[other])
                        key = (round(local, 9), i)
                        if best_local is None or key < best_local:
                            best_local = key
                            best_index = i
                    if best_index != assign[stem]:
                        assign[stem] = best_index
                        changed = True
                if not changed:
                    break
            energy = sum(nodes[s].hypotheses[assign[s]].unary for s in stems)
            for kind, a, b in component_edges:
                energy += pair_e(kind, a, assign[a], b, assign[b])
            flips = sum(1 for s in stems if assign[s] != published_start[s])
            results.append((round(energy, 9), flips, assign))
        results.sort(key=lambda r: (r[0], r[1]))
        assignment.update(results[0][2])
    return assignment


def load_channel_records(volume: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """(snap records by target, street-solve records by stem) from the artifact stores."""
    snap: dict[str, dict] = {}
    path = volume / "artifacts" / "osm_snap" / "candidates.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                snap[record["target"]] = record
    streets: dict[str, dict] = {}
    path = volume / "artifacts" / "street_solve" / "candidates.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                streets[record["stem"]] = record
    return snap, streets


def build_nodes(volume: Path, sidecar_dir: Path, vctx) -> dict[str, PageNode]:
    """PageNodes for every base page and split panel with any hypothesis."""
    snap_records, street_records = load_channel_records(volume)
    nodes: dict[str, PageNode] = {}
    for unit in [*vctx.units, *vctx.panel_units]:
        base = panel_base(unit.stem)
        if base is not None and any(u.stem == unit.stem for u in vctx.units):
            continue
        hypotheses, published = collect_hypotheses(
            sidecar_dir,
            unit.stem,
            snap_records.get(unit.stem),
            street_records.get(unit.stem),
            page_size=(unit.width, unit.height),
        )
        if len(hypotheses) <= 1 and published is None:
            continue  # nothing to arbitrate: no pose exists anywhere
        nodes[unit.stem] = PageNode(
            unit=unit,
            is_panel=base is not None,
            base=base,
            hypotheses=hypotheses,
            published_index=published,
        )
    return nodes


def score_nodes(vctx, nodes: dict[str, PageNode], note_ratios: dict) -> None:
    """Uniformly score every pose hypothesis and fill in unary energies."""
    from mapsnap.osm_snap_experiment import build_page_context, page_keymap_data

    fitted_log2 = [
        pose_scale_log2(affine)
        for node in nodes.values()
        if node.published_index is not None
        and (affine := node.hypotheses[node.published_index].affine) is not None
    ]
    family_log2 = float(np.median(fitted_log2)) if len(fitted_log2) >= 3 else None

    for stem in sorted(nodes):
        node = nodes[stem]
        ctx, status = build_page_context(vctx, node.unit)
        centers, regions = page_keymap_data(vctx, node.unit)
        for hypothesis in node.hypotheses:
            if hypothesis.affine is None:
                continue
            if ctx is not None:
                evaluation = evaluate_pose(ctx, vctx.feature_index, hypothesis.affine)
                if evaluation is not None:
                    hypothesis.scores["verification"] = evaluation["verification"]
                    hypothesis.scores["name"] = (evaluation.get("name") or {}).get(
                        "score", 0.0
                    )
            else:
                hypothesis.scores["context"] = status
            if regions:
                hypothesis.scores["containment"] = region_containment_frac(
                    hypothesis.affine, (node.unit.width, node.unit.height), regions
                )
            distance = keymap_distance_m(
                hypothesis.affine, node.unit.width, node.unit.height, centers
            )
            if distance is not None:
                hypothesis.scores["keymap_dist_m"] = round(distance, 1)
            if node.unit.truth is not None:
                hypothesis.scores["grid_rmse_ft"] = round(
                    grid_rmse_ft_between(
                        node.unit.truth.affine_local,
                        hypothesis.affine,
                        node.unit.width,
                        node.unit.height,
                    ),
                    1,
                )
        transfer_gcp_tiers(node)
        apply_ambiguity_penalty(node)
        for i, hypothesis in enumerate(node.hypotheses):
            unary_energy(
                hypothesis,
                i == node.published_index,
                node.unit.keymap_radius_m or vctx.radius_m,
                family_log2,
                note_ratios.get(stem),
                page_placed=node.published_index is not None,
            )


def build_edges(
    nodes: dict[str, PageNode], adjacency: dict
) -> list[tuple[str, str, str]]:
    """(kind, a, b) edges: mutual-stamp pairs, no duplicates."""
    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a, b in adjacency.get("adjacency", []) if adjacency else []:
        if a in nodes and b in nodes:
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                edges.append(("stamp", key[0], key[1]))
    return edges


def write_outputs(
    volume: Path,
    nodes: dict[str, PageNode],
    assignment: dict[str, int],
    out_dir: Path,
) -> tuple[int, list[dict]]:
    """verdicts.jsonl + report.md; returns (flip count, verdict rows)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in sorted(nodes):
        node = nodes[stem]
        chosen = assignment[stem]
        published = node.published_index
        chosen_h = node.hypotheses[chosen]
        published_h = node.hypotheses[published] if published is not None else None
        rivals = sorted(
            (h.unary, i) for i, h in enumerate(node.hypotheses) if i != chosen
        )
        near_flip = bool(rivals) and rivals[0][0] - chosen_h.unary < 0.25
        rows.append(
            {
                "stem": stem,
                "chosen": chosen_h.source,
                "chosen_merged": chosen_h.merged_sources,
                "published": published_h.source if published_h else UNPLACED,
                "changed": chosen
                != (published if published is not None else len(node.hypotheses) - 1),
                "unary_terms": chosen_h.unary_terms,
                "published_terms": published_h.unary_terms if published_h else None,
                "grid_rmse_ft_chosen": chosen_h.scores.get("grid_rmse_ft"),
                "grid_rmse_ft_published": (
                    published_h.scores.get("grid_rmse_ft") if published_h else None
                ),
                "near_flip": near_flip,
                "unverified": bool(chosen_h.scores.get("unverified")),
                "hypotheses": [
                    {
                        "source": h.source,
                        "unary": round(h.unary, 4),
                        "verification": h.scores.get("verification"),
                        "grid_rmse_ft": h.scores.get("grid_rmse_ft"),
                    }
                    for h in node.hypotheses
                ],
            }
        )
    (out_dir / "verdicts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    flips = [row for row in rows if row["changed"]]
    lines = [
        f"# reconcile report: {volume.name}",
        "",
        (
            f"{len(rows)} pages arbitrated, {len(flips)} decisions flipped, "
            f"{sum(1 for r in rows if r['near_flip'])} near-flips (review these)."
        ),
        "",
    ]
    for row in flips:
        lines.append(
            f"## {row['stem']}: {row['published']} -> {row['chosen']}"
            + (
                f"  (grid rmse {row['grid_rmse_ft_published']} -> "
                f"{row['grid_rmse_ft_chosen']} ft — diagnostic only)"
                if row["grid_rmse_ft_chosen"] is not None
                or row["grid_rmse_ft_published"] is not None
                else ""
            )
        )
        lines.append("")
        lines.append("| hypothesis | unary | verification | grid rmse ft |")
        lines.append("|---|---|---|---|")
        for h in row["hypotheses"]:
            lines.append(
                f"| {h['source']} | {h['unary']} | {h['verification']} | "
                f"{h['grid_rmse_ft']} |"
            )
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    return len(flips), rows


def publish(
    volume: Path, nodes: dict[str, PageNode], assignment: dict[str, int]
) -> tuple[int, int]:
    """Write the arbitrated answer as ``pN.georef-final.json``, one per page.

    The ONLY mode that writes to the volume root, and the only sidecar
    ``fit`` publishes from. Every arbitrated page gets a file, including the
    ones left unplaced: those carry ``corners: null``, which
    ``expand_georef_globs`` skips while still claiming the page key.

    An explicit non-fit is what lets the arbiter *unpublish*. Previously that
    took renaming each channel's sidecar out of the glob's way — the pipeline
    could only express "unplaced" as the absence of a file, so suppressing a
    pose meant hiding it. Here the decision is a written record, and the
    channels' own sidecars stay exactly where they are.
    """
    written = unplaced = 0
    for stale in volume.glob("p*.georef-final.json"):
        stale.unlink()
    for stem in sorted(nodes):
        node = nodes[stem]
        hypothesis = node.hypotheses[assignment[stem]]
        w, h = node.unit.width, node.unit.height
        a = hypothesis.affine
        corners = (
            None
            if a is None
            else [
                [
                    a[0, 0] * x + a[0, 1] * y + a[0, 2],
                    a[1, 0] * x + a[1, 1] * y + a[1, 2],
                ]
                for x, y in [(0, 0), (w, 0), (w, h), (0, h)]
            ]
        )
        (volume / f"{stem}.georef-final.json").write_text(
            json.dumps(
                {
                    "width": w,
                    "height": h,
                    "corners": corners,
                    "streets": [],
                    "intersections": [],
                    "reconcile": {
                        "source": hypothesis.source,
                        "merged": hypothesis.merged_sources,
                        "unary": round(hypothesis.unary, 4),
                        "terms": {
                            k: round(v, 4) for k, v in hypothesis.unary_terms.items()
                        },
                    },
                },
                indent=1,
            )
        )
        if corners is None:
            unplaced += 1
        else:
            written += 1
    return written, unplaced


def materialize_and_grade(
    volume: Path, nodes: dict[str, PageNode], assignment: dict[str, int], out_dir: Path
) -> None:
    """Write chosen poses as minimal sidecars, build an IIIF, run the real compare."""
    from mapsnap.fit import find_ref_iiif

    mat_dir = out_dir / "materialized"
    if mat_dir.exists():
        shutil.rmtree(mat_dir)
    mat_dir.mkdir(parents=True)
    for stem in sorted(nodes):
        node = nodes[stem]
        hypothesis = node.hypotheses[assignment[stem]]
        if hypothesis.affine is None:
            continue
        a = hypothesis.affine
        w, h = node.unit.width, node.unit.height
        corners = [
            [a[0, 0] * x + a[0, 1] * y + a[0, 2], a[1, 0] * x + a[1, 1] * y + a[1, 2]]
            for x, y in [(0, 0), (w, 0), (w, h), (0, h)]
        ]
        (mat_dir / f"{stem}.georef.json").write_text(
            json.dumps(
                {
                    "width": w,
                    "height": h,
                    "corners": corners,
                    "streets": [],
                    "intersections": [],
                    "reconcile": {
                        "source": hypothesis.source,
                        "merged": hypothesis.merged_sources,
                    },
                },
                indent=1,
            )
        )
    # compare resolves split-panel polygons from the generated dir.
    for panels in volume.glob("p*.panels.json"):
        shutil.copy2(panels, mat_dir / panels.name)
    ref = find_ref_iiif(volume)
    iiif_path = out_dir / "reconcile.iiif.json"
    subprocess.run(
        [
            "mapsnap",
            "iiif",
            str(ref),
            str(mat_dir / "*.georef.json"),
            "--output",
            str(iiif_path),
        ],
        check=True,
    )
    print(f"wrote {iiif_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report-only joint arbitration of all candidate poses (#270)."
    )
    parser.add_argument("volume", type=Path)
    parser.add_argument(
        "--sidecars-from",
        type=Path,
        default=None,
        metavar="DIR",
        help="Read georef sidecars from this dir (e.g. artifacts/reconcile-base) "
        "instead of the volume root — the archived-baseline mode.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Write the arbitrated poses as pN.georef-final.json in the "
        "volume root (the only mode that writes there). fit --reconcile does "
        "this for you.",
    )
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Materialize the chosen poses and build an IIIF for "
        "grading with the real compare.",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=None,
        help="Override the entry bar (sweep support).",
    )
    parser.add_argument(
        "--pages", nargs="*", default=None, help="Restrict to these stems (debug)."
    )
    args = parser.parse_args()

    global ENTRY_PENALTY
    if args.gate is not None:
        ENTRY_PENALTY = args.gate

    from mapsnap.osm_snap_experiment import load_volume_context, printed_note_ratios

    volume = args.volume
    sidecar_dir = args.sidecars_from or volume
    if not sidecar_dir.is_absolute() and not sidecar_dir.exists():
        sidecar_dir = volume / sidecar_dir
    vctx = load_volume_context(volume)
    nodes = build_nodes(volume, sidecar_dir, vctx)
    if args.pages:
        nodes = {stem: node for stem, node in nodes.items() if stem in args.pages}
    print(
        f"{volume.name}: {len(nodes)} pages, "
        f"{sum(len(n.hypotheses) for n in nodes.values())} hypotheses"
    )
    snap_records, _ = load_channel_records(volume)
    note_ratios = printed_note_ratios(volume, list(snap_records.values()))
    score_nodes(vctx, nodes, note_ratios)
    adjacency = vctx.adjacency or {}
    edges = build_edges(nodes, adjacency)
    fitted_logs = [
        pose_scale_log2(affine) * math.log(2)
        for n in nodes.values()
        if n.published_index is not None
        and (affine := n.hypotheses[n.published_index].affine) is not None
    ]
    median_log_scale = float(np.median(fitted_logs)) if fitted_logs else 0.0
    origin = None
    for node in nodes.values():
        for hypothesis in node.hypotheses:
            if hypothesis.affine is not None:
                origin = (hypothesis.affine[0, 2], hypothesis.affine[1, 2])
                break
        if origin:
            break
    overlap_soft = calibrate_overlap_soft(nodes, adjacency, origin or (0.0, 0.0))
    print(
        f"overlap soft threshold (p90 of published adjacent pairs): {overlap_soft:.3f}"
    )
    assignment = solve(
        nodes,
        edges,
        adjacency,
        median_log_scale,
        origin or (0.0, 0.0),
        overlap_soft=overlap_soft,
    )
    out_dir = volume / "artifacts" / "reconcile"
    flips, _ = write_outputs(volume, nodes, assignment, out_dir)
    print(f"{flips} decisions flipped -> {out_dir / 'report.md'}")
    if args.publish:
        written, unplaced = publish(volume, nodes, assignment)
        print(f"published {written} reconcile sidecars, {unplaced} unplaced markers")
    if args.grade:
        materialize_and_grade(volume, nodes, assignment, out_dir)


if __name__ == "__main__":
    main()
