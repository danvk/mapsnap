"""Tests for the snap harness's production decision core.

arbitrate_challenge / refine_adoption decide whether to REPLACE a placed
RANSAC fit — the riskiest action in the pipeline — so their gates are pinned
here with the failure modes that motivated them (see the osm-snap PR).
"""

import dataclasses

import numpy as np
import pytest

from mapsnap import osm_snap_experiment
from mapsnap.edge_join_experiment import PageUnit
from mapsnap.feature_index import FeatureIndex
from mapsnap.osm_snap import ScalePrior
from mapsnap.osm_snap_experiment import (
    SNAP_LOG_BEGIN,
    SNAP_LOG_END,
    VolumeContext,
    affine_m_per_px,
    append_snap_logs,
    arbitrate_challenge,
    candidates_record_fresh,
    canonicalize_refine_keys,
    decision_block,
    init_worker,
    refine_adopt_set,
    refine_adoption,
    refine_eligible_features,
    refine_rule_outcome,
    rung_flip,
    snap_one_page,
    with_incumbent_scale,
)

# A page-local degree scale of ~0.6 m/px at the test latitude.
KX = 111_320.0 * 0.766  # cos(40 deg)
SCALE_DEG = 0.6 / KX


def affine(lon_shift_m: float = 0.0) -> list[list[float]]:
    """A north-up page->lonlat affine, optionally slid east by metres."""
    return [
        [SCALE_DEG, 0.0, -74.0 + lon_shift_m / KX],
        [0.0, -SCALE_DEG * 111_320.0 * 0.766 / 110_540.0, 40.0],
    ]


def fitted_record(
    incumbent_ver: float,
    challenger_ver: float,
    shift_m: float,
    incumbent_name: float = 0.1,
    challenger_name: float = 0.3,
    select: float = 2.0,
) -> dict:
    return {
        "target": "p1",
        "status": "ok",
        "fit_state": "fitted",
        "width": 1000,
        "height": 1000,
        "incumbent": {
            "world_affine": affine(),
            "verification": incumbent_ver,
            "name": {"score": incumbent_name, "n_hits": 1, "n_labels": 3},
        },
        "candidates": [
            {
                "world_affine": affine(shift_m),
                "center": [-74.0 + (500 * SCALE_DEG) + shift_m / KX, 39.99],
                "theta_deg": 0.0,
                "theta_source": "label-pair-exact",
                "center_dist_m": 50.0,
                "select_score": select,
                "verification": challenger_ver,
                "name": {"score": challenger_name, "n_hits": 2, "n_labels": 3},
            },
        ],
    }


def test_challenge_requires_indefensible_incumbent():
    # A plausible incumbent (ver >= 0.1) is never overturned, however strong
    # the challenger: the Chicago modern-OSM trap (wrong poses matching
    # today's grid better than the truth does).
    record = fitted_record(incumbent_ver=0.6, challenger_ver=1.5, shift_m=100.0)
    assert arbitrate_challenge(record, 1.5) is None
    # An OSM-contradicted incumbent with a strongly disagreeing, evidence-
    # winning challenger is replaced.
    record = fitted_record(incumbent_ver=-0.3, challenger_ver=1.5, shift_m=100.0)
    challenge = arbitrate_challenge(record, 1.5)
    assert challenge is not None and challenge["challenge"]
    assert challenge["disagreement_ft"] > 100.0


def test_challenge_requires_real_disagreement_and_name_parity():
    # Agreement (< 100ft ~ 30m) is refinement territory, not a challenge.
    record = fitted_record(incumbent_ver=-0.3, challenger_ver=1.5, shift_m=15.0)
    assert arbitrate_challenge(record, 1.5) is None
    # Names must not get worse.
    record = fitted_record(
        incumbent_ver=-0.3,
        challenger_ver=1.5,
        shift_m=100.0,
        incumbent_name=0.5,
        challenger_name=0.2,
    )
    assert arbitrate_challenge(record, 1.5) is None


def test_refine_adopts_agreeing_evidence_winner_only():
    # An agreeing challenger clearly winning verification is adopted.
    record = fitted_record(incumbent_ver=0.8, challenger_ver=1.2, shift_m=15.0)
    adoption = refine_adoption(record)
    assert adoption is not None and adoption["refine"]
    # Within the margin (<= +0.05): keep the incumbent — no churn on good fits.
    record = fitted_record(incumbent_ver=0.8, challenger_ver=0.85, shift_m=15.0)
    assert refine_adoption(record) is None
    # Far apart is arbitration territory, not refinement.
    record = fitted_record(incumbent_ver=0.8, challenger_ver=1.2, shift_m=100.0)
    assert refine_adoption(record) is None


def test_refine_adoption_margin_override():
    # The sweep harness's margin overrides: -inf adopts every agreeing
    # challenger, +inf adopts none, and a tighter margin rejects a winner
    # the production margin would take.
    record = fitted_record(incumbent_ver=0.8, challenger_ver=0.85, shift_m=15.0)
    assert refine_adoption(record, margin=-float("inf")) is not None
    record = fitted_record(incumbent_ver=0.8, challenger_ver=1.2, shift_m=15.0)
    assert refine_adoption(record, margin=float("inf")) is None
    assert refine_adoption(record, margin=0.5) is None


def test_refine_eligible_features_mirrors_select():
    records = [
        # Adoptable at SOME margin: eligible.
        fitted_record(incumbent_ver=0.8, challenger_ver=0.85, shift_m=15.0),
        # Claimed by arbitration (indefensible incumbent, strong disagreeing
        # challenger): not refinement's to sweep.
        fitted_record(incumbent_ver=-0.3, challenger_ver=1.6, shift_m=100.0),
    ]
    records[1]["target"] = "p2"
    eligible = refine_eligible_features(records)
    assert set(eligible) == {"p1"}
    features = eligible["p1"]
    assert features["incumbent_verification"] == 0.8
    assert features["challenger_verification"] == 0.85
    assert features["challenger_name"] == 0.3


def test_refine_adopt_set_rules():
    eligible = {
        "p1": {
            "incumbent_verification": 0.2,
            "challenger_verification": 0.5,
            "incumbent_name": 0.4,
            "challenger_name": 0.1,
        },
        "p2": {
            "incumbent_verification": 1.0,
            "challenger_verification": 1.2,
            "incumbent_name": 0.2,
            "challenger_name": 0.6,
        },
    }
    assert refine_adopt_set(eligible, 0.1) == {"p1", "p2"}
    assert refine_adopt_set(eligible, 0.25) == {"p1"}
    # Name parity drops p1 (its challenger loses the name head-to-head).
    assert refine_adopt_set(eligible, 0.1, name_parity=True) == {"p2"}
    # Band-aware: permissive below the verification edge, closed above it.
    band = (0.5, 0.0, float("inf"))
    assert refine_adopt_set(eligible, 0.1, band=band) == {"p1"}


def test_refine_rule_outcome_buckets():
    volume_data = {
        "total_land_m2": 100.0,
        "items": [
            # Mid-tier -> good under the challenger: +10 land.
            {
                "key": "p1",
                "gen_key": "p1",
                "land_m2": 10.0,
                "rmse_none": 40.0,
                "rmse_all": 12.0,
            },
            # Good -> disaster: -20 land, and it counts as a loss.
            {
                "key": "p2",
                "gen_key": "p2",
                "land_m2": 10.0,
                "rmse_none": 20.0,
                "rmse_all": 250.0,
            },
            # Not adopted: no contribution however much it would move.
            {
                "key": "p3",
                "gen_key": "p3",
                "land_m2": 50.0,
                "rmse_none": 40.0,
                "rmse_all": 12.0,
            },
        ],
    }
    delta, land, gains, losses = refine_rule_outcome(volume_data, {"p1", "p2"})
    assert delta == 10.0 - 2 * 10.0
    assert land == 100.0
    assert (gains, losses) == (1, 1)


def test_canonicalize_refine_keys_joins_cases():
    # Sidecar stems are lowercase (chicago p101w) while truth page keys carry
    # the case (p101W); the remap makes rule outcomes see those adoptions.
    result = {
        "eligible": {"p101w": {}},
        "items": [
            {"key": "p101W", "gen_key": "p101W"},
            {"key": "p5", "gen_key": None},
            {"key": "p7", "gen_key": "p7"},
        ],
    }
    canonicalize_refine_keys(result)
    assert [item["gen_key"] for item in result["items"]] == ["p101w", None, "p7"]


def make_unit(fit_state: str) -> PageUnit:
    return PageUnit(
        stem="p1",
        number=1,
        width=1000,
        height=1000,
        fit_state=fit_state,
        truth=None,
        split_truth=False,
        gen_affine=np.array(affine()) if fit_state == "fitted" else None,
        inlier_intersections=0,
        inlier_streets=0,
        keymap_centers=[],
        keymap_radius_m=0.0,
    )


def test_candidates_record_fresh_tracks_fit_changes():
    record = {"fit_state": "fitted", "georef_mtime": 111, "status": "ok"}
    assert candidates_record_fresh(record, make_unit("fitted"), 111)
    # A re-run georef rewrote the sidecar: the cached incumbent is stale.
    assert not candidates_record_fresh(record, make_unit("fitted"), 222)
    # The page's fit STATE changed (fitted -> nofit or vice versa).
    assert not candidates_record_fresh(record, make_unit("nofit"), 111)
    # Legacy records (no georef_mtime) always recompute once.
    assert not candidates_record_fresh(
        {"fit_state": "fitted", "status": "ok"}, make_unit("fitted"), 111
    )
    # Failures are always retried: they turn on the P(road) cache and the key
    # map's sidecars, which this check does not track, so caching one would pin
    # the page behind a stale failure even after that input is fixed.
    base = {"fit_state": "fitted", "georef_mtime": 111}
    for status in ("no_prob", "no_keymap", "implausible"):
        assert not candidates_record_fresh(
            {**base, "status": status}, make_unit("fitted"), 111
        )
    assert candidates_record_fresh({**base, "status": "ok"}, make_unit("fitted"), 111)
    # A contradiction hint appeared (or was rewritten by a re-demotion): the
    # cached candidates predate the stamp-consistency gate and must recompute.
    assert not candidates_record_fresh(
        {**base, "status": "ok"}, make_unit("fitted"), 111, hint_mtime=555
    )
    hinted = {**base, "status": "ok", "contradiction_mtime": 555}
    assert candidates_record_fresh(hinted, make_unit("fitted"), 111, hint_mtime=555)
    assert not candidates_record_fresh(hinted, make_unit("fitted"), 111, hint_mtime=777)
    # The key map's sidecars moved (the #213 assignment repair rewrites them
    # without touching any page's georef): the cached search centers are stale.
    keyed = {**hinted, "keymap_mtime": 999}
    assert candidates_record_fresh(
        keyed, make_unit("fitted"), 111, hint_mtime=555, keymap_mtime=999
    )
    assert not candidates_record_fresh(
        keyed, make_unit("fitted"), 111, hint_mtime=555, keymap_mtime=1000
    )


def test_init_worker_indexes_pages_and_panels(monkeypatch, tmp_path):
    """snap_one_page must reach panels too, and never rebuild a context it was given."""
    panel = dataclasses.replace(make_unit("nofit"), stem="p1__1")
    context = VolumeContext(
        volume=tmp_path,
        units=[make_unit("fitted")],
        panel_units=[panel],
        features=[],
        feature_index=FeatureIndex([]),
        locator=None,
        volume_m_per_px=0.6,
        adjacency={},
        region_centroids={},
        filter_params={},
        radius_m=150.0,
        radius_source="calibrated",
        median_theta_deg=None,
    )

    def fail(_volume):
        raise AssertionError("a supplied context must not be rebuilt")

    monkeypatch.setattr(osm_snap_experiment, "load_volume_context", fail)
    init_worker(tmp_path, context)
    assert set(osm_snap_experiment.worker_state["units"]) == {"p1", "p1__1"}

    seen = []
    monkeypatch.setattr(
        osm_snap_experiment,
        "page_record",
        lambda vctx, unit: seen.append(unit.stem) or {"target": unit.stem},
    )
    stem, record = snap_one_page("p1__1")
    assert (stem, record["target"]) == ("p1__1", "p1__1")
    assert record["elapsed_s"] >= 0  # every record carries its own wall-clock cost
    assert seen == ["p1__1"]


def test_append_snap_logs_is_idempotent(tmp_path):
    record = fitted_record(incumbent_ver=0.8, challenger_ver=1.2, shift_m=15.0)
    selection = refine_adoption(record)
    assert selection is not None
    (tmp_path / "p1.txt").write_text("georef log line\n")
    for _ in range(2):
        append_snap_logs(tmp_path, [record], [selection], "arbitrate")
    text = (tmp_path / "p1.txt").read_text()
    assert text.startswith("georef log line\n")
    assert text.count(SNAP_LOG_BEGIN) == 1
    assert text.count(SNAP_LOG_END) == 1
    assert "refine: candidate #1 accepted" in text


def rung_record(
    scale_ratio: float,
    incumbent_ver: float = 0.3,
    challenger_ver: float = 0.9,
    select: float = 1.2,
    incumbent_name: float = 0.3,
    challenger_name: float = 0.3,
) -> dict:
    """A fitted record whose sole candidate differs from the incumbent by scale_ratio."""
    record = fitted_record(
        incumbent_ver=incumbent_ver,
        challenger_ver=challenger_ver,
        shift_m=15.0,
        incumbent_name=incumbent_name,
        challenger_name=challenger_name,
        select=select,
    )
    candidate = record["candidates"][0]
    candidate["world_affine"] = [
        [v * scale_ratio for v in row[:2]] + [row[2]]
        for row in candidate["world_affine"]
    ]
    return record


def test_rung_flip_adopts_a_doubled_candidate():
    # The calibrated signature: half-scale incumbent, doubled candidate that
    # wins verification with name parity and a confident select score.
    flip = rung_flip(rung_record(2.0))
    assert flip is not None and flip["rung"] and flip["reason"] == "rung"
    assert flip["chosen"] == 0


def test_rung_flip_never_flips_down():
    # Verification has a small-footprint bias: every would-be down flip in the
    # twelve-volume calibration was a break. Direction is the load-bearing gate.
    assert rung_flip(rung_record(0.5)) is None
    assert rung_flip(rung_record(0.5, challenger_ver=5.0)) is None


def test_rung_flip_requires_margin_parity_and_confidence():
    # Same-scale candidates are not rung disputes.
    assert rung_flip(rung_record(1.0)) is None
    # Verification margin below RUNG_VER_MARGIN keeps the incumbent.
    assert rung_flip(rung_record(2.0, incumbent_ver=0.85, challenger_ver=0.9)) is None
    # A name regression beyond the floor keeps the incumbent.
    assert rung_flip(rung_record(2.0, incumbent_name=0.5, challenger_name=0.2)) is None
    # An unconfident candidate keeps the incumbent.
    assert rung_flip(rung_record(2.0, select=0.5)) is None


def test_rung_flip_note_authority_unlocks_down_flips():
    # A doubled-scale incumbent whose page prints the standard note
    # (note_ratio = expected/incumbent = 0.5): the matching half-scale
    # candidate may flip down.
    record = rung_record(0.5, incumbent_ver=0.3, challenger_ver=0.9)
    assert rung_flip(record) is None  # no note: down stays blocked
    flip = rung_flip(record, note_ratio=0.5)
    assert flip is not None and flip["rung"]
    # A note that ENDORSES the incumbent changes nothing.
    assert rung_flip(record, note_ratio=1.0) is None
    # A condemning note only admits candidates that MATCH it.
    assert rung_flip(rung_record(0.5), note_ratio=2.0) is None


def test_cluster_search_centers_merges_overlapping_discs():
    from mapsnap.osm_snap import cluster_search_centers

    # Three centers within 100m of each other and one 2km away.
    base = (-74.0, 40.0)
    kx = 111_320.0 * 0.766
    near = [
        base,
        (base[0] + 80 / kx, base[1]),
        (base[0], base[1] + 80 / 110_540.0),
    ]
    far = [(base[0] + 2000 / kx, base[1])]
    merged = cluster_search_centers(near + far, link_m=120.0)
    assert len(merged) == 2
    # Singletons and empties pass through untouched.
    assert cluster_search_centers(far, 120.0) == far
    assert cluster_search_centers([], 120.0) == []


def test_stamp_corroborated_rescue_relaxes_the_gates():
    from mapsnap.osm_snap_experiment import (
        PRODUCTION_GATE_MARGIN,
        PRODUCTION_GATE_SCORE,
        select_argmax,
    )

    def cand(score, sep=None, center=(-90.0, 30.0), theta=0.0, median=None):
        c = {
            "select_score": score,
            "center": list(center),
            "theta_deg": theta,
        }
        if sep is not None:
            c["stamp_separation_m"] = sep
            c["stamp_median_m"] = median if median is not None else sep
        return c

    def rec(candidates, fit_state="nofit"):
        return {
            "target": "p9",
            "status": "ok",
            "fit_state": fit_state,
            "candidates": candidates,
        }

    # KC p551 shape: true pose within the stamp bound, rivals gated
    # implausible (select_score None) -> adopted under the relaxed bar.
    # (0.77 was its v1-era score; the v3 recalibration shifted the bars and
    # the scores together, so the fixture tracks: 0.87 vs the 0.8 bar.)
    (choice,) = select_argmax(
        [rec([cand(0.87, sep=24.0), {"select_score": None}])],
        PRODUCTION_GATE_SCORE,
        PRODUCTION_GATE_MARGIN,
    )
    assert choice["chosen"] == 0 and choice["reason"] == "stamp-corroborated"

    # NO p125 shape: a corroborated twin close behind must NOT margin-block,
    # but an uncorroborated rival within the margin still does.
    twin = [cand(1.82, sep=30.0), cand(1.61, sep=40.0), cand(1.55, theta=20.0)]
    (choice,) = select_argmax(
        [rec(twin)], PRODUCTION_GATE_SCORE, PRODUCTION_GATE_MARGIN
    )
    assert choice["chosen"] == 0 and choice["reason"] == "stamp-corroborated"
    rival = [cand(0.9, sep=30.0), cand(0.8, theta=20.0)]
    (choice,) = select_argmax(
        [rec(rival)], PRODUCTION_GATE_SCORE, PRODUCTION_GATE_MARGIN
    )
    assert choice["chosen"] is None and "margin" in choice["reason"]

    # Without stamp corroboration the normal bar stands (0.87 < 1.35)...
    (choice,) = select_argmax(
        [rec([cand(0.87)])], PRODUCTION_GATE_SCORE, PRODUCTION_GATE_MARGIN
    )
    assert choice["chosen"] is None
    # ...and a fitted page's record never takes the relaxed path.
    (choice,) = select_argmax(
        [rec([cand(0.77, sep=24.0)], fit_state="fitted")],
        PRODUCTION_GATE_SCORE,
        PRODUCTION_GATE_MARGIN,
    )
    assert choice["chosen"] is None

    # Nashville p8 shape: the wrong pose matches ONE of four scattered junk
    # stamps (min 66m) but the median partner is far out -- not corroborated.
    (choice,) = select_argmax(
        [rec([cand(0.99, sep=66.0, median=420.0)])],
        PRODUCTION_GATE_SCORE,
        PRODUCTION_GATE_MARGIN,
    )
    assert choice["chosen"] is None and "score" in choice["reason"]


def test_refine_ineligible_on_one_gcp_incumbent():
    # #277: a deferred/1-effective-GCP incumbent is a rung-guess; an agreeing
    # challenger from the local search confirms it by construction, so refine
    # must decline and let rung_flip / keep-incumbent decide (nashville p4:
    # refine blessed the half-scale pose at 406 ft that rung_flip had been
    # flipping to 26 ft).
    record = fitted_record(incumbent_ver=0.2, challenger_ver=0.35, shift_m=15.0)
    record["incumbent"]["effective_gcps"] = 1
    assert refine_adoption(record) is None
    # Two distinct intersections is a real fit: refinement applies again.
    record["incumbent"]["effective_gcps"] = 2
    assert refine_adoption(record) is not None
    # Records cached before effective_gcps existed keep the old behavior.
    del record["incumbent"]["effective_gcps"]
    assert refine_adoption(record) is not None


def test_refine_requires_informative_challenger_evidence():
    # #291: two negative verifications differ by more than the margin, but the
    # difference is noise — P(road) supports neither pose. richmond p380 lost
    # a 16 ft fit to a 65 ft one exactly this way.
    record = fitted_record(incumbent_ver=-0.249, challenger_ver=-0.154, shift_m=15.0)
    record["incumbent"]["effective_gcps"] = 14
    assert refine_adoption(record) is None
    # A challenger with real evidence still refines.
    record = fitted_record(incumbent_ver=-0.249, challenger_ver=0.35, shift_m=15.0)
    record["incumbent"]["effective_gcps"] = 14
    assert refine_adoption(record) is not None


def test_load_page_units_populates_demoted_affine(tmp_path):
    """A declined pose (misscale) loads as demoted_affine; a fitted one as gen_affine.

    Two near-identical loaders build PageUnits (this module's panel loader and
    edge_join_experiment.load_page_units for base pages); the #315 fix was
    silently a no-op TWICE because only one of them populated the field. This
    pins the base loader.
    """
    import json as json_module

    from PIL import Image

    from mapsnap.osm_snap_experiment import load_page_units

    corners = [[-96.0, 46.0], [-95.9, 46.0], [-95.9, 45.9], [-96.0, 45.9]]
    for stem, doc in [
        ("p1", {"width": 40, "height": 30, "status": "misscale", "corners": corners}),
        ("p2", {"width": 40, "height": 30, "corners": corners}),
        ("p3", {"width": 40, "height": 30, "status": "nofit"}),
    ]:
        Image.new("RGB", (40, 30)).save(tmp_path / f"{stem}.jpg")
        (tmp_path / f"{stem}.georef.json").write_text(json_module.dumps(doc))
    units = {u.stem: u for u in load_page_units(tmp_path)}
    assert units["p1"].fit_state == "misscale"
    assert units["p1"].demoted_affine is not None
    assert units["p1"].gen_affine is None
    assert units["p2"].fit_state == "fitted"
    assert units["p2"].gen_affine is not None
    assert units["p2"].demoted_affine is None
    assert units["p3"].demoted_affine is None


def test_runner_up_affines_load_for_fitted_pages_only(tmp_path):

    from mapsnap.edge_join_experiment import runner_up_affines_of

    corners = [[-80.0, 25.0], [-79.99, 25.0], [-79.99, 24.99], [-80.0, 24.99]]
    georef = {"runner_up_poses": [{"corners": corners, "score_ratio": 0.98}]}
    affines = runner_up_affines_of(georef, 1000, 1000)
    assert len(affines) == 1
    assert affines[0].shape == (2, 3)
    assert runner_up_affines_of(None, 1000, 1000) == []
    assert runner_up_affines_of({"runner_up_poses": [{"bad": 1}]}, 1000, 1000) == []


# --- #325 phase 2: the decision trace --------------------------------------


def rescue_record(top_select: float, rival_select: float | None = None) -> dict:
    """A rescue-state record with a plausible top candidate and optional rival."""
    candidates = [
        {
            "world_affine": affine(),
            "center": [-74.0 + 500 * SCALE_DEG, 39.99],
            "theta_deg": 0.0,
            "select_score": top_select,
            "verification": top_select,
            "plausible": True,
            "gate_reasons": [],
        }
    ]
    if rival_select is not None:
        candidates.append(
            {
                "world_affine": affine(500.0),
                "center": [-74.0 + 500 * SCALE_DEG + 500.0 / KX, 39.99],
                "theta_deg": 0.0,
                "select_score": rival_select,
                "verification": rival_select,
                "plausible": True,
                "gate_reasons": [],
            }
        )
    margin = top_select - rival_select if rival_select is not None else top_select
    return {
        "target": "p1",
        "status": "ok",
        "fit_state": "none",
        "width": 1000,
        "height": 1000,
        "candidates": candidates,
        "margin": round(margin, 4),
    }


def bars_by_rule(decision: dict) -> dict[str, dict]:
    return {bar["rule"]: bar for bar in decision["bars"]}


def test_decision_block_rescue_accepts_a_clear_winner():
    decision = decision_block(rescue_record(2.0, 1.0))
    assert decision["path"] == "rescue"
    assert decision["page_verdict"] == "rescue"
    bars = bars_by_rule(decision)
    assert bars["select"]["verdict"] == "pass" and bars["select"]["got"] == 2.0
    assert bars["margin"]["verdict"] == "pass" and bars["margin"]["got"] == 1.0
    assert bars["stamp-corroborated"]["verdict"] == "n/a"
    assert {s["rule"] for s in decision["skipped"]} == {"volume-energy"}


def test_decision_block_rescue_reports_the_failing_bar():
    # Below the production score gate: the verdict and the bar agree with
    # select_argmax's own reason.
    decision = decision_block(rescue_record(1.0, 0.2))
    assert decision["page_verdict"] == "abstain"
    assert bars_by_rule(decision)["select"]["verdict"] == "fail"
    assert decision["argmax_reason"].startswith("score 1.00 <")
    # Ambiguous near-tie: score clears, margin does not.
    decision = decision_block(rescue_record(2.0, 1.9))
    assert decision["page_verdict"] == "abstain"
    bars = bars_by_rule(decision)
    assert bars["select"]["verdict"] == "pass"
    assert bars["margin"]["verdict"] == "fail"


def test_decision_block_rescue_stamp_corroboration_relaxes_the_bar():
    record = rescue_record(1.0)
    record["candidates"][0]["stamp_median_m"] = 40.0
    decision = decision_block(record)
    bars = bars_by_rule(decision)
    assert bars["stamp-corroborated"]["verdict"] == "pass"
    assert "stamp-corroborated bar" in bars["select"]["need"]
    assert decision["page_verdict"] == "rescue"


def test_decision_block_challenge_matches_arbitrate_challenge():
    record = fitted_record(incumbent_ver=-0.3, challenger_ver=1.5, shift_m=100.0)
    decision = decision_block(record)
    assert decision["path"] == "challenge"
    assert decision["page_verdict"] == "challenge"
    bars = bars_by_rule(decision)
    assert all(
        bars[rule]["verdict"] == "pass"
        for rule in (
            "challenge/select",
            "challenge/margin",
            "challenge/disagreement",
            "challenge/incumbent-indefensible",
            "challenge/verification",
            "challenge/name-parity",
        )
    )
    # A defensible incumbent blocks the challenge and is named as the reason.
    record = fitted_record(incumbent_ver=0.6, challenger_ver=1.5, shift_m=100.0)
    decision = decision_block(record)
    assert decision["page_verdict"] == "keep"
    assert (
        bars_by_rule(decision)["challenge/incumbent-indefensible"]["verdict"] == "fail"
    )
    assert {s["rule"] for s in decision["skipped"]} >= {"remote-search"}


def test_decision_block_refine_matches_refine_adoption():
    record = fitted_record(incumbent_ver=0.6, challenger_ver=0.8, shift_m=15.0)
    decision = decision_block(record)
    assert decision["page_verdict"] == "refine"
    bars = bars_by_rule(decision)
    assert bars["refine/agreement"]["verdict"] == "pass"
    assert bars["refine/verification-margin"]["verdict"] == "pass"
    assert bars["challenge/disagreement"]["verdict"] == "fail"
    # Same lock, no verification edge: keep.
    record = fitted_record(incumbent_ver=0.6, challenger_ver=0.62, shift_m=15.0)
    assert decision_block(record)["page_verdict"] == "keep"


def test_decision_block_unsearched_page_is_skipped():
    decision = decision_block(
        {"target": "p1", "status": "no_prob", "fit_state": "none"}
    )
    assert decision["page_verdict"] == "abstain"
    assert decision["bars"] == []
    assert decision["skipped"][0]["rule"] == "all"


# --- incumbent scale rung -----------------------------------------------------


def test_affine_m_per_px_reads_the_fixture_scale():
    # The fixture is built at ~0.6 m/px.
    assert affine_m_per_px(np.array(affine())) == pytest.approx(0.6, abs=0.01)


def test_with_incumbent_scale_adds_a_missing_rung_and_dedupes_an_existing_one():
    ladder = [ScalePrior(0.3, 0.05, "volume-median")]
    # The fixture's 0.6 m/px is a 2x rung the ladder lacks: appended, median first.
    added = with_incumbent_scale(ladder, np.array(affine()))
    assert [p.source for p in added] == ["volume-median", "incumbent"]
    assert added[1].m_per_px == pytest.approx(0.6, abs=0.01)
    # Within ~16% of an existing rung: the ladder is returned unchanged.
    same = with_incumbent_scale(
        [ScalePrior(0.55, 0.05, "volume-median")], np.array(affine())
    )
    assert [p.source for p in same] == ["volume-median"]
