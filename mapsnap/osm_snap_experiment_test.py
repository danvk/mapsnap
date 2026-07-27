"""Tests for the snap harness's production decision core.

arbitrate_challenge / refine_adoption decide whether to REPLACE a placed
RANSAC fit — the riskiest action in the pipeline — so their gates are pinned
here with the failure modes that motivated them (see the osm-snap PR).
"""

import numpy as np

from mapsnap.edge_join_experiment import PageUnit
from mapsnap.osm_snap_experiment import (
    SNAP_LOG_BEGIN,
    SNAP_LOG_END,
    append_snap_logs,
    arbitrate_challenge,
    candidates_record_fresh,
    canonicalize_refine_keys,
    refine_adopt_set,
    refine_adoption,
    refine_eligible_features,
    refine_rule_outcome,
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
