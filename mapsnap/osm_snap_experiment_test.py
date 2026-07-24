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
    refine_adoption,
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
    # Within the margin (< +0.1): keep the incumbent — no churn on good fits.
    record = fitted_record(incumbent_ver=0.8, challenger_ver=0.85, shift_m=15.0)
    assert refine_adoption(record) is None
    # Far apart is arbitration territory, not refinement.
    record = fitted_record(incumbent_ver=0.8, challenger_ver=1.2, shift_m=100.0)
    assert refine_adoption(record) is None


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
    record = {"fit_state": "fitted", "georef_mtime": 111}
    assert candidates_record_fresh(record, make_unit("fitted"), 111)
    # A re-run georef rewrote the sidecar: the cached incumbent is stale.
    assert not candidates_record_fresh(record, make_unit("fitted"), 222)
    # The page's fit STATE changed (fitted -> nofit or vice versa).
    assert not candidates_record_fresh(record, make_unit("nofit"), 111)
    # Legacy records (no georef_mtime) always recompute once.
    assert not candidates_record_fresh(
        {"fit_state": "fitted"}, make_unit("fitted"), 111
    )


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
