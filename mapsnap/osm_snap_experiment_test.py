"""Tests for osm_snap_experiment: the refine-rule sweep harness."""

from mapsnap.osm_snap_experiment import (
    canonicalize_refine_keys,
    refine_adopt_set,
    refine_eligible_features,
    refine_rule_outcome,
)
from mapsnap.snap_volume_test import (
    fitted_record,
)


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
