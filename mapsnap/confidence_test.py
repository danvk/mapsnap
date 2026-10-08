"""Tests for the per-pose confidence score."""

import json
import math
from pathlib import Path

import pytest

from mapsnap.confidence import (
    CATEGORICAL,
    NUMERIC,
    annotate,
    feature_vector,
    load_model,
    p_good,
    page_features,
    panel_fraction,
    source_kind,
    volume_stats,
)


def hypothesis(source, center, *, chosen=False, gcps=0, verification=None, **extra):
    return {
        "source": source,
        "chosen": chosen,
        "center": center,
        "effective_gcps": gcps,
        "verification": verification,
        "name": extra.get("name"),
        "containment": extra.get("containment"),
        "keymap_dist_m": extra.get("keymap_dist_m"),
        "rung": extra.get("rung"),
        "terms": extra.get("terms", {}),
    }


def record(stem, decision, hypotheses, **evidence):
    return {
        "stem": stem,
        "decision": decision,
        "snap_verdict": "refine",
        "merged": ["snap:0"],
        "hypotheses": hypotheses,
        "evidence": {
            "fit_state": "fitted",
            "inlier_intersections": 6,
            "inlier_streets": 5,
            "keymap": "georeferenced",
            "keymap_radius_m": 600.0,
            "mutual_edges": 2,
            "stamp_agreement": {"neighbours": 4, "median_m": 30.0, "agree_100m": 3},
            **evidence,
        },
    }


def strong_page(stem="p1"):
    return record(
        stem,
        "placed",
        [
            hypothesis(
                "georef-snap",
                [-90.0, 30.0],
                chosen=True,
                gcps=7,
                verification=1.6,
                name=0.8,
                containment=0.9,
                keymap_dist_m=60.0,
                rung={"verdict": "on rung", "rung_distance": 0.01},
            ),
            hypothesis("georef", [-90.0001, 30.0], gcps=7, verification=1.2),
            hypothesis("snap:1", [-90.01, 30.0], verification=0.4),
        ],
    )


def weak_page(stem="p2"):
    return record(
        stem,
        "placed",
        [
            hypothesis(
                "snap:0",
                [-90.0, 30.0],
                chosen=True,
                verification=0.2,
                keymap_dist_m=900.0,
                rung={"verdict": "between rungs", "rung_distance": 0.4},
                terms={"ambiguity": 0.5},
            ),
            hypothesis("georef", [-90.02, 30.01], gcps=1, verification=-0.4),
        ],
        inlier_intersections=1,
        inlier_streets=2,
        mutual_edges=0,
        stamp_agreement={"neighbours": 2, "median_m": 800.0, "agree_100m": 0},
    )


def test_source_kind_groups_channels():
    assert source_kind("snap:2") == "snap"
    assert source_kind("georef:contradicted") == "georef"
    assert source_kind("georef-snap") == "georef-snap"
    assert source_kind("streets-candidate") == "streets"


def test_page_features_reads_the_chosen_pose_and_its_rivals():
    features = page_features(strong_page(), 1.0, {"volume_placed_frac": 0.9})
    assert features is not None
    assert features["source"] == "georef-snap"
    assert features["gcps"] == 7 and features["fit_gcps"] == 7
    assert features["agree_50m"] == 1  # the georef pose ~10 m away
    assert features["agree_200m"] == 1  # snap:1 is ~1 km away
    assert features["keymap_dist_rel"] == pytest.approx(0.1)
    assert features["stamp_agree_frac"] == pytest.approx(0.75)
    assert features["volume_placed_frac"] == 0.9
    assert not features["panel"]


def test_page_features_is_none_without_a_chosen_pose():
    abstained = record("p3", "abstained", [hypothesis("unplaced", None, chosen=True)])
    assert page_features(abstained, 1.0, {}) is None


def test_feature_vector_one_hots_categoricals_and_keeps_missing_as_none():
    features = page_features(weak_page(), 1.0, {})
    assert features is not None
    values, names = feature_vector(features)
    assert (
        len(values)
        == len(names)
        == len(NUMERIC) + sum(len(levels) for levels in CATEGORICAL.values())
    )
    assert values[names.index("source=snap")] == 1.0
    assert values[names.index("source=georef")] == 0.0
    assert values[names.index("name")] is None  # not recorded


def test_p_good_applies_fill_standardization_and_missingness():
    columns = {
        name: {"fill": 0.0, "mean": 0.0, "scale": 1.0, "coef": 0.0}
        for name in feature_vector(page_features(strong_page(), 1.0, {}) or {})[1]
    }
    columns["gcps"] = {"fill": 0.0, "mean": 2.0, "scale": 2.0, "coef": 1.0}
    columns["name"] = {
        "fill": 0.5,
        "mean": 0.5,
        "scale": 1.0,
        "coef": 3.0,
        "missing_coef": -1.0,
    }
    model = {"version": "test", "intercept": 0.5, "columns": columns}
    strong = page_features(strong_page(), 1.0, {}) or {}
    # gcps 7 -> (7 - 2) / 2 = 2.5; name 0.8 -> 3 * 0.3 = 0.9; intercept 0.5.
    assert p_good(strong, model) == pytest.approx(1 / (1 + math.exp(-3.9)))
    weak = page_features(weak_page(), 1.0, {}) or {}
    # gcps 0 -> -1.0; name missing -> fill 0.5 (no change) and -1.0; intercept 0.5.
    assert p_good(weak, model) == pytest.approx(1 / (1 + math.exp(1.5)))


def test_the_shipped_model_ranks_a_strong_fit_above_a_weak_one():
    model = load_model()
    assert model is not None, "models/confidence.json is missing"
    stats = {"volume_placed_frac": 0.85, "volume_median_verification": 1.2}
    strong = p_good(page_features(strong_page(), 1.0, stats) or {}, model)
    weak = p_good(page_features(weak_page(), 1.0, stats) or {}, model)
    assert 0.0 < weak < strong < 1.0
    assert strong > 0.8 and weak < 0.5


def test_panel_fraction_reads_the_sheets_panels(tmp_path: Path):
    (tmp_path / "p5.panels.json").write_text(
        json.dumps(
            {
                "width": 100,
                "height": 100,
                "panels": [
                    [[0, 0], [100, 0], [100, 25], [0, 25]],
                    [[0, 25], [100, 25], [100, 100], [0, 100]],
                ],
            }
        )
    )
    assert panel_fraction(tmp_path, "p5__1") == pytest.approx(0.25)
    assert panel_fraction(tmp_path, "p5__2") == pytest.approx(0.75)
    assert panel_fraction(tmp_path, "p5") == 1.0
    assert panel_fraction(tmp_path, "p9__1") == 1.0  # no panels.json: treated as whole


def test_volume_stats_skip_superseded_sheets():
    split = {"stem": "p4", "decision": "superseded", "hypotheses": []}
    abstained = {"stem": "p3", "decision": "abstained", "hypotheses": []}
    stats = volume_stats([strong_page(), weak_page(), split, abstained])
    assert stats["volume_placed_frac"] == pytest.approx(2 / 3)
    assert stats["volume_median_verification"] == pytest.approx(0.9)


def test_annotate_scores_placed_pages_and_marks_panels_provisional(tmp_path: Path):
    panel = strong_page("p7__1")
    abstained = record("p8", "abstained", [hypothesis("unplaced", None, chosen=True)])
    records = [strong_page(), panel, abstained]
    model = load_model()
    annotate(records, tmp_path, model)
    page_confidence, panel_confidence = (
        records[0]["confidence"],
        records[1]["confidence"],
    )
    assert 0 < page_confidence["p_good"] < 1
    assert page_confidence["provisional"] is False
    assert panel_confidence["provisional"] is True
    assert page_confidence["model"] == (model or {})["version"]
    assert "confidence" not in records[2]


def test_annotate_without_a_model_changes_nothing(tmp_path: Path, monkeypatch):
    from mapsnap import confidence

    monkeypatch.setattr(confidence, "load_model", lambda: None)
    records = [strong_page()]
    annotate(records, tmp_path)
    assert "confidence" not in records[0]
