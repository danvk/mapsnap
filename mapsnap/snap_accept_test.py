import json
import math

from mapsnap.snap_accept import (
    AcceptModel,
    LogisticModel,
    load_accept_model,
    replace_terms,
    rescue_terms,
)

KX = 111_320.0 * 0.766  # cos(40 deg)
SCALE_DEG = 0.6 / KX  # ~0.6 m/px


def affine(lon_shift_m: float = 0.0) -> list[list[float]]:
    """A north-up page->lonlat affine near 40N, optionally slid east by meters."""
    return [
        [SCALE_DEG, 0.0, -74.0 + lon_shift_m / KX],
        [0.0, -SCALE_DEG * KX / 110_540.0, 40.0],
    ]


def candidate(shift_m: float = 0.0, select: float | None = 1.0, **overrides) -> dict:
    """A scored candidate with every field the accept model reads."""
    fields = {
        "world_affine": affine(shift_m),
        "center": [-74.0, 40.0],
        "theta_deg": 0.0,
        "select_score": select,
        "verification": select,
        "inlier_frac": 0.6,
        "ncc_fine": 0.5,
        "chamfer_mean_m": 8.0,
        "n_points": 2000,
        "scale_m_per_px": 0.6,
        "scale_adjust": 1.0,
        "center_dist_m": 40.0,
    }
    fields.update(overrides)
    return fields


def record(target: str = "p1", candidates: list[dict] | None = None, **fields) -> dict:
    """A candidates.jsonl record for a 1000 x 1000 px page."""
    return {
        "target": target,
        "status": "ok",
        "fit_state": "none",
        "width": 1000,
        "height": 1000,
        "candidates": candidates if candidates is not None else [candidate()],
        **fields,
    }


def one_term_model(name: str, coef: float, threshold: float = 0.5) -> LogisticModel:
    """A model reading one term, standardized to itself."""
    return LogisticModel(
        0.0, {name: {"mean": 0.0, "scale": 1.0, "coef": coef}}, threshold
    )


def test_logistic_model_probability_and_json_round_trip(tmp_path):
    model = LogisticModel(
        0.5, {"ncc_fine": {"mean": 0.4, "scale": 0.2, "coef": 2.0}}, threshold=0.3
    )
    expected = 1 / (1 + math.exp(-(0.5 + 2.0 * (0.8 - 0.4) / 0.2)))
    assert math.isclose(model.probability({"ncc_fine": 0.8}), expected)
    path = tmp_path / "snap_accept.json"
    path.write_text(
        json.dumps(
            {"version": "test", "rescue": model.to_json(), "replace": model.to_json()}
        )
    )
    loaded = load_accept_model(path)
    assert loaded.version == "test"
    assert loaded.rescue.threshold == 0.3
    assert math.isclose(loaded.replace.probability({"ncc_fine": 0.8}), expected)


def test_rescue_terms_read_the_top_candidate_and_its_rivals():
    page = record(
        target="p4__2",
        candidates=[
            candidate(select=1.4),
            candidate(shift_m=3.0, select=1.3),  # the same lock: agrees
            candidate(shift_m=800.0, select=0.9),  # a distinct rival
        ],
    )
    terms = rescue_terms(page)
    assert terms["panel"] == 1.0
    assert terms["agree"] == 1.0
    assert math.isclose(terms["log_distinct_margin"], math.log1p(0.5))
    # 1000 x 1000 px at 0.6 m/px is 0.36 km^2.
    assert math.isclose(terms["log_footprint"], math.log(0.36))
    assert math.isclose(terms["log_center_dist"], math.log1p(40.0))
    assert terms["abs_log_scale_adjust"] == 0.0


def test_rescue_terms_cap_a_page_with_no_rival():
    terms = rescue_terms(record())
    assert math.isclose(terms["log_distinct_margin"], math.log1p(3.0))
    assert terms["agree"] == 0.0
    assert terms["panel"] == 0.0


def test_replace_terms_add_the_incumbent_head_to_head():
    page = record(
        fit_state="fitted",
        candidates=[candidate(shift_m=300.0, select=1.8, verification=1.2)],
        incumbent={
            "world_affine": affine(),
            "verification": -0.2,
            "ncc_fine": 0.1,
            "inlier_frac": 0.3,
            "chamfer_mean_m": 45.0,
            "effective_gcps": 4,
            "name": {"score": 0.2},
        },
    )
    terms = replace_terms(page)
    assert math.isclose(terms["verification_gain"], 1.4)
    assert terms["incumbent_chamfer_m"] == 30.0  # capped
    assert math.isclose(terms["log_incumbent_gcps"], math.log1p(4))
    assert math.isclose(terms["name_gain"], -0.2)
    # A 300 m slide east is ~984 ft of grid RMSE.
    assert math.isclose(math.expm1(terms["log_disagreement"]), 984.0, rel_tol=0.01)


def test_unscorable_records_get_no_probability():
    model = AcceptModel(
        "test", one_term_model("ncc_fine", 1.0), one_term_model("ncc_fine", 1.0)
    )
    assert model.rescue_probability(record(candidates=[candidate(select=None)])) is None
    assert model.rescue_probability(record(candidates=[])) is None
    # A fitted page needs an incumbent pose to be compared against.
    assert model.replace_probability(record(fit_state="fitted")) is None
