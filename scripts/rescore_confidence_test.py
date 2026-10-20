"""Tests for rescore_confidence.py."""

import json
from pathlib import Path

from rescore_confidence import output_path, page_key, rescore_page, volume_confidences

from mapsnap.confidence import load_model


def annotation(label: str, metadata: list[dict]) -> dict:
    """A published annotation with just the fields re-scoring reads."""
    return {"type": "Annotation", "label": label, "metadata": metadata}


def placed_record(stem: str) -> dict:
    """A minimal provenance record for a page reconcile placed."""
    return {
        "stem": stem,
        "decision": "placed",
        "hypotheses": [
            {
                "source": "georef",
                "chosen": True,
                "center": [-122.18, 37.45],
                "effective_gcps": 6,
                "verification": 1.5,
            }
        ],
        "evidence": {"fit_state": "fitted", "inlier_intersections": 6},
    }


def test_output_path_names_the_model_and_never_the_input():
    iiif = Path("/v/mapsnap.iiif.json")
    assert output_path(iiif, "corpus-v2") == Path(
        "/v/mapsnap.confidence-corpus-v2.iiif.json"
    )
    assert output_path(iiif, "corpus-v2") != iiif


def test_rescore_page_replaces_confidence_in_place_and_keeps_the_rest():
    page = {
        "items": [
            annotation(
                "Menlo Park, California | 1897 | sanborn00680_003 p2 [1]",
                [
                    {"label": "streets", "value": "4"},
                    {"label": "confidence", "value": "0.04"},
                    {"label": "confidence_provisional", "value": "true"},
                    {"label": "split_canvas_x", "value": "0.0"},
                ],
            )
        ]
    }
    confidences = {"p2__1": {"p_good": 0.61, "provisional": False, "model": "v9"}}
    out, counts = rescore_page(page, confidences)
    assert [e["label"] for e in out["items"][0]["metadata"]] == [
        "streets",
        "confidence",
        "confidence_model",
        "split_canvas_x",
    ]
    assert out["items"][0]["metadata"][1]["value"] == "0.61"
    assert counts == {"rescored": 1}
    # The input is untouched.
    assert page["items"][0]["metadata"][1]["value"] == "0.04"


def test_rescore_page_drops_a_score_the_new_model_does_not_give():
    page = {
        "items": [
            annotation(
                "Town | 1900 | item p7",
                [
                    {"label": "confidence", "value": "0.90"},
                    {"label": "streets", "value": "3"},
                ],
            )
        ]
    }
    out, counts = rescore_page(page, {})
    assert out["items"][0]["metadata"] == [{"label": "streets", "value": "3"}]
    assert counts == {"unscored": 1}


def test_volume_confidences_scores_placed_pages_by_lowercased_key(tmp_path: Path):
    (tmp_path / "p1A.provenance.json").write_text(json.dumps(placed_record("p1A")))
    unplaced = placed_record("p2") | {"decision": "unplaced"}
    (tmp_path / "p2.provenance.json").write_text(json.dumps(unplaced))
    model = load_model()
    assert model is not None
    confidences = volume_confidences(tmp_path, model)
    assert set(confidences) == {"p1a"}
    assert 0.0 < confidences["p1a"]["p_good"] < 1.0
    assert confidences["p1a"]["model"] == model["version"]


def test_page_key_keeps_long_loc_suffixes():
    assert (
        page_key("San Francisco, California | 1887 | sanborn00813_002 p0028AL")
        == "p0028AL"
    )
    assert (
        page_key("Menlo Park, California | 1897 | sanborn00680_003 p2 [1]") == "p2__1"
    )
    assert page_key("Town | 1950 | item p0139snew [3]") == "p0139snew__3"
    assert page_key("no page here") is None
