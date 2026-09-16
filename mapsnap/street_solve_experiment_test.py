"""Tests for the street-constraint georeferencer (mapsnap.street_solve_experiment)."""


def test_load_posed_candidates_distinguishes_missing_from_empty(tmp_path) -> None:
    """A file with nothing posed is an empty result; no file at all is an error."""
    import json

    import pytest

    from mapsnap.street_solve_experiment import load_posed_candidates

    path = tmp_path / "candidates.jsonl"
    with pytest.raises(SystemExit):
        load_posed_candidates(path)
    path.write_text(
        json.dumps({"stem": "p1", "status": "unposed"})
        + "\n"
        + json.dumps({"stem": "p2", "status": "posed", "corners": None})
        + "\n"
    )
    assert load_posed_candidates(path) == {}
    path.write_text(
        json.dumps({"stem": "p3", "status": "posed", "corners": [[0, 0]]}) + "\n"
    )
    assert list(load_posed_candidates(path)) == ["p3"]
