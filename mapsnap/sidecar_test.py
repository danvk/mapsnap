"""Tests for the georef sidecar contract (verdict inside the file, #270 phase 3)."""

import json

from mapsnap import sidecar


def posed_doc() -> dict:
    return {
        "width": 100,
        "height": 80,
        "corners": [[0, 1], [1, 1], [1, 0], [0, 0]],
        "intersections": [],
    }


def test_absent_status_reads_as_accepted():
    # A stage with nothing to complain about writes no status, so every
    # pre-existing sidecar in the corpus reads as accepted.
    assert sidecar.status(posed_doc()) == sidecar.ACCEPTED
    assert sidecar.accepted(posed_doc())


def test_poseless_doc_is_never_accepted():
    # A neighborhood-only sidecar records that the channel was asked and had no
    # answer; it must not be mistaken for a fit just because no status is set.
    assert not sidecar.accepted({"keymap": {"lat": 1, "lon": 2}})


def test_demote_keeps_the_pose_and_records_why(tmp_path):
    path = tmp_path / "p1.georef.json"
    path.write_text(json.dumps(posed_doc()))
    sidecar.demote(path, sidecar.MISSCALE, {"px_per_ft": 0.5})
    doc = json.loads(path.read_text())
    assert sidecar.status(doc) == sidecar.MISSCALE
    assert not sidecar.accepted(doc)
    # The whole point: the pose survives its demotion, so the arbiter can weigh
    # it. Under the rename convention this file simply ceased to exist.
    assert doc["corners"] == posed_doc()["corners"]
    assert doc["status_detail"]["px_per_ft"] == 0.5


def test_demote_twice_merges_detail(tmp_path):
    # georef demotes on scale, then the adjacency gate demotes the same page:
    # the later verdict wins but neither reason is lost.
    path = tmp_path / "p1.georef.json"
    path.write_text(json.dumps(posed_doc()))
    sidecar.demote(path, sidecar.MISSCALE, {"px_per_ft": 0.5})
    sidecar.demote(path, sidecar.CONTRADICTED, {"reason": "uncorroborated"})
    doc = json.loads(path.read_text())
    assert sidecar.status(doc) == sidecar.CONTRADICTED
    assert doc["status_detail"] == {"px_per_ft": 0.5, "reason": "uncorroborated"}


def test_attach_rejected_carries_a_second_pose(tmp_path):
    # One channel, two poses: the key-map retry's winner at the top level and
    # the pose it beat alongside it, each with its own verdict.
    path = tmp_path / "p55.georef.json"
    path.write_text(json.dumps(posed_doc()))
    loser = {**posed_doc(), "status": sidecar.KEYMAP_OUTLIER}
    sidecar.attach_rejected(path, [loser])
    doc = json.loads(path.read_text())
    assert sidecar.accepted(doc)
    (rejected,) = sidecar.rejected_poses(doc)
    assert sidecar.status(rejected) == sidecar.KEYMAP_OUTLIER
    assert not sidecar.accepted(rejected)
