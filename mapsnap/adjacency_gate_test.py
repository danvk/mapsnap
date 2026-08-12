import json
from pathlib import Path

import pytest

from mapsnap import sidecar
from mapsnap.adjacency_gate import (
    arbitrate_suspects,
    contradiction_centers,
    demote,
    load_fitted_pages,
)

# A row of 1000x800 pages, 1 px = 1e-5 deg (~1.1 m/px): p1 | p2 | p3 west-to-east.
PAGE_LON_SPAN = 0.01
LAT0 = 40.0


def georef_doc(lon0: float, lat0: float = LAT0, n_gcps: int = 5) -> dict:
    corners = [
        [lon0, lat0],
        [lon0 + PAGE_LON_SPAN, lat0],
        [lon0 + PAGE_LON_SPAN, lat0 - 0.008],
        [lon0, lat0 - 0.008],
    ]
    intersections = [
        {"x": 100.0 + 150.0 * i, "y": 100.0 + 100.0 * i, "inlier": True}
        for i in range(n_gcps)
    ]
    return {
        "width": 1000,
        "height": 800,
        "corners": corners,
        "intersections": intersections,
    }


def claim(key: str, x_frac: float, y_frac: float = 0.5) -> dict:
    return {
        "key": key,
        "number": int("".join(c for c in key if c.isdigit())),
        "claim": True,
        "x_frac": x_frac,
        "y_frac": y_frac,
    }


def write_volume(tmp_path: Path, p3_lat_shift: float, p3_gcps: int) -> Path:
    """Three mutually-claiming pages; p3's fit optionally displaced north."""
    adjacency = {
        "pages": {
            "p1": {"detections": [claim("2", 0.98)]},
            "p2": {"detections": [claim("1", 0.02), claim("3", 0.98)]},
            "p3": {"detections": [claim("2", 0.02)]},
        },
        "adjacency": [["p1", "p2"], ["p2", "p3"]],
    }
    (tmp_path / "adjacency.json").write_text(json.dumps(adjacency))
    for i, (stem, shift, gcps) in enumerate(
        [("p1", 0.0, 5), ("p2", 0.0, 5), ("p3", p3_lat_shift, p3_gcps)]
    ):
        doc = georef_doc(i * PAGE_LON_SPAN, LAT0 + shift, gcps)
        (tmp_path / f"{stem}.georef.json").write_text(json.dumps(doc))
    return tmp_path


def gate(volume: Path):
    adjacency = json.loads((volume / "adjacency.json").read_text())
    pages = load_fitted_pages(volume, adjacency)
    return pages, arbitrate_suspects(adjacency, pages)


def test_compatible_row_produces_no_verdicts(tmp_path):
    _, verdicts = gate(write_volume(tmp_path, p3_lat_shift=0.0, p3_gcps=1))
    assert verdicts == []


def test_displaced_weak_page_is_demoted_with_hints(tmp_path):
    # p3's fit sits ~330 m north of where p2's printed claim says it belongs,
    # and only 1 GCP supports it: uncorroborated suspect, hard signal agrees.
    volume = write_volume(tmp_path, p3_lat_shift=0.003, p3_gcps=1)
    pages, verdicts = gate(volume)
    assert [v.stem for v in verdicts] == ["p3"]
    verdict = verdicts[0]
    assert verdict.reason == "uncorroborated"
    assert verdict.signal == "gcps=1"
    # The hint is p2's stamp mapped through p2's (trusted) fit: near the p2/p3 seam.
    (partner,) = {p["stem"] for p in verdict.partners}
    assert partner == "p2"
    lon, lat = verdict.partners[0]["stamp"]
    assert abs(lon - (PAGE_LON_SPAN + 0.98 * PAGE_LON_SPAN)) < 1e-6
    assert abs(lat - (LAT0 - 0.004)) < 1e-6

    demote(volume, pages["p3"], verdict)
    # The pose stays in its own sidecar; what changes is that georef no longer
    # claims it, so it stops counting as a published fit and the arbiter can
    # still weigh it (with the contradiction against it).
    doc = json.loads((volume / "p3.georef.json").read_text())
    assert sidecar.status(doc) == sidecar.CONTRADICTED
    assert not sidecar.internally_valid(doc)
    assert doc["status_detail"]["reason"] == "uncorroborated"
    assert doc["corners"]
    centers = contradiction_centers(volume, "p3")
    assert len(centers) == 1 and abs(centers[0][0] - lon) < 1e-9
    assert contradiction_centers(volume, "p2") == []


def test_structurally_sound_suspect_is_never_demoted(tmp_path):
    # Same displacement, but p3 is a 5-GCP fit with median rotation and scale:
    # the stamp alone must not demote it.
    _, verdicts = gate(write_volume(tmp_path, p3_lat_shift=0.003, p3_gcps=5))
    assert verdicts == []


def test_stamp_gate_accepts_true_pose_and_rejects_the_alias(tmp_path):
    import numpy as np

    from mapsnap.adjacency_gate import load_stamp_gate

    volume = write_volume(tmp_path, p3_lat_shift=0.003, p3_gcps=1)
    pages, verdicts = gate(volume)
    demote(volume, pages["p3"], verdicts[0])
    stamp_gate = load_stamp_gate(volume, "p3", 1000, 800)
    assert stamp_gate is not None
    # p3's own claim of p2 exists, so the strict pairwise check applies.
    assert stamp_gate.own_claims == [[(0.02 * 1000, 0.5 * 800)]]

    def affine(lon0: float, lat0: float) -> np.ndarray:
        scale = PAGE_LON_SPAN / 1000
        return np.array([[scale, 0.0, lon0], [0.0, -0.008 / 800, lat0]])

    true_pose = affine(2 * PAGE_LON_SPAN, LAT0)  # back where p2's stamp says
    alias = affine(2 * PAGE_LON_SPAN, LAT0 + 0.003)  # the old wrong pose
    good = stamp_gate.separation_m(true_pose)
    bad = stamp_gate.separation_m(alias)
    assert good is not None and good < 100.0
    assert bad is not None and bad > 100.0


def test_stamp_gate_footprint_fallback_when_own_read_missing(tmp_path):
    import json as json_module

    import numpy as np

    from mapsnap.adjacency_gate import load_stamp_gate

    volume = write_volume(tmp_path, p3_lat_shift=0.003, p3_gcps=1)
    pages, verdicts = gate(volume)
    demote(volume, pages["p3"], verdicts[0])
    # Erase p3's own claim of p2: the neighbor's stamp must then at least
    # touch the candidate footprint.
    adjacency = json_module.loads((volume / "adjacency.json").read_text())
    adjacency["pages"]["p3"]["detections"] = []
    (volume / "adjacency.json").write_text(json_module.dumps(adjacency))
    stamp_gate = load_stamp_gate(volume, "p3", 1000, 800)
    assert stamp_gate is not None and stamp_gate.own_claims == [[]]

    def affine(lon0: float, lat0: float) -> np.ndarray:
        scale = PAGE_LON_SPAN / 1000
        return np.array([[scale, 0.0, lon0], [0.0, -0.008 / 800, lat0]])

    touching = stamp_gate.separation_m(affine(2 * PAGE_LON_SPAN, LAT0))
    far = stamp_gate.separation_m(affine(2 * PAGE_LON_SPAN + 0.05, LAT0))
    assert touching is not None and touching < 100.0
    assert far is not None and far > 100.0


def test_corroborated_pair_blames_the_edge(tmp_path):
    # p3 gains a compatible neighbor p4: both sides of the contradicted p2~p3
    # edge are vouched for, so the edge is junk and nobody is demoted.
    volume = write_volume(tmp_path, p3_lat_shift=0.003, p3_gcps=1)
    adjacency = json.loads((volume / "adjacency.json").read_text())
    adjacency["pages"]["p3"]["detections"].append(claim("4", 0.98))
    adjacency["pages"]["p4"] = {"detections": [claim("3", 0.02)]}
    adjacency["adjacency"].append(["p3", "p4"])
    (volume / "adjacency.json").write_text(json.dumps(adjacency))
    # p4 fitted consistently with p3's (displaced) pose: their edge agrees.
    (volume / "p4.georef.json").write_text(
        json.dumps(georef_doc(3 * PAGE_LON_SPAN, LAT0 + 0.003, 5))
    )
    _, verdicts = gate(volume)
    assert verdicts == []


def coarse_doc(lon0: float, lat0: float = LAT0, n_gcps: int = 5) -> dict:
    """A double-scale sheet: the same 1000x800 px covering twice the ground span."""
    doc = georef_doc(lon0, lat0, n_gcps)
    doc["corners"] = [
        [lon0, lat0],
        [lon0 + 2 * PAGE_LON_SPAN, lat0],
        [lon0 + 2 * PAGE_LON_SPAN, lat0 - 0.016],
        [lon0, lat0 - 0.016],
    ]
    return doc


def write_mixed_scale_volume(tmp_path: Path, p5_lat_shift: float) -> Path:
    """Three fine pages (the scale median) plus a coarse mutual pair p4~p5."""
    adjacency = {
        "pages": {
            "p1": {"detections": [claim("2", 0.98)]},
            "p2": {"detections": [claim("1", 0.02), claim("3", 0.98)]},
            "p3": {"detections": [claim("2", 0.02)]},
            "p4": {"detections": [claim("5", 0.98)]},
            "p5": {"detections": [claim("4", 0.02)]},
        },
        "adjacency": [["p1", "p2"], ["p2", "p3"], ["p4", "p5"]],
    }
    (tmp_path / "adjacency.json").write_text(json.dumps(adjacency))
    for i, stem in enumerate(["p1", "p2", "p3"]):
        doc = georef_doc(i * PAGE_LON_SPAN)
        (tmp_path / f"{stem}.georef.json").write_text(json.dumps(doc))
    (tmp_path / "p4.georef.json").write_text(json.dumps(coarse_doc(4 * PAGE_LON_SPAN)))
    (tmp_path / "p5.georef.json").write_text(
        json.dumps(coarse_doc(6 * PAGE_LON_SPAN, LAT0 + p5_lat_shift))
    )
    return tmp_path


def test_edge_scale_factor_widens_for_coarse_never_narrows_for_fine():
    import math as math_module

    from mapsnap.adjacency_gate import edge_scale_factor

    def page(log_scale: float):
        import numpy as np

        from mapsnap.adjacency_gate import FittedPage

        return FittedPage(
            stem="p",
            affine=np.zeros((2, 3)),
            width=1,
            height=1,
            channel_paths=[],
            gcps=0,
            theta_deg=0.0,
            log_scale=log_scale,
        )

    median = -13.0
    # A coarse pair at 2x the median widens the bar 2x; the coarser page rules.
    assert edge_scale_factor(
        page(median + math_module.log(2)), page(median), median
    ) == pytest.approx(2.0)
    # A fine pair never narrows below the calibrated floor.
    assert (
        edge_scale_factor(
            page(median - math_module.log(2)), page(median - math_module.log(2)), median
        )
        == 1.0
    )


def test_coarse_pair_at_metric_bar_is_not_contradicted(tmp_path):
    from mapsnap.adjacency_gate import edge_contradictions

    # ~112 m stamp separation: over the fixed 100 m bar, but the same *pixel*
    # error a fine sheet would show at ~56 m. The doubled bar clears it.
    volume = write_mixed_scale_volume(tmp_path, p5_lat_shift=0.0008)
    adjacency = json.loads((volume / "adjacency.json").read_text())
    pages = load_fitted_pages(volume, adjacency)
    contradictions, edge_flags = edge_contradictions(adjacency, pages)
    assert contradictions == []
    assert edge_flags["p4"] == [False]


def test_coarse_pair_far_over_scaled_bar_is_still_contradicted(tmp_path):
    from mapsnap.adjacency_gate import edge_contradictions

    # ~270 m separation exceeds even the doubled bar: still a contradiction.
    volume = write_mixed_scale_volume(tmp_path, p5_lat_shift=0.0024)
    adjacency = json.loads((volume / "adjacency.json").read_text())
    pages = load_fitted_pages(volume, adjacency)
    contradictions, _ = edge_contradictions(adjacency, pages)
    assert [(c.a, c.b) for c in contradictions] == [("p4", "p5")]


def test_fine_pair_over_metric_bar_is_still_contradicted(tmp_path):
    from mapsnap.adjacency_gate import edge_contradictions

    # The same displacement on median-scale sheets keeps the original bar.
    volume = write_volume(tmp_path, p3_lat_shift=0.0012, p3_gcps=5)
    adjacency = json.loads((volume / "adjacency.json").read_text())
    pages = load_fitted_pages(volume, adjacency)
    contradictions, _ = edge_contradictions(adjacency, pages)
    assert [(c.a, c.b) for c in contradictions] == [("p2", "p3")]
