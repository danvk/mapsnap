import json
from pathlib import Path

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
    assert not (volume / "p3.georef.json").exists()
    assert (volume / "p3.georef-contradicted.json").exists()
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
