from pathlib import Path

from mapsnap.page_adjacency import (
    classify_edge,
    is_claim,
    mutual_edges,
    volume_page_images,
)


def make_detection(
    number: int, edge: str = "R", height: float = 45.0, confidence: float = 0.9
) -> dict:
    return {"number": number, "edge": edge, "height": height, "confidence": confidence}


def test_classify_edge_bands():
    # Image-relative sides (page orientation is unknown at this stage): T/B/L/R.
    assert classify_edge(0.5, 0.5) == "center"
    assert classify_edge(0.1, 0.5) == "L"
    assert classify_edge(0.9, 0.5) == "R"
    assert classify_edge(0.5, 0.1) == "T"
    assert classify_edge(0.5, 0.9) == "B"
    assert classify_edge(0.05, 0.05) == "TL"
    assert classify_edge(0.95, 0.05) == "TR"
    assert classify_edge(0.05, 0.95) == "BL"
    assert classify_edge(0.95, 0.95) == "BR"


def test_is_claim_accepts_large_edge_number():
    assert is_claim(make_detection(50), own_number=49)


def test_is_claim_rejects_own_number_center_small_and_lowconf():
    assert not is_claim(make_detection(49), own_number=49)  # the sheet's own number
    assert not is_claim(make_detection(50, edge="center"), own_number=49)
    assert not is_claim(make_detection(50, height=12.0), own_number=49)
    assert not is_claim(make_detection(50, confidence=0.01), own_number=49)


def test_mutual_edges_requires_reciprocity():
    claims = {"p49": {50, 51}, "p50": {49}, "p51": set()}
    # 49<->50 reciprocate; 49->51 is unreciprocated and produces no edge.
    assert mutual_edges(claims) == [("p49", "p50")]


def test_mutual_edges_lettered_stems():
    # Chicago-style stems: numbers resolve to the lettered page stem.
    claims = {"p59w": {60}, "p60w": {59}}
    assert mutual_edges(claims) == [("p59w", "p60w")]


def test_mutual_edges_empty_when_one_sided():
    assert mutual_edges({"p1": {2}, "p2": set()}) == []


def test_volume_page_images_skips_splits_and_keymaps(tmp_path: Path):
    for name in ["p0.jpg", "p1.jpg", "p2.jpg", "p2__1.jpg", "p2__2.jpg"]:
        (tmp_path / name).write_bytes(b"")
    # p0 is a key map (detections sidecar under raw/): excluded from the scan.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "p0.keymap.json").write_text("{}")
    names = [p.name for p in volume_page_images(tmp_path)]
    # p2 itself stays (the parent sheet carries the margin references); panels are skipped.
    assert names == ["p1.jpg", "p2.jpg"]
