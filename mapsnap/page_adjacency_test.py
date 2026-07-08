import math
from pathlib import Path

from mapsnap.page_adjacency import (
    classify_edge,
    is_claim,
    mutual_edges,
    polygon_rotation_deg,
    single_digit_height_band,
    volume_page_images,
)


def axis_polygon(width: float = 40.0, height: float = 45.0) -> list[list[float]]:
    return [[0, 0], [width, 0], [width, height], [0, height]]


def rotated_polygon(degrees: float, width: float = 40.0) -> list[list[float]]:
    dx = width * math.cos(math.radians(degrees))
    dy = width * math.sin(math.radians(degrees))
    return [[0, 0], [dx, dy], [dx - dy, dy + dx], [-dy, dx]]


def make_detection(
    number: int,
    edge: str = "R",
    height: float = 45.0,
    confidence: float = 0.95,
    polygon: list[list[float]] | None = None,
) -> dict:
    return {
        "number": number,
        "edge": edge,
        "height": height,
        "confidence": confidence,
        "polygon": polygon if polygon is not None else axis_polygon(),
    }


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


def test_polygon_rotation_deg():
    assert polygon_rotation_deg(axis_polygon()) == 0.0
    assert abs(polygon_rotation_deg(rotated_polygon(17.0)) - 17.0) < 0.01
    # Vertical text boxes are axis-aligned too: fold 90 degrees back to 0.
    assert abs(polygon_rotation_deg(rotated_polygon(90.0))) < 0.01


def test_is_claim_accepts_large_edge_number():
    assert is_claim(make_detection(50), own_number=49)


def test_is_claim_rejects_own_number_center_small_and_lowconf():
    assert not is_claim(make_detection(49), own_number=49)  # the sheet's own number
    assert not is_claim(make_detection(50, edge="center"), own_number=49)
    assert not is_claim(make_detection(50, height=12.0), own_number=49)
    assert not is_claim(make_detection(50, confidence=0.01), own_number=49)


def test_is_claim_rejects_rotated_quads():
    # Every rotated candidate inspected was a misread street name or pipe annotation.
    rotated = make_detection(50, polygon=rotated_polygon(17.0))
    assert not is_claim(rotated, own_number=49)


def test_is_claim_single_digit_confidence_floor():
    # Junk single-digit reads reciprocate by coincidence; require high confidence.
    assert not is_claim(make_detection(6, confidence=0.5), own_number=49)
    assert is_claim(make_detection(6, confidence=0.95), own_number=49)
    # Multi-digit numbers keep the permissive floor.
    assert is_claim(make_detection(50, confidence=0.5), own_number=49)


def test_is_claim_single_digit_height_band():
    band = (30.0, 65.0)
    # A 130px "1" is a frame rule, not a reference; a 45px one matches the printed size.
    assert not is_claim(
        make_detection(1, height=130.0), own_number=49, height_band=band
    )
    assert is_claim(make_detection(1, height=45.0), own_number=49, height_band=band)
    # The band applies only to single digits; multi-digit heights are already trustworthy.
    assert is_claim(make_detection(50, height=130.0), own_number=49, height_band=band)
    # Without a band (no confirmed references), single digits pass on confidence alone.
    assert is_claim(make_detection(1, height=130.0), own_number=49, height_band=None)


def test_single_digit_height_band_from_median():
    band = single_digit_height_band([42.0, 46.0, 48.0])
    assert band is not None
    assert abs(band[0] - 0.65 * 46.0) < 1e-9
    assert abs(band[1] - 1.4 * 46.0) < 1e-9


def test_single_digit_height_band_empty():
    assert single_digit_height_band([]) is None


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
