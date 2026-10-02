import json
from pathlib import Path

import cv2
import numpy as np

from mapsnap.edition_transfer import (
    Edition,
    Sheet,
    Transfer,
    corner_distance_m,
    decide,
    donors_for,
    edition_keys,
    edition_votes,
    neighbour_keys,
    read_georef,
    same_sheet,
    sheet_keys,
    transfer_sheet,
)

# A 400 x 560 px page at 0.5 m/px, north-up, near Long Island City.
WIDTH, HEIGHT = 400, 560
ORIGIN = (-73.95, 40.75)
KX = 111_320.0 * np.cos(np.radians(ORIGIN[1]))
KY = 110_540.0


def corners_for(x_m: float, y_m: float, m_per_px: float = 0.5) -> list[list[float]]:
    """North-up corner quad (TL, TR, BR, BL) with its top-left x_m, y_m from ORIGIN."""
    w, h = WIDTH * m_per_px, HEIGHT * m_per_px
    quad = [(0, 0), (w, 0), (w, -h), (0, -h)]
    return [
        [ORIGIN[0] + (x_m + dx) / KX, ORIGIN[1] + (y_m + dy) / KY] for dx, dy in quad
    ]


def road_map(shift_px: tuple[int, int] = (0, 0)) -> np.ndarray:
    """An irregular network of roads, so a match has one right answer."""
    prob = np.zeros((HEIGHT + 200, WIDTH + 200), np.float32)
    rng = np.random.default_rng(7)
    for _ in range(14):
        x0, y0, x1, y1 = rng.integers(0, max(prob.shape), 4)
        cv2.line(prob, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 9)
    x, y = 100 + shift_px[0], 100 + shift_px[1]
    return prob[y : y + HEIGHT, x : x + WIDTH].copy()


def transfer(**values) -> Transfer:
    """A Transfer that passes every gate, with any field replaced."""
    fields = {
        "key": "p1",
        "donor": "d",
        "donor_key": "p1",
        "corners": corners_for(0, 0),
        "width": WIDTH,
        "height": HEIGHT,
        "ncc": 0.5,
        "ncc_fine": 0.9,
        "inlier_frac": 0.9,
        "chamfer_mean_m": 1.0,
        "scale_adjust": 1.0,
        "rotation_deg": 0.0,
        "scale_ratio": 1.0,
        "shift_m": 5.0,
        "overlap_frac": 0.98,
    }
    return Transfer(**(fields | values))


def test_transfer_recovers_a_shifted_scan_of_the_same_sheet():
    donor = Sheet(
        "p1",
        road_map(),
        {"width": WIDTH, "height": HEIGHT, "corners": corners_for(0, 0)},
    )
    # The target scan's paper starts 20 px (10 m) further east and 10 px (5 m)
    # further south, so its true top-left sits there on the ground.
    target = Sheet("p1", road_map(shift_px=(20, 10)), None)
    result = transfer_sheet(target, donor, "donor")
    assert result is not None
    assert result.accepted
    assert corner_distance_m(result.corners, corners_for(10, -5)) < 3
    assert 5 < result.shift_m < 15


def test_corner_distance_is_in_metres():
    assert corner_distance_m(corners_for(0, 0), corners_for(0, 0)) == 0
    assert abs(corner_distance_m(corners_for(0, 0), corners_for(30, 40)) - 50) < 0.01


def test_a_transfer_needs_road_agreement_and_to_land_on_its_donor():
    assert transfer().accepted
    assert not transfer(inlier_frac=0.3).accepted
    assert not transfer(ncc_fine=0.2).accepted
    # A neighbouring sheet on a regular grid: good road agreement, slid over.
    assert not transfer(shift_m=150.0, overlap_frac=0.5).accepted


def test_transfer_georef_is_a_sidecar_that_remembers_its_match():
    georef = transfer().georef()
    assert georef["corners"] == corners_for(0, 0)
    assert (georef["width"], georef["height"]) == (WIDTH, HEIGHT)
    assert georef["edition_transfer"]["donor"] == "d"


def test_donors_are_newer_editions_nearest_first_then_older(tmp_path):
    editions = [Edition(tmp_path, year, {}, {}) for year in (1898, 1915, 1936, 1950)]
    assert [e.year for e in donors_for(editions[2], editions)] == [1950, 1915, 1898]
    # 1898 is nearer in time, but newer editions go first.
    assert [e.year for e in donors_for(editions[1], editions)] == [1936, 1950, 1898]
    assert [e.year for e in donors_for(editions[3], editions)] == [1936, 1915, 1898]


def test_neighbour_keys_are_the_numbers_either_side():
    assert neighbour_keys("p77") == ["p76", "p78"]
    assert neighbour_keys("p1") == ["p2"]
    assert neighbour_keys("p5N") == ["p4N", "p6N"]
    assert neighbour_keys("pind1") == []


def test_sheets_match_across_editions_by_number_and_suffix():
    assert same_sheet("p5", "p5N")  # Chicago vol. 1: 1906 vs 1950's North part
    assert same_sheet("p0005N", "p5N")
    assert same_sheet("p10", "p10Sa")
    assert not same_sheet("p5N", "p5W")
    assert not same_sheet("p5", "p6")
    assert same_sheet("pind1", "pind1")


def test_edition_keys_prefer_an_exact_key_then_namesakes(tmp_path):
    placed = {"corners": corners_for(0, 0)}
    edition = Edition(
        tmp_path,
        1950,
        {"p5N": placed, "p10Sa": placed, "p10Sb": placed},
        {"p7": placed},
    )
    assert edition_keys(edition, "p5") == ["p5N"]
    assert edition_keys(edition, "p7") == ["p7"]
    assert edition_keys(edition, "p10") == ["p10Sa", "p10Sb"]
    assert edition_keys(edition, "p11") == []


def test_read_georef_skips_unplaced_pages(tmp_path):
    (tmp_path / "p1.georef-final.json").write_text(
        json.dumps({"corners": corners_for(0, 0)})
    )
    (tmp_path / "p2.georef-final.json").write_text(json.dumps({"corners": None}))
    assert read_georef(tmp_path, "p1") == {"corners": corners_for(0, 0)}
    assert read_georef(tmp_path, "p2") is None
    assert read_georef(tmp_path, "p3") is None


def test_sheet_keys_are_numbered_sheets_without_panels_or_the_key_map(tmp_path: Path):
    for name in ("p0", "p2", "p10", "p1", "p3__1", "p3"):
        (tmp_path / f"{name}.roadprob.jpg").write_bytes(b"")
    assert sheet_keys(tmp_path) == ["p1", "p2", "p3", "p10"]


def test_decide_fills_gaps_and_replaces_own_fits_only_when_outvoted():
    good, bad = transfer(), transfer(inlier_frac=0.1)
    assert decide(False, good) == "transfer"
    assert decide(False, bad) == "unplaced"
    assert decide(False, None) == "unplaced"
    assert decide(True, None) == "own"
    assert decide(True, bad, 2000.0, (0, 3)) == "own"
    assert decide(True, good, 3.0) == "own"
    # One of two disagreeing fits is wrong: other editions' fits decide.
    assert decide(True, good, 2317.0, (0, 3)) == "replaced"
    assert decide(True, good, 600.0, (2, 1)) == "own"
    # A tie, as with only two editions, keeps the fit the sheet has.
    assert decide(True, good, 600.0, (0, 0)) == "own"
    assert decide(True, good, 600.0, (1, 1)) == "own"


def test_edition_votes_count_other_editions_fits_near_a_pose(tmp_path):
    here, there = corners_for(0, 0), corners_for(500, 0)
    editions = [
        Edition(tmp_path, 1898, {"p1": {"corners": there}}, {}),
        Edition(tmp_path, 1936, {"p1": {"corners": corners_for(20, 0)}}, {}),
        Edition(tmp_path, 1947, {"p1": {"corners": here}}, {}),
        Edition(tmp_path, 1950, {}, {}),
    ]
    voters = editions[1:]
    assert edition_votes("p1", here, voters) == [1936, 1947]
    assert edition_votes("p1", there, voters) == []
