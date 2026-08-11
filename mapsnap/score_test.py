"""Tests for mapsnap.score."""

import json
from pathlib import Path

import pytest
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from mapsnap.score import (
    LocalFrame,
    PageScore,
    land_fraction,
    sheet_portions,
    summarize,
    truth_footprint_ring,
)


def _identity_georef_item(polygon: list[tuple[float, float]] | None) -> dict:
    """An annotation whose GCPs make pixel == world; optionally with a selector."""
    item: dict = {
        "target": {"source": {"width": 100, "height": 100}},
        "body": {
            "features": [
                {
                    "properties": {"resourceCoords": [px, py]},
                    "geometry": {"coordinates": [px, py]},
                }
                for px, py in [(0, 0), (100, 0), (100, 100), (0, 100)]
            ],
        },
    }
    if polygon is not None:
        points = " ".join(f"{x},{y}" for x, y in polygon)
        item["target"]["selector"] = {
            "type": "SvgSelector",
            "value": f'<svg><polygon points="{points}" /></svg>',
        }
    return item


# truth_footprint_ring


def test_footprint_uses_selector_when_present():
    poly = [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)]
    assert truth_footprint_ring(_identity_georef_item(poly)) == [
        [10.0, 20.0],
        [30.0, 20.0],
        [30.0, 40.0],
        [10.0, 40.0],
    ]


def test_footprint_falls_back_to_gcp_rectangle():
    # No selector (Grand Rapids-style truth): the full source rect through the
    # identity GCP transform is the footprint.
    ring = truth_footprint_ring(_identity_georef_item(None))
    assert ring is not None
    assert [[round(x), round(y)] for x, y in ring] == [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
    ]


def test_footprint_none_without_gcps_or_selector():
    item = _identity_georef_item(None)
    item["body"]["features"] = []
    assert truth_footprint_ring(item) is None


# land_fraction


def test_land_fraction_near_street_band():
    footprint = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    one_street = STRtree([shapely.LineString([(100, -10), (100, 1010)])])
    fraction = land_fraction(footprint, one_street, near_m=120.0)
    # Only the ~220m-wide band around x=100 is "land".
    assert 0.1 < fraction < 0.4


def test_land_fraction_dense_streets_is_all_land():
    footprint = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    grid_streets = STRtree(
        [shapely.LineString([(x, -10), (x, 1010)]) for x in range(0, 1100, 100)]
    )
    assert land_fraction(footprint, grid_streets, near_m=120.0) == 1.0


def test_land_fraction_far_streets_and_empty_tree_are_zero():
    footprint = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    far = STRtree([shapely.LineString([(9000, 0), (9000, 1000)])])
    assert land_fraction(footprint, far, near_m=120.0) == 0.0
    assert land_fraction(footprint, STRtree([]), near_m=120.0) == 0.0


# summarize


def test_summarize_net_score_subtracts_disasters():
    pages = [
        PageScore("p1", area_m2=100.0, land_m2=100.0, rmse_ft=10.0),  # good
        PageScore("p2", area_m2=100.0, land_m2=100.0, rmse_ft=100.0),  # mid: no credit
        PageScore("p3", area_m2=100.0, land_m2=100.0, rmse_ft=500.0),  # disaster
        PageScore("p4", area_m2=100.0, land_m2=100.0, rmse_ft=None),  # unplaced
    ]
    s = summarize(pages, good_ft=25.0, disaster_ft=200.0)
    assert s.n_pages == 4 and s.n_placed == 3
    assert s.good_share == 0.25
    assert s.disaster_share == 0.25
    assert s.net_score == 0.0  # the disaster cancels the success


def test_summarize_ignores_sheet_size():
    # A sheet's SIZE must not decide anything: one placed and one unplaced
    # sheet is 0.5 whether the placed one is four times the other's area or
    # a quarter of it. Under land-AREA weighting this was 0.8 and 0.2 --
    # which is how hudson scored 50.5% while placing 76% of its pages.
    big_good = [
        PageScore("big", area_m2=400.0, land_m2=400.0, rmse_ft=10.0),
        PageScore("small", area_m2=100.0, land_m2=100.0, rmse_ft=None),
    ]
    small_good = [
        PageScore("big", area_m2=400.0, land_m2=400.0, rmse_ft=None),
        PageScore("small", area_m2=100.0, land_m2=100.0, rmse_ft=10.0),
    ]
    assert summarize(big_good).net_score == 0.5
    assert summarize(small_good).net_score == 0.5


def test_summarize_still_discounts_water():
    # Size is out, but the water discount stays: a half-water sheet carries
    # half the weight of a dry one. This is what keeps brooklyn's mostly-water
    # unplaced sheets from costing a full page each (-1.5 rather than -9.3).
    pages = [
        PageScore("dry", area_m2=100.0, land_m2=100.0, rmse_ft=10.0),
        PageScore("harbour", area_m2=100.0, land_m2=50.0, rmse_ft=None),
    ]
    assert summarize(pages).net_score == pytest.approx(1.0 / 1.5)


def test_split_panels_share_one_sheet():
    # Four panels of one sheet weigh the same, in total, as one unsplit sheet:
    # a volume cannot gain or lose score by how finely its sheets were cut.
    quarters = [
        PageScore(
            f"p1__{i}", area_m2=25.0, land_m2=25.0, rmse_ft=10.0, sheet_portion=0.25
        )
        for i in range(1, 5)
    ]
    whole = [PageScore("p2", area_m2=100.0, land_m2=100.0, rmse_ft=None)]
    assert sum(p.weight for p in quarters) == pytest.approx(1.0)
    assert summarize(quarters + whole).net_score == pytest.approx(0.5)


def test_split_panels_weigh_by_paper_share():
    # A 95/5 split weighs 0.95/0.05: losing the sliver costs almost nothing,
    # losing the main panel costs almost the whole sheet.
    lost_sliver = [
        PageScore(
            "p1__1", area_m2=95.0, land_m2=95.0, rmse_ft=10.0, sheet_portion=0.95
        ),
        PageScore("p1__2", area_m2=5.0, land_m2=5.0, rmse_ft=None, sheet_portion=0.05),
    ]
    lost_main = [
        PageScore(
            "p1__1", area_m2=95.0, land_m2=95.0, rmse_ft=None, sheet_portion=0.95
        ),
        PageScore("p1__2", area_m2=5.0, land_m2=5.0, rmse_ft=10.0, sheet_portion=0.05),
    ]
    assert summarize(lost_sliver).net_score == pytest.approx(0.95)
    assert summarize(lost_main).net_score == pytest.approx(0.05)


def test_summarize_empty_is_zero():
    s = summarize([])
    assert s.net_score == 0.0 and s.weight == 0.0


# LocalFrame sanity


def test_local_frame_metres_scale():
    frame = LocalFrame(lon0=-118.0, lat0=34.0)
    x, y = frame.to_xy(-118.0, 34.01)
    assert abs(x) < 1e-6 and 1000 < y < 1200  # ~1.1km per 0.01 deg lat


# sheet_portions


def write_panels(oim: Path, sheet: str, rings: list[list[list[float]]]) -> None:
    oim.mkdir(parents=True, exist_ok=True)
    (oim / f"{sheet}.panels.json").write_text(
        json.dumps({"width": 100, "height": 100, "panels": rings})
    )


def box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_sheet_portions_unsplit_sheet_is_one(tmp_path):
    assert sheet_portions(tmp_path / "oim", ["p1", "p2"]) == {"p1": 1.0, "p2": 1.0}


def test_sheet_portions_split_by_paper_area(tmp_path):
    # A sheet cut 80/20 by paper area, from the OIM region boundaries.
    write_panels(tmp_path / "oim", "p1", [box(0, 0, 80, 100), box(80, 0, 100, 100)])
    portions = sheet_portions(tmp_path / "oim", ["p1__1", "p1__2"])
    assert portions["p1__1"] == pytest.approx(0.8)
    assert portions["p1__2"] == pytest.approx(0.2)
    assert sum(portions.values()) == pytest.approx(1.0)


def test_sheet_portions_normalize_when_panels_overlap(tmp_path):
    # miami p62's shape: two panels each covering nearly the whole sheet. The
    # sheet still carries weight 1, shared between them, rather than 2.
    write_panels(tmp_path / "oim", "p1", [box(0, 0, 100, 100), box(0, 0, 100, 100)])
    portions = sheet_portions(tmp_path / "oim", ["p1__1", "p1__2"])
    assert sum(portions.values()) == pytest.approx(1.0)
    assert portions["p1__1"] == pytest.approx(0.5)


def test_sheet_portions_missing_boundaries_split_evenly(tmp_path, capsys):
    # No panels.json: weight the panels equally and SAY so, rather than
    # silently dropping the sheet out of the denominator.
    portions = sheet_portions(tmp_path / "oim", ["p1__1", "p1__2", "p1__3"])
    assert portions == {"p1__1": 1 / 3, "p1__2": 1 / 3, "p1__3": 1 / 3}
    assert "no panel boundaries" in capsys.readouterr().err
