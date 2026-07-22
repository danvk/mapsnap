"""Tests for mapsnap.score."""

import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from mapsnap.score import (
    LocalFrame,
    PageScore,
    land_fraction,
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


def test_summarize_weights_by_land_not_pages():
    pages = [
        PageScore("big", area_m2=400.0, land_m2=400.0, rmse_ft=10.0),
        PageScore("small", area_m2=100.0, land_m2=100.0, rmse_ft=None),
    ]
    s = summarize(pages)
    assert s.net_score == 0.8  # 400 of 500 land, not 1 of 2 pages


def test_summarize_empty_is_zero():
    s = summarize([])
    assert s.net_score == 0.0 and s.land_m2 == 0.0


# LocalFrame sanity


def test_local_frame_metres_scale():
    frame = LocalFrame(lon0=-118.0, lat0=34.0)
    x, y = frame.to_xy(-118.0, 34.01)
    assert abs(x) < 1e-6 and 1000 < y < 1200  # ~1.1km per 0.01 deg lat
