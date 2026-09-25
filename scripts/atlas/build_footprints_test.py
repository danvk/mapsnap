"""Tests for build_footprints.py."""

import math

from build_footprints import (
    METRES_PER_DEGREE,
    newest_coverage,
    page_transform,
    plausible_pages,
    volume_footprint,
)
from shapely.geometry import MultiPolygon, box

LATITUDE = 40.0
# Degrees of longitude and latitude in a metre, at LATITUDE.
LON_PER_M = 1 / (METRES_PER_DEGREE * math.cos(math.radians(LATITUDE)))
LAT_PER_M = 1 / METRES_PER_DEGREE


def ground_box(x0: float, y0: float, x1: float, y1: float):
    """A box given in metres east and north of (-75, LATITUDE)."""
    return box(
        -75 + x0 * LON_PER_M,
        LATITUDE + y0 * LAT_PER_M,
        -75 + x1 * LON_PER_M,
        LATITUDE + y1 * LAT_PER_M,
    )


def test_two_gcps_make_a_similarity_on_the_ground():
    # 1 px = 1 m, north up: (100, 0) is 100 m east of (0, 0).
    item = {
        "body": {
            "features": [
                {
                    "properties": {"resourceCoords": [0, 0]},
                    "geometry": {"coordinates": [-75, LATITUDE]},
                },
                {
                    "properties": {"resourceCoords": [100, 0]},
                    "geometry": {"coordinates": [-75 + 100 * LON_PER_M, LATITUDE]},
                },
            ]
        }
    }
    to_geo = page_transform(item)
    assert to_geo is not None
    lon, lat = to_geo(0, 100)
    # 100 px down the page is 100 m south -- not 100/cos(lat) m, as a
    # similarity fitted in raw degrees would put it.
    assert math.isclose(lon, -75, abs_tol=1e-9)
    assert math.isclose((LATITUDE - lat) / LAT_PER_M, 100, rel_tol=1e-6)


def test_plausible_pages_drop_oversized_and_isolated_sheets():
    sheet = 300  # a typical sheet, ~420 m corner to corner
    town = [(ground_box(i * sheet, 0, (i + 1) * sheet, sheet), 420.0) for i in range(4)]
    stray = (ground_box(20_000, 0, 20_300, 300), 420.0)
    huge = (ground_box(0, 0, 300, 300), 17_300.0)
    kept = plausible_pages([*town, stray, huge])
    assert kept == [polygon for polygon, _ in town]


def test_volume_footprint_closes_seams_and_fills_holes():
    # Eight sheets around a missing ninth, with 10 m gaps between them.
    pages = [
        ground_box(x * 310, y * 310, x * 310 + 300, y * 310 + 300)
        for x in range(3)
        for y in range(3)
        if (x, y) != (1, 1)
    ]
    footprint = volume_footprint(pages)
    assert isinstance(footprint, MultiPolygon)
    assert len(footprint.geoms) == 1
    assert len(footprint.geoms[0].interiors) == 0


def test_newest_coverage_hides_what_later_volumes_cover():
    whole = MultiPolygon([ground_box(0, 0, 2000, 1000)])
    west = MultiPolygon([ground_box(0, 0, 1000, 1000)])
    display = newest_coverage({"old": (1890, whole), "new": (1950, west)})
    assert display["new"] is not None
    assert display["new"].equals(west)
    shown = display["old"]
    assert shown is not None
    # What is left of the old atlas is its eastern half, less the margin.
    west_edge = (shown.bounds[0] - -75) / LON_PER_M
    assert 1000 < west_edge < 1100
    covered = newest_coverage({"old": (1890, west), "new": (1950, whole)})
    assert covered["old"] is None
