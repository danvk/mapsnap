"""Tests for build_footprints.py."""

import math

from build_footprints import (
    METRES_PER_DEGREE,
    Edition,
    Page,
    Series,
    group_series,
    judge_pages,
    number_key,
    page_bar,
    page_key,
    page_transform,
    plausible_pages,
    town_record,
    visible_parts,
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


def sheet_page(key: str, x: float, y: float, confidence: float | None = 0.95) -> Page:
    """A 300 m sheet with its southwest corner x, y meters from the origin."""
    return Page(key, ground_box(x, y, x + 300, y + 300), 420.0, confidence)


def test_plausible_pages_drop_oversized_and_isolated_sheets():
    town = [sheet_page(f"p{i}", i * 300, 0) for i in range(4)]
    stray = sheet_page("p8", 20_000, 0)
    huge = Page("p9", ground_box(0, 0, 300, 300), 17_300.0)
    assert plausible_pages([*town, stray, huge]) == town


def test_page_key_reads_the_label():
    assert page_key("Abbeville, Alabama | 1907 | sanborn00001_001 p1") == "p1"
    assert page_key("Abbeville, Alabama | 1907 | sanborn00001_001 p2 [5]") == "p2__5"


def test_page_bar_rises_for_each_doubt():
    # The main body of an edition, on ground no other volume claims, is free.
    assert page_bar(disconnected=False, claims=0, uncorroborated=True) == 0.0
    assert math.isclose(
        page_bar(disconnected=False, claims=1, uncorroborated=False), 0.8
    )
    # Hyde Park: apart from town, and in no other edition.
    assert math.isclose(
        page_bar(disconnected=True, claims=0, uncorroborated=True), 0.85
    )
    # Claims count up to two.
    assert math.isclose(
        page_bar(disconnected=False, claims=5, uncorroborated=False), 0.9
    )


def test_judge_pages_drops_a_doubtful_island_and_keeps_a_sure_one():
    # Two editions of one town, the newer with a Hyde Park-like annex 3 km off
    # (sure pages) and a stray on another volume's ground (unsure).
    town_1900 = [sheet_page(f"p{i}", i * 300, 0, 0.3) for i in range(4)]
    town_1950 = [sheet_page(f"p{i}", i * 300, 0) for i in range(4)]
    annex = [sheet_page("p86", 4000, 0, 0.85), sheet_page("p87", 4300, 0, 0.92)]
    stray = sheet_page("p27", 0, 3000, 0.73)
    other = [sheet_page(f"q{i}", i * 300, 3000) for i in range(2)]
    editions = [
        Edition("t1900", 1900, "", pages=town_1900),
        Edition("t1950", 1950, "", pages=[*town_1950, *annex, stray]),
        Edition("north", 1950, "2", pages=other),
    ]
    series = [
        Series("", ["t1900", "t1950"]),
        Series("2", ["north"]),
    ]
    kept = {(v.item, v.page.key): v.kept for v in judge_pages(editions, series)}
    assert kept[("t1950", "p86")] and kept[("t1950", "p87")]
    assert not kept[("t1950", "p27")]
    assert all(kept[("t1900", f"p{i}")] for i in range(4))


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


def ground_shape(x0: float, y0: float, x1: float, y1: float) -> MultiPolygon:
    """ground_box as a MultiPolygon, as footprints are."""
    return MultiPolygon([ground_box(x0, y0, x1, y1)])


def test_visible_parts_hides_what_earlier_shapes_cover():
    whole = ground_shape(0, 0, 2000, 1000)
    west = ground_shape(0, 0, 1000, 1000)
    display = visible_parts({"whole": whole, "west": west}, ["west", "whole"])
    assert display["west"] is west
    shown = display["whole"]
    assert shown is not None
    # What is left of the whole is its eastern half, less the margin.
    west_edge = (shown.bounds[0] - -75) / LON_PER_M
    assert 1000 < west_edge < 1100
    covered = visible_parts({"whole": whole, "west": west}, ["whole", "west"])
    assert covered["west"] is None


def test_visible_parts_keeps_a_one_sheet_town_whole():
    # 100 m across: opening by the 60 m margin would erase it.
    sheet = ground_shape(0, 0, 100, 100)
    assert visible_parts({"town": sheet}, ["town"])["town"] is sheet


def test_visible_parts_leaves_out_shapes_not_in_the_order():
    shapes = {"a": ground_shape(0, 0, 500, 500), "b": ground_shape(0, 0, 900, 900)}
    assert visible_parts(shapes, ["a"])["b"] is None


def test_number_key_sorts_numbers_then_letters():
    numbers = ["B", "10", "2", "10a", "A"]
    assert sorted(numbers, key=number_key) == ["2", "10", "10a", "A", "B"]


def test_group_series_groups_numbered_volumes_by_number():
    editions = [
        Edition("vol1-1906", 1906, "1", ground_shape(0, 0, 1000, 1000)),
        Edition("vol2-1910", 1910, "2", ground_shape(1000, 0, 2000, 1000)),
        Edition("vol1-1950", 1950, "1", ground_shape(0, 0, 1100, 1000)),
        Edition("vol2-1935", 1935, "2"),  # digitized, nothing placed
    ]
    series = group_series(editions)
    assert [(s.number, s.items) for s in series] == [
        ("1", ["vol1-1906", "vol1-1950"]),
        ("2", ["vol2-1910", "vol2-1935"]),
    ]


def test_group_series_clusters_unnumbered_editions_by_ground():
    editions = [
        Edition("town-1890", 1890, "", ground_shape(0, 0, 1000, 1000)),
        Edition("town-1900", 1900, "", ground_shape(0, 0, 1500, 1000)),
        # A mill district well away from town.
        Edition("mill-1901", 1901, "", ground_shape(5000, 0, 5500, 500)),
        Edition("town-1920", 1920),  # unplaced: joins the largest cluster
    ]
    series = group_series(editions)
    assert [s.items for s in series] == [
        ["town-1890", "town-1900", "town-1920"],
        ["mill-1901"],
    ]


def test_group_series_flags_an_overview_atlas():
    numbered = [
        Edition(f"vol{n}", 1905, str(n), ground_shape(n * 1000, 0, n * 1000 + 900, 900))
        for n in range(1, 5)
    ]
    overview = Edition("borough", 1905, "", ground_shape(0, 0, 6000, 1000))
    series = group_series([*numbered, overview])
    assert [(s.items, s.overview) for s in series][-1] == (["borough"], True)
    assert not any(s.overview for s in series[:-1])


def test_group_series_leaves_unnumbered_unplaced_editions_of_a_numbered_town_out():
    editions = [
        Edition("vol1", 1950, "1", ground_shape(0, 0, 1000, 1000)),
        Edition("additions", 1920),
    ]
    assert [s.items for s in group_series(editions)] == [["vol1"]]


def test_town_record_draws_the_smaller_series_whole():
    def edition(item: str, year: int, number: str, east_m: float) -> Edition:
        page = Page("p1", ground_box(0, 0, east_m, 1000), 1400.0, 0.95)
        return Edition(item, year, number, MultiPolygon([page.outline]), [page])

    editions = [edition("old-town", 1890, "", 2000), edition("vol1", 1957, "1", 1000)]
    record = town_record(editions)
    assert set(record["volumes"]) == {"old-town", "vol1"}
    vol1, old = record["series"]
    assert vol1["volume"] == "1"
    assert vol1["display"] is True
    assert old["items"] == ["old-town"]
    assert isinstance(old["display"], list)  # its eastern half
    assert "volume" not in old
