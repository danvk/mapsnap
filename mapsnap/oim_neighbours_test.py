"""Tests for OIM-based neighbour prediction."""

import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from mapsnap.oim_neighbours import (
    OimPage,
    affine_3x3,
    load_pages,
    longest_line,
    predicted_neighbours,
    selector_polygon,
    sheet_number,
    transform,
)

# One canvas pixel = 1e-5 degrees; canvases are 1000 x 1000.
SCALE = 1e-5


def annotation(
    page: int, origin: tuple[float, float], selector: str | None = None
) -> dict:
    """A minimal OIM annotation whose canvas maps to lon/lat by translation + scale."""
    lon0, lat0 = origin
    corners = [(0, 0), (1000, 0), (0, 1000), (1000, 1000)]
    features = [
        {
            "properties": {"resourceCoords": [x, y]},
            "geometry": {"coordinates": [lon0 + x * SCALE, lat0 - y * SCALE]},
        }
        for x, y in corners
    ]
    target: dict = {
        "source": {
            "id": f"https://example.org/iiif/service:gmd:x:0001_1900-{page:04d}",
            "width": 1000,
            "height": 1000,
        }
    }
    if selector:
        target["selector"] = {"type": "SvgSelector", "value": selector}
    return {
        "label": f"Town | 1900 p{page}",
        "target": target,
        "body": {"features": features},
    }


def oim_page(page: int, origin: tuple[float, float]) -> OimPage:
    """An OimPage built the way load_pages builds one."""
    item = annotation(page, origin)
    gcps = [
        (
            (
                f["properties"]["resourceCoords"][0],
                f["properties"]["resourceCoords"][1],
            ),
            tuple(f["geometry"]["coordinates"]),
        )
        for f in item["body"]["features"]
    ]
    to_world = affine_3x3(gcps)
    selector = selector_polygon(item)
    return OimPage(
        f"p{page}",
        np.linalg.inv(to_world),
        selector,
        transform(selector, to_world),
        1000,
        1000,
    )


def test_affine_3x3_recovers_an_exact_mapping():
    gcps = [
        ((0.0, 0.0), (10.0, 20.0)),
        ((100.0, 0.0), (11.0, 20.0)),
        ((0.0, 100.0), (10.0, 19.0)),
    ]
    matrix = affine_3x3(gcps)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix @ [100.0, 100.0, 1.0], [11.0, 19.0, 1.0])


def test_selector_polygon_reads_svg_points_and_defaults_to_the_canvas():
    masked = annotation(
        1, (0, 0), '<svg><polygon points="100,100 900,100 900,900 100,900"/></svg>'
    )
    assert selector_polygon(masked).area == 800 * 800
    assert selector_polygon(annotation(1, (0, 0))).area == 1000 * 1000


def test_transform_maps_every_vertex():
    shift = np.array([[1.0, 0, 5], [0, 1.0, -5], [0, 0, 1]])
    moved = transform(Polygon([(0, 0), (10, 0), (10, 10)]), shift)
    assert moved.bounds == (5.0, -5.0, 15.0, 5.0)


def test_sheet_number_strips_prefix_and_panel_suffix():
    assert sheet_number("p12") == "12"
    assert sheet_number("p381a__2") == "381A"


def test_longest_line_picks_the_longest_piece():
    pieces = MultiLineString([[(0, 0), (1, 0)], [(0, 5), (10, 5)]])
    line = longest_line(pieces)
    assert line is not None and line.length == 10
    assert longest_line(LineString()) is None
    assert longest_line(Point(0, 0)) is None


def test_predicted_neighbours_finds_edge_sharers_but_not_corner_touchers():
    width = 1000 * SCALE
    centre = oim_page(5, (0.0, 0.0))
    east = oim_page(6, (width, 0.0))  # shares the right edge
    south = oim_page(9, (0.0, -width))  # shares the bottom edge
    diagonal = oim_page(7, (width, -width))  # touches one corner only
    neighbours = predicted_neighbours(centre, [centre, east, south, diagonal])
    assert set(neighbours) == {"6", "9"}
    # The east neighbour's stretch runs down the right edge of the page.
    right = neighbours["6"]
    # (It wraps BORDER_BUFFER x width round each corner.)
    assert right.bounds[0] >= 1000 - 31 and right.bounds[2] == 1000
    assert right.length > 900


def test_load_pages_skips_annotations_without_three_gcps(tmp_path: Path):
    good = annotation(3, (0.0, 0.0))
    sparse = annotation(4, (0.0, 0.0))
    sparse["body"]["features"] = sparse["body"]["features"][:2]
    export = tmp_path / "main.iiif.json"
    export.write_text(json.dumps({"items": [good, sparse]}))
    pages = load_pages(export)
    assert [p.key for p in pages] == ["p3"]
    assert pages[0].world.area > 0
