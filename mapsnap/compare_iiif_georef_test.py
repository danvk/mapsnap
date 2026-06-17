"""Tests for split association in mapsnap.compare_iiif_georef."""

import json

from shapely.geometry import Polygon as ShapelyPolygon

from mapsnap.compare_iiif_georef import (
    annotation_split_index,
    load_split_polygons,
    match_split_pairs,
    polygon_iou,
)


def _square(x: float, y: float, side: float) -> ShapelyPolygon:
    return ShapelyPolygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


def test_polygon_iou_identical_and_disjoint():
    a = _square(0, 0, 10)
    assert polygon_iou(a, a) == 1.0
    assert polygon_iou(a, _square(20, 20, 10)) == 0.0


def test_polygon_iou_partial_overlap():
    # Two 10×10 squares overlapping in a 5×5 corner: inter=25, union=175.
    assert polygon_iou(_square(0, 0, 10), _square(5, 5, 10)) == 25 / 175


def test_annotation_split_index_parses_id():
    assert annotation_split_index({"id": "http://x/p4__2/georef"}) == 2
    assert annotation_split_index({"id": "http://x/p4/georef"}) is None
    assert annotation_split_index({}) is None


def _write_panels(path, width, height, rings):
    path.write_text(
        json.dumps(
            {"image": "p.jpg", "width": width, "height": height, "panels": rings}
        )
    )


def test_load_split_polygons_scales_generated_panels(tmp_path):
    # A 50×100 ring in a 100×200 (25%) frame scales ×4 to an 800×1600 canvas.
    panels = tmp_path / "p4.panels.json"
    _write_panels(panels, 100, 200, [[[0, 0], [50, 0], [50, 100], [0, 100]]])

    truth = load_split_polygons(panels)  # no scaling
    assert truth[1].bounds == (0.0, 0.0, 50.0, 100.0)

    gen = load_split_polygons(panels, source_dims=(800.0, 1600.0))  # ×8, ×8
    assert gen[1].bounds == (0.0, 0.0, 400.0, 800.0)


def test_match_split_pairs_associates_by_polygon_overlap_ignoring_numbering():
    # Truth split 1 is the left half, split 2 the right half (canvas coords).
    truth_polygons = {1: _square(0, 0, 100), 2: _square(100, 0, 100)}
    truth_items = [{"label": "P [1]"}, {"label": "P [2]"}]
    # Our numbering is reversed: gen __1 sits on the right, __2 on the left.
    gen_items = [
        {"id": "http://x/p__1/georef"},
        {"id": "http://x/p__2/georef"},
    ]
    gen_polygons = {1: _square(100, 0, 100), 2: _square(0, 0, 100)}

    pairs, unmatched = match_split_pairs(
        truth_items, gen_items, truth_polygons, gen_polygons
    )

    assert unmatched == []
    paired = {t["label"]: g["id"] for t, g in pairs}
    assert paired == {"P [1]": "http://x/p__2/georef", "P [2]": "http://x/p__1/georef"}


def test_match_split_pairs_unmatched_when_no_gen_polygon():
    truth_polygons = {1: _square(0, 0, 100)}
    truth_items = [{"label": "P [1]"}]
    pairs, unmatched = match_split_pairs(truth_items, [], truth_polygons, {})
    assert pairs == []
    assert unmatched == truth_items
