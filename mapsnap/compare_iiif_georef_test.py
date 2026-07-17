"""Tests for split association in mapsnap.compare_iiif_georef."""

import json

import pytest

from shapely.geometry import Polygon as ShapelyPolygon

from mapsnap.compare_iiif_georef import (
    annotation_split_index,
    annotations_by_source,
    load_split_polygons,
    match_split_pairs,
    page_label,
    parse_svg_polygon,
    polygon_iou,
    split_numbers_disagree,
    truth_page_number,
    truth_polygon_world,
    truth_polygons_by_page,
    truth_polygons_world,
)


def test_annotations_by_source_null_ids_key_by_label(tmp_path):
    # OIM volumes with null source.id must group by the label's page (not collapse to one
    # None key), with splits of a page sharing the parent entry.
    doc = {
        "items": [
            {
                "label": "Grand Rapids, Mich. | 1953 | Vol. 7 p714",
                "target": {"source": {"id": None}},
            },
            {
                "label": "Grand Rapids, Mich. | 1953 | Vol. 7 p721 [1]",
                "target": {"source": {"id": None}},
            },
            {
                "label": "Grand Rapids, Mich. | 1953 | Vol. 7 p721 [2]",
                "target": {"source": {"id": None}},
            },
        ]
    }
    path = tmp_path / "main.iiif.json"
    path.write_text(json.dumps(doc))
    groups = annotations_by_source(path)
    assert sorted(groups) == ["p714", "p721"]
    assert len(groups["p721"]) == 2  # both splits share the parent entry


def test_split_numbering_annotations():
    # Numbers agree → plain key, no disagreement.
    agree = {"page_key": "p13__1", "gen_page_key": "p13__1"}
    assert split_numbers_disagree(agree) is False
    assert page_label(agree) == "p13__1"

    # Numbers disagree → '(t)' marker on the truth key.
    disagree = {"page_key": "p13__1", "gen_page_key": "p13__2"}
    assert split_numbers_disagree(disagree) is True
    assert page_label(disagree) == "p13__1 (t)"

    # Full pages (no gen_page_key, e.g. missing rows) are never flagged.
    full = {"page_key": "p13"}
    assert split_numbers_disagree(full) is False
    assert page_label(full) == "p13"


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


def test_match_split_pairs_unsplit_page_matches_largest_truth_split():
    # OIM split the page into 3 panels of different sizes; we kept it whole, so the
    # generated annotation has no split index and stands in as the full canvas.
    truth_polygons = {
        1: _square(0, 0, 100),  # area 10_000
        2: _square(0, 0, 200),  # area 40_000 — largest
        3: _square(0, 0, 50),  # area 2_500 — below MIN_SPLIT_IOU vs the canvas
    }
    truth_items = [{"label": "P [3]"}, {"label": "P [2]"}, {"label": "P [1]"}]
    gen_items = [{"id": "http://x/p3/georef"}]  # no __N → unsplit page
    canvas = _square(0, 0, 300)  # area 90_000

    pairs, unmatched = match_split_pairs(
        truth_items, gen_items, truth_polygons, {}, canvas_polygon=canvas
    )

    assert len(pairs) == 1
    truth, gen = pairs[0]
    assert truth["label"] == "P [2]"  # the largest split
    assert gen["id"] == "http://x/p3/georef"
    assert {t["label"] for t in unmatched} == {"P [1]", "P [3]"}


def test_match_split_pairs_unmatched_when_no_gen_polygon():
    truth_polygons = {1: _square(0, 0, 100)}
    truth_items = [{"label": "P [1]"}]
    pairs, unmatched = match_split_pairs(truth_items, [], truth_polygons, {})
    assert pairs == []
    assert unmatched == truth_items


def test_match_split_pairs_picks_best_overlap_not_file_order():
    # The single generated panel overlaps truth [1] strongly; truth [3], listed first,
    # only grazes it. The grazing split must not claim the generated panel ahead of the
    # real match (regression for the champaign p4__3 mismatch).
    truth_polygons = {
        1: _square(0, 0, 100),
        2: _square(200, 0, 100),
        3: _square(99, 0, 2),  # tiny overlap with the generated panel's right edge
    }
    truth_items = [{"label": "P [3]"}, {"label": "P [2]"}, {"label": "P [1]"}]
    gen_items = [{"id": "http://x/p__1/georef"}]
    gen_polygons = {1: _square(0, 0, 100)}

    pairs, unmatched = match_split_pairs(
        truth_items, gen_items, truth_polygons, gen_polygons
    )

    assert len(pairs) == 1
    truth, gen = pairs[0]
    assert truth["label"] == "P [1]"
    assert gen["id"] == "http://x/p__1/georef"
    assert {t["label"] for t in unmatched} == {"P [2]", "P [3]"}


def test_parse_svg_polygon():
    value = '<svg><polygon points="10,20 30.5,40 50,60.25" /></svg>'
    assert parse_svg_polygon(value) == [(10.0, 20.0), (30.5, 40.0), (50.0, 60.25)]
    assert parse_svg_polygon("<svg></svg>") == []


def _identity_georef_item(polygon: list[tuple[float, float]]) -> dict:
    # An annotation whose GCPs make pixel==world (affine = identity), so the world ring
    # equals the input polygon; four GCPs so the polynomial fit is well-determined.
    points = " ".join(f"{x},{y}" for x, y in polygon)
    return {
        "target": {
            "selector": {
                "type": "SvgSelector",
                "value": f'<svg><polygon points="{points}" /></svg>',
            },
        },
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


def test_truth_polygon_world_applies_annotation_transform():
    poly = [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)]
    ring = truth_polygon_world(_identity_georef_item(poly))
    assert ring == [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]


def test_truth_polygon_world_none_without_selector_or_gcps():
    # No SvgSelector.
    assert truth_polygon_world({"target": {"selector": {"type": "Other"}}}) is None
    # Fewer than three GCPs.
    item = _identity_georef_item([(0, 0), (1, 0), (1, 1)])
    item["body"]["features"] = item["body"]["features"][:2]
    assert truth_polygon_world(item) is None


def test_truth_polygons_world_reads_all_items(tmp_path):
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    doc = {
        "items": [
            _identity_georef_item(poly),
            {"target": {"selector": {"type": "Other"}}},  # skipped: no polygon
            _identity_georef_item(poly),
        ]
    }
    path = tmp_path / "main.iiif.json"
    path.write_text(json.dumps(doc))
    rings = truth_polygons_world(path)
    assert len(rings) == 2  # two usable annotations, one skipped
    assert rings[0] == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]


def test_truth_page_number():
    assert truth_page_number({"label": "New Orleans | 1896 | Vol. 2 p156"}) == 156
    assert truth_page_number({"label": "New Orleans | 1896 | Vol. 2 p73 [1]"}) == 73
    assert truth_page_number({"label": "no page here"}) is None


def test_truth_page_number_uppercase_direction_suffix():
    # Chicago-style labels carry uppercase direction suffixes; the number must
    # still parse (a regression here silently emptied truth_polygons_by_page).
    assert truth_page_number({"label": "Chicago, Ill. | 1950 | Vol. 1 p103W"}) == 103
    assert truth_page_number({"label": "Chicago, Ill. | 1950 | Vol. 1 p22N"}) == 22
    assert truth_page_number({"label": "Hudson | 1950 p6n"}) == 6


def test_redundant_skeleton_keys_prefers_whichever_fit():
    from mapsnap.compare_iiif_georef import redundant_skeleton_keys

    truth = {"p153", "p153s", "p90"}
    # Full-color page fit: the skeleton is redundant.
    assert redundant_skeleton_keys(truth, {"p153"}) == {"p153s"}
    # Only the skeleton fit: keep it, drop the full-color truth row.
    assert redundant_skeleton_keys(truth, {"p153s"}) == {"p153"}
    # Neither fit: the miss counts once, against the full-color page.
    assert redundant_skeleton_keys(truth, set()) == {"p153s"}
    # Both fit: the full-color page wins.
    assert redundant_skeleton_keys(truth, {"p153", "p153s"}) == {"p153s"}
    # A lone skeleton (no full-color counterpart in truth) is never dropped.
    assert redundant_skeleton_keys({"p12s"}, set()) == set()


def test_redundant_skeleton_keys_rejects_compound_suffixes():
    from mapsnap.compare_iiif_georef import redundant_skeleton_keys

    with pytest.raises(AssertionError):
        redundant_skeleton_keys({"p6ns", "p6n"}, set())


def test_truth_polygons_by_page_groups_splits(tmp_path):
    def labeled(label: str, poly: list[tuple[float, float]]) -> dict:
        item = _identity_georef_item(poly)
        item["label"] = label
        return item

    a = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    b = [(20.0, 0.0), (30.0, 0.0), (30.0, 10.0)]
    c = [(0.0, 20.0), (10.0, 20.0), (10.0, 30.0)]
    doc = {
        "items": [
            labeled("Vol p73 [1]", a),
            labeled("Vol p73 [2]", b),  # same page 73, second split
            labeled("Vol p90", c),
            {"target": {"selector": {"type": "Other"}}, "label": "Vol p99"},  # skipped
        ]
    }
    path = tmp_path / "main.iiif.json"
    path.write_text(json.dumps(doc))
    by_page = truth_polygons_by_page(path)
    assert set(by_page) == {73, 90}  # p99 skipped (no polygon)
    assert len(by_page[73]) == 2  # both splits of page 73
    assert by_page[73][0] == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    assert len(by_page[90]) == 1


def test_compare_pages_ignores_skeletons_with_full_color_counterparts(tmp_path):
    from mapsnap.compare_iiif_georef import compare_pages

    def item(page_id: str, label: str) -> dict:
        return {
            "label": label,
            "target": {
                "source": {
                    "id": f"https://loc.gov/x/g123-{page_id}/info.json",
                    "width": 1000,
                    "height": 1000,
                },
            },
            "body": {
                "transformation": {"type": "polynomial", "options": {"order": 1}},
                "features": [
                    {
                        "properties": {"resourceCoords": [x, y]},
                        "geometry": {
                            "coordinates": [-74.0 + x * 1e-6, 40.0 + y * 1e-6]
                        },
                    }
                    for x, y in [(0, 0), (900, 0), (0, 900)]
                ],
            },
        }

    truth = {"items": [item("0153", "x p153"), item("0153s", "x p153s")]}
    generated = {"items": [item("0153", "x p153")]}
    truth_path = tmp_path / "main.iiif.json"
    gen_path = tmp_path / "gen.iiif.json"
    truth_path.write_text(json.dumps(truth))
    gen_path.write_text(json.dumps(generated))
    rows, missing = compare_pages(truth_path, gen_path)
    # p153s maps the same ground as p153 and `mapsnap iiif` never emits it;
    # it must not count as a missing page.
    assert [r["page_key"] for r in rows] == ["p153"]
    assert missing == []
