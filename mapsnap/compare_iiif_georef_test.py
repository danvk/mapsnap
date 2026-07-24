"""Tests for split association in mapsnap.compare_iiif_georef."""

import json

import pytest

from shapely.geometry import Polygon as ShapelyPolygon

from mapsnap.compare_iiif_georef import (
    annotation_split_index,
    annotations_by_source,
    load_split_polygons,
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


def _split_volume(tmp_path, gen_items, gen_panels=None):
    """Write a truth file with two split panels at DIFFERENT transforms.

    Truth p7 [1] covers the left half of a 400x200 canvas; p7 [2] the right
    half, georeferenced ~0.01 deg (~1km) away — an inset composite. Returns
    (truth_path, gen_path).
    """

    def features(origin_lon):
        return [
            {
                "properties": {"resourceCoords": [x, y]},
                "geometry": {"coordinates": [origin_lon + x * 1e-6, 40.0 + y * 1e-6]},
            }
            for x, y in [(0, 0), (150, 0), (0, 150)]
        ]

    def item(label, origin_lon, item_id="https://x/p7/georef"):
        return {
            "id": item_id,
            "label": label,
            "target": {
                "source": {
                    "id": "https://loc.gov/x/g123-0007/info.json",
                    "width": 400,
                    "height": 200,
                },
            },
            "body": {
                "transformation": {"type": "polynomial", "options": {"order": 1}},
                "features": features(origin_lon),
            },
        }

    truth = {"items": [item("x p7 [1]", -74.0), item("x p7 [2]", -73.99)]}
    (tmp_path / "oim").mkdir(exist_ok=True)
    _write_panels(
        tmp_path / "oim" / "p7.panels.json",
        400,
        200,
        [
            [[0, 0], [200, 0], [200, 200], [0, 200]],
            [[200, 0], [400, 0], [400, 200], [200, 200]],
        ],
    )
    gen = {"items": [item(*args) for args in gen_items]}
    if gen_panels is not None:
        _write_panels(tmp_path / "p7.panels.json", 400, 200, gen_panels)
    truth_path = tmp_path / "main.iiif.json"
    gen_path = tmp_path / "gen.iiif.json"
    truth_path.write_text(json.dumps(truth))
    gen_path.write_text(json.dumps(gen))
    return truth_path, gen_path


def test_split_regions_whole_page_on_one_inset_is_a_disaster(tmp_path):
    # Scenario A: our splitter failed and the whole page georeferenced onto
    # truth panel 1's location. Panel 1's region grades well — but panel 2's
    # imagery is DISPLAYED ~1km from where it belongs, and grading its region
    # under the same transform must expose that as a huge error rather than
    # scoring identically to an honest abstention.
    from mapsnap.compare_iiif_georef import compare_pages

    truth_path, gen_path = _split_volume(tmp_path, [("x p7", -74.0)])
    rows, missing = compare_pages(truth_path, gen_path)
    assert missing == []
    by_key = {r["page_key"]: r for r in rows}
    assert by_key["p7__1"]["rmse_ft"] < 25.0
    assert by_key["p7__2"]["rmse_ft"] > 1000.0


def test_split_regions_honest_abstention_is_merely_missing(tmp_path):
    # Scenario B: we split correctly, placed panel 1, declined panel 2.
    from mapsnap.compare_iiif_georef import compare_pages

    truth_path, gen_path = _split_volume(
        tmp_path,
        [("x p7 [1]", -74.0, "https://x/p7__1/georef")],
        gen_panels=[[[0, 0], [200, 0], [200, 200], [0, 200]]],
    )
    rows, missing = compare_pages(truth_path, gen_path)
    assert [r["page_key"] for r in rows] == ["p7__1"]
    assert rows[0]["rmse_ft"] < 25.0
    assert len(missing) == 1


def test_split_regions_correct_whole_page_credits_every_panel(tmp_path):
    # A contiguous split placed correctly as a whole page: BOTH truth panels
    # grade well (the old largest-split stand-in rule credited only one).
    from mapsnap.compare_iiif_georef import compare_pages

    def features(origin_lon):
        return [
            {
                "properties": {"resourceCoords": [x, y]},
                "geometry": {"coordinates": [origin_lon + x * 1e-6, 40.0 + y * 1e-6]},
            }
            for x, y in [(0, 0), (150, 0), (0, 150)]
        ]

    def item(label):
        return {
            "id": "https://x/p7/georef",
            "label": label,
            "target": {
                "source": {
                    "id": "https://loc.gov/x/g123-0007/info.json",
                    "width": 400,
                    "height": 200,
                },
            },
            "body": {
                "transformation": {"type": "polynomial", "options": {"order": 1}},
                "features": features(-74.0),
            },
        }


    truth = {"items": [item("x p7 [1]"), item("x p7 [2]")]}
    gen = {"items": [item("x p7")]}
    (tmp_path / "oim").mkdir(exist_ok=True)
    _write_panels(
        tmp_path / "oim" / "p7.panels.json",
        400,
        200,
        [
            [[0, 0], [200, 0], [200, 200], [0, 200]],
            [[200, 0], [400, 0], [400, 200], [200, 200]],
        ],
    )
    truth_path = tmp_path / "main.iiif.json"
    gen_path = tmp_path / "gen.iiif.json"
    truth_path.write_text(json.dumps(truth))
    gen_path.write_text(json.dumps(gen))
    rows, missing = compare_pages(truth_path, gen_path)
    assert missing == []
    assert sorted(r["page_key"] for r in rows) == ["p7__1", "p7__2"]
    assert all(r["rmse_ft"] < 25.0 for r in rows)


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


def test_compare_pages_warns_on_truth_splits_without_oim_panels(tmp_path, capsys):
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

    # Truth split items but no oim/p52.panels.json: the generated placement can
    # never be matched — that silent gap once understated KC by ~4 net points,
    # so compare (and through it `mapsnap score`) must say so out loud.
    truth = {"items": [item("0052", "x p52 [1]"), item("0052", "x p52 [2]")]}
    generated = {"items": [item("0052", "x p52")]}
    truth_path = tmp_path / "main.iiif.json"
    gen_path = tmp_path / "gen.iiif.json"
    truth_path.write_text(json.dumps(truth))
    gen_path.write_text(json.dumps(generated))
    rows, missing = compare_pages(truth_path, gen_path)
    assert rows == []
    assert len(missing) == 2
    err = capsys.readouterr().err
    assert "p52" in err and "oim-split-truth" in err
