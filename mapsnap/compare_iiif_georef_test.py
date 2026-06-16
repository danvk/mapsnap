"""Tests for split association in mapsnap.compare_iiif_georef."""

from mapsnap.compare_iiif_georef import (
    gen_split_region,
    match_split_pairs,
    rect_iou,
)


def test_rect_iou_identical_and_disjoint():
    assert rect_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert rect_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_rect_iou_partial_overlap():
    # Two 10×10 squares overlapping in a 5×5 corner: inter=25, union=175.
    assert rect_iou((0, 0, 10, 10), (5, 5, 10, 10)) == 25 / 175


def _gen_item(label: str, region: tuple[float, float, float, float]) -> dict:
    x, y, w, h = region
    return {
        "label": label,
        "metadata": [
            {"label": "split_canvas_x", "value": str(x)},
            {"label": "split_canvas_y", "value": str(y)},
            {"label": "split_canvas_w", "value": str(w)},
            {"label": "split_canvas_h", "value": str(h)},
        ],
    }


def test_gen_split_region_reads_metadata():
    assert gen_split_region(_gen_item("p4 [1]", (1, 2, 3, 4))) == (1.0, 2.0, 3.0, 4.0)
    assert gen_split_region({"label": "p4", "metadata": []}) is None


def test_match_split_pairs_associates_by_overlap_ignoring_numbering():
    # OIM truth: split 1 is the left region, split 2 the right.
    truth_regions = {1: (0.0, 0.0, 100.0, 200.0), 2: (100.0, 0.0, 100.0, 200.0)}
    truth_items = [{"label": "P [1]"}, {"label": "P [2]"}]
    # Our numbering is reversed: gen "[1]" sits on the right, "[2]" on the left.
    gen_items = [
        _gen_item("P [1]", (100, 0, 100, 200)),
        _gen_item("P [2]", (0, 0, 100, 200)),
    ]

    pairs, unmatched = match_split_pairs(truth_items, gen_items, truth_regions)

    assert unmatched == []
    paired = {t["label"]: g["label"] for t, g in pairs}
    assert paired == {"P [1]": "P [2]", "P [2]": "P [1]"}


def test_match_split_pairs_unmatched_when_no_gen_region():
    truth_regions = {1: (0.0, 0.0, 100.0, 200.0)}
    truth_items = [{"label": "P [1]"}]
    pairs, unmatched = match_split_pairs(truth_items, [], truth_regions)
    assert pairs == []
    assert unmatched == truth_items
