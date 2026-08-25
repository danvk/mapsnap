"""Tests for the page content-region model's label construction (#226)."""

import numpy as np

from mapsnap.region_model import letterbox, truth_region_mask


def selector_item(points, label="x p7", source=(400, 200), gcps=None):
    item = {
        "label": label,
        "target": {
            "source": {
                "id": "https://x/p7/info.json",
                "width": source[0],
                "height": source[1],
            },
            "selector": {
                "type": "SvgSelector",
                "value": '<svg><polygon points="'
                + " ".join(f"{x},{y}" for x, y in points)
                + '" /></svg>',
            },
        },
        "body": {"features": gcps or []},
    }
    return item


def gcp(x, y):
    return {
        "properties": {"resourceCoords": [x, y]},
        "geometry": {"coordinates": [0.0, 0.0]},
    }


def test_truth_region_mask_scales_source_to_jpg():
    # Selector covers the left half of a 400x200 source; jpg is 100x50.
    item = selector_item([(0, 0), (200, 0), (200, 200), (0, 200)])
    mask = truth_region_mask([item], (100, 50))
    assert mask is not None
    assert mask[25, 10] == 255 and mask[25, 80] == 0


def test_truth_region_mask_gates_split_selectors_on_gcps():
    # A split item ('[1]') whose selector contains none of its own GCPs is the
    # OIM#402 crop-frame hazard and must not paint.
    bad = selector_item(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        label="x p7 [1]",
        gcps=[gcp(300, 150), gcp(350, 180)],
    )
    assert truth_region_mask([bad], (100, 50)) is None
    good = selector_item(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        label="x p7 [1]",
        gcps=[gcp(50, 50), gcp(20, 80)],
    )
    assert truth_region_mask([good], (100, 50)) is not None


def test_letterbox_preserves_aspect_on_white():
    image = np.zeros((100, 200, 3), np.uint8)
    boxed = letterbox(image, size=64)
    assert boxed.shape == (64, 64, 3)
    assert boxed[10, 10].tolist() == [0, 0, 0]  # content, top-left anchored
    assert boxed[50, 10].tolist() == [255, 255, 255]  # pad below a 2:1 image
