import json

import numpy as np

from mapsnap.fix_truth_splits import (
    fix_annotation_page,
    gcp_containment,
    ring_offset,
    shifted_selector,
)


def test_ring_offset_exact_vertexwise():
    crop_polygon = np.array(
        [[10.0, 20.0], [110.0, 20.0], [110.0, 220.0], [10.0, 220.0]]
    )
    ring = [[3010.1, 520.0], [3110.1, 520.0], [3110.1, 720.0], [3010.1, 720.0]]
    offset = ring_offset(ring, crop_polygon)
    assert offset is not None
    assert abs(offset[0] - 3000.1) < 0.2 and abs(offset[1] - 500.0) < 0.2


def test_ring_offset_handles_closed_ring_and_vertex_mismatch():
    crop_polygon = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]])
    closed = [
        [200.0, 300.0],
        [300.0, 300.0],
        [300.0, 350.0],
        [200.0, 350.0],
        [200.0, 300.0],
    ]
    assert ring_offset(closed, crop_polygon) == (200.0, 300.0)
    # Different vertex count -> bbox-corner fallback.
    pentagon = [
        [200.0, 300.0],
        [300.0, 300.0],
        [310.0, 320.0],
        [300.0, 350.0],
        [200.0, 350.0],
    ]
    assert ring_offset(pentagon, crop_polygon) == (200.0, 300.0)
    assert ring_offset([[0, 0]], np.empty((0, 2))) is None


def _item(gcps, selector_points, label="X p1 [2]"):
    points = " ".join(f"{x},{y}" for x, y in selector_points)
    return {
        "label": label,
        "target": {
            "source": {"id": None, "width": 6000, "height": 8000},
            "selector": {
                "type": "SvgSelector",
                "value": f'<svg><polygon points="{points}" /></svg>',
            },
        },
        "body": {
            "features": [
                {
                    "properties": {"resourceCoords": [px, py]},
                    "geometry": {"coordinates": [0.0, 0.0]},
                }
                for px, py in gcps
            ]
        },
    }


def test_gcp_containment():
    square = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)]
    inside = _item([(500, 500), (10, 10)], square)
    outside = _item([(5000, 5000), (4000, 4000)], square)
    assert gcp_containment(inside, square) == 1.0
    assert gcp_containment(outside, square) == 0.0


def test_shifted_selector_round_trips():
    value = '<svg><polygon points="1,2 3,4 5,6" /></svg>'
    new_value, points = shifted_selector(value, (10.0, 20.0))
    assert points == [(11.0, 22.0), (13.0, 24.0), (15.0, 26.0)]
    assert "11.0,22.0" in new_value


def test_fix_applies_only_when_containment_improves(tmp_path):
    # A synthetic 200x100 crop: white margins, one grey panel at (20,10)-(180,90).
    crop = np.full((100, 200), 255, dtype=np.uint8)
    crop[10:90, 20:180] = 128
    from PIL import Image

    oim = tmp_path
    Image.fromarray(crop).save(oim / "p1__1.jpg")
    Image.fromarray(crop).save(oim / "p1__2.jpg")
    from mapsnap.oim_truth import panel_polygon

    polygon = panel_polygon(np.asarray(Image.open(oim / "p1__1.jpg").convert("L")))
    assert polygon is not None
    # Split 1 sits at the canvas origin, split 2 at x=3000: rings = polygon + offset.
    rings = [polygon.tolist(), (polygon + [3000.0, 0.0]).tolist()]
    (oim / "p1.panels.json").write_text(json.dumps({"panels": rings}))

    panel = [(20, 10), (180, 10), (180, 90), (20, 90)]  # crop-frame selector
    # [1]: selector already in canvas frame, GCPs inside -> untouched.
    # [2]: GCPs live at x~3000 (full frame) but the selector is crop-frame -> fixed.
    doc = {
        "items": [
            _item([(100, 50)], panel, label="X p1 [1]"),
            _item([(3100, 50), (3150, 80)], panel, label="X p1 [2]"),
        ]
    }
    log = fix_annotation_page(doc, oim)
    assert len(log) == 1 and "p1 [2]" in log[0] and "shifted by (3000, 0)" in log[0]
    fixed = doc["items"][1]["target"]["selector"]["value"]
    assert "3020.0,10.0" in fixed
    unfixed = doc["items"][0]["target"]["selector"]["value"]
    assert "20,10" in unfixed
    # Idempotent: a second pass changes nothing.
    assert fix_annotation_page(doc, oim) == []
