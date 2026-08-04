import json

import cv2
import numpy as np

from mapsnap.keymap.road_prob import (
    KeymapSheet,
    buffered_scores,
    coverage_mask,
    keymap_sheets,
    mapped_extent_mask,
    measure_stroke_px,
    within_px,
)
from mapsnap.road_model import UNet, load_model, rasterize_road_mask

UNIT_GEOREF = {
    # One degree per 1000 px in both axes, anchored at (0, 0): a georef whose
    # world-to-pixel transform is trivially checkable.
    "width": 1000,
    "height": 1000,
    "corners": [[0.0, 0.0], [1.0, 0.0], [1.0, -1.0], [0.0, -1.0]],
}


def line_feature(*points):
    return {
        "geometry": {"type": "LineString", "coordinates": [list(p) for p in points]}
    }


def test_rasterize_width_px_overrides_ground_width():
    features = [line_feature((0.1, -0.5), (0.9, -0.5))]
    mask = rasterize_road_mask(UNIT_GEOREF, features, width_px=21)
    column = mask[:, 500]
    assert column.max() == 255
    # Stroke height ~21 px regardless of the georef's ground scale.
    assert 18 <= int((column > 0).sum()) <= 24
    # And the metre path still works and differs.
    thin = rasterize_road_mask(UNIT_GEOREF, features)
    assert (thin[:, 500] > 0).sum() != (column > 0).sum()


def test_measure_stroke_px_reads_the_drawn_corridor_width():
    # A synthetic sheet: saturated "blocks" separated by 30px paper corridors.
    image = np.full((1200, 1200, 3), 250, np.uint8)
    for y0 in range(0, 1200, 180):
        for x0 in range(0, 1200, 180):
            cv2.rectangle(image, (x0, y0), (x0 + 150, y0 + 150), (180, 200, 255), -1)
    stroke = measure_stroke_px(image)
    assert 24 <= stroke <= 36


def test_within_px_matches_dilation_semantics():
    mask = np.zeros((100, 100), np.uint8)
    mask[50, 50] = 1
    near = within_px(mask, 10)
    assert near[50, 60] and near[58, 50]
    assert not near[50, 62] and not near[70, 70]


def test_buffered_scores_tolerate_label_misalignment():
    # Two parallel 20px ribbons offset by 30px: plain IoU is 0, but at a 40px
    # tolerance each is fully explained by the other.
    shape = (400, 400)
    label = np.zeros(shape, np.float32)
    label[100:120, :] = 1.0
    predicted = np.zeros(shape, np.float32)
    predicted[130:150, :] = 1.0
    valid = np.ones(shape, np.uint8)
    scores = buffered_scores(predicted, label, valid, tolerance_px=40)
    assert scores["iou"] == 0.0
    assert scores["completeness"] == 1.0 and scores["correctness"] == 1.0
    # Beyond the tolerance nothing matches.
    far = buffered_scores(predicted, label, valid, tolerance_px=5)
    assert far["completeness"] == 0.0 and far["correctness"] == 0.0


def test_buffered_scores_stratify_by_background():
    shape = (200, 400)
    label = np.zeros(shape, np.float32)
    label[40:60, :] = 1.0  # a road over paper
    label[140:160, :] = 1.0  # a road over a pastel fill
    image = np.full((*shape, 3), 250, np.uint8)
    image[100:, :] = (170, 200, 250)  # saturated lower half
    predicted = np.zeros(shape, np.float32)
    predicted[40:60, :] = 1.0  # only the paper road is found
    scores = buffered_scores(
        predicted, label, np.ones(shape, np.uint8), image, tolerance_px=10
    )
    assert scores["completeness_paper"] == 1.0
    assert scores["completeness_fill"] == 0.0


def test_colour_checkpoint_round_trip(tmp_path):
    import torch

    model = UNet(base=8, in_channels=3)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = load_model(path, "cpu")
    first = loaded.enc1.block[0]
    assert first.weight.shape[1] == 3 and first.weight.shape[0] == 8


def write_sheet(tmp_path, inliers=30, with_regions=True):
    raw = tmp_path / "vol" / "raw"
    raw.mkdir(parents=True)
    georef = dict(
        UNIT_GEOREF,
        intersections=[{"x": 0, "y": 0, "inlier": True}] * inliers,
    )
    (raw / "p0.georef.json").write_text(json.dumps(georef))
    (raw / "p0.keymap.json").write_text(json.dumps({"streets": []}))
    cv2.imwrite(str(raw / "p0.jpg"), np.full((1000, 1000, 3), 250, np.uint8))
    (tmp_path / "vol" / "centerlines.geojson").write_text(
        json.dumps({"features": [line_feature((0.2, -0.2), (0.8, -0.8))]})
    )
    if with_regions:
        (raw / "p0.regions.panels.json").write_text(
            json.dumps(
                {
                    "labels": ["1", "2"],
                    "panels": [
                        [[300, 300], [500, 300], [500, 500], [300, 500]],
                        [[520, 300], [700, 300], [700, 500], [520, 500]],
                    ],
                }
            )
        )
    return KeymapSheet(tmp_path / "vol", "p0")


def test_keymap_sheets_gates_on_inliers(tmp_path):
    write_sheet(tmp_path, inliers=30)
    assert [s.stem for s in keymap_sheets(tmp_path)] == ["p0"]
    (tmp_path / "vol" / "raw" / "p0.georef.json").write_text(
        json.dumps(dict(UNIT_GEOREF, intersections=[]))
    )
    assert keymap_sheets(tmp_path) == []


def test_mapped_extent_hugs_the_regions_and_drops_stray_clusters(tmp_path):
    sheet = write_sheet(tmp_path)
    mask = mapped_extent_mask(sheet, margin_px=60)
    assert mask[400, 500]  # inside a region
    assert mask[400, 510]  # in the corridor between the two regions
    assert not mask[50, 50]  # the far corner is furniture
    # A small far-away cluster (an inset the segmenter mislabelled) is dropped:
    # only the largest connected component survives.
    regions = json.loads(
        (tmp_path / "vol" / "raw" / "p0.regions.panels.json").read_text()
    )
    regions["panels"].append([[30, 30], [70, 30], [70, 70], [30, 70]])
    regions["labels"].append("9")
    (tmp_path / "vol" / "raw" / "p0.regions.panels.json").write_text(
        json.dumps(regions)
    )
    again = mapped_extent_mask(sheet, margin_px=60)
    assert not again[50, 50] and again[400, 400]


def test_coverage_mask_excludes_ground_past_the_osm_bbox(tmp_path):
    sheet = write_sheet(tmp_path)
    features = json.loads(sheet.centerlines_path.read_text())["features"]
    mask = coverage_mask(sheet.georef(), features)
    assert mask[500, 500]  # inside the centerlines bbox
    assert not mask[950, 950]  # the sheet extends past the download
