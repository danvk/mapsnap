"""Tests for the key-map snap target (#211): loading and rasterising."""

import json
from pathlib import Path

import cv2
import numpy as np

from mapsnap.edge_join import FrameSpec
from mapsnap.osm_snap import (
    KeymapTarget,
    frame_around,
    keymap_rasters,
    load_keymap_targets,
    nearest_keymap_target,
)


def affine_georef(width: int, height: int) -> dict:
    """A key-map georef whose corners are a plain north-up affine, no GCPs."""
    lon0, lat0, per_px = -90.1, 29.95, 1e-5
    corners = [
        [lon0, lat0],
        [lon0 + width * per_px, lat0],
        [lon0 + width * per_px, lat0 - height * per_px],
        [lon0, lat0 - height * per_px],
    ]
    return {"width": width, "height": height, "corners": corners, "intersections": []}


def make_target(tmp_path: Path, stem: str = "p0", with_roadprob: bool = True) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / f"{stem}.keymap.json").write_text("{}")
    (raw / f"{stem}.georef.json").write_text(json.dumps(affine_georef(400, 300)))
    if with_roadprob:
        prob = np.zeros((300, 400), np.uint8)
        cv2.line(prob, (0, 150), (399, 150), 255, 6)  # one east-west road
        cv2.line(prob, (200, 0), (200, 299), 255, 6)  # one north-south road
        cv2.imwrite(str(raw / f"{stem}.roadprob.png"), prob)
    return raw


def test_load_keymap_targets_needs_georef_and_roadprob(tmp_path):
    make_target(tmp_path, "p0")
    make_target(tmp_path, "pb", with_roadprob=False)
    targets = load_keymap_targets(tmp_path)
    assert [t.stem for t in targets] == ["p0"]
    assert targets[0].prob.shape == (300, 400)
    assert targets[0].prob.max() == 1.0


def test_keymap_rasters_put_the_roads_where_the_georef_says(tmp_path):
    make_target(tmp_path, "p0")
    target = load_keymap_targets(tmp_path)[0]
    georef = affine_georef(400, 300)
    # A frame around the key map's centre: the two roads cross there.
    centre = (georef["corners"][0][0] + 200e-5, georef["corners"][0][1] - 150e-5)
    frame = frame_around(centre, half_m=200.0, res_m=2.0)
    prob, valid, skeleton = keymap_rasters(frame, target)
    assert prob.shape == frame.shape and valid.shape == frame.shape
    assert valid.mean() > 0.5  # the frame lies mostly on the sheet
    # The east-west road runs through the frame centre row; sample it.
    rows, cols = frame.shape
    assert prob[rows // 2, cols // 4] > 0.5
    assert prob[rows // 2, 3 * cols // 4] > 0.5
    assert prob[rows // 4, cols // 4] < 0.1  # off-road stays dark
    assert skeleton.any()
    # Skeleton cells sit on inked cells.
    assert prob[skeleton].mean() > 0.5


def test_keymap_rasters_mark_cells_off_the_sheet_invalid(tmp_path):
    make_target(tmp_path, "p0")
    target = load_keymap_targets(tmp_path)[0]
    georef = affine_georef(400, 300)
    # A frame centred on the sheet's top-left corner: three quarters off-sheet.
    frame = frame_around(tuple(georef["corners"][0]), half_m=150.0, res_m=2.0)
    prob, valid, _skeleton = keymap_rasters(frame, target)
    assert 0.15 < valid.mean() < 0.4
    assert not prob[~valid].any()


def test_nearest_keymap_target_prefers_the_sheet_that_contains_the_point(tmp_path):
    make_target(tmp_path, "p0")
    far = tmp_path / "far"
    far.mkdir()
    make_target(far, "p0")
    near_target = load_keymap_targets(tmp_path)[0]
    far_target = load_keymap_targets(far)[0]
    far_target.georef["corners"] = [
        [c[0] + 1.0, c[1]] for c in far_target.georef["corners"]
    ]
    inside = (-90.1 + 100e-5, 29.95 - 100e-5)
    assert nearest_keymap_target([far_target, near_target], inside) is near_target
    assert nearest_keymap_target([], inside) is None


def test_keymap_target_builds_its_model_lazily_and_leaves_it_out_of_state(
    tmp_path,
):
    make_target(tmp_path, "p0")
    target = load_keymap_targets(tmp_path)[0]
    assert target.model_cache is None
    world = target.model(np.array([[0.0, 0.0]]))
    assert np.allclose(world[0], affine_georef(400, 300)["corners"][0])
    assert target.model_cache is not None
    # What a worker receives: the georef, never the unpicklable spline.
    state = target.__getstate__()
    assert state["model_cache"] is None and state["georef"] is target.georef
    clone = KeymapTarget(**{k: v for k, v in state.items()})
    assert np.allclose(clone.model(np.array([[0.0, 0.0]]))[0], world[0])
    assert isinstance(FrameSpec, type)
