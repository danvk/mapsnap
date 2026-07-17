import numpy as np
import torch

from mapsnap.road_model import (
    UNet,
    invert_affine,
    normalize_patch,
    page_world_affine,
    rasterize_road_mask,
    road_mask,
    skeleton_junctions,
)

# A page whose pixels map to a 0.01 x 0.01 degree quad at the equator, axis-aligned
# with latitude decreasing down the image (TL, TR, BR, BL).
GEOREF = {
    "corners": [[0.0, 0.01], [0.01, 0.01], [0.01, 0.0], [0.0, 0.0]],
    "width": 500,
    "height": 500,
}


def test_page_world_affine_roundtrip():
    affine = page_world_affine(GEOREF)
    inverse = invert_affine(affine)
    px = np.array([123.0, 456.0])
    world = px @ affine[:, :2].T + affine[:, 2]
    back = world @ inverse[:, :2].T + inverse[:, 2]
    assert np.allclose(back, px, atol=1e-9)


def test_rasterize_road_mask_draws_centerline():
    # A horizontal street across the middle of the page (lat 0.005 = pixel row 250).
    features = [
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [[0.0, 0.005], [0.01, 0.005]],
            }
        }
    ]
    mask = rasterize_road_mask(GEOREF, features, width_m=25.0)
    assert mask.shape == (500, 500)
    assert mask[250, 250] == 255  # on the centerline
    assert mask[100, 250] == 0  # far from any street
    # Width ~25 m at ~2.2 m/px is ~11 px: row 250 +- 20 should bracket the stroke.
    column = mask[:, 250]
    assert column[230:271].sum() > 0
    assert column[:200].sum() == 0


def test_rasterize_skips_far_away_features():
    features = [
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [[5.0, 5.0], [5.01, 5.0]],  # nowhere near the page
            }
        }
    ]
    mask = rasterize_road_mask(GEOREF, features)
    assert mask.sum() == 0


def test_unet_output_shape():
    model = UNet(base=8)
    x = torch.zeros(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_normalize_patch_range():
    gray = np.array([[0, 255], [128, 210]], np.uint8)
    normalized = normalize_patch(gray)
    assert normalized.min() >= -1.0 and normalized.max() <= 1.0
    assert normalized.dtype == np.float32


def test_road_mask_keeps_large_drops_small():
    # A big high-probability square (area 2500) plus a 3x3 speck (area 9).
    probabilities = np.zeros((100, 100), np.float32)
    probabilities[10:60, 10:60] = 0.9  # large blob, above threshold
    probabilities[80:83, 80:83] = 0.9  # tiny blob
    mask = road_mask(probabilities, threshold=0.5, min_area=500)
    assert mask.dtype == np.uint8
    assert mask[30, 30] == 255  # inside the large blob, kept
    assert mask[81, 81] == 0  # tiny speck dropped
    assert mask[0, 0] == 0  # background


def test_skeleton_junctions_finds_a_cross():
    # A 1-px plus sign: one intersection at its center.
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[10, :] = True  # horizontal arm
    skeleton[:, 10] = True  # vertical arm
    junctions = skeleton_junctions(skeleton)
    assert len(junctions) == 1
    assert np.allclose(junctions[0], [10, 10], atol=1.5)  # (x, y)


def test_skeleton_junctions_none_on_a_straight_line():
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[10, :] = True  # a single street, no branch points
    assert len(skeleton_junctions(skeleton)) == 0
