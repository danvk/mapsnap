"""Tests for the region-match candidate generator (#226 phase 2)."""


import numpy as np
from shapely.geometry import Polygon

from mapsnap.region_match import match_page, outline_area_px, page_outline


def test_page_outline_takes_the_largest_blob():
    prob = np.zeros((100, 100), np.float32)
    prob[10:60, 10:70] = 0.9  # main content
    prob[80:85, 80:85] = 0.9  # speck: a neighbour-sheet sliver
    outline = page_outline(prob)
    assert outline is not None
    assert 2500 < outline_area_px(outline) < 3300  # the 50x60 blob, not the speck


def test_page_outline_none_when_nothing_predicted():
    assert page_outline(np.zeros((100, 100), np.float32)) is None


def test_match_page_recovers_a_known_pose():
    # A 400x200 page whose content is an off-centre rectangle, and a region
    # that is that rectangle on the ground: 100 m wide, north-up.
    prob = np.zeros((200, 400), np.float32)
    prob[40:160, 60:340] = 0.9
    outline = page_outline(prob)
    assert outline is not None
    # Region in the metre frame: 280x120 px at 0.5 m/px = 140x60 m.
    region = Polygon([(-70, -30), (70, -30), (70, 30), (-70, 30)])
    candidates = match_page(outline, region, (400, 200), (-90.0, 30.0))
    assert candidates, "a rectangle should match its own region"
    best = candidates[0]
    assert best.iou > 0.9
    assert 0.4 < best.scale_m_per_px < 0.6
    # Corners are lon/lat around the origin, and the page is right side up.
    assert all(
        abs(c[0] + 90.0) < 0.01 and abs(c[1] - 30.0) < 0.01 for c in best.corners
    )


def test_match_page_emits_a_ladder_without_upside_down_aliases():
    # A rectangle matches its region at several orientations, so the channel
    # emits a ladder rather than forcing one answer -- but the 180-degree
    # alias is corpus-impossible (#324) and must never be offered.
    prob = np.zeros((200, 400), np.float32)
    prob[40:160, 60:340] = 0.9
    outline = page_outline(prob)
    assert outline is not None
    region = Polygon([(-70, -30), (70, -30), (70, 30), (-70, 30)])
    candidates = match_page(outline, region, (400, 200), (-90.0, 30.0))
    assert len(candidates) >= 2, "the ladder is the point: emit, do not resolve"
    assert candidates == sorted(candidates, key=lambda c: -c.iou)
    assert all(abs(c.rotation_deg) < 90 for c in candidates), "no upside-down poses"
