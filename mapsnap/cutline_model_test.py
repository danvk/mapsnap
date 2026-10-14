"""Tests for mapsnap.cutline_model's mask-to-panels helpers."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, box

from mapsnap.cutline_model import (
    MODEL_PATH,
    boundary_support,
    cutline_model_path,
    cutline_probability,
    letterbox,
    merge_unsupported,
)


def test_cutline_model_path_defaults_to_the_model(monkeypatch):
    monkeypatch.delenv("MAPSNAP_SPLITTER", raising=False)
    monkeypatch.delenv("MAPSNAP_CUTLINE_WEIGHTS", raising=False)
    assert cutline_model_path() == MODEL_PATH
    monkeypatch.setenv("MAPSNAP_CUTLINE_WEIGHTS", "/tmp/other.pt")
    assert cutline_model_path() == Path("/tmp/other.pt")
    monkeypatch.setenv("MAPSNAP_SPLITTER", "classical")
    assert cutline_model_path() is None
    monkeypatch.setenv("MAPSNAP_SPLITTER", "nonsense")
    with pytest.raises(ValueError):
        cutline_model_path()


def test_letterbox_keeps_aspect_on_a_white_square():
    boxed = letterbox(np.zeros((200, 100, 3), dtype=np.uint8), size=64)
    assert boxed.shape == (64, 64, 3)
    assert boxed[:, :32].max() == 0 and boxed[:, 32:].min() == 255


def test_cutline_probability_is_a_page_sized_map():
    page = np.full((300, 240, 3), 255, dtype=np.uint8)
    page[:, 118:124] = 0  # a heavy vertical rule
    prob = cutline_probability(page, MODEL_PATH)
    assert prob.shape == (300, 240) and prob.dtype == np.uint8


def test_boundary_support_is_the_covered_share_of_a_line():
    near_line = np.zeros((100, 100), dtype=bool)
    near_line[:, 50] = True
    assert boundary_support(LineString([(50, 0), (50, 99)]), near_line) == 1.0
    assert boundary_support(LineString([(20, 0), (20, 99)]), near_line) == 0.0


def test_merge_unsupported_keeps_drawn_cuts_and_dissolves_the_rest():
    prob = np.zeros((100, 300), dtype=np.uint8)
    prob[:, 100] = 255  # the network drew the cut at x = 100 only
    left, middle, right = (
        box(0, 0, 100, 100),
        box(100, 0, 200, 100),
        box(200, 0, 300, 100),
    )
    panels = merge_unsupported([left, middle, right], prob)
    # The undrawn boundary at x = 200 is dissolved; the drawn one at x = 100 stays.
    assert sorted(p.bounds for p in panels) == [(0, 0, 100, 100), (100, 0, 300, 100)]


def test_merge_unsupported_leaves_a_single_panel_alone():
    whole = box(0, 0, 10, 10)
    assert merge_unsupported([whole], np.zeros((10, 10), dtype=np.uint8)) == [whole]
