"""Tests for split_review.py."""

import argparse
import json
from pathlib import Path

import pytest
from score_splits_oim import Case, write_panels
from shapely.geometry import box
from split_review import (
    Arm,
    changed,
    page_record,
    parse_arm,
    read_titles,
    rings,
    truth_panels,
)

HALVES = {
    "width": 200,
    "height": 100,
    "panels": [
        [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]],
        [[100, 0], [200, 0], [200, 100], [100, 100], [100, 0]],
    ],
}


def test_parse_arm_splits_label_from_directory():
    assert parse_arm("cutline model=/tmp/a=b") == Arm("cutline model", Path("/tmp/a=b"))
    with pytest.raises(argparse.ArgumentTypeError):
        parse_arm("/tmp/no-label")


def test_rings_are_open_and_rounded():
    assert rings([box(0, 0, 10.04, 5)]) == [
        [[10.0, 0.0], [10.0, 5.0], [0.0, 5.0], [0.0, 0.0]]
    ]


def test_changed_needs_another_count_or_moved_panels():
    halves = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
    nudged = [box(0, 0, 101, 100), box(101, 0, 200, 100)]
    moved = [box(0, 0, 150, 100), box(150, 0, 200, 100)]
    assert not changed(halves, nudged)
    assert changed(halves, moved)
    assert changed(halves, [box(0, 0, 200, 100)])


def test_truth_panels_scale_oim_truth_and_default_to_the_whole_page(
    tmp_path: Path,
):
    truth = tmp_path / "a__p1.panels.json"
    truth.write_text(json.dumps(HALVES))
    split = Case("a__p1", "a", "p1", tmp_path / "a__p1.jpg", truth)
    assert [p.bounds for p in truth_panels(split, (100, 50))] == [
        (0, 0, 50, 50),
        (50, 0, 100, 50),
    ]
    unsplit = Case("b__p2", "b", "p2", tmp_path / "b__p2.jpg")
    assert [p.bounds for p in truth_panels(unsplit, (100, 50))] == [(0, 0, 100, 50)]


def test_page_record_scores_each_arm_against_truth(tmp_path: Path):
    truth = tmp_path / "a__p1.panels.json"
    truth.write_text(json.dumps(HALVES))
    image = tmp_path / "a__p1.jpg"
    case = Case("a__p1", "a", "p1", image, truth)
    right, whole = tmp_path / "right", tmp_path / "whole"
    right.mkdir()
    whole.mkdir()  # no file: the arm left the page whole
    write_panels(right, image, [box(0, 0, 100, 100), box(100, 0, 200, 100)], (200, 100))
    arms = [Arm("A", whole), Arm("B", right)]
    record = page_record(case, {"a__p1": "Champaign, Ill. | 1909"}, arms, (200, 100))
    assert record["image"] == "images/a__p1.jpg"
    assert record["title"] == "Champaign, Ill. | 1909"
    assert record["label"] == "split"
    assert len(record["truth"]) == 2
    assert [len(arm["panels"]) for arm in record["arms"]] == [1, 2]
    # Left whole: A / (2P - A) for its largest truth panel, half the page.
    assert record["arms"][0]["iou"] == pytest.approx(1 / 3, abs=1e-4)
    assert record["arms"][1]["iou"] == 1.0


def test_read_titles_keys_by_image_stem(tmp_path: Path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("image\ttitle\nimages/a__p1.jpg\tChampaign\n")
    assert read_titles(manifest) == {"a__p1": "Champaign"}
