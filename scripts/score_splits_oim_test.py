"""Tests for score_splits_oim.py."""

import json
from pathlib import Path

import pytest
from PIL import Image
from score_splits_oim import (
    Case,
    manifest_cases,
    outcome,
    panels_in_frame,
    run_panels,
    score_case,
    score_record,
    summarize,
    unfinished_names,
    write_panels,
)
from shapely.geometry import box

from mapsnap.split import SheetContext

HALVES = {
    "width": 200,
    "height": 100,
    "panels": [
        [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]],
        [[100, 0], [200, 0], [200, 100], [100, 100], [100, 0]],
    ],
}


def write_benchmark(root: Path) -> Path:
    """A two-row benchmark: one split page with truth, one unsplit page."""
    (root / "images").mkdir()
    (root / "labels").mkdir()
    for name in ("a__p1", "b__p2"):
        Image.new("RGB", (100, 50)).save(root / "images" / f"{name}.jpg")
    (root / "labels" / "a__p1.panels.json").write_text(json.dumps(HALVES))
    manifest = root / "manifest.tsv"
    manifest.write_text(
        "image\titem\tpage\tlabel\tweight\tfold\tsize_band\tvolume_has_p0\tvolume_sheets\n"
        "images/a__p1.jpg\ta\tp1\tsplit\t1\ttrain\t1-5\tfalse\t4\n"
        "images/b__p2.jpg\tb\tp2\tunsplit\t1\ttest\t41+\ttrue\t90\n"
    )
    return manifest


def test_manifest_cases_pairs_split_pages_with_their_labels(tmp_path: Path):
    cases = manifest_cases(write_benchmark(tmp_path))
    assert [(c.name, c.truth is not None, c.fold, c.size_band) for c in cases] == [
        ("a__p1", True, "train", "1-5"),
        ("b__p2", False, "test", "41+"),
    ]
    assert cases[0].truth == tmp_path / "labels" / "a__p1.panels.json"
    assert [c.sheet() for c in cases] == [
        SheetContext("p1", 4, False),
        SheetContext("p2", 90, True),
    ]


def test_panels_in_frame_scales_truth_to_the_image():
    halves = panels_in_frame(HALVES, (100, 50))
    assert [p.bounds for p in halves] == [(0, 0, 50, 50), (50, 0, 100, 50)]


def test_score_case_is_perfect_for_identical_panels_and_lenient_on_a_miss():
    halves = [box(0, 0, 50, 50), box(50, 0, 100, 50)]
    assert score_case(halves, halves)[0] == pytest.approx(1.0)
    # Left whole, the page matches one half: 2500 / (5000 + the other half's 2500).
    iou, per_truth = score_case(halves, [box(0, 0, 100, 50)])
    assert iou == pytest.approx(1 / 3)
    assert sorted(per_truth) == pytest.approx([0.0, 0.5])


def test_run_panels_reads_a_runs_cut_and_treats_absence_as_whole(tmp_path: Path):
    case = Case("a__p1", "a", "p1", tmp_path / "a__p1.jpg")
    assert [p.bounds for p in run_panels(tmp_path, case, (100, 50))] == [
        (0, 0, 100, 50)
    ]
    (tmp_path / "a__p1.panels.json").write_text(json.dumps(HALVES))
    assert len(run_panels(tmp_path, case, (100, 50))) == 2


def test_unfinished_names_reads_the_fetch_scripts_list(tmp_path: Path):
    assert unfinished_names(tmp_path) == set()
    (tmp_path / "unfinished.json").write_text(json.dumps(["x__p3"]))
    assert unfinished_names(tmp_path) == {"x__p3"}


def test_score_record_marks_an_unsplit_page_ok_only_when_left_whole(tmp_path: Path):
    manifest = write_benchmark(tmp_path)
    split_case, unsplit_case = manifest_cases(manifest)
    halves = panels_in_frame(HALVES, (100, 50))
    assert score_record(unsplit_case, halves, (100, 50))["ok"] is False
    record = score_record(split_case, halves, (100, 50))
    assert (record["iou"], record["n_truth"], record["n_gen"]) == (1.0, 2, 2)


@pytest.mark.parametrize(
    ("n_truth", "n_gen", "expected"),
    [(2, 1, "whole"), (4, 2, "too few"), (3, 3, "right"), (2, 3, "too many")],
)
def test_outcome(n_truth: int, n_gen: int, expected: str):
    assert outcome(n_truth, n_gen) == expected


def test_summarize_reports_folds_bands_and_outcomes():
    records = [
        {
            "kind": "positive",
            "fold": "train",
            "size_band": "1-5",
            "weight": 1.0,
            "iou": 1.0,
            "n_truth": 2,
            "n_gen": 2,
            "small_panel_ious": [0.9],
        },
        {
            "kind": "positive",
            "fold": "test",
            "size_band": "1-5",
            "weight": 3.0,
            "iou": 0.5,
            "n_truth": 4,
            "n_gen": 1,
            "small_panel_ious": [0.0],
        },
        {
            "kind": "negative",
            "fold": "test",
            "size_band": "41+",
            "weight": 1.0,
            "iou": 1.0,
            "n_truth": 1,
            "n_gen": 1,
            "ok": True,
        },
    ]
    lines = summarize(records)
    # Weighting by 1 / sampling weight: (1.0 + 0.5 / 3) / (1 + 1 / 3) = 0.875.
    assert lines[0] == (
        "positives 2: IoU 0.750 (corrected 0.875), cut 50.0%, right count 50.0%, "
        "IoU>=0.9 50.0%, small panels 1/2; negatives 1: left whole 100.0%"
    )
    text = "\n".join(lines)
    assert "by fold:" in text and "by size_band:" in text
    assert "     2      1  1.000      0.0%      0.0%    100.0%      0.0%" in text


def test_write_panels_round_trips_through_run_panels(tmp_path: Path):
    image = tmp_path / "images" / "a__p1.jpg"
    halves = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
    write_panels(tmp_path, image, halves, (200, 100))
    data = json.loads((tmp_path / "a__p1.panels.json").read_text())
    assert (data["image"], data["width"], data["height"]) == ("a__p1.jpg", 200, 100)
    case = Case("a__p1", "a", "p1", image)
    panels = run_panels(tmp_path, case, (200, 100))
    assert [p.area for p in panels] == [10000, 10000]
