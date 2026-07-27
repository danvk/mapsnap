import json
from pathlib import Path

import pytest

from mapsnap.keymap.check_truth_labels import (
    check_truth_file,
    normalize_label,
    sheet_label,
    suggest,
    truth_files,
    volume_pages,
)


def test_sheet_label():
    assert sheet_label("p35") == ("35", None)
    assert sheet_label("p35B") == ("35B", None)
    assert sheet_label("p35b") == ("35B", None)  # stem case varies across volumes
    assert sheet_label("p35B__2") == ("35B", 2)
    assert sheet_label("p1499L") == ("1499L", None)
    # Front matter and other non-page images.
    assert sheet_label("pcover") is None
    assert sheet_label("06372_01_1957-ind1") is None


def test_normalize_label():
    assert normalize_label("  35b ") == "35B"
    assert normalize_label("") == ""


def test_suggest_only_fixes_mechanical_slips():
    known = {"35A", "60"}
    # The labeller's own uncertainty marker.
    assert suggest("35A?", known) == "35A"
    # Stripping punctuation must still land on a real sheet.
    assert suggest("99?", known) is None
    # Already valid: nothing to suggest.
    assert suggest("60", known) is None
    # Not a fuzzy matcher — a genuine misreading gets no guess.
    assert suggest("36A", known) is None


@pytest.fixture
def volume(tmp_path: Path) -> Path:
    """A volume with a key map (p0), a split sheet (p4), and plain pages."""
    (tmp_path / "raw").mkdir()
    for stem in ["p0", "p1", "p2", "p3A", "p4", "p4__1", "p4__2", "pcover"]:
        (tmp_path / f"{stem}.jpg").write_bytes(b"jpeg")
    (tmp_path / "raw" / "p0.keymap.json").write_text("{}")
    return tmp_path


def write_truth(volume: Path, texts: list[str]) -> Path:
    path = volume / "raw" / "truth" / "p0.labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"labels": [{"x": 0, "y": 0, "text": t} for t in texts]})
    )
    return path


def test_volume_pages_indexes_sheets_panels_and_keymaps(volume: Path):
    pages = volume_pages(volume)
    assert set(pages.stems) == {"0", "1", "2", "3A", "4"}
    assert pages.panels["4"] == 2  # p4__1, p4__2
    assert pages.panels["1"] == 0
    assert pages.keymaps == {"0"}  # p0 has a keymap.json sidecar
    assert pages.skipped == ["pcover"]


def test_clean_truth_reports_nothing(volume: Path):
    # The key map does not label itself, and the split sheet is drawn per panel.
    path = write_truth(volume, ["1", "2", "3A", "4", "4"])
    report = check_truth_file(path, volume)
    assert report.ok
    assert report.n_sheets == 4  # p0 excluded as a key map


def test_unknown_label_is_reported_with_a_suggestion(volume: Path):
    path = write_truth(volume, ["1", "2", "3A?", "4", "4"])
    report = check_truth_file(path, volume)
    assert report.unknown == [("3A?", "3A")]
    # The sheet it was meant to name is now also unlabelled.
    assert report.missing == ["3A"]


def test_missing_page_is_reported(volume: Path):
    path = write_truth(volume, ["1", "3A", "4", "4"])
    report = check_truth_file(path, volume)
    assert report.missing == ["2"]
    assert not report.unknown


def test_duplicate_is_reported_only_for_unsplit_sheets(volume: Path):
    path = write_truth(volume, ["1", "1", "2", "3A", "4", "4", "4"])
    report = check_truth_file(path, volume)
    # p1 is not split, so twice is a mistake; p4 has panels, so three times is fine.
    assert report.duplicated == [("1", 2)]


def test_truth_files_accepts_a_volume_or_a_labels_file(volume: Path):
    path = write_truth(volume, ["1"])
    assert truth_files(volume) == [(path, volume)]
    assert truth_files(path) == [(path, volume)]
