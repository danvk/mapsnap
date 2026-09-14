"""Tests for the key-map raw-sheet fetcher (mapsnap.loc_raw)."""

from pathlib import Path

import pytest

from mapsnap.loc_raw import Wanted, keymap_keys, missing_raw, read_list, write_list


def test_missing_raw_wants_only_the_sheets_with_no_copy() -> None:
    present = ["p1.jpg", "p2.jpg", "raw/p0.jpg", "keymaps.json"]
    assert missing_raw(present, ["p1"]) == ["p1"]
    assert missing_raw(present, ["p0"]) == []  # already mirrored
    assert missing_raw(present, ["p0", "p1"]) == ["p1"]
    assert missing_raw(present, []) == []


def test_missing_raw_asks_for_a_panel_s_parent_sheet() -> None:
    """A split panel is cut locally from the parent, so the parent is fetched."""
    assert missing_raw(["p1.jpg"], ["p1__1", "p1__2"]) == ["p1"]
    assert missing_raw(["raw/p1.jpg"], ["p1__1", "p1__2"]) == []


def test_keymap_keys_reads_the_record_and_survives_rubbish() -> None:
    assert keymap_keys(["keymaps.json"], '{"keys": ["p1", "p1b"]}') == ["p1", "p1b"]
    assert keymap_keys(["keymaps.json"], '{"keys": []}') == []
    assert keymap_keys(["keymaps.json"], "not json") == []
    assert keymap_keys(["p1.jpg"], '{"keys": ["p1"]}') == []  # no record present


def test_list_round_trips_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "wanted.tsv"
    write_list(
        path,
        [Wanted("sanborn2", "p1"), Wanted("sanborn1", "p3"), Wanted("sanborn1", "p1")],
    )
    assert path.read_text().splitlines()[0] == "item\tkey"
    assert read_list(path) == [
        Wanted("sanborn1", "p1"),
        Wanted("sanborn1", "p3"),
        Wanted("sanborn2", "p1"),
    ]


def test_read_list_rejects_a_file_without_the_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("item\tstate\nx\ty\n")
    with pytest.raises(SystemExit, match="'key'"):
        read_list(path)
