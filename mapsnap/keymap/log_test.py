"""Tests for the key-map decision log."""

from pathlib import Path

from mapsnap.keymap.log import (
    append_keymap_log,
    keymap_log_path,
    read_section,
    section_markers,
)


def test_keymap_log_path_sits_under_raw_for_either_image(tmp_path: Path):
    volume = tmp_path / "vol"
    assert keymap_log_path(volume / "p0.jpg") == volume / "raw" / "p0.keymap.txt"
    assert (
        keymap_log_path(volume / "raw" / "p0.jpg") == volume / "raw" / "p0.keymap.txt"
    )
    assert keymap_log_path(volume / "p0__1.jpg").name == "p0__1.keymap.txt"
    assert keymap_log_path(volume / "raw" / "pa.jpg").name == "pa.keymap.txt"


def test_append_keymap_log_replaces_a_stage_section_and_keeps_the_rest(tmp_path: Path):
    image = tmp_path / "vol" / "p0.jpg"
    append_keymap_log(
        image, "split", ["split rejected — panel 2 is flush with 1 sheet edge(s)"]
    )
    append_keymap_log(image, "keymap-detect", ["p0: key map by convention"])
    # A rerun of the first stage replaces its section in place, not at the end.
    append_keymap_log(image, "split", ["split accepted: 2 panels"])
    text = keymap_log_path(image).read_text()
    begin_split, end_split = section_markers("split")
    assert text.count(begin_split) == 1 and text.count(end_split) == 1
    assert "split rejected" not in text
    assert text.index(begin_split) < text.index(section_markers("keymap-detect")[0])
    assert read_section(image, "split") == ["split accepted: 2 panels"]
    assert read_section(image, "keymap-detect") == ["p0: key map by convention"]
    assert read_section(image, "inset") is None
    # The header records when the decision was made.
    assert "==== mapsnap split (20" in text


def test_append_keymap_log_creates_the_raw_directory(tmp_path: Path):
    image = tmp_path / "vol" / "p1N.jpg"
    path = append_keymap_log(image, "cartouche", ["nothing read"])
    assert path.exists() and path.parent.name == "raw"
