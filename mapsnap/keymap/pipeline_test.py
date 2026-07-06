from pathlib import Path

from mapsnap.keymap.pipeline import (
    format_page_spec,
    keymap_volume_dir,
    valid_page_spec,
)


def test_format_page_spec_single_run():
    assert format_page_spec([1, 2, 3, 4, 5]) == "1-5"


def test_format_page_spec_with_gaps():
    assert format_page_spec([1, 2, 3, 5, 7, 8, 9]) == "1-3,5,7-9"


def test_format_page_spec_dedups_and_sorts():
    assert format_page_spec([10, 1, 2, 2, 1]) == "1-2,10"


def test_format_page_spec_singletons():
    assert format_page_spec([1, 3, 5]) == "1,3,5"


def test_format_page_spec_empty():
    assert format_page_spec([]) == ""


def test_keymap_volume_dir_under_raw():
    # A full-resolution key map lives in <volume>/raw/, so the volume is the grandparent.
    assert keymap_volume_dir(Path("data/chicago/raw/p0b.jpg")) == Path("data/chicago")


def test_keymap_volume_dir_flat():
    # A key map alongside the scaled pages has the volume as its immediate parent.
    assert keymap_volume_dir(Path("data/chicago/p0b.jpg")) == Path("data/chicago")


def test_valid_page_spec_from_volume_images(tmp_path: Path):
    volume = tmp_path / "vol"
    (volume / "raw").mkdir(parents=True)
    for name in ["p1.jpg", "p2.jpg", "p3.jpg", "p10.jpg", "p0b.jpg", "covr.jpg"]:
        (volume / name).write_bytes(b"")
    keymap = volume / "raw" / "p0b.jpg"
    keymap.write_bytes(b"")
    # covr has no page number; p0b's "page 0" is dropped as non-positive.
    assert valid_page_spec([keymap]) == "1-3,10"


def test_valid_page_spec_unions_multiple_volumes(tmp_path: Path):
    first = tmp_path / "a"
    first.mkdir()
    for name in ["p1.jpg", "p2.jpg", "p0b.jpg"]:
        (first / name).write_bytes(b"")
    second = tmp_path / "b"
    second.mkdir()
    for name in ["p4.jpg", "p5.jpg", "p0b.jpg"]:
        (second / name).write_bytes(b"")
    assert valid_page_spec([first / "p0b.jpg", second / "p0b.jpg"]) == "1-2,4-5"
