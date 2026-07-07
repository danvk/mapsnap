"""Tests for mapsnap.utils."""

from pathlib import Path

import pytest

from mapsnap.utils import (
    Step,
    default_centerlines,
    image_stem,
    list_pages,
    mark_step_done,
    source_id_to_page_key,
    step_done,
)


def test_default_centerlines_in_same_dir(tmp_path):
    centerlines = tmp_path / "centerlines.geojson"
    centerlines.touch()
    assert default_centerlines(tmp_path) == centerlines


def test_default_centerlines_in_parent_dir(tmp_path):
    # Split panels live in a subdirectory; the file sits in the volume (parent) dir.
    centerlines = tmp_path / "centerlines.geojson"
    centerlines.touch()
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    assert default_centerlines(panels_dir) == centerlines


def test_default_centerlines_missing_returns_none(tmp_path):
    assert default_centerlines(tmp_path) is None


def test_list_pages_splits_supersede_parent(tmp_path):
    for name in ("p1.jpg", "p2.jpg", "p2__1.jpg", "p2__2.jpg", "p3.jpg"):
        (tmp_path / name).touch()
    # raw/ and oim/ subdirectories are ignored.
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "p1.jpg").touch()

    names = [p.name for p in list_pages(tmp_path)]
    assert names == ["p1.jpg", "p2__1.jpg", "p2__2.jpg", "p3.jpg"]


def test_list_pages_empty_dir(tmp_path):
    assert list_pages(tmp_path) == []


def test_single_extension():
    assert image_stem("photo.jpg") == "photo"


def test_multi_extension():
    assert image_stem("p50n.2048px.jpg") == "p50n"


def test_with_directory():
    assert image_stem("/path/to/p50n.2048px.jpg") == "p50n"


def test_no_extension():
    assert image_stem("p50n") == "p50n"


def test_hidden_file_no_extension():
    assert image_stem(".hidden") == ""


# source_id_to_page_key


def test_source_id_numeric():
    assert (
        source_id_to_page_key(
            "https://loc.gov/resource/g4164nm.g4164nm1950-0425/info.json", ""
        )
        == "p425"
    )


def test_source_id_strips_leading_zeros():
    assert (
        source_id_to_page_key(
            "https://loc.gov/resource/g4164nm.g4164nm1950-0006/info.json", ""
        )
        == "p6"
    )


def test_source_id_letter_suffix():
    assert (
        source_id_to_page_key(
            "https://loc.gov/resource/g4164nm.g4164nm1950-0006N/info.json", ""
        )
        == "p6N"
    )


def test_source_id_without_info_json():
    assert (
        source_id_to_page_key("https://loc.gov/resource/g4164nm.g4164nm1939-0027s", "")
        == "p27s"
    )


def test_source_id_split_label():
    assert (
        source_id_to_page_key(
            "https://loc.gov/resource/g4164nm.g4164nm1950-0156/info.json",
            "Page 156 [2]",
        )
        == "p156__2"
    )


# source_id_to_page_key — sb format (e.g. Washington DC 1916)

_DC = "https://tile.loc.gov/image-services/iiif/service:gmd:gmd385m:g3851m:g3851gm:g01227003"


def test_source_id_sb():
    assert source_id_to_page_key(f"{_DC}:sb001250", "") == "p125"
    assert source_id_to_page_key(f"{_DC}:sb002160", "") == "p216"
    assert source_id_to_page_key(f"{_DC}:sb00154s", "") == "p154s"
    assert source_id_to_page_key(f"{_DC}:sb00001a", "") == "p1a"


# source_id_to_page_key — Chicago 1950 format (e.g. 01790_01N_1950-0003N)

_CHI = "https://tile.loc.gov/image-services/iiif/service:gmd:gmd410m:g4104m:g4104cm:g01790195001N"


def test_source_id_chicago():
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-0001N", "") == "p1N"
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-0013N", "") == "p13N"
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-0110W", "") == "p110W"


def test_source_id_chicago_non_sheet():
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-covr", "") == "covr"
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-titl", "") == "titl"
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-ind1", "") == "ind1"


def test_step_done_and_mark(tmp_path: Path):
    assert not step_done(tmp_path, "scale")
    mark_step_done(tmp_path, "scale")
    assert step_done(tmp_path, "scale")
    assert not step_done(tmp_path, "split")  # independent of other steps


def test_step_runs_body_and_stamps(tmp_path: Path):
    step = Step(tmp_path)
    ran = []
    with step("osm"):
        ran.append(1)
    assert ran == [1]  # body ran
    assert step_done(tmp_path, "osm")  # and was stamped


def test_step_skips_completed_body(tmp_path: Path):
    mark_step_done(tmp_path, "osm")
    step = Step(tmp_path)
    ran = []
    with step("osm"):
        ran.append(1)  # must not execute
    assert ran == []  # whole body was skipped
    line_after_block = True  # code after the block still runs normally
    assert line_after_block


def test_step_force_reruns_completed(tmp_path: Path):
    mark_step_done(tmp_path, "osm")
    step = Step(tmp_path, force=True)
    ran = []
    with step("osm"):
        ran.append(1)
    assert ran == [1]


def test_step_leaves_no_stamp_when_body_raises(tmp_path: Path):
    step = Step(tmp_path)
    with pytest.raises(ValueError):
        with step("osm"):
            raise ValueError("boom")
    assert not step_done(tmp_path, "osm")  # interrupted step re-runs next time
