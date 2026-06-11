"""Tests for mapsnap.utils."""

from mapsnap.utils import image_stem, source_id_to_page_key


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
    # Non-sheet pages (cover, title, index) have no digit after the final hyphen
    # and fall through to the split("/")[-2] fallback, which returns "iiif".
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-covr", "") == "iiif"
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-titl", "") == "iiif"
    assert source_id_to_page_key(f"{_CHI}:01790_01N_1950-ind1", "") == "iiif"
