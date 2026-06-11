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
