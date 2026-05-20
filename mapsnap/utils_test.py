"""Tests for mapsnap.utils."""

from mapsnap.utils import image_stem


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
