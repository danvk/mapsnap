"""Tests for grade_snap_candidates.py."""

from grade_snap_candidates import write_stand_in_jpeg

from mapsnap.utils import jpeg_dimensions


def test_stand_in_jpeg_reports_the_page_size(tmp_path):
    path = tmp_path / "p12__2.jpg"
    write_stand_in_jpeg(path, 1613, 982)
    assert jpeg_dimensions(path) == (1613, 982)
