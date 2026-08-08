"""Unit tests for the OIM region-boundary panels builder (#273)."""

import json

from mapsnap.oim_panels import (
    embedded_json,
    region_division,
    region_ring,
    write_page_files,
)


def region(division: int, ring: list, georeferenced: bool = True) -> dict:
    return {
        "id": 1000 + division,
        "division_number": str(division),
        "georeferenced": georeferenced,
        "boundary": {"type": "Polygon", "coordinates": [ring]},
    }


def test_embedded_json_decodes_one_balanced_value():
    html = 'junk "cutlines": [[[1, 2], [3, 4]]], "other": 5 junk'
    assert embedded_json(html, "cutlines") == [[[1, 2], [3, 4]]]
    assert embedded_json(html, "missing") is None


def test_region_ring_closes_and_floats():
    r = region(1, [[0, 0], [10, 0], [10, 5], [0, 5]])
    ring = region_ring(r)
    assert ring[0] == ring[-1]
    assert ring[0] == [0.0, 0.0]


def test_region_ring_rejects_non_polygon():
    assert (
        region_ring({"boundary": {"type": "MultiPolygon", "coordinates": []}}) is None
    )
    assert region_ring({}) is None


def test_region_division_tolerates_strings_and_absence():
    assert region_division({"division_number": "3"}) == 3
    assert region_division({}) == 0


def test_write_page_files_orders_by_division(tmp_path):
    regions = [
        region(2, [[5, 0], [10, 0], [10, 10], [5, 10]]),
        region(1, [[0, 0], [5, 0], [5, 10], [0, 10]], georeferenced=False),
    ]
    regions.sort(key=region_division)
    assert write_page_files(tmp_path, "p12", regions, [[[5, 0], [5, 10]]], [10, 10])
    data = json.loads((tmp_path / "oim" / "p12.panels.json").read_text())
    assert data["width"] == 10 and data["height"] == 10
    assert len(data["panels"]) == 2
    assert data["panels"][0][0] == [0.0, 0.0]  # division 1 first
    assert data["georeferenced"] == [False, True]
    cut = json.loads((tmp_path / "oim" / "p12.cutlines.json").read_text())
    assert cut["cutlines"] == [[[5, 0], [5, 10]]]


def test_write_page_files_skips_uncut_pages(tmp_path):
    only = [region(1, [[0, 0], [10, 0], [10, 10], [0, 10]])]
    assert not write_page_files(tmp_path, "p1", only, [], [10, 10])
    assert not (tmp_path / "oim" / "p1.panels.json").exists()
