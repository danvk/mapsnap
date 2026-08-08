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


def test_region_ring_flips_y_and_closes():
    # OIM boundaries are y-up; a region at the TOP of a height-10 canvas has
    # boundary y in [5, 10] and must come out at image y in [0, 5].
    r = region(1, [[0, 5], [10, 5], [10, 10], [0, 10]])
    ring = region_ring(r, 10)
    assert ring is not None
    assert ring[0] == ring[-1]
    ys = [y for _, y in ring]
    assert min(ys) == 0.0 and max(ys) == 5.0


def test_region_ring_rejects_non_polygon():
    assert (
        region_ring({"boundary": {"type": "MultiPolygon", "coordinates": []}}, 10)
        is None
    )
    assert region_ring({}, 10) is None


def test_region_division_tolerates_strings_and_absence():
    assert region_division({"division_number": "3"}) == 3
    assert region_division({}) == 0


def test_write_page_files_orders_by_division(tmp_path):
    regions = [
        region(2, [[5, 0], [10, 0], [10, 10], [5, 10]]),
        region(1, [[0, 0], [5, 0], [5, 10], [0, 10]], georeferenced=False),
    ]  # full-height cuts: y-flip-invariant, so ring assertions stay simple
    regions.sort(key=region_division)
    assert write_page_files(tmp_path, "p12", regions, [[[5, 0], [5, 10]]], [10, 10])
    data = json.loads((tmp_path / "oim" / "p12.panels.json").read_text())
    assert data["width"] == 10 and data["height"] == 10
    assert len(data["panels"]) == 2
    assert data["panels"][0][0] == [0.0, 10.0]  # division 1 first (y flipped)
    assert data["georeferenced"] == [False, True]
    cut = json.loads((tmp_path / "oim" / "p12.cutlines.json").read_text())
    assert cut["cutlines"] == [[[5.0, 10.0], [5.0, 0.0]]]  # y flipped


def test_write_page_files_skips_uncut_pages(tmp_path):
    only = [region(1, [[0, 0], [10, 0], [10, 10], [0, 10]])]
    assert not write_page_files(tmp_path, "p1", only, [], [10, 10])
    assert not (tmp_path / "oim" / "p1.panels.json").exists()


def test_title_page_key_handles_plain_and_sb_formats():
    from mapsnap.oim_panels import title_page_key

    assert title_page_key("Fargo, N.D. | 1958 p12") == "p12"
    assert title_page_key("Washington, D.C. | 1916 | Vol. 2 psb002600") == "p260"
    assert title_page_key("Washington, D.C. | 1916 | Vol. 2 psb00103w") == "p103w"
    assert title_page_key("Washington, D.C. | 1916 | Vol. 2") is None
