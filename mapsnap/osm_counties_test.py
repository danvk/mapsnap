"""Tests for the per-county OSM extractor (mapsnap.osm_counties)."""

import json
from pathlib import Path

import pytest

from mapsnap.osm_counties import (
    County,
    buffered_rings,
    extract_entry,
    load_boundaries,
    natural_earth_geojson,
    pending,
    read_counties,
)

COUNTIES_TSV = (
    "fips\tstate\tne_name\tne_type\titems\tsheets\tloc_names\n"
    "US17031\tIL\tCook\tCounty\t191\t11822\tcook county\n"
    "US06037\tCA\tLos Angeles\tCounty\t490\t8673\tlos angeles county\n"
)

# A one-degree square around (40N, 100W), as Natural Earth would give it.
SQUARE = {
    "type": "Polygon",
    "coordinates": [
        [[-100.0, 40.0], [-99.0, 40.0], [-99.0, 41.0], [-100.0, 41.0], [-100.0, 40.0]]
    ],
}


def test_read_counties_keeps_what_the_extractor_needs(tmp_path: Path) -> None:
    path = tmp_path / "counties.tsv"
    path.write_text(COUNTIES_TSV)
    counties = read_counties(path)
    assert [c.fips for c in counties] == ["US17031", "US06037"]
    assert counties[0] == County("US17031", "IL", "Cook", 11822)
    assert counties[0].filename == "US17031.osm.pbf"


def test_read_counties_rejects_a_file_without_the_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("fips\tstate\nUS1\tIL\n")
    with pytest.raises(SystemExit, match="ne_name"):
        read_counties(path)


def test_buffer_grows_the_boundary_by_roughly_the_asked_distance() -> None:
    """A degree of longitude is 85 km at 40N, so buffering in raw degrees would skew."""
    rings = buffered_rings(SQUARE, buffer_km=3.0)
    assert len(rings) == 1
    xs = [x for x, _ in rings[0][0]]
    ys = [y for _, y in rings[0][0]]
    # 3 km north-south is 0.027 degrees; east-west at 40N it is 0.035.
    grew_south = 40.0 - min(ys)
    grew_north = max(ys) - 41.0
    grew_east = max(xs) - (-99.0)
    assert 0.024 < grew_south < 0.031, grew_south
    assert grew_north == pytest.approx(grew_south, abs=1e-3)
    assert 0.031 < grew_east < 0.040, grew_east


def test_buffer_is_wider_in_degrees_the_further_north_it_is() -> None:
    north = dict(
        SQUARE,
        coordinates=[
            [
                [-100.0, 60.0],
                [-99.0, 60.0],
                [-99.0, 61.0],
                [-100.0, 61.0],
                [-100.0, 60.0],
            ]
        ],
    )
    south = buffered_rings(SQUARE, 3.0)[0][0]
    far = buffered_rings(north, 3.0)[0][0]
    south_width = max(x for x, _ in south) - (-99.0)
    north_width = max(x for x, _ in far) - (-99.0)
    assert north_width > south_width * 1.5


def test_a_multipolygon_county_becomes_a_multipolygon_entry() -> None:
    islands = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[-100.0, 40.0], [-99.9, 40.0], [-99.9, 40.1], [-100.0, 40.0]]],
            [[[-95.0, 40.0], [-94.9, 40.0], [-94.9, 40.1], [-95.0, 40.0]]],
        ],
    }
    county = County("US1", "XX", "Islands", 1)
    entry = extract_entry(county, buffered_rings(islands, 1.0))
    assert entry["output"] == "US1.osm.pbf"
    assert entry["output_format"] == "pbf"
    assert "multipolygon" in entry and len(entry["multipolygon"]) == 2

    single = extract_entry(county, buffered_rings(SQUARE, 1.0))
    assert "polygon" in single and "multipolygon" not in single


def test_pending_skips_counties_already_on_disk(tmp_path: Path) -> None:
    counties = [County("US1", "IL", "A", 1), County("US2", "IL", "B", 2)]
    assert pending(counties, tmp_path) == counties
    (tmp_path / "US1.osm.pbf").write_bytes(b"pbf")
    assert [c.fips for c in pending(counties, tmp_path)] == ["US2"]


def test_load_boundaries_keys_on_fips(tmp_path: Path) -> None:
    path = tmp_path / "ne.geojson"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {"properties": {"FIPS": "US17031"}, "geometry": SQUARE},
                    {"properties": {}, "geometry": SQUARE},  # no FIPS, skipped
                ]
            }
        )
    )
    boundaries = load_boundaries(path)
    assert list(boundaries) == ["US17031"]


def test_natural_earth_geojson_passes_a_geojson_through(tmp_path: Path) -> None:
    path = tmp_path / "ne.geojson"
    path.write_text("{}")
    assert natural_earth_geojson(path) == path


def test_natural_earth_geojson_rejects_a_directory_with_no_shapefile(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="no .shp"):
        natural_earth_geojson(tmp_path)


def test_a_zero_byte_extract_does_not_count_as_done(tmp_path: Path) -> None:
    """A killed osmium pass leaves stubs; treating them as done would skip them forever."""
    from mapsnap.osm_counties import is_cut

    counties = [County("US1", "IL", "A", 1), County("US2", "IL", "B", 2)]
    (tmp_path / "US1.osm.pbf").write_bytes(b"")  # created, never filled
    (tmp_path / "US2.osm.pbf").write_bytes(b"pbf")
    assert not is_cut(tmp_path / "US1.osm.pbf")
    assert is_cut(tmp_path / "US2.osm.pbf")
    assert [c.fips for c in pending(counties, tmp_path)] == ["US1"]
