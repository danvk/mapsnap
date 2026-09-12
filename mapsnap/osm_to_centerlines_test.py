"""Tests for osm_to_centerlines helpers."""

import json
import sys
from pathlib import Path

import osmium
import pytest

from mapsnap import osm_to_centerlines
from mapsnap.osm_to_centerlines import (
    _cluster_coords,
    centerlines_from_osm,
    compute_street_intersections,
    load_centerlines,
)
from mapsnap.utils import default_centerlines, require_centerlines

# ---------------------------------------------------------------------------
# _cluster_coords
# ---------------------------------------------------------------------------


def test_cluster_coords_single_point():
    result = _cluster_coords([(1.0, 40.0)])
    assert len(result) == 1


def test_cluster_coords_nearby_points_merged():
    # Two points only ~1ft apart should collapse to one centroid.
    result = _cluster_coords([(0.0, 40.0), (0.000001, 40.000001)])
    assert len(result) == 1


def test_cluster_coords_distant_points_separate():
    # Points ~600ft apart (roughly 0.001 deg lon at lat 40) stay separate.
    result = _cluster_coords([(0.0, 40.0), (0.01, 40.0)])
    assert len(result) == 2


def test_cluster_coords_empty():
    assert _cluster_coords([]) == []


# ---------------------------------------------------------------------------
# compute_street_intersections
# ---------------------------------------------------------------------------

# Minimal GeoJSON features: two streets that share node (0.0, 40.0).
_FEAT_A = {
    "properties": {"street_name": "Main Street"},
    "geometry": {"type": "LineString", "coordinates": [[-1.0, 40.0], [0.0, 40.0]]},
}
_FEAT_B = {
    "properties": {"street_name": "Oak Avenue"},
    "geometry": {"type": "LineString", "coordinates": [[0.0, 40.0], [0.0, 41.0]]},
}
_FEAT_C = {
    "properties": {"street_name": "Elm Street"},
    "geometry": {"type": "LineString", "coordinates": [[5.0, 45.0], [6.0, 46.0]]},
}


def test_compute_intersections_finds_shared_node():
    result = compute_street_intersections([_FEAT_A, _FEAT_B])
    assert len(result) == 1
    street_a, street_b, lon, lat = result[0]
    assert street_a == "MAIN STREET"
    assert street_b == "OAK AVENUE"
    assert abs(lon) < 0.001
    assert abs(lat - 40.0) < 0.001


def test_compute_intersections_no_shared_node():
    result = compute_street_intersections([_FEAT_A, _FEAT_C])
    assert result == []


def test_compute_intersections_street_names_normalized():
    # Raw OSM name "Main Street" should appear as "MAIN STREET" in output.
    result = compute_street_intersections([_FEAT_A, _FEAT_B])
    assert result[0][0] == "MAIN STREET"
    assert result[0][1] == "OAK AVENUE"


def test_compute_intersections_street_a_lt_street_b():
    # street_a should always be alphabetically before street_b.
    result = compute_street_intersections([_FEAT_A, _FEAT_B])
    assert result[0][0] <= result[0][1]


def test_compute_intersections_no_self_pairs():
    # A street that doubles back on itself should not produce a self-intersection.
    feat = {
        "properties": {"street_name": "Loop Road"},
        "geometry": {
            "type": "LineString",
            "coordinates": [[0.0, 40.0], [1.0, 40.0], [0.0, 40.0]],
        },
    }
    result = compute_street_intersections([feat])
    assert result == []


def test_compute_intersections_clusters_nearby_nodes():
    # Two pairs of shared nodes just 1ft apart should cluster to one intersection.
    feat_a = {
        "properties": {"street_name": "Main Street"},
        "geometry": {
            "type": "LineString",
            "coordinates": [[0.0, 40.0], [0.000001, 40.000001]],
        },
    }
    feat_b = {
        "properties": {"street_name": "Oak Avenue"},
        "geometry": {
            "type": "LineString",
            "coordinates": [[0.0, 40.0], [0.000001, 40.000001]],
        },
    }
    result = compute_street_intersections([feat_a, feat_b])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# OSM file input (pyosmium) and the loader every stage goes through
# ---------------------------------------------------------------------------

OSM_XML = """<?xml version='1.0' encoding='UTF-8'?>
<osm version="0.6" generator="test">
  <node id="1" lat="40.0000000" lon="-73.0000000"/>
  <node id="2" lat="40.0010000" lon="-73.0000000"/>
  <node id="3" lat="40.0010000" lon="-73.0010000"/>
  <node id="4" lat="40.0020000" lon="-73.0010000"/>
  <way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/><tag k="name" v="Hooper Street"/></way>
  <way id="11"><nd ref="2"/><nd ref="3"/><tag k="highway" v="footway"/><tag k="name" v="Park Path"/></way>
  <way id="12"><nd ref="3"/><nd ref="4"/><tag k="highway" v="service"/><tag k="service" v="alley"/><tag k="name" v="Back Alley"/></way>
  <way id="13"><nd ref="1"/><nd ref="3"/><tag k="highway" v="service"/><tag k="service" v="driveway"/><tag k="name" v="Drive"/></way>
  <way id="14"><nd ref="1"/><nd ref="4"/><tag k="highway" v="residential"/></way>
  <way id="15"><nd ref="2"/><nd ref="99"/><tag k="highway" v="residential"/><tag k="name" v="Cut Street"/></way>
</osm>
"""


def write_osm(tmp_path: Path) -> tuple[Path, Path]:
    """The fixture as .osm XML and as .osm.pbf (written by pyosmium from the XML)."""
    xml = tmp_path / "streets.osm"
    xml.write_text(OSM_XML)
    pbf = tmp_path / "centerlines.osm.pbf"
    with osmium.SimpleWriter(str(pbf)) as writer:
        for obj in osmium.FileProcessor(str(xml)):
            writer.add(obj)
    return xml, pbf


def test_centerlines_from_osm_keeps_named_drivable_ways_with_geometry(tmp_path: Path):
    xml, pbf = write_osm(tmp_path)
    for path in (xml, pbf):
        collection = centerlines_from_osm(path)
        names = [f["properties"]["street_name"] for f in collection["features"]]
        # footway dropped, driveway dropped, alley kept, unnamed dropped,
        # the way with a node the file has no location for dropped
        assert names == ["Hooper Street", "Back Alley"]
        hooper = collection["features"][0]["geometry"]
        assert hooper["type"] == "LineString"
        assert hooper["coordinates"] == [[-73.0, 40.0], [-73.0, 40.001]]


def test_load_centerlines_dispatches_on_suffix(tmp_path: Path):
    xml, pbf = write_osm(tmp_path)
    from_pbf = load_centerlines(pbf)
    geojson = tmp_path / "centerlines.geojson"
    geojson.write_text(json.dumps(from_pbf))
    assert load_centerlines(geojson) == from_pbf
    assert load_centerlines(str(xml)) == from_pbf
    # The intersection finder keys on coordinates, which survive both readers.
    rows = compute_street_intersections(from_pbf["features"])
    assert rows == [("BACK ALLEY", "HOOPER STREET", -73.0, 40.001)] or rows == []


def test_default_centerlines_finds_geojson_first_then_pbf_then_parent(tmp_path: Path):
    volume = tmp_path / "county" / "item"
    volume.mkdir(parents=True)
    assert default_centerlines(volume) is None
    with pytest.raises(SystemExit):
        require_centerlines(volume)
    county_pbf = tmp_path / "county" / "centerlines.osm.pbf"
    county_pbf.write_bytes(b"")
    assert default_centerlines(volume) == county_pbf  # the parent's extract
    own_pbf = volume / "centerlines.osm.pbf"
    own_pbf.write_bytes(b"")
    assert default_centerlines(volume) == own_pbf
    own_geojson = volume / "centerlines.geojson"
    own_geojson.write_text("{}")
    assert require_centerlines(volume) == own_geojson  # GeoJSON wins over the extract


def test_cli_converts_an_osm_file_and_can_skip_the_debug_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, pbf = write_osm(tmp_path)
    out = tmp_path / "vol" / "centerlines.geojson"
    out.parent.mkdir()
    monkeypatch.setattr(sys, "argv", ["osm-to-geojson", str(pbf), "--output", str(out)])
    osm_to_centerlines.main()
    assert len(json.loads(out.read_text())["features"]) == 2
    assert (out.parent / "streets.txt").read_text() == "BACK ALLEY\nHOOPER STREET\n"
    assert (out.parent / "intersections.csv").exists()
    (out.parent / "streets.txt").unlink()
    (out.parent / "intersections.csv").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        ["osm-to-geojson", str(pbf), "--output", str(out), "--no-debug-files"],
    )
    osm_to_centerlines.main()
    assert out.exists()
    assert not (out.parent / "streets.txt").exists()
    assert not (out.parent / "intersections.csv").exists()
