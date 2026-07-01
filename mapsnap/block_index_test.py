"""Unit tests for build_block_index and find_intersection_gcps.

Test data: testdata/chicago_p29n_centerlines.geojson — cook-county-centerlines.geojson
clipped to the bounding box of Chicago p29n (Streeterville / Lake Shore Drive area, 1950).
Eight streets appear: East Erie, East Grand Avenue, East Illinois, East Ohio, East Ontario,
North Lake Shore Drive, North McClurg Court, North Peshtigo Court.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mapsnap.georef_from_labels import LabelFeature, find_intersection_gcps
from mapsnap.streets import build_block_index

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TESTDATA = Path(__file__).parent.parent / "testdata"


def _load_p29n() -> dict:
    return json.loads((TESTDATA / "chicago_p29n_centerlines.geojson").read_text())


def _make_geojson(*name_and_lines: tuple[str, list[list[float]]]) -> dict:
    """Build a minimal GeoJSON FeatureCollection from (street_name, coords) pairs."""
    features = [
        {
            "type": "Feature",
            "properties": {"street_name": name},
            "geometry": {"type": "LineString", "coordinates": coords},
        }
        for name, coords in name_and_lines
    ]
    return {"type": "FeatureCollection", "features": features}


def _feat(text: str, cx: float, cy: float, dir_pix: float) -> LabelFeature:
    """Build a LabelFeature with canonical text, pixel center, and label direction."""
    return LabelFeature(
        raw_text=text,
        text=text,
        center=(cx, cy),
        dir_pix=dir_pix,
        long_side=200.0,
        short_side=20.0,
    )


EW = 0.0  # dir_pix for an east-west label (horizontal)
NS = math.pi / 2  # dir_pix for a north-south label (vertical)


# ---------------------------------------------------------------------------
# build_block_index — basic construction
# ---------------------------------------------------------------------------


def test_build_index_all_raw_streets_indexed():
    index = build_block_index(_load_p29n())
    for name in [
        "EAST GRAND AVENUE",
        "NORTH LAKE SHORE DRIVE",
        "NORTH PESHTIGO COURT",
        "EAST OHIO STREET",
        "EAST ILLINOIS STREET",
        "EAST ONTARIO STREET",
        "EAST ERIE STREET",
        "NORTH MCCLURG COURT",
    ]:
        assert name in index, f"{name!r} missing from index"


def test_build_index_block_counts():
    index = build_block_index(_load_p29n())
    assert len(index["EAST GRAND AVENUE"]) == 11
    assert len(index["NORTH LAKE SHORE DRIVE"]) == 4
    assert len(index["NORTH PESHTIGO COURT"]) == 1


def test_build_index_coords_are_numpy_arrays():
    index = build_block_index(_load_p29n())
    for block in index["EAST GRAND AVENUE"]:
        assert isinstance(block.coords, np.ndarray)
        assert block.coords.ndim == 2
        assert block.coords.shape[1] == 2


def test_build_index_street_type_alias_added():
    # e.g. "EAST GRAND AVENUE" → also "EAST GRAND" (type suffix stripped).
    index = build_block_index(_load_p29n())
    assert "EAST GRAND" in index
    assert "EAST OHIO" in index
    assert "NORTH PESHTIGO" in index


def test_build_index_direction_alias_added():
    # e.g. "EAST GRAND AVENUE" → also "GRAND AVENUE" and "GRAND" (direction stripped).
    index = build_block_index(_load_p29n())
    assert "GRAND AVENUE" in index
    assert "GRAND" in index
    assert "OHIO STREET" in index
    assert "OHIO" in index
    assert "LAKE SHORE DRIVE" in index
    assert "LAKE SHORE" in index


def test_build_index_alias_shares_same_block_list():
    # Aliases share the same list object so that id()-based deduplication in
    # process_image correctly collapses them to a single canonical entry.
    index = build_block_index(_load_p29n())
    assert index["GRAND"] is index["EAST GRAND AVENUE"]
    assert index["GRAND AVENUE"] is index["EAST GRAND AVENUE"]
    assert index["EAST GRAND"] is index["EAST GRAND AVENUE"]
    assert index["OHIO"] is index["EAST OHIO STREET"]
    assert index["LAKE SHORE"] is index["NORTH LAKE SHORE DRIVE"]


def test_build_index_ambiguous_bare_name_not_aliased():
    # "MAIN" is the bare name for both "North Main Street" and "South Main Street"
    # → ambiguous, so "MAIN" and "MAIN STREET" must not be added.
    data = _make_geojson(
        ("North Main Street", [[-90.0, 30.0], [-90.0, 30.1]]),
        ("South Main Street", [[-90.0, 29.9], [-90.0, 30.0]]),
    )
    index = build_block_index(data)
    assert "NORTH MAIN STREET" in index
    assert "SOUTH MAIN STREET" in index
    assert "MAIN STREET" not in index
    assert "MAIN" not in index


def test_build_index_unambiguous_bare_name_aliased():
    # "ELM" is unambiguous (only one Elm Street) → alias is added and shares the list.
    data = _make_geojson(("Elm Street", [[-90.0, 30.0], [-90.1, 30.0]]))
    index = build_block_index(data)
    assert "ELM STREET" in index
    assert "ELM" in index
    assert index["ELM"] is index["ELM STREET"]


def test_build_index_empty_geojson():
    assert build_block_index({"type": "FeatureCollection", "features": []}) == {}


def test_build_index_multilinestring():
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"street_name": "Elm Street"},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[-90.0, 30.0], [-90.1, 30.0]],
                        [[-90.2, 30.0], [-90.3, 30.0]],
                    ],
                },
            }
        ],
    }
    index = build_block_index(data)
    assert "ELM STREET" in index
    assert len(index["ELM STREET"]) == 2


# ---------------------------------------------------------------------------
# find_intersection_gcps — basic intersection detection
# ---------------------------------------------------------------------------


def test_find_gcps_empty_features():
    assert find_intersection_gcps([], build_block_index(_load_p29n())) == []


def test_find_gcps_no_matching_streets():
    # Features whose text doesn't match any block_index key produce no GCPs.
    index = build_block_index(_load_p29n())
    feats = [_feat("NONEXISTENT STREET", 500.0, 500.0, EW)]
    assert find_intersection_gcps(feats, index) == []


def test_find_gcps_grand_x_peshtigo():
    # East Grand Avenue (E-W) and North Peshtigo Court (N-S) share one GeoJSON
    # node: (-87.6151976, 41.8918751).  E-W line through (802, 2215) × N-S line
    # through (1354, 2372) = pixel crossing (1354, 2215).
    index = build_block_index(_load_p29n())
    feats = [
        _feat("EAST GRAND AVENUE", 802.0, 2215.0, EW),
        _feat("NORTH PESHTIGO COURT", 1354.0, 2372.0, NS),
    ]
    gcps = find_intersection_gcps(feats, index)
    assert len(gcps) == 1
    gcp = gcps[0]
    assert {gcp.label_a, gcp.label_b} == {"EAST GRAND AVENUE", "NORTH PESHTIGO COURT"}
    assert gcp.geo == pytest.approx((-87.6151976, 41.8918751), abs=1e-6)
    assert gcp.pixel == pytest.approx((1354.0, 2215.0), abs=1e-6)


def test_find_gcps_lake_shore_x_ohio():
    # North Lake Shore Drive (N-S) and East Ohio Street (E-W) share one node:
    # (-87.6145965, 41.8926846).  Pixel crossing: (1883, 1591).  Lake Shore Drive curves
    # only mildly here, so the straight-axis correction (~42 ft) stays below its floor and
    # the raw shared node is kept.
    index = build_block_index(_load_p29n())
    feats = [
        _feat("NORTH LAKE SHORE DRIVE", 1883.0, 1969.0, NS),
        _feat("EAST OHIO STREET", 780.0, 1591.0, EW),
    ]
    gcps = find_intersection_gcps(feats, index)
    assert len(gcps) == 1
    gcp = gcps[0]
    assert {gcp.label_a, gcp.label_b} == {"NORTH LAKE SHORE DRIVE", "EAST OHIO STREET"}
    assert gcp.geo == pytest.approx((-87.6145965, 41.8926846), abs=1e-6)
    assert gcp.pixel == pytest.approx((1883.0, 1591.0), abs=1e-6)


def test_find_gcps_parallel_streets_produce_no_gcp():
    # East Grand Avenue and East Ohio Street run parallel (both E-W) and share
    # no GeoJSON coordinate node.
    index = build_block_index(_load_p29n())
    feats = [
        _feat("EAST GRAND AVENUE", 802.0, 2215.0, EW),
        _feat("EAST OHIO STREET", 780.0, 1591.0, EW),
    ]
    assert find_intersection_gcps(feats, index) == []


def test_curved_junction_straightening_skips_jogs(monkeypatch):
    # A jog — the same street pair crossing twice — must not be straightened: the dominant-axis
    # window would span both branches and fabricate a bogus shift (cf. Brooklyn p26, COURT
    # jogging across BRYANT). straight_intersection_geo is called for the unique MAIN x SINGLE
    # crossing but skipped for the two-node MAIN x JOG pair.
    import mapsnap.georef_from_labels as module

    calls: list[tuple[float, float]] = []
    real = module.straight_intersection_geo

    def spy(geo, blocks_a, blocks_b, **kwargs):
        calls.append(geo)
        return real(geo, blocks_a, blocks_b, **kwargs)

    monkeypatch.setattr(module, "straight_intersection_geo", spy)

    geojson = _make_geojson(
        (
            "MAIN STREET",
            [[-90.002, 30.0], [-90.001, 30.0], [-90.0, 30.0], [-89.999, 30.0]],
        ),
        ("SINGLE STREET", [[-90.0, 30.0], [-90.0, 29.999]]),  # crosses MAIN once
        # Touches MAIN at two nodes ~630 ft apart -> two clusters (a jog).
        (
            "JOG STREET",
            [[-90.001, 30.0], [-90.001, 29.9995], [-89.999, 29.9995], [-89.999, 30.0]],
        ),
    )
    index = build_block_index(geojson)
    feats = [
        _feat("MAIN STREET", 500.0, 500.0, EW),
        _feat("SINGLE STREET", 700.0, 300.0, NS),
        _feat("JOG STREET", 300.0, 300.0, NS),
    ]
    module.find_intersection_gcps(feats, index)
    assert len(calls) == 1  # only the unique crossing was a straightening candidate
    assert calls[0] == pytest.approx((-90.0, 30.0), abs=1e-6)


def test_find_gcps_sorted_by_pixel_dist():
    # Grand×Peshtigo (pixel_dist ≈ 574) comes before Lake Shore×Ohio (≈ 1166).
    index = build_block_index(_load_p29n())
    feats = [
        _feat("EAST GRAND AVENUE", 802.0, 2215.0, EW),
        _feat("NORTH PESHTIGO COURT", 1354.0, 2372.0, NS),
        _feat("NORTH LAKE SHORE DRIVE", 1883.0, 1969.0, NS),
        _feat("EAST OHIO STREET", 780.0, 1591.0, EW),
    ]
    gcps = find_intersection_gcps(feats, index)
    assert len(gcps) == 2
    assert gcps[0].pixel_dist < gcps[1].pixel_dist
    assert {gcps[0].label_a, gcps[0].label_b} == {
        "EAST GRAND AVENUE",
        "NORTH PESHTIGO COURT",
    }


# ---------------------------------------------------------------------------
# find_intersection_gcps — one GCP per (label_a, label_b, cluster)
# ---------------------------------------------------------------------------


def test_find_gcps_three_lake_instances_one_gcp():
    # Three detections of "NORTH LAKE SHORE DRIVE" at different pixel positions.
    # Combined with one EAST OHIO STREET, exactly one GCP must be returned —
    # the (fa, fb) pair with the smallest pixel_dist.
    #
    #   (1883, 1969) × (780, 1591)  pixel_dist ≈ 1166
    #   (1606,  613) × (780, 1591)  pixel_dist ≈ 1280
    #   (1186, 1775) × (780, 1591)  pixel_dist ≈  446  ← selected
    index = build_block_index(_load_p29n())
    feats = [
        _feat("NORTH LAKE SHORE DRIVE", 1883.0, 1969.0, NS),
        _feat("NORTH LAKE SHORE DRIVE", 1606.0, 613.0, NS),
        _feat("NORTH LAKE SHORE DRIVE", 1186.0, 1775.0, NS),
        _feat("EAST OHIO STREET", 780.0, 1591.0, EW),
    ]
    gcps = find_intersection_gcps(feats, index)
    assert len(gcps) == 1
    gcp = gcps[0]
    assert {gcp.label_a, gcp.label_b} == {"NORTH LAKE SHORE DRIVE", "EAST OHIO STREET"}
    # Best crossing: N-S line through (1186, 1775) × E-W line through (780, 1591).
    assert gcp.pixel == pytest.approx((1186.0, 1591.0), abs=1e-6)
    assert gcp.pixel_dist == pytest.approx(445.7, abs=1.0)


def test_find_gcps_two_instances_selects_closer_pair():
    # Two Peshtigo labels: the one closer to the Grand label wins.
    index = build_block_index(_load_p29n())
    feats = [
        _feat("NORTH PESHTIGO COURT", 1354.0, 2372.0, NS),  # pixel_dist ≈  574
        _feat("NORTH PESHTIGO COURT", 1354.0, 500.0, NS),  # pixel_dist ≈ 1716
        _feat("EAST GRAND AVENUE", 802.0, 2215.0, EW),
    ]
    gcps = find_intersection_gcps(feats, index)
    assert len(gcps) == 1
    assert gcps[0].pixel == pytest.approx((1354.0, 2215.0), abs=1e-6)
    assert gcps[0].pixel_dist < 600.0
