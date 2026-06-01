"""Tests for osm_to_centerlines helpers."""

from mapsnap.osm_to_centerlines import _cluster_coords, compute_street_intersections

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
