"""Unit tests for georef_gcp_points."""

import numpy as np
import pytest

from mapsnap.make_iiif_georef import _two_gcp_affine, georef_gcp_points

GCP = tuple[tuple[float, float], tuple[float, float]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CORNERS = [[-90.0, 30.1], [-89.9, 30.1], [-89.9, 30.0], [-90.0, 30.0]]


def make_georef(
    width: int,
    height: int,
    intersections: list[dict],
    corners: list | None = None,
) -> dict:
    return {
        "width": width,
        "height": height,
        "corners": corners if corners is not None else _CORNERS,
        "intersections": intersections,
    }


def make_intersection(
    label_a: str,
    label_b: str,
    x: float,
    y: float,
    *,
    lon: float = -90.0,
    lat: float = 30.0,
    inlier: bool = True,
    initial: bool = False,
) -> dict:
    return {
        "label_a": label_a,
        "label_b": label_b,
        "x": x,
        "y": y,
        "lon": lon,
        "lat": lat,
        "inlier": inlier,
        "initial": initial,
    }


def pixels(pts: list[GCP]) -> list[tuple[float, float]]:
    return [pt[0] for pt in pts]


# ---------------------------------------------------------------------------
# Fallback to corners
# ---------------------------------------------------------------------------


def test_no_initials_returns_corners():
    georef = make_georef(2000, 2000, [])
    pts = georef_gcp_points(georef)
    assert len(pts) == 4
    assert pts[0][0] == (0.0, 0.0)
    assert pts[1][0] == (2000.0, 0.0)
    assert pts[2][0] == (2000.0, 2000.0)
    assert pts[3][0] == (0.0, 2000.0)


def test_one_initial_returns_corners():
    georef = make_georef(
        2000, 2000, [make_intersection("A", "B", 100, 500, initial=True)]
    )
    assert len(georef_gcp_points(georef)) == 4


def test_coincident_initials_returns_corners():
    # Both initial intersections at the same pixel → degenerate, fall back to corners.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 500, 500, initial=True),
            make_intersection("C", "D", 500, 500, initial=True),
        ],
    )
    assert len(georef_gcp_points(georef)) == 4


# ---------------------------------------------------------------------------
# First two GCPs always match the initial intersections
# ---------------------------------------------------------------------------


def test_first_two_gcps_are_initials():
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 100, 500, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 500, lon=-89.9, lat=30.1, initial=True),
        ],
    )
    pts = georef_gcp_points(georef)
    assert pts[0] == ((100.0, 500.0), (-90.0, 30.1))
    assert pts[1] == ((900.0, 500.0), (-89.9, 30.1))


# ---------------------------------------------------------------------------
# Option 1 — cross-intersection from streets already in the initial pair set
# ---------------------------------------------------------------------------


def test_option1_uses_cross_intersection():
    # P1: A×B at (100,500). P2: C×D at (900,500). A×C at (500,900) shares one
    # street from each initial and is well off the P1-P2 line.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 100, 500, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 500, lon=-89.9, lat=30.1, initial=True),
            make_intersection("A", "C", 500, 900),
        ],
    )
    pts = georef_gcp_points(georef)
    assert len(pts) == 3
    assert pts[2][0] == (500.0, 900.0)


def test_option1_picks_candidate_closest_to_center():
    # Two valid option-1 candidates; the one closer to the image center wins.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 100, 500, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 500, lon=-89.9, lat=30.1, initial=True),
            make_intersection("A", "C", 200, 1800),  # far from center (1000,1000)
            make_intersection("B", "D", 700, 1200),  # closer to center
        ],
    )
    pts = georef_gcp_points(georef)
    assert pts[2][0] == (700.0, 1200.0)


def test_option1_collinear_alias_duplicate_rejected():
    # Reproduces the KORTE bug: "KORTE AVENUE" and "KORTE STREET" are aliases for
    # the same physical street, so "KORTE AVENUE×PHILIP STREET" appears as an
    # option-1 candidate at the exact same pixel as the first initial GCP.
    # All intersections share y=929 (fully collinear), so option 3 must be used.
    georef = make_georef(
        2012,
        2476,
        [
            make_intersection(
                "KORTE AVENUE", "MANISTIQUE STREET", 937, 929, inlier=True
            ),
            make_intersection(
                "KORTE STREET",
                "PHILIP STREET",
                341,
                929,
                lon=-82.935707,
                lat=42.36392,
                inlier=True,
                initial=True,
            ),
            make_intersection(
                "KORTE AVENUE",
                "PHILIP STREET",
                341,
                929,
                lon=-82.935707,
                lat=42.36392,
                inlier=True,
            ),
            make_intersection(
                "KORTE AVENUE",
                "ASHLAND STREET",
                1532,
                929,
                lon=-82.933654,
                lat=42.464666,
                inlier=True,
                initial=True,
            ),
        ],
    )
    pts = georef_gcp_points(georef)
    assert len(pts) == 3
    # The third GCP must not duplicate either initial GCP's pixel position.
    assert pts[2][0] != (341.0, 929.0)
    assert pts[2][0] != (1532.0, 929.0)
    # All three pixel positions must be distinct.
    assert len(set(pixels(pts))) == 3


# ---------------------------------------------------------------------------
# Option 2 — any other intersection (non-collinear, inliers preferred)
# ---------------------------------------------------------------------------


def test_option2_inlier_preferred_over_non_inlier():
    # No option-1 candidates (E, F, G, H not in streets_set = {A,B,C,D}).
    # Inlier at (200,200) is farther from center than non-inlier at (500,200),
    # but inliers are ranked first.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 100, 500, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 500, lon=-89.9, lat=30.1, initial=True),
            make_intersection("E", "F", 500, 200, inlier=False),
            make_intersection("G", "H", 200, 200, inlier=True),
        ],
    )
    pts = georef_gcp_points(georef)
    assert pts[2][0] == (200.0, 200.0)


def test_option2_collinear_candidate_rejected():
    # E×F lies on the P1-P2 line (y=500). It must be rejected; G×H at y=200 passes.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 100, 500, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 500, lon=-89.9, lat=30.1, initial=True),
            make_intersection("E", "F", 500, 500, inlier=True),  # collinear
            make_intersection("G", "H", 500, 200, inlier=True),  # non-collinear
        ],
    )
    pts = georef_gcp_points(georef)
    assert pts[2][0] == (500.0, 200.0)


# ---------------------------------------------------------------------------
# Option 3 — perpendicular offset from the P1-P2 midpoint
# ---------------------------------------------------------------------------


def test_option3_used_when_all_candidates_collinear():
    # Only intersection other than initials is on the P1-P2 line.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 100, 500, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 500, lon=-89.9, lat=30.1, initial=True),
            make_intersection("E", "F", 500, 500),  # collinear
        ],
    )
    pts = georef_gcp_points(georef)
    p3x, p3y = pts[2][0]
    # Midpoint is (500, 500); perpendicular offset gives x≈500, y≠500.
    assert abs(p3x - 500.0) < 1.0
    assert p3y != pytest.approx(500.0)


def test_option3_picks_side_closer_to_image_center():
    # P1-P2 horizontal at y=900 in a 1000×1000 image; center=(500,500).
    # Perp offset goes to y=1700 or y=100; y=100 is closer to center.
    georef = make_georef(
        1000,
        1000,
        [
            make_intersection("A", "B", 100, 900, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 900, lon=-89.9, lat=30.1, initial=True),
        ],
    )
    pts = georef_gcp_points(georef)
    _, p3y = pts[2][0]
    assert p3y < 900.0  # side above P1-P2, toward center at y=500


# ---------------------------------------------------------------------------
# Third GCP geo comes from the 2-GCP affine, not the intersection's stored coords
# ---------------------------------------------------------------------------


def test_third_gcp_geo_is_projected_not_raw():
    # The option-1 intersection has obviously bogus stored geo (-99, 99);
    # the returned geo must come from the 2-GCP affine projection instead.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 100, 500, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 500, lon=-89.9, lat=30.1, initial=True),
            make_intersection("A", "C", 500, 900, lon=-99.0, lat=99.0),
        ],
    )
    pts = georef_gcp_points(georef)
    _, geo3 = pts[2]
    assert geo3 != (-99.0, 99.0)
    # Geo should be consistent with the 2-GCP affine applied to (500, 900).
    A = _two_gcp_affine((100.0, 500.0), (-90.0, 30.1), (900.0, 500.0), (-89.9, 30.1))
    expected = A @ np.array([500.0, 900.0, 1.0])
    assert geo3[0] == pytest.approx(expected[0], abs=1e-7)
    assert geo3[1] == pytest.approx(expected[1], abs=1e-7)
