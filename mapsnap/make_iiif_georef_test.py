"""Unit tests for make_iiif_georef helpers."""

from mapsnap.make_iiif_georef import (
    GcpPoint,
    _load_oim_index,
    _service_url_to_page_key,
    georef_gcp_points,
    georef_path_to_page_key,
)

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


def pixels(pts: list[GcpPoint]) -> list[tuple[float, float]]:
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
    assert all(p[2] == "corner" for p in pts)


def test_one_initial_returns_corners_plus_initial():
    # One initial: 4 corners + 1 initial GCP.
    georef = make_georef(
        2000, 2000, [make_intersection("A", "B", 100, 500, initial=True)]
    )
    pts = georef_gcp_points(georef)
    assert len(pts) == 5
    assert all(p[2] == "corner" for p in pts[:4])
    assert pts[4][0] == (100.0, 500.0)
    assert pts[4][2] == "gcp"


def test_deferred_single_gcp_includes_inlier():
    # Deferred image: one inlier intersection with initial=False (as written by
    # process_deferred_image). Should appear as "gcp" alongside the four corners.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection(
                "A", "B", 183, 1953, lon=-82.97, lat=42.37, inlier=True, initial=False
            )
        ],
    )
    pts = georef_gcp_points(georef)
    assert len(pts) == 5
    assert all(p[2] == "corner" for p in pts[:4])
    assert pts[4][0] == (183.0, 1953.0)
    assert pts[4][2] == "gcp"


def test_coincident_initials_returns_corners():
    # Both initial intersections at the same pixel → degenerate, fall back to corners
    # plus both initial GCPs.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 500, 500, initial=True),
            make_intersection("C", "D", 500, 500, initial=True),
        ],
    )
    pts = georef_gcp_points(georef)
    assert len(pts) == 6
    assert all(p[2] == "corner" for p in pts[:4])
    assert all(p[2] == "gcp" for p in pts[4:])


# ---------------------------------------------------------------------------
# Two non-coincident initials → exactly 2 GCPs, no synthetic third point
# ---------------------------------------------------------------------------


def test_two_initials_returns_exactly_two_gcps():
    # Normal case: two non-coincident initials → exactly those two, no third point.
    georef = make_georef(
        2000,
        2000,
        [
            make_intersection("A", "B", 100, 500, lon=-90.0, lat=30.1, initial=True),
            make_intersection("C", "D", 900, 500, lon=-89.9, lat=30.1, initial=True),
            make_intersection("A", "C", 500, 900),  # extra intersection — ignored
        ],
    )
    pts = georef_gcp_points(georef)
    assert len(pts) == 2
    assert pts[0] == ((100.0, 500.0), (-90.0, 30.1), "gcp")
    assert pts[1] == ((900.0, 500.0), (-89.9, 30.1), "gcp")


def test_two_initials_collinear_case_still_two_gcps():
    # Extra intersections all collinear with initials — no third point is synthesised;
    # only the two initials are returned.
    georef = make_georef(
        2012,
        2476,
        [
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
                "ASHLAND STREET",
                1532,
                929,
                lon=-82.933654,
                lat=42.464666,
                inlier=True,
                initial=True,
            ),
            make_intersection(
                "KORTE AVENUE", "MANISTIQUE STREET", 937, 929, inlier=True
            ),
        ],
    )
    pts = georef_gcp_points(georef)
    assert len(pts) == 2
    assert pts[0][0] == (341.0, 929.0)
    assert pts[1][0] == (1532.0, 929.0)


# ---------------------------------------------------------------------------
# _service_url_to_page_key
# ---------------------------------------------------------------------------


def test_service_url_oim_with_info_json():
    # OIM format: /info.json suffix is stripped before parsing.
    url = "https://tile.loc.gov/image-services/iiif/service:gmd:g4104cm:g01790195001N:01790_01N_1950-0006N/info.json"
    assert _service_url_to_page_key(url) == "p6n"


def test_service_url_loc_no_info_json():
    # LOC manifest format: no /info.json suffix.
    url = "https://tile.loc.gov/image-services/iiif/service:gmd:g4104cm:g01790195001N:01790_01N_1950-0006N"
    assert _service_url_to_page_key(url) == "p6n"


def test_service_url_large_page_number():
    assert _service_url_to_page_key("...:01790_01N_1950-0103W") == "p103w"


def test_service_url_lowercase_suffix():
    # Brooklyn-style: lowercase sequential letter suffix.
    assert _service_url_to_page_key("...:05791_02_1939-0027s") == "p27s"


def test_service_url_no_suffix_letter():
    assert _service_url_to_page_key("...:01790_01N_1950-0050") == "p50"


def test_service_url_strips_leading_zeros():
    assert _service_url_to_page_key("...-0001N") == "p1n"


def test_service_url_non_sheet_returns_none():
    # Covers, indexes, title pages start with a letter after "-" — not a page number.
    assert _service_url_to_page_key("...-covr") is None
    assert _service_url_to_page_key("...-titl") is None


# ---------------------------------------------------------------------------
# georef_path_to_page_key
# ---------------------------------------------------------------------------


def test_georef_path_simple():
    assert georef_path_to_page_key("data/vol/p16.georef.json") == "p16"


def test_georef_path_with_direction_suffix():
    assert georef_path_to_page_key("data/vol/p16s.georef.json") == "p16s"


def test_georef_path_with_underscore_prefix():
    assert georef_path_to_page_key("data/vol/chicago_p428.georef.json") == "p428"


def test_georef_path_with_gcps_infix():
    assert georef_path_to_page_key("data/vol/p16s.gcps.georef.json") == "p16s"


def test_georef_path_split_page():
    assert georef_path_to_page_key("data/vol/p20__2.georef.json") == "p20__2"


def test_georef_path_split_page_multi_digit():
    assert georef_path_to_page_key("data/vol/p4__10.georef.json") == "p4__10"


def test_georef_path_no_match():
    assert georef_path_to_page_key("data/vol/streets.json") is None
    assert georef_path_to_page_key("data/vol/p16.streets.json") is None


# ---------------------------------------------------------------------------
# _load_oim_index
# ---------------------------------------------------------------------------

_BASE_URL = "https://tile.loc.gov/image-services/iiif/service:gmd:g4104cm:g01790195001N:01790_01N_1950-0006N/info.json"


def _make_oim_item(url: str, label: str) -> dict:
    return {
        "label": label,
        "target": {"source": {"id": url}},
    }


def test_load_oim_index_simple():
    data = {"items": [_make_oim_item(_BASE_URL, "Page 6")]}
    index = _load_oim_index(data)
    assert list(index.keys()) == ["p6n"]


def test_load_oim_index_split_label():
    # Label " [1]" suffix → page key gets "__1" appended.
    data = {"items": [_make_oim_item(_BASE_URL, "Page 6 [1]")]}
    index = _load_oim_index(data)
    assert list(index.keys()) == ["p6n__1"]


def test_load_oim_index_split_label_second_half():
    data = {"items": [_make_oim_item(_BASE_URL, "Page 6 [2]")]}
    index = _load_oim_index(data)
    assert list(index.keys()) == ["p6n__2"]


def test_load_oim_index_non_sheet_skipped():
    cover_url = (
        "https://tile.loc.gov/image-services/iiif/service:gmd:...-covr/info.json"
    )
    data = {"items": [_make_oim_item(cover_url, "Cover")]}
    assert _load_oim_index(data) == {}
