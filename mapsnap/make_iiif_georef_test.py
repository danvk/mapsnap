"""Unit tests for make_iiif_georef helpers."""

from pathlib import Path

from shapely.geometry import box

from mapsnap.make_iiif_georef import (
    GcpPoint,
    _load_oim_index,
    _service_url_to_page_key,
    expand_georef_globs,
    fill_missing_source_ids,
    georef_gcp_points,
    georef_path_to_page_key,
    make_annotation,
)
from mapsnap.split import write_panels_json

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


_DC = "https://tile.loc.gov/image-services/iiif/service:gmd:gmd385m:g3851m:g3851gm:g01227003"


def test_service_url_sb_format():
    # Washington DC 1916 uses sb-format: sb{5-digit page}{suffix char}
    assert _service_url_to_page_key(f"{_DC}:sb001250") == "p125"
    assert _service_url_to_page_key(f"{_DC}:sb002160") == "p216"
    assert _service_url_to_page_key(f"{_DC}:sb00154s") == "p154s"
    assert _service_url_to_page_key(f"{_DC}:sb00001a") == "p1a"


# ---------------------------------------------------------------------------
# georef_path_to_page_key
# ---------------------------------------------------------------------------


def test_georef_path_simple():
    assert georef_path_to_page_key("data/vol/p16.georef.json") == "p16"


def test_georef_path_with_direction_suffix():
    assert georef_path_to_page_key("data/vol/p16s.georef.json") == "p16s"


def test_georef_path_with_left_right_suffix():
    assert georef_path_to_page_key("data/vol/p10L.georef.json") == "p10l"
    assert georef_path_to_page_key("data/vol/p10R.georef.json") == "p10r"


def test_georef_path_left_right_suffix_zero():
    assert georef_path_to_page_key("data/vol/p0L.georef.json") == "p0l"


def test_georef_path_left_right_split_page():
    assert georef_path_to_page_key("data/vol/p4L__2.georef.json") == "p4l__2"


def test_georef_path_with_underscore_prefix():
    assert georef_path_to_page_key("data/vol/chicago_p428.georef.json") == "p428"


def test_georef_path_with_gcps_infix():
    assert georef_path_to_page_key("data/vol/p16s.gcps.georef.json") == "p16s"


def test_georef_path_with_sequence_letter_suffix():
    # Sanborn sheets run past the directional letters into a/b/c/…; every letter
    # suffix must be kept, not just s/n/e/w/l/r.
    assert georef_path_to_page_key("data/vol/p1499o.georef.json") == "p1499o"
    assert georef_path_to_page_key("data/vol/p1499a.georef.json") == "p1499a"
    assert georef_path_to_page_key("data/vol/p1499q.georef.json") == "p1499q"


def test_georef_path_sequence_letter_split():
    assert georef_path_to_page_key("data/vol/p1499q__2.georef.json") == "p1499q__2"


def test_georef_path_multi_letter_suffix_parses():
    # A compound suffix parses (rather than silently dropping the page);
    # drop_redundant_skeletons is what raises on the ambiguous trailing 's'.
    assert georef_path_to_page_key("data/vol/p6ns.georef.json") == "p6ns"


def test_georef_path_split_page():
    assert georef_path_to_page_key("data/vol/p20__2.georef.json") == "p20__2"


def test_georef_path_split_page_multi_digit():
    assert georef_path_to_page_key("data/vol/p4__10.georef.json") == "p4__10"


def test_georef_path_neighbor_variant():
    assert georef_path_to_page_key("data/vol/p147.georef-neighbor.json") == "p147"
    assert georef_path_to_page_key("data/vol/p16s.georef-neighbor.json") == "p16s"


def test_georef_path_osm_variant():
    assert georef_path_to_page_key("data/vol/p147.georef-osm.json") == "p147"
    assert georef_path_to_page_key("data/vol/p4l__2.georef-osm.json") == "p4l__2"


def test_georef_path_other_variants_do_not_match():
    assert georef_path_to_page_key("data/vol/p16.georef-nofit.json") is None
    assert georef_path_to_page_key("data/vol/p16.georef-misscale.json") is None


def test_expand_georef_globs_first_glob_wins(tmp_path):
    for name in [
        "p1.georef.json",
        "p1.georef-neighbor.json",
        "p2.georef-neighbor.json",
    ]:
        (tmp_path / name).write_text("{}")
    pattern = f"{tmp_path}/p*.georef.json,{tmp_path}/p*.georef-neighbor.json"
    paths = [Path(p).name for p in expand_georef_globs(pattern)]
    assert paths == ["p1.georef.json", "p2.georef-neighbor.json"]


def test_expand_georef_globs_warns_on_unparsable_key(tmp_path, capsys):
    # A file whose name encodes no page key must not vanish silently: it is
    # skipped, but with a warning naming the file (regression guard for the
    # suffix bug that once dropped pages with no trace).
    (tmp_path / "p16.georef.json").write_text("{}")
    (tmp_path / "key.georef.json").write_text("{}")
    paths = [Path(p).name for p in expand_georef_globs(f"{tmp_path}/*.georef.json")]
    assert paths == ["p16.georef.json"]
    err = capsys.readouterr().err
    assert "could not parse" in err
    assert "key.georef.json" in err


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


def test_load_oim_index_split_label_keys_by_parent():
    # Split labels are keyed by the unsplit parent page; the "[N]" suffix is dropped.
    data = {"items": [_make_oim_item(_BASE_URL, "Page 6 [1]")]}
    index = _load_oim_index(data)
    assert list(index.keys()) == ["p6n"]


def test_load_oim_index_splits_share_one_parent_entry():
    # Both halves of a split page collapse to a single parent canvas entry.
    data = {
        "items": [
            _make_oim_item(_BASE_URL, "Page 6 [1]"),
            _make_oim_item(_BASE_URL, "Page 6 [2]"),
        ]
    }
    index = _load_oim_index(data)
    assert list(index.keys()) == ["p6n"]


def test_load_oim_index_null_source_id_falls_back_to_label():
    # Some OIM volumes (e.g. Grand Rapids 1953 vol 7) carry a null source.id; the page key
    # must then come from the label's trailing "pNNN" token.
    data = {
        "items": [
            {
                "label": "Grand Rapids, Mich. | 1953 | Vol. 7 p714",
                "target": {"source": {"id": None, "width": 6660, "height": 8070}},
            }
        ]
    }
    index = _load_oim_index(data)
    assert list(index.keys()) == ["p714"]


def test_load_oim_index_null_source_id_splits_key_by_parent():
    # Null-source split labels collapse to one parent-keyed entry, matching URL-keyed behavior.
    data = {
        "items": [
            {
                "label": "Grand Rapids, Mich. | 1953 | Vol. 7 p721 [1]",
                "target": {"source": {"id": None}},
            },
            {
                "label": "Grand Rapids, Mich. | 1953 | Vol. 7 p721 [2]",
                "target": {"source": {"id": None}},
            },
        ]
    }
    index = _load_oim_index(data)
    assert list(index.keys()) == ["p721"]


def test_load_oim_index_skips_item_with_no_key():
    # No source id and an unparseable label -> item is skipped, not a crash.
    data = {"items": [{"label": "cover", "target": {"source": {"id": None}}}]}
    assert _load_oim_index(data) == {}


def _metadata_value(annotation: dict, label: str) -> str | None:
    for entry in annotation["metadata"]:
        if entry["label"] == label:
            return entry["value"]
    return None


def test_make_annotation_split_uses_panels_json(tmp_path):
    # Parent page is 200×400 at 25%; the full canvas is 4× larger (800×1600).
    write_panels_json(
        tmp_path / "p4.jpg", [box(10, 20, 60, 120)], width=200, height=400
    )
    item = {
        "label": "P4",
        "target": {
            "source": {
                "id": "http://example/p4/info.json",
                "type": "ImageService3",
                "width": 800,
                "height": 1600,
            }
        },
    }
    georef = make_georef(width=50, height=100, intersections=[])

    annotation = make_annotation(
        item,
        georef,
        "p4__1",
        tmp_path / "p4__1.jpg",
        creator_url="http://example/me",
        now="2026-01-01T00:00:00Z",
    )

    # Panel bbox (10,20)-(60,120) in the 25% frame scales ×4 to the full canvas.
    assert _metadata_value(annotation, "split_canvas_x") == "40.0"
    assert _metadata_value(annotation, "split_canvas_y") == "80.0"
    assert _metadata_value(annotation, "split_canvas_w") == "200.0"
    assert _metadata_value(annotation, "split_canvas_h") == "400.0"
    assert annotation["id"] == "http://example/p4__1/georef"
    # The first corner GCP (pixel 0,0) maps to the panel's top-left on the canvas.
    assert annotation["body"]["features"][0]["properties"]["resourceCoords"] == [
        40.0,
        80.0,
    ]


def test_make_annotation_null_source_id_uses_item_id(tmp_path):
    # A null source.id (OIM annotation with no linked image service): the canvas id falls
    # back to the item's own id (trailing slash trimmed) so annotation ids stay unique.
    item = {
        "id": "https://oldinsurancemaps.net/iiif/resource/54270/",
        "label": "Grand Rapids, Mich. | 1953 | Vol. 7 p703",
        "target": {
            "id": "https://oldinsurancemaps.net/iiif/selector/54270/",
            "source": {
                "id": None,
                "type": "ImageService2",
                "width": 800,
                "height": 1600,
            },
        },
    }
    georef = make_georef(width=50, height=100, intersections=[])
    annotation = make_annotation(
        item, georef, "p703", tmp_path / "p703.jpg", "http://x", "now"
    )
    assert annotation["id"] == "https://oldinsurancemaps.net/iiif/resource/54270/georef"
    assert annotation["target"]["source"]["id"] is None


def test_make_annotation_split_missing_panels_raises(tmp_path):
    item = {
        "label": "P4",
        "target": {
            "source": {
                "id": "http://example/p4/info.json",
                "width": 800,
                "height": 1600,
            }
        },
    }
    georef = make_georef(width=50, height=100, intersections=[])
    try:
        make_annotation(
            item, georef, "p4__1", tmp_path / "p4__1.jpg", "http://x", "now"
        )
    except ValueError as exc:
        assert "panels.json" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing panels.json")


def test_load_oim_index_non_sheet_skipped():
    cover_url = (
        "https://tile.loc.gov/image-services/iiif/service:gmd:...-covr/info.json"
    )
    data = {"items": [_make_oim_item(cover_url, "Cover")]}
    assert _load_oim_index(data) == {}


def _item_with_id(source_id):
    return {"target": {"source": {"id": source_id, "width": 6660, "height": 8070}}}


def test_fill_missing_source_ids_extrapolates_loc_pattern():
    prefix = "https://tile.loc.gov/image-services/iiif/service:gmd:g04023195307:04023_07_1953"
    index = {
        "p715": _item_with_id(f"{prefix}-0715"),
        "p712": _item_with_id(f"{prefix}-0712"),
        "p714": _item_with_id(None),
        "p6n": _item_with_id(None),
    }
    fill_missing_source_ids(index)
    assert index["p714"]["target"]["source"]["id"] == f"{prefix}-0714"
    assert index["p6n"]["target"]["source"]["id"] == f"{prefix}-0006N"


def test_fill_missing_source_ids_requires_one_unambiguous_pattern():
    index = {
        "p1": _item_with_id("https://x/a-0001"),
        "p2": _item_with_id("https://y/b-0002"),
        "p3": _item_with_id(None),
    }
    fill_missing_source_ids(index)
    assert index["p3"]["target"]["source"]["id"] is None


def test_fill_missing_source_ids_skips_sb_format():
    index = {
        "p125": _item_with_id("https://tile.loc.gov/x/g01227003:sb001250"),
        "p126": _item_with_id(None),
    }
    fill_missing_source_ids(index)
    assert index["p126"]["target"]["source"]["id"] is None
