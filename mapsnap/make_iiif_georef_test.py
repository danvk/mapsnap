"""Unit tests for make_iiif_georef helpers."""

import json
from pathlib import Path

from shapely.geometry import box

from mapsnap.make_iiif_georef import (
    GcpPoint,
    _load_oim_index,
    _service_url_to_page_key,
    drop_redundant_skeletons,
    expand_georef_globs,
    fill_missing_source_ids,
    georef_gcp_points,
    georef_path_to_page_key,
    glob_matched_anything,
    make_annotation,
    own_label,
)
from mapsnap.split import write_panels_json

# A minimal sidecar that carries a pose (expand_georef_globs skips poseless ones).
_POSED = {"corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}

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
    assert georef_path_to_page_key("data/vol/p147.georef-snap.json") == "p147"
    assert georef_path_to_page_key("data/vol/p4l__2.georef-snap.json") == "p4l__2"
    # A variant absent from the pattern yields no page key, so the glob matches the
    # file and the annotation silently omits the page -- worth a test per variant.
    assert georef_path_to_page_key("data/vol/p147.georef-street.json") == "p147"
    assert georef_path_to_page_key("data/vol/p4l__2.georef-street.json") == "p4l__2"


def test_georef_path_parses_any_variant():
    # The variant list used to be a whitelist, so renaming a channel silently
    # dropped every page of it. Parsing is now general; what must not be
    # published is decided by the sidecar's CONTENT (has_pose), not its name.
    assert georef_path_to_page_key("data/vol/p16.georef-final.json") == "p16"
    assert georef_path_to_page_key("data/vol/p16.georef-nofit.json") == "p16"
    assert georef_path_to_page_key("data/vol/p16.georef-misscale.json") == "p16"
    assert georef_path_to_page_key("data/vol/p16.notgeoref.json") is None


def test_expand_globs_skips_and_claims_poseless_sidecars(tmp_path):
    """A poseless sidecar leaves the page unplaced AND blocks later globs."""
    (tmp_path / "p1.georef-final.json").write_text(json.dumps({"corners": None}))
    (tmp_path / "p1.georef.json").write_text(
        json.dumps({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    )
    (tmp_path / "p2.georef-final.json").write_text(
        json.dumps({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    )
    pattern = f"{tmp_path}/p*.georef-final.json,{tmp_path}/p*.georef.json"
    assert [Path(p).name for p in expand_georef_globs(pattern)] == [
        "p2.georef-final.json"
    ]


def test_expand_georef_globs_first_glob_wins(tmp_path):
    for name in [
        "p1.georef.json",
        "p1.georef-neighbor.json",
        "p2.georef-neighbor.json",
    ]:
        (tmp_path / name).write_text(json.dumps(_POSED))
    pattern = f"{tmp_path}/p*.georef.json,{tmp_path}/p*.georef-neighbor.json"
    paths = [Path(p).name for p in expand_georef_globs(pattern)]
    assert paths == ["p1.georef.json", "p2.georef-neighbor.json"]


def test_expand_georef_globs_warns_on_unparsable_key(tmp_path, capsys):
    # A file whose name encodes no page key must not vanish silently: it is
    # skipped, but with a warning naming the file (regression guard for the
    # suffix bug that once dropped pages with no trace).
    (tmp_path / "p16.georef.json").write_text(json.dumps(_POSED))
    (tmp_path / "key.georef.json").write_text(json.dumps(_POSED))
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


def test_own_label_carries_our_split_marker_not_the_reference_items():
    """The label says what the id says: our panel number, or nothing (#343, #306).

    The reference index keeps one OIM item per parent page (last one wins), so
    copying its label verbatim leaked "[2]" onto unsplit placements and onto
    both of our own panels.
    """
    reference = "Fargo, N.D. | 1958 p10 [2]"
    # We fit the whole sheet: no marker at all, whatever the reference carried.
    assert own_label(reference, None) == "Fargo, N.D. | 1958 p10"
    # We fit panel 1: OUR number, not the reference item's.
    assert own_label(reference, 1) == "Fargo, N.D. | 1958 p10 [1]"
    # A reference with no marker still gains ours when we split the sheet.
    assert own_label("Fargo, N.D. | 1958 p10", 2) == "Fargo, N.D. | 1958 p10 [2]"
    # Untouched when nothing needs changing (the common whole-page case).
    assert own_label("Chicago, Ill. | 1950 | Vol. 1 p16N", None) == (
        "Chicago, Ill. | 1950 | Vol. 1 p16N"
    )
    # Stray whitespace around the marker does not survive into the output.
    assert own_label("Page 6  [1] ", None) == "Page 6"


def test_make_annotation_labels_match_their_ids(tmp_path):
    """Label marker and id suffix never disagree — the ambiguity compare had to
    heuristically resolve (annotation_is_own_output) is gone at the source."""
    write_panels_json(
        tmp_path / "p4.jpg",
        [box(0, 0, 200, 200), box(0, 200, 200, 400)],
        width=200,
        height=400,
    )
    item = {
        "label": "Test | 1900 p4 [2]",  # last-wins reference label
        "target": {
            "source": {
                "id": "http://example/p4/info.json",
                "type": "ImageService3",
                "width": 800,
                "height": 1600,
            }
        },
    }
    georef = make_georef(width=200, height=200, intersections=[])
    panel_one = make_annotation(
        item, georef, "p4__1", tmp_path / "p4__1.jpg", "http://x", "now"
    )
    assert panel_one["id"].endswith("p4__1/georef")
    assert panel_one["label"] == "Test | 1900 p4 [1]"
    whole = make_annotation(
        item,
        make_georef(width=200, height=400, intersections=[]),
        "p4",
        tmp_path / "p4.jpg",
        "http://x",
        "now",
    )
    assert whole["id"].endswith("p4/georef")
    assert whole["label"] == "Test | 1900 p4"


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


# --- the mirror's metadata.json as a reference (#354) --------------------------


def _metadata(**overrides) -> dict:
    data = {
        "item": "sanborn01971_002",
        "loc_url": "https://www.loc.gov/item/sanborn01971_002/",
        "state": "illinois",
        "year": "1893",
        "city": "lewistown",
        "storage_dir": "gmd/gmd410m/g4104m/g4104lm/g019711893",
        "sheets": [
            {
                "seq": 1,
                "stem": "01971_1893-0001",
                "key": "p1",
                "storage_dir": "gmd/gmd410m/g4104m/g4104lm/g019711893",
                "width": 1613,
                "height": 1913,
            }
        ],
    }
    data.update(overrides)
    return data


def test_loc_service_id_matches_the_manifest_form() -> None:
    from mapsnap.make_iiif_georef import loc_service_id

    assert loc_service_id(
        "gmd/gmd410m/g4104m/g4104lm/g019711893", "01971_1893-0001"
    ) == (
        "https://tile.loc.gov/image-services/iiif/service"
        ":gmd:gmd410m:g4104m:g4104lm:g019711893:01971_1893-0001"
    )
    # A stray leading or trailing slash must not produce an empty segment.
    assert loc_service_id("/gmd/x/", "stem").endswith(":gmd:x:stem")


def test_metadata_index_points_at_loc_with_a_full_res_canvas() -> None:
    from mapsnap.make_iiif_georef import _load_metadata_index

    index = _load_metadata_index(_metadata())
    assert list(index) == ["p1"]
    source = index["p1"]["target"]["source"]
    assert source["id"].startswith("https://tile.loc.gov/image-services/iiif/service:")
    assert source["id"].endswith("01971_1893-0001/info.json")
    assert source["type"] == "ImageService2"
    assert (source["width"], source["height"]) == (1613 * 4, 1913 * 4)
    assert "Lewistown" in index["p1"]["label"]


def test_metadata_index_keeps_an_uppercase_page_suffix() -> None:
    """10,882 corpus sheets are keyed p5S; parsing the URL back would lowercase it,
    and the georef sidecars are named in the mirror's case."""
    from mapsnap.make_iiif_georef import _load_metadata_index, _service_url_to_page_key

    data = _metadata()
    data["sheets"][0]["key"] = "p5S"
    data["sheets"][0]["stem"] = "00015_01_1951-0005S"
    index = _load_metadata_index(data)
    assert "p5S" in index, "the mirror's own casing, which names the sidecars"
    # The URL parser is where the case would have been lost.
    assert _service_url_to_page_key(index["p5S"]["target"]["source"]["id"]) == "p5s"
    # ...and that lowercased form is exactly what a georef path parses to, so
    # the index answers it too and resolves to the same sheet. Keeping only the
    # mirror's case published Chicago 1950 vol 1 with no annotations at all: it
    # fitted 108 of 123 pages and every one of its sheets is lettered.
    assert index["p5s"] is index["p5S"]


def test_metadata_index_skips_a_sheet_missing_any_field() -> None:
    from mapsnap.make_iiif_georef import _load_metadata_index

    data = _metadata()
    data["sheets"] += [
        {"seq": 2, "stem": "x", "key": "p2", "width": 10},  # no height
        {"seq": 3, "key": "p3", "width": 10, "height": 10},  # no stem
    ]
    assert list(_load_metadata_index(data)) == ["p1"]


def test_metadata_index_falls_back_to_the_item_storage_dir() -> None:
    from mapsnap.make_iiif_georef import _load_metadata_index

    data = _metadata()
    del data["sheets"][0]["storage_dir"]
    assert (
        "g019711893:01971_1893-0001"
        in _load_metadata_index(data)["p1"]["target"]["source"]["id"]
    )


# --- the page's own label and report card (#354) ------------------------------


def test_volume_label_from_each_reference_shape() -> None:
    from mapsnap.make_iiif_georef import volume_label

    assert volume_label(_metadata(city="madison", state="indiana", year="1904")) == (
        "Madison, Indiana | 1904"
    )
    assert volume_label({"label": "Sanborn ... Columbus, Ohio."}).startswith("Sanborn")
    assert volume_label({}) == ""


def _provenance(stem, decision, source, *, panels=None, tier=0, keymap="georeferenced"):
    return {
        "stem": stem,
        "decision": decision,
        "source": source,
        "panels": panels,
        "hypotheses": [],
        "evidence": {"keymap": keymap},
        "approximate": None
        if decision == "superseded"
        else {"tier": tier, "basis": "b", "lonlat": [-85.4, 38.7], "radius_m": 100.0},
    }


def test_volume_report_counts_and_names_the_abstentions(tmp_path) -> None:
    """An unplaced page cannot be an item -- a georeference annotation's body is
    its control points -- so the page-level metadata is where it is recorded."""
    import json

    from mapsnap.make_iiif_georef import volume_report

    records = [
        _provenance("p1", "placed", "georef"),
        _provenance("p2", "placed", "georef-snap"),
        _provenance("p3", "abstained", "unplaced", tier=2),
        _provenance("p4", "superseded", "unplaced", panels=["p4__1", "p4__2"]),
        _provenance("p4__1", "placed", "georef"),
        _provenance("p4__2", "abstained", "unplaced", tier=1),
    ]
    for r in records:
        (tmp_path / f"{r['stem']}.provenance.json").write_text(json.dumps(r))
    report = {m["label"]: m["value"] for m in volume_report(tmp_path, "2026-09-16")}
    assert report["generated"] == "2026-09-16"
    assert report["pages"] == "5"  # the superseded parent is not a page to place
    assert report["placed"] == "3"
    assert report["unplaced"] == "2"
    assert report["fit sources"] == "georef 2, georef-snap 1"
    assert report["key map"] == "georeferenced"
    assert report["split sheets"] == "1 sheet(s) cut into 2 panels"
    assert "p3 (tier 2," in report["unplaced pages"]
    assert "p4__2 (tier 1," in report["unplaced pages"]


def test_volume_report_without_records_still_dates_the_run(tmp_path) -> None:
    from mapsnap.make_iiif_georef import volume_report

    assert volume_report(tmp_path, "2026-09-16") == [
        {"label": "generated", "value": "2026-09-16"}
    ]
    assert volume_report(None, "2026-09-16") == []


def test_glob_matched_anything_separates_a_wrong_path_from_an_empty_volume(
    tmp_path,
) -> None:
    """Gardiner NY 1913 placed nothing; that is a result, not a bad glob."""
    from mapsnap.make_iiif_georef import expand_georef_globs, glob_matched_anything

    poseless = tmp_path / "p1.georef-final.json"
    poseless.write_text(json.dumps({"width": 10, "height": 10, "corners": None}))
    pattern = str(tmp_path / "*.georef-final.json")
    assert expand_georef_globs(pattern) == []  # nothing publishable
    assert glob_matched_anything(pattern) is True  # but the sidecar is there
    assert glob_matched_anything(str(tmp_path / "nope-*.json")) is False


def test_glob_matched_anything_handles_a_comma_list(tmp_path) -> None:
    from mapsnap.make_iiif_georef import glob_matched_anything

    (tmp_path / "p1.georef.json").write_text("{}")
    both = f"{tmp_path}/nope-*.json,{tmp_path}/*.georef.json"
    assert glob_matched_anything(both) is True
    assert glob_matched_anything(f"{tmp_path}/a-*.json,{tmp_path}/b-*.json") is False


def test_volume_report_survives_a_volume_that_placed_nothing(tmp_path) -> None:
    """Gardiner has no annotations at all, which is when the card matters most."""
    from mapsnap.make_iiif_georef import volume_report

    for stem in ("p0__1", "p0__2"):
        (tmp_path / f"{stem}.provenance.json").write_text(
            json.dumps(_provenance(stem, "abstained", "unplaced", tier=5))
        )
    report = {m["label"]: m["value"] for m in volume_report(tmp_path, "2026-09-16")}
    assert report["placed"] == "0"
    assert report["unplaced"] == "2"
    assert report["fit sources"] == "none"
    assert "p0__1 (tier 5," in report["unplaced pages"]


def test_report_card_names_the_run_when_tagged(tmp_path) -> None:
    """A published fit must be traceable to the corpus pass that produced it."""
    from mapsnap.make_iiif_georef import volume_report

    (tmp_path / "p1.provenance.json").write_text(
        json.dumps(_provenance("p1", "placed", "georef"))
    )
    card = {
        m["label"]: m["value"] for m in volume_report(tmp_path, "2026-09-16", "v1.3")
    }
    assert card["run"] == "v1.3"
    assert card["generated"] == "2026-09-16"
    # Untagged runs say nothing rather than saying "None".
    untagged = {m["label"]: m["value"] for m in volume_report(tmp_path, "2026-09-16")}
    assert "run" not in untagged


def test_report_card_names_the_run_with_no_provenance(tmp_path) -> None:
    """The tag survives the early return taken when a volume has no records."""
    from mapsnap.make_iiif_georef import volume_report

    card = {
        m["label"]: m["value"] for m in volume_report(tmp_path, "2026-09-16", "v1.3")
    }
    assert card["run"] == "v1.3"


def test_label_note_distinguishes_a_volume_second_annotation_page(tmp_path) -> None:
    """A volume publishes its sheets AND its key map; the labels must differ."""
    from mapsnap.make_iiif_georef import volume_label

    source = {"label": "Madison, Indiana | 1904"}
    label = volume_label(source)
    for note in ("key map", None):
        page_label = " | ".join(
            part for part in (label, note, "mapsnap generated fit (2026-09-16)") if part
        )
        if note:
            assert page_label == (
                "Madison, Indiana | 1904 | key map | mapsnap generated fit (2026-09-16)"
            )
        else:
            assert "key map" not in page_label


def test_page_key_lower_matches_what_a_georef_path_parses_to() -> None:
    """The two sides must agree, or a lettered page is silently unpublished."""
    from mapsnap.make_iiif_georef import georef_path_to_page_key, page_key_lower

    for key in ("p97W", "p9N", "p5S", "p1499H"):
        assert page_key_lower(key) == georef_path_to_page_key(
            f"data/vol/{key}.georef-final.json"
        )
    # Keys with no suffix, and split panels, are untouched.
    assert page_key_lower("p20") == "p20"
    assert page_key_lower("p20__3") == "p20__3"
    assert page_key_lower("p97W__2") == "p97w__2"


def test_expand_georef_globs_accepts_a_comma_in_the_path(tmp_path):
    """The mirror holds an item named sanborn09511_002,5 (#515)."""
    volume = tmp_path / "sanborn09511_002,5"
    volume.mkdir()
    (volume / "p1.georef-final.json").write_text(json.dumps(_POSED))
    pattern = f"{volume}/*.georef-final.json"
    assert [Path(p).name for p in expand_georef_globs(pattern)] == [
        "p1.georef-final.json"
    ]
    assert glob_matched_anything(pattern)


def test_skeleton_rule_keeps_pages_it_cannot_judge_when_publishing():
    """'p0005ls' is ambiguous; publishing keeps it rather than asserting (#512)."""
    items = [(key, None, None, None, None) for key in ["p1l", "p0005ls", "p7", "p7s"]]
    kept = [item[0] for item in drop_redundant_skeletons(items)]
    # The unambiguous skeleton pair is still resolved; the ambiguous key stays.
    assert kept == ["p1l", "p0005ls", "p7"]
