"""Tests for building truth AnnotationPages from OIM API data."""

from typing import Any

from mapsnap.oim_api_truth import (
    PLACEHOLDER_RING,
    boundary_extent,
    build_annotation_page,
    flip_ring,
    split_panels,
    transformation_body,
    truth_annotation,
)

# A rectangular region boundary in OIM's bottom-left-origin document frame,
# occupying x in [100, 2100] and y in [1000, 4000] of a 4000x5000 document.
BOUNDARY = {
    "type": "Polygon",
    "coordinates": [
        [
            [100.0, 1000.0],
            [2100.0, 1000.0],
            [2100.0, 4000.0],
            [100.0, 4000.0],
            [100.0, 1000.0],
        ]
    ],
}


def make_document(document_id: int = 87464) -> dict[str, Any]:
    return {
        "id": document_id,
        "slug": "washington_dc_1916_vol_2_psb001250",
        "iiif_info": (
            "https://tile.loc.gov/image-services/iiif/"
            "service:gmd:gmd385m:g3851m:g3851gm:g01227003:sb001250/info.json"
        ),
        "image_size": [4000, 5000],
        "page_number": "sb001250",
    }


def make_session(
    division: int | None = 2,
    transformation: str = "helmert",
    boundary: dict[str, Any] = BOUNDARY,
) -> dict[str, Any]:
    return {
        "status": "success",
        "reg2": {
            "id": 75103,
            "document_id": 87464,
            "title": "Washington, D.C. | 1916 | Vol. 2 psb001250"
            + (f" [{division}]" if division else ""),
            "created_by": "danvk",
            "boundary": boundary,
            # The API serializes division_number as a string.
            "division_number": str(division) if division else None,
        },
        "lyr2": {"id": 86621},
        "map": {
            "identifier": "sanborn01227_003",
            "title": "Washington, D.C. | 1916 | Vol. 2",
        },
        "data": {
            "epsg": 3857,
            "transformation": transformation,
            "gcps": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"image": [50, 60], "id": "x"},
                        "geometry": {"type": "Point", "coordinates": [-77.018, 38.921]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"image": [1900, 2800], "id": "y"},
                        "geometry": {"type": "Point", "coordinates": [-77.013, 38.917]},
                    },
                ],
            },
        },
    }


def test_boundary_extent() -> None:
    assert boundary_extent(BOUNDARY) == (100.0, 1000.0, 2100.0, 4000.0)


def test_flip_ring() -> None:
    assert flip_ring([[0.0, 0.0], [10.0, 500.0]], 500.0) == [[0.0, 500.0], [10.0, 0.0]]


def test_transformation_body() -> None:
    assert transformation_body("helmert", 2) == {"type": "helmert"}
    assert transformation_body("tps", 8) == {"type": "thinPlateSpline"}
    assert transformation_body("poly1", 4) == {
        "type": "polynomial",
        "options": {"order": 1},
    }
    assert transformation_body("poly3", 12)["options"] == {"order": 3}
    # Unknown or missing names fall back by GCP count.
    assert transformation_body(None, 2) == {"type": "helmert"}
    assert transformation_body("mystery", 5) == {
        "type": "polynomial",
        "options": {"order": 1},
    }


def test_truth_annotation_offsets_gcps_into_canvas_frame() -> None:
    item = truth_annotation(make_session(), make_document())
    # xmin=100; y_offset = canvas_height - ymax = 5000 - 4000 = 1000.
    coords = [f["properties"]["resourceCoords"] for f in item["body"]["features"]]
    assert coords == [[150, 1060], [2000, 3800]]
    # Geo coordinates pass through untouched.
    assert item["body"]["features"][0]["geometry"]["coordinates"] == [-77.018, 38.921]
    assert item["body"]["transformation"] == {"type": "helmert"}
    assert item["label"].endswith("psb001250 [2]")
    source = item["target"]["source"]
    assert source["width"] == 4000 and source["height"] == 5000
    assert source["id"].endswith(":sb001250/info.json")
    # Selector ring is the boundary flipped into top-left-origin coordinates.
    assert (
        'points="100,4000 2100,4000 2100,1000 100,1000 100,4000"'
        in (item["target"]["selector"]["value"])
    )


def test_split_panels_indexes_by_division_with_placeholders() -> None:
    sessions = [make_session(division=1), make_session(division=3)]
    panels = split_panels(sessions, make_document(), "p125")
    assert panels["image"] == "p125.jpg"
    assert panels["width"] == 4000 and panels["height"] == 5000
    assert len(panels["panels"]) == 3
    assert panels["panels"][1] == PLACEHOLDER_RING  # missing division 2
    assert panels["panels"][0][0] == [100.0, 4000.0]  # flipped boundary


def test_build_annotation_page() -> None:
    sessions = [
        make_session(division=2),
        make_session(division=1, transformation="poly1"),
    ]
    page, panels = build_annotation_page(
        sessions, [make_document()], "sanborn01227_003"
    )
    assert page["type"] == "AnnotationPage"
    assert page["id"].endswith("/mosaic/sanborn01227_003/main-content/")
    assert "Washington, D.C. | 1916 | Vol. 2" in page["label"]
    # Items sorted by (document, division): the poly1 division-1 item first.
    assert len(page["items"]) == 2
    assert page["items"][0]["label"].endswith("[1]")
    assert page["items"][0]["body"]["transformation"]["type"] == "polynomial"
    # Split panels keyed by the parent page key derived from the LOC service URL.
    assert list(panels) == ["p125"]
    assert len(panels["p125"]["panels"]) == 2


def test_build_annotation_page_skips_missing_documents() -> None:
    page, panels = build_annotation_page([make_session()], [], "sanborn01227_003")
    assert page["items"] == []
    assert panels == {}
