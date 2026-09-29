import json
import math
from pathlib import Path

import pytest
from publish_run import (
    CDN_BASE,
    Destination,
    add_names,
    drop_keymap_pages,
    filter_pages,
    for_cdn,
    haversine_m,
    item_id,
    item_status,
    page_center,
    page_key,
    page_transform,
    read_locations,
    read_mapping,
    read_volume_numbers,
    sheet_extent_m,
    sheets_placed,
    strip_creators,
    update_report,
    volume_number,
    with_page_fields,
)

from mapsnap.make_iiif_georef import CREATOR, RIGHTS

SERVICE = "https://tile.loc.gov/image-services/iiif/service:gmd:x:05664_1886-0001"
# Metres per degree of latitude on the sphere haversine_m uses.
M_PER_DEGREE = math.pi * 6_371_008.8 / 180
TOWN = (-75.15, 39.84)


def annotation(
    m_per_px: float = 0.5,
    center: tuple[float, float] = TOWN,
    label: str = "Woodbury, New Jersey | 1886 | sanborn05664_001 p4",
    transformation: str = "helmert",
) -> dict:
    """A 1000x1000 px page, north up, centered on `center`, with a GCP at each corner."""
    lon0, lat0 = center
    k = math.cos(math.radians(lat0))

    def lonlat(x: float, y: float) -> list[float]:
        return [
            lon0 + (x - 500) * m_per_px / (M_PER_DEGREE * k),
            lat0 - (y - 500) * m_per_px / M_PER_DEGREE,
        ]

    corners = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    return {
        "id": f"{SERVICE}/georef",
        "type": "Annotation",
        "label": label,
        "creator": [{"id": "https://example.org/me", "type": "Person"}],
        "target": {
            "id": f"{SERVICE}/selector",
            "type": "SpecificResource",
            "source": {
                "id": f"{SERVICE}/info.json",
                "type": "ImageService2",
                "width": 1000,
                "height": 1000,
            },
            "selector": {
                "type": "SvgSelector",
                "value": '<svg><polygon points="0,0 1000,0 1000,1000 0,1000" /></svg>',
            },
        },
        "body": {
            "id": f"{SERVICE}/gcps",
            "type": "FeatureCollection",
            "transformation": {"type": transformation},
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "resourceCoords": [x, y],
                        "type": "corner",
                        "creator": {"id": "https://example.org/me"},
                    },
                    "geometry": {"type": "Point", "coordinates": lonlat(x, y)},
                }
                for x, y in corners
            ],
        },
    }


def test_haversine_m_is_a_degree_of_latitude_per_degree():
    assert haversine_m((0, 0), (0, 1)) == pytest.approx(M_PER_DEGREE)


@pytest.mark.parametrize("transformation", ["helmert", "polynomial"])
def test_page_transform_maps_each_gcp_home(transformation):
    page = annotation(transformation=transformation)
    transform = page_transform(page)
    assert transform is not None
    for feature in page["body"]["features"]:
        lon, lat = transform(*feature["properties"]["resourceCoords"])
        assert (lon, lat) == pytest.approx(feature["geometry"]["coordinates"])


def test_page_transform_needs_two_gcps():
    page = annotation()
    page["body"]["features"] = page["body"]["features"][:1]
    assert page_transform(page) is None


def test_sheet_extent_m_is_the_ground_diagonal():
    page = annotation(m_per_px=0.5)
    transform = page_transform(page)
    assert transform is not None
    assert sheet_extent_m(page, transform) == pytest.approx(
        500 * math.sqrt(2), rel=1e-3
    )


def test_page_center_is_the_middle_of_the_clip_outline():
    page = annotation(m_per_px=1.0)
    page["target"]["selector"]["value"] = (
        '<svg><polygon points="0,0 200,0 200,200 0,200" /></svg>'
    )
    transform = page_transform(page)
    assert transform is not None
    # The clip's middle, (100, 100) px, is 400 m west and 400 m north of the sheet's.
    lon, lat = page_center(page, transform)
    assert haversine_m((lon, TOWN[1]), TOWN) == pytest.approx(400, rel=1e-3)
    assert haversine_m((TOWN[0], lat), TOWN) == pytest.approx(400, rel=1e-3)
    assert lon < TOWN[0] and lat > TOWN[1]


def test_page_key_reads_the_stem_and_panel_from_the_label():
    assert page_key({"label": "Town | 1900 | sanborn1_001 p4"}) == "p4"
    assert page_key({"label": "Town | 1900 | sanborn1_001 p4 [2]"}) == "p4__2"


def test_item_id_keeps_a_suffixed_loc_id_whole():
    assert item_id(Path("sanborn04424_001.iiif.json")) == "sanborn04424_001"
    assert item_id(Path("sanborn04424_001.5.iiif.json")) == "sanborn04424_001.5"


def test_filter_pages_drops_a_sheet_too_large_to_be_real():
    # 10 m/px: a 14 km sheet, over #503's 6 km.
    pages = filter_pages([annotation(m_per_px=10.0), annotation()], 40, None)
    assert len(pages.kept) == 1
    assert pages.too_large == 1
    assert pages.withheld == [("p4", "sheet 14.1 km across")]


def test_filter_pages_drops_a_far_page_only_in_a_small_volume():
    far = annotation(center=(TOWN[0], TOWN[1] + 0.1))  # 11 km north
    near = annotation()
    small = filter_pages([far, near], 5, TOWN)
    assert small.kept == [near]
    assert small.too_far == 1
    assert small.withheld[0][1].endswith("km from the catalogue location")
    # Eleven sheets is not a small volume; neither is one of unknown size.
    assert len(filter_pages([far, near], 11, TOWN).kept) == 2
    assert len(filter_pages([far, near], 0, TOWN).kept) == 2
    # Nor is a volume with no catalogue location.
    assert len(filter_pages([far, near], 5, None).kept) == 2


def test_filter_pages_keeps_a_page_it_cannot_fit():
    page = annotation(m_per_px=10.0)
    page["body"]["features"] = page["body"]["features"][:1]
    assert filter_pages([page], 5, TOWN).kept == [page]


def test_item_status_says_why_an_item_is_withheld():
    page = annotation()
    assert item_status([page], [page], False) == "published"
    assert item_status([page], [page], True) == "withheld: key map under 2.5 km"
    assert item_status([], [], False) == "no page placed"
    assert item_status([page], [], False) == "withheld: every page filtered"
    # A key map is still worth publishing when no page is left beside it.
    assert item_status([page], [], has_keymap=True) == "published: key map only"
    assert item_status([], [], has_keymap=True) == "published: key map only"
    assert item_status([page], [], True, has_keymap=True) == (
        "withheld: key map under 2.5 km"
    )


def test_drop_keymap_pages_leaves_the_key_map_to_its_own_file():
    keymap = [annotation(label="Town | 1888 | sanborn00007_002 p1 [1]")]
    pages = [
        annotation(label="Town | 1888 | sanborn00007_002 p1 [1]"),
        annotation(label="Town | 1888 | sanborn00007_002 p1 [2]"),
    ]
    kept, dropped = drop_keymap_pages(pages, keymap)
    assert kept == [pages[1]]
    assert dropped == ["p1__1"]
    assert drop_keymap_pages(pages, []) == (pages, [])


def test_strip_creators_removes_every_creator():
    page = annotation()
    page["rights"] = "x"
    strip_creators(page)
    assert "creator" not in page and "rights" not in page
    assert all("creator" not in f["properties"] for f in page["body"]["features"])


def test_for_cdn_rescales_into_the_cdns_quarter_size_frame():
    page = annotation()
    page["target"]["source"].update(width=6450, height=7650)
    page["body"]["features"][2]["properties"]["resourceCoords"] = [6450, 7650]
    page["target"]["selector"]["value"] = (
        '<svg><polygon points="0,0 6450,0 6450,7650 0,7650" /></svg>'
    )
    cdn = for_cdn(page)
    # ceil(6450 / 4) x ceil(7650 / 4), not rounded: 1613 x 1913.
    assert cdn["target"]["source"] == {
        "id": f"{CDN_BASE}/service:gmd:x:05664_1886-0001",
        "type": "ImageService3",
        "width": 1613,
        "height": 1913,
    }
    assert cdn["body"]["features"][2]["properties"]["resourceCoords"] == [1613, 1913]
    assert (
        'points="0.0,0.0 1613.0,0.0 1613.0,1913.0 0.0,1913.0"'
        in (cdn["target"]["selector"]["value"])
    )
    # The original is untouched.
    assert page["target"]["source"]["width"] == 6450


def test_for_cdn_repoints_the_annotations_own_ids_split_panel_included():
    page = annotation()
    page["id"] = f"{SERVICE}__2/georef"
    cdn = for_cdn(page)
    assert cdn["id"] == f"{CDN_BASE}/service:gmd:x:05664_1886-0001__2/georef"
    assert cdn["target"]["id"] == f"{CDN_BASE}/service:gmd:x:05664_1886-0001/selector"
    assert cdn["body"]["id"] == f"{CDN_BASE}/service:gmd:x:05664_1886-0001/gcps"


def test_for_cdn_refuses_an_image_not_on_loc_gov():
    page = annotation()
    page["target"]["source"]["id"] = "https://example.org/iiif/p4"
    with pytest.raises(ValueError):
        for_cdn(page)


def test_with_page_fields_puts_the_url_creator_and_license_first():
    page = {
        "id": "https://www.loc.gov/item/x/generated",
        "type": "AnnotationPage",
        "@context": ["http://www.w3.org/ns/anno.jsonld"],
        "label": "Town | 1900",
        "metadata": [],
        "items": [],
    }
    published = with_page_fields(
        page, "https://mapsnap.org/runs/v1.3/iiif/loc/x.main.iiif.json"
    )
    assert list(published)[:6] == [
        "id",
        "type",
        "@context",
        "label",
        "creator",
        "rights",
    ]
    assert published["id"] == "https://mapsnap.org/runs/v1.3/iiif/loc/x.main.iiif.json"
    assert published["creator"] == CREATOR
    assert published["rights"] == RIGHTS
    assert published["items"] == []


def test_update_report_counts_only_what_is_published():
    page = {
        "metadata": [
            {"label": "pages", "value": "10"},
            {"label": "placed", "value": "5"},
            {"label": "unplaced", "value": "5"},
        ]
    }
    update_report(page, 4, [("p1__2", "11.5 km from the catalogue location")])
    assert page["metadata"] == [
        {"label": "pages", "value": "10"},
        {"label": "placed", "value": "4"},
        {"label": "unplaced", "value": "6"},
        {"label": "withheld", "value": "p1__2 (11.5 km from the catalogue location)"},
    ]


def test_destination_writes_both_sources_minified_with_their_own_urls(tmp_path):
    for source in ("loc", "chronoscope"):
        (tmp_path / "iiif" / source).mkdir(parents=True)
    page = {
        "id": "old",
        "type": "AnnotationPage",
        "@context": ["http://www.w3.org/ns/anno.jsonld"],
        "items": [annotation()],
    }
    Destination(tmp_path, "https://mapsnap.org/runs/v1.3/").write(
        "x.main.iiif.json", page
    )
    loc_text = (tmp_path / "iiif/loc/x.main.iiif.json").read_text()
    loc = json.loads(loc_text)
    assert loc_text == json.dumps(loc, separators=(",", ":"), ensure_ascii=False)
    cdn = json.loads((tmp_path / "iiif/chronoscope/x.main.iiif.json").read_text())
    assert loc["id"] == "https://mapsnap.org/runs/v1.3/iiif/loc/x.main.iiif.json"
    assert (
        cdn["id"] == "https://mapsnap.org/runs/v1.3/iiif/chronoscope/x.main.iiif.json"
    )
    assert loc["items"][0]["target"]["source"]["id"].startswith("https://tile.loc.gov/")
    assert cdn["items"][0]["target"]["source"]["id"].startswith(CDN_BASE)


def test_read_mapping_counts_each_items_sheets(tmp_path):
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text(
        "item\tstate\tyear\tcity\tseq\n"
        "sanborn1_001\tnew-jersey\t1886\twoodbury\t1\n"
        "sanborn1_001\tnew-jersey\t1886\twoodbury\t2\n"
        "sanborn1_001.5\tnew-jersey\t1890\twoodbury\t1\n"
    )
    items = read_mapping(mapping)
    assert items["sanborn1_001"]["sheets"] == 2
    assert items["sanborn1_001.5"] == {
        "state_slug": "new-jersey",
        "city_slug": "woodbury",
        "year": "1890",
        "sheets": 1,
    }


def test_add_names_takes_the_atlas_names_and_a_postal_code(tmp_path):
    (tmp_path / "volumes").mkdir()
    (tmp_path / "places.json").write_text(
        json.dumps(
            {
                "places": [
                    {
                        "id": "new-jersey/woodbury",
                        "name": "Woodbury",
                        "state": "New Jersey",
                    }
                ]
            }
        )
    )
    (tmp_path / "volumes/new-jersey.json").write_text(
        json.dumps(
            {"new-jersey/woodbury": [{"item": "sanborn1_001", "date": "1886-11"}]}
        )
    )
    items = {
        "sanborn1_001": {"state_slug": "new-jersey", "city_slug": "woodbury"},
        "sanborn2_001": {"state_slug": "new-york", "city_slug": "glens-falls"},
    }
    add_names(items, tmp_path)
    assert items["sanborn1_001"] == {
        "state_slug": "new-jersey",
        "city_slug": "woodbury",
        "city": "Woodbury",
        "state": "New Jersey",
        "date": "1886-11",
        "title": "",
        "postal": "NJ",
    }
    # An item the atlas does not list falls back to its slugs.
    assert items["sanborn2_001"]["city"] == "Glens Falls"
    assert items["sanborn2_001"]["postal"] == "NY"


def test_read_locations_is_lon_lat_by_exact_item(tmp_path):
    locations = tmp_path / "locations.tsv"
    locations.write_text("item\tlat\tlon\nsanborn1_001\t39.84\t-75.15\n")
    assert read_locations(locations) == {"sanborn1_001": (-75.15, 39.84)}


def test_page_transform_does_not_mirror_a_two_point_helmert_page():
    page = annotation()
    corners = page["body"]["features"]
    page["body"]["features"] = [corners[0], corners[2]]
    transform = page_transform(page)
    assert transform is not None
    # The corners it was not given land where they belong, not mirrored across the diagonal.
    for feature in (corners[1], corners[3]):
        lon, lat = transform(*feature["properties"]["resourceCoords"])
        assert (lon, lat) == pytest.approx(feature["geometry"]["coordinates"])


def test_volume_number_reads_the_catalogue_notes():
    assert volume_number(["Vol.1  1915  Republished 1939.", "115 sheet(s)."]) == "1"
    assert volume_number(["128 sheet(s).", "Vol. 2, 1915; Republished 1939."]) == "2"
    assert volume_number(["Volume 3A"]) == "3A"
    assert volume_number(["47 skeleton maps. Bound."]) == ""


def test_read_volume_numbers_keys_by_item(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    records = [
        {"Id": "http://www.loc.gov/item/sanborn05791_054/", "Notes": ["Vol. 2, 1915"]},
        {"Id": "http://www.loc.gov/item/sanborn04424_001.5/", "Notes": ["Vol. 1"]},
        {"Id": "http://www.loc.gov/item/sanborn00001_001/", "Notes": ["2 sheet(s)."]},
    ]
    metadata.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    assert read_volume_numbers(metadata) == {
        "sanborn05791_054": "2",
        "sanborn04424_001.5": "1",
    }


def test_sheets_placed_counts_a_split_sheet_once():
    pages = [
        annotation(label="Town | 1900 | x p4 [1]"),
        annotation(label="Town | 1900 | x p4 [2]"),
        annotation(label="Town | 1900 | x p5"),
    ]
    assert sheets_placed(pages) == 2
