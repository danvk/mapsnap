"""Tests for the independent-city mapping (mapsnap.loc_cities)."""

import json
from pathlib import Path

from mapsnap.loc_cities import (
    Boundary,
    Item,
    boundary_from_tags,
    city_index,
    match_items,
    normalize_place,
    read_boundaries,
    read_independent_city_items,
    state_by_fips_prefix,
    write_boundaries_geojson,
    write_counties_tsv,
    write_items_tsv,
)

RICHMOND = Boundary(
    relation_id=3864712, name="Richmond", fips="51760", border_type="city"
)
RICHMOND_COUNTY = Boundary(
    relation_id=206413, name="Richmond County", fips="51159", border_type="county"
)
NEWPORT_NEWS = Boundary(
    relation_id=3864704, name="Newport News", fips="51700", border_type="city"
)
SAINT_LOUIS = Boundary(
    relation_id=1180533, name="Saint Louis", fips="29510", border_type="city"
)


def test_normalize_place_folds_punctuation_and_saint() -> None:
    assert normalize_place("Newport  News") == "newport news"
    assert normalize_place("St. Louis") == "saint louis"
    assert normalize_place("Falls-Church") == "falls church"
    assert normalize_place("O'Fallon") == "ofallon"


def test_read_independent_city_items_catches_every_loc_spelling(tmp_path: Path) -> None:
    """The LoC spells it four ways, and "hampton (independent city)" is one of them."""
    path = tmp_path / "skipped.tsv"
    path.write_text(
        "item\tstate\tcounty\tcity\tcountry\tsheets\treason\n"
        "a\tVirginia\tindependent cities\tRichmond\tus\t27\tunmatched\n"
        "b\tVirginia\tindependent city\tSalem\tus\t9\tunmatched\n"
        "c\tVirginia\tindependent cities (prior to 2001)\tClifton Forge\tus\t9\tx\n"
        "d\tVirginia\thampton (independent city)\tHampton\tus\t2\tunmatched\n"
        "e\tAlaska\tketchikan census division\tKetchikan\tus\t5\tunmatched\n"
    )
    items = read_independent_city_items(path)
    assert [item.item for item in items] == ["a", "b", "c", "d"]
    assert items[0] == Item("a", "virginia", "independent cities", "Richmond", 27)


def test_state_by_fips_prefix_reads_the_matched_items(tmp_path: Path) -> None:
    path = tmp_path / "items.tsv"
    path.write_text(
        "item\tstate\tcounty\tcity\tsheets\tmatch\tfips\tne_name\n"
        "a\tvirginia\tfairfax county\tvienna\t3\texact\tUS51059\tFairfax\n"
        "b\tmaryland\tcarroll county\twestminster\t2\texact\tUS24013\tCarroll\n"
        "c\tvirginia\tbath county\thot springs\t1\texact\tUS51017\tBath\n"
        "d\tguam\t\t\t1\texact\tbadfips\t\n"
    )
    assert state_by_fips_prefix(path) == {"51": "virginia", "24": "maryland"}


def test_city_index_keeps_cities_and_drops_counties() -> None:
    """Richmond the city and Richmond County share a name; only the city is wanted."""
    index = city_index(
        [RICHMOND, RICHMOND_COUNTY, SAINT_LOUIS], {"51": "virginia", "29": "missouri"}
    )
    assert index == {
        ("virginia", "richmond"): RICHMOND,
        ("missouri", "saint louis"): SAINT_LOUIS,
    }


def test_city_index_accepts_a_compound_border_type() -> None:
    both = Boundary(
        relation_id=1, name="Anchorage", fips="02020", border_type="borough;city"
    )
    assert city_index([both], {"02": "alaska"}) == {("alaska", "anchorage"): both}


def test_city_index_skips_a_state_it_cannot_name() -> None:
    assert city_index([RICHMOND], {}) == {}


def test_match_items_prefers_a_name_match() -> None:
    items = [Item("a", "virginia", "independent cities", "Richmond", 27)]
    index = {("virginia", "richmond"): RICHMOND}
    matched, unmatched = match_items(items, index, {})
    assert unmatched == []
    assert matched[0].boundary == RICHMOND
    assert matched[0].kind == "exact"


def test_match_items_redirects_a_city_that_no_longer_exists() -> None:
    """Warwick merged into Newport News in 1958, so its streets are there now."""
    items = [Item("a", "virginia", "independent cities", "Warwick", 1)]
    matched, unmatched = match_items(items, {}, {"51700": NEWPORT_NEWS})
    assert unmatched == []
    assert matched[0].boundary == NEWPORT_NEWS
    assert matched[0].kind == "successor"


def test_match_items_reports_what_it_cannot_place() -> None:
    items = [Item("a", "virginia", "independent cities", "Atlantis", 4)]
    matched, unmatched = match_items(items, {}, {})
    assert matched == []
    assert unmatched == items


def test_boundary_county_fips_matches_loc_counties() -> None:
    assert RICHMOND.county_fips == "US51760"


def test_write_items_tsv_sorts_by_item_and_names_the_relation(tmp_path: Path) -> None:
    path = tmp_path / "out.tsv"
    items = [
        Item("b", "virginia", "independent cities", "Richmond", 27),
        Item("a", "missouri", "independent cities", "Saint Louis", 5),
    ]
    matched, _ = match_items(
        items,
        {
            ("virginia", "richmond"): RICHMOND,
            ("missouri", "saint louis"): SAINT_LOUIS,
        },
        {},
    )
    write_items_tsv(path, matched)
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == [
        "item",
        "state",
        "county",
        "city",
        "sheets",
        "match",
        "fips",
        "osm_relation",
        "osm_name",
    ]
    assert lines[1].split("\t") == [
        "a",
        "missouri",
        "independent cities",
        "Saint Louis",
        "5",
        "exact",
        "US29510",
        "1180533",
        "Saint Louis",
    ]
    assert lines[2].startswith("b\tvirginia")


def test_boundary_from_tags_accepts_a_county_tier_city() -> None:
    tags = {
        "boundary": "administrative",
        "admin_level": "6",
        "border_type": "city",
        "name": "Baltimore",
        "nist:fips_code": "24510",
    }
    assert boundary_from_tags(133345, tags) == Boundary(
        133345, "Baltimore", "24510", "city"
    )


def test_boundary_from_tags_rejects_what_cannot_be_joined() -> None:
    base = {
        "boundary": "administrative",
        "admin_level": "6",
        "nist:fips_code": "51760",
    }
    assert boundary_from_tags(1, base | {"admin_level": "8"}) is None  # a town
    assert boundary_from_tags(1, base | {"boundary": "census"}) is None
    assert (
        boundary_from_tags(1, {k: v for k, v in base.items() if k != "nist:fips_code"})
        is None
    )


def test_read_boundaries_reads_a_real_osm_file(tmp_path: Path) -> None:
    """Exercises the pyosmium path, not just the tag logic."""
    path = tmp_path / "boundaries.osm"
    path.write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        '<osm version="0.6" generator="test">\n'
        '  <node id="1" version="1" lat="39.3" lon="-76.6"/>\n'
        '  <way id="1" version="1"><nd ref="1"/></way>\n'
        '  <relation id="133345" version="1">\n'
        '    <member type="way" ref="1" role="outer"/>\n'
        '    <tag k="boundary" v="administrative"/>\n'
        '    <tag k="admin_level" v="6"/>\n'
        '    <tag k="border_type" v="city"/>\n'
        '    <tag k="name" v="Baltimore"/>\n'
        '    <tag k="nist:fips_code" v="24510"/>\n'
        "  </relation>\n"
        '  <relation id="999" version="1">\n'
        '    <member type="way" ref="1" role="outer"/>\n'
        '    <tag k="boundary" v="administrative"/>\n'
        '    <tag k="admin_level" v="8"/>\n'
        '    <tag k="name" v="Some Town"/>\n'
        "  </relation>\n"
        "</osm>\n"
    )
    assert read_boundaries(path) == [Boundary(133345, "Baltimore", "24510", "city")]


def test_write_counties_tsv_aggregates_items_per_boundary(tmp_path: Path) -> None:
    """Several volumes share a city, and osm-counties wants one row per boundary."""
    path = tmp_path / "counties.tsv"
    items = [
        Item("a", "virginia", "independent cities", "Richmond", 27),
        Item("b", "virginia", "independent cities", "Richmond", 13),
        Item("c", "missouri", "independent cities", "Saint Louis", 5),
    ]
    matched, _ = match_items(
        items,
        {
            ("virginia", "richmond"): RICHMOND,
            ("missouri", "saint louis"): SAINT_LOUIS,
        },
        {},
    )
    write_counties_tsv(path, matched)
    lines = [line.split("\t") for line in path.read_text().splitlines()]
    assert lines[0] == ["fips", "state", "ne_name", "ne_type", "items", "sheets"]
    # Ordered by sheets, so the biggest job is visible first in a long run.
    assert lines[1] == ["US51760", "virginia", "Richmond", "City", "2", "40"]
    assert lines[2] == ["US29510", "missouri", "Saint Louis", "City", "1", "5"]


def test_write_counties_tsv_labels_a_successor_county(tmp_path: Path) -> None:
    path = tmp_path / "counties.tsv"
    bedford = Boundary(2532615, "Bedford County", "51019", "county")
    items = [Item("a", "virginia", "independent cities", "Bedford", 3)]
    matched, _ = match_items(items, {}, {"51019": bedford})
    write_counties_tsv(path, matched)
    assert path.read_text().splitlines()[1].split("\t")[3] == "County"


def test_write_boundaries_geojson_is_a_feature_collection(tmp_path: Path) -> None:
    path = tmp_path / "b.geojson"
    features = [
        {
            "type": "Feature",
            "properties": {"FIPS": "US51760", "name": "Richmond"},
            "geometry": {"type": "MultiPolygon", "coordinates": []},
        }
    ]
    write_boundaries_geojson(path, features)
    loaded = json.loads(path.read_text())
    assert loaded["type"] == "FeatureCollection"
    assert loaded["features"][0]["properties"]["FIPS"] == "US51760"
