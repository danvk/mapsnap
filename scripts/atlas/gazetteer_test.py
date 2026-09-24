"""Tests for gazetteer.py."""

from pathlib import Path

from gazetteer import (
    COUNTY_FILE,
    PLACE_FILE,
    SUBDIVISION_FILE,
    Gazetteer,
    county_name,
    normalize_name,
    read_gazetteer,
    strip_place_suffix,
)


def test_normalize_name_matches_the_census_spelling():
    assert normalize_name("Saint Louis") == normalize_name("St. Louis")
    assert normalize_name("De Kalb") == normalize_name("DeKalb")
    assert normalize_name("Staten Island (Borough Of Richmond)") == "statenisland"


def test_strip_place_suffix_takes_the_longest_status():
    assert strip_place_suffix("St. Louis city") == ("St. Louis", "city")
    assert strip_place_suffix("Juneau city and borough") == (
        "Juneau",
        "city and borough",
    )
    assert strip_place_suffix("Urban Honolulu CDP") == ("Urban Honolulu", "CDP")


def test_county_name_reads_the_catalogue_abbreviation():
    assert county_name("Hudson Co.") == "hudson"
    assert county_name("Queens County") == "queens"
    assert county_name("Baltimore") is None


# A few real Gazetteer points, by table.
GAZETTEER = Gazetteer(
    places={
        ("MO", "stlouis"): (38.635699, -90.244582),
        ("NY", "newyork"): (40.6635, -73.9387),  # in Brooklyn
        ("HI", "urbanhonolulu"): (21.324347, -157.84764),
    },
    subdivisions={("CT", "westport"): (41.120831, -73.343464)},
    counties={
        ("NY", "newyork"): (40.776642, -73.970187),
        ("NY", "queens"): (40.654658, -73.841209),
        ("NJ", "hudson"): (40.731375, -74.078601),
    },
)


def test_locate_prefers_a_place():
    located = GAZETTEER.locate("Saint Louis", "Missouri", ["Saint Louis County"])
    assert located is not None and located.source == "place"
    assert (located.lat, located.lon) == (38.635699, -90.244582)


def test_locate_puts_manhattan_in_manhattan_not_in_brooklyn():
    located = GAZETTEER.locate("New York", "New York", ["Bronx, Manhattan"])
    assert located is not None and located.source == "county"
    assert located.lat == 40.776642


def test_locate_falls_back_to_subdivisions_and_counties():
    westport = GAZETTEER.locate("Westport", "Connecticut", ["Fairfield County"])
    assert westport is not None and westport.source == "subdivision"
    hudson = GAZETTEER.locate("Hudson Co.", "New Jersey", ["Hudson County"])
    assert hudson is not None and hudson.source == "county"
    # A village since absorbed into a city: its record's county.
    jamaica = GAZETTEER.locate("Jamaica", "New York", ["Queens County"])
    assert jamaica is not None and jamaica.lat == 40.654658


def test_locate_applies_aliases_and_point_overrides():
    honolulu = GAZETTEER.locate("Honolulu", "Hawaii", ["Honolulu County"])
    assert honolulu is not None and honolulu.lat == 21.324347
    # San Francisco's Gazetteer point is in the Pacific.
    san_francisco = GAZETTEER.locate("San Francisco", "California", [])
    assert san_francisco is not None and san_francisco.lon > -122.5


def test_locate_gives_up_on_what_it_cannot_place():
    assert (
        GAZETTEER.locate("New Jersey Coast", "New Jersey", ["New Jersey Coast"]) is None
    )
    assert GAZETTEER.locate("Nogales", "Sonora", ["Magdalena District"]) is None


def test_read_gazetteer_prefers_the_incorporated_place(tmp_path: Path):
    header = "USPS\tGEOID\tANSICODE\tNAME\tLSAD\tFUNCSTAT\tALAND\tAWATER\tALAND_SQMI\tAWATER_SQMI\tINTPTLAT\tINTPTLONG   \n"
    (tmp_path / PLACE_FILE).write_text(
        header
        + "VA\t1\t1\tRichmond CDP\t57\tS\t999999999\t0\t0\t0\t36.0\t-80.0\n"
        + "VA\t2\t2\tRichmond city\t25\tA\t1000\t0\t0\t0\t37.531399\t-77.476009   \n"
    )
    (tmp_path / SUBDIVISION_FILE).write_text(
        "USPS\tGEOID\tANSICODE\tNAME\tFUNCSTAT\tALAND\tAWATER\tALAND_SQMI\tAWATER_SQMI\tINTPTLAT\tINTPTLONG\n"
        "CT\t1\t1\tWestport town\tA\t1\t0\t0\t0\t41.120831\t-73.343464\n"
    )
    (tmp_path / COUNTY_FILE).write_text(
        "USPS\tGEOID\tANSICODE\tNAME\tALAND\tAWATER\tALAND_SQMI\tAWATER_SQMI\tINTPTLAT\tINTPTLONG\n"
        "NY\t36081\t1\tQueens County\t1\t0\t0\t0\t40.654658\t-73.841209\n"
    )
    gazetteer = read_gazetteer(tmp_path)
    assert gazetteer.places[("VA", "richmond")] == (37.531399, -77.476009)
    assert gazetteer.subdivisions[("CT", "westport")] == (41.120831, -73.343464)
    assert gazetteer.counties[("NY", "queens")] == (40.654658, -73.841209)
