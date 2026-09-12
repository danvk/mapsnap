"""Tests for the LoC item -> Natural Earth county mapping."""

import csv
import json
from pathlib import Path

from mapsnap.loc_counties import (
    Item,
    load_counties,
    load_items,
    load_sheet_counts,
    match_items,
    normalize,
    report,
    resolve,
    skip_reason,
    write_outputs,
)


def feature(fips: str, postal: str, name: str, kind: str = "County") -> dict:
    return {
        "type": "Feature",
        "properties": {"FIPS": fips, "REGION": postal, "NAME": name, "TYPE": kind},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
    }


def write_ne(path: Path) -> Path:
    features = [
        feature("US17031", "IL", "Cook"),
        feature("US30049", "MT", "Lewis and Clark"),
        feature("US12086", "FL", "Miami-Dade"),
        feature("US20131", "KS", "Nemaha"),
        feature("US31055", "NE", "Douglas"),
        feature("US31153", "NE", "Sarpy"),
        feature("US29189", "MO", "Saint Louis"),
        feature("US29510", "MO", "Saint Louis", "City"),
        feature("US24510", "MD", "Baltimore", "City"),
        feature("US24005", "MD", "Baltimore"),
        feature("US51191", "VA", "Washington"),
        feature("US47163", "TN", "Sullivan"),
        feature("US11001", "DC", "Washington DC", "District of Columbia"),
        feature("US36061", "NY", "New York"),
        feature("US36005", "NY", "Bronx"),
        feature("US22071", "LA", "Orleans", "Parish"),
        feature("US32510", "NV", "Carson", "City"),
        feature("US51159", "VA", "Richmond"),
        feature("US51760", "VA", "Richmond", "City"),
        feature("US10005", "DE", "Sussex"),
        feature("US24045", "MD", "Wicomico"),
        feature("US40139", "OK", "Texas"),
        feature("US48421", "TX", "Sherman"),
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_normalize_strips_suffix_and_spells_saint():
    assert normalize("Cook County") == "cook"
    assert normalize("st. louis county") == "saint louis"
    assert normalize("St. Louis") == "saint louis"
    assert normalize("orleans parish") == "orleans"
    # Punctuation goes; the split into parts happens on the raw string, in resolve.
    assert normalize("jackson, clay, and platte counties") == "jackson clay and platte"
    assert normalize("independent cities (prior to 2001)") == "independent cities"
    assert normalize("washington county (va)") == "washington (va)"
    assert normalize("Miami-Dade") == "miami-dade"
    assert normalize("orleans parish county") == "orleans"  # two suffixes


def test_city_and_county_of_the_same_name_keep_the_county_under_the_bare_key(tmp_path):
    index = load_counties(write_ne(tmp_path / "ne.geojson"))
    # Feature order differs: MO lists the county first, MD the city first.
    assert index["MO"]["saint louis"].fips == "US29189"
    assert index["MO"]["saint louis city"].fips == "US29510"
    assert index["MD"]["baltimore"].fips == "US24005"
    assert index["MD"]["baltimore city"].fips == "US24510"
    # Virginia has both a rural Richmond County and the independent city.
    assert index["VA"]["richmond"].fips == "US51159"
    assert index["VA"]["richmond city"].fips == "US51760"
    # A city with no namesake county answers to both keys.
    assert index["NV"]["carson"].fips == "US32510"
    assert index["NV"]["carson city"].fips == "US32510"


def test_resolve_order_whole_then_alias_then_split_then_fuzzy(tmp_path):
    index = load_counties(write_ne(tmp_path / "ne.geojson"))
    exact = resolve("cook county", "IL", index)
    assert exact.kind == "exact" and exact.counties[0].fips == "US17031"
    # The whole name wins before any split on "and".
    whole = resolve("lewis and clark county", "MT", index)
    assert whole.kind == "exact" and whole.counties[0].name == "Lewis and Clark"
    alias = resolve("dade county", "FL", index)
    assert alias.kind == "alias" and alias.counties[0].name == "Miami-Dade"
    multi = resolve("douglas and sarpy county", "NE", index)
    assert multi.kind == "multi" and [c.fips for c in multi.counties] == [
        "US31055",
        "US31153",
    ]
    fuzzy = resolve("nemaba county", "KS", index)
    assert fuzzy.kind == "fuzzy" and fuzzy.counties[0].name == "Nemaha"
    assert resolve("new jersey coast", "NJ", index).kind == "unmatched"
    # A part that fails leaves the whole string unmatched rather than half-mapped.
    assert resolve("douglas and nowhere county", "NE", index).kind == "unmatched"


def test_catalog_conventions_manhattan_double_suffix_carson_city(tmp_path):
    index = load_counties(write_ne(tmp_path / "ne.geojson"))
    nyc = resolve("bronx, manhattan", "NY", index)
    assert nyc.kind == "multi" and [c.name for c in nyc.counties] == [
        "Bronx",
        "New York",
    ]
    assert resolve("orleans parish county", "LA", index).counties[0].fips == "US22071"
    assert resolve("carson city county", "NV", index).counties[0].fips == "US32510"


def test_state_tokens_inside_multi_county_strings(tmp_path):
    index = load_counties(write_ne(tmp_path / "ne.geojson"))
    match = resolve("sussex county, del., and wicomico county, md", "DE", index)
    assert [(c.postal, c.name) for c in match.counties] == [
        ("DE", "Sussex"),
        ("MD", "Wicomico"),
    ]
    match = resolve("texas county, oklahoma and sherman county, texas", "TX", index)
    assert [(c.postal, c.name) for c in match.counties] == [
        ("OK", "Texas"),
        ("TX", "Sherman"),
    ]


def test_parenthetical_state_overrides_the_item_state(tmp_path):
    index = load_counties(write_ne(tmp_path / "ne.geojson"))
    match = resolve("washington county (va) and sullivan county (tn)", "VA", index)
    assert match.kind == "multi"
    assert [(c.postal, c.name) for c in match.counties] == [
        ("VA", "Washington"),
        ("TN", "Sullivan"),
    ]


def test_dc_resolves_through_the_alias_table(tmp_path):
    index = load_counties(write_ne(tmp_path / "ne.geojson"))
    match = resolve("district of columbia", "DC", index)
    assert match.kind == "alias" and match.counties[0].fips == "US11001"


def test_skip_reasons():
    us = Item("a", "illinois", "cook county", "chicago", "united states")
    assert skip_reason(us) is None
    assert (
        skip_reason(Item("b", "coahuila", "", "piedras negras", "mexico")) == "foreign"
    )
    assert (
        skip_reason(
            Item("c", "virginia", "independent cities", "richmond", "united states")
        )
        == "independent-city"
    )
    assert (
        skip_reason(
            Item("d", "virginia", "hampton (independent city)", "", "united states")
        )
        == "independent-city"
    )
    assert (
        skip_reason(Item("e", "multiple states - us", "", "", "united states"))
        == "no-state"
    )
    assert skip_reason(Item("f", "ohio", "", "", "united states")) == "no-county"
    # No country field at all is treated as domestic.
    assert skip_reason(Item("g", "ohio", "licking county", "", "")) is None


def write_catalog(root: Path) -> None:
    records = {
        ("illinois", "1905", "sanborn01790_001"): {
            "location_state": ["illinois"],
            "location_county": ["cook county"],
            "location_city": ["chicago"],
            "location_country": ["united states"],
        },
        ("nebraska", "1910", "sanborn04600_001"): {
            "location_state": ["nebraska"],
            "location_county": ["douglas and sarpy county"],
            "location_city": ["omaha"],
            "location_country": ["united states"],
        },
        ("coahuila", "1905", "sanborn09999_001"): {
            "location_state": ["coahuila"],
            "location_county": [],
            "location_city": ["piedras negras"],
            "location_country": ["mexico"],
        },
        ("new-jersey", "1890", "sanborn05500_001"): {
            "location_state": ["new jersey"],
            "location_county": ["new jersey coast"],
            "location_city": ["asbury park"],
            "location_country": ["united states"],
        },
    }
    for (state, year, item), record in records.items():
        path = root / state / year / f"{item}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"item": record}))


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_end_to_end_outputs(tmp_path: Path):
    index = load_counties(write_ne(tmp_path / "ne.geojson"))
    write_catalog(tmp_path / "meta")
    items = load_items(tmp_path / "meta")
    assert [i.item for i in items] == [
        "sanborn09999_001",
        "sanborn01790_001",
        "sanborn04600_001",
        "sanborn05500_001",
    ]
    mapping = tmp_path / "map.tsv"
    mapping.write_text(
        "item\tstate\tyear\tcity\tseq\tstem\tpage_key\tsource\tbytes\tstorage_dir\n"
        "sanborn01790_001\tillinois\t1905\tchicago\t1\ts-0001\tp1\ttorrent-jp2\t1\td\n"
        "sanborn01790_001\tillinois\t1905\tchicago\t2\ts-0002\tp2\ttorrent-jp2\t1\td\n"
        "sanborn01790_001\tillinois\t1905\tchicago\t3\ts-covr\tpcovr\ttorrent-jp2\t1\td\n"
        "sanborn04600_001\tnebraska\t1910\tomaha\t1\ts-0001\tp1\ttorrent-jp2\t1\td\n"
    )
    counts = load_sheet_counts(mapping)
    assert counts == {"sanborn01790_001": 2, "sanborn04600_001": 1}  # cover not kept
    for item in items:
        item.sheets = counts.get(item.item, 0)
    matched, skipped = match_items(items, index)
    assert [(i.item, m.kind) for i, m in matched] == [
        ("sanborn01790_001", "exact"),
        ("sanborn04600_001", "multi"),
    ]
    assert [(i.item, r) for i, r in skipped] == [
        ("sanborn09999_001", "foreign"),
        ("sanborn05500_001", "unmatched"),
    ]
    write_outputs(tmp_path / "out", matched, skipped)
    rows = read_tsv(tmp_path / "out" / "items.tsv")
    assert (
        rows[1]["fips"] == "US31055;US31153" and rows[1]["ne_name"] == "Douglas;Sarpy"
    )
    counties = read_tsv(tmp_path / "out" / "counties.tsv")
    # Sorted by sheets: Cook (2) first, then Douglas and Sarpy (1 each).
    assert [(c["fips"], c["items"], c["sheets"]) for c in counties] == [
        ("US17031", "1", "2"),
        ("US31055", "1", "1"),
        ("US31153", "1", "1"),
    ]
    assert counties[0]["loc_names"] == "cook county"
    skipped_rows = read_tsv(tmp_path / "out" / "skipped.tsv")
    assert {r["reason"] for r in skipped_rows} == {"foreign", "unmatched"}
    text = report(matched, skipped)
    assert "4 items, 3 sheets" in text
    assert (
        "unmatched county strings (1 distinct)" in text and "new jersey coast" in text
    )
