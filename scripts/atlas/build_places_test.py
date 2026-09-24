"""Tests for build_places.py's loc.gov sheet index."""

from collections import defaultdict
from pathlib import Path

from build_places import (
    Lookups,
    Place,
    compact_stems,
    loc_resource,
    read_loc_sheets,
    read_record,
)
from gazetteer import Gazetteer

HEADER = "item\tstate\tyear\tcity\tseq\tstem\tpage_key\tsource\tbytes\tstorage_dir\n"


def mapping_row(item: str, seq: int, stem: str, storage_dir: str) -> str:
    """One mapping TSV row with only the fields read_loc_sheets reads filled in."""
    return f"{item}\tindiana\t1917\tsouth-bend\t{seq}\t{stem}\tp\ttorrent-jp2\t0\t{storage_dir}\n"


def test_loc_resource_joins_the_last_two_directories():
    assert (
        loc_resource("gmd/gmd409m/g4094m/g4094sm/g4094sm_g025021917")
        == "g4094sm.g4094sm_g025021917"
    )
    assert loc_resource("gmd/gmd412m/g4124m/g4124pm/g096701899") == "g4124pm.g096701899"
    assert loc_resource("") is None


def test_compact_stems_round_trips():
    stems = ["08492_01_1921-0001", "08492_02_1921-0001", "08492_02_1921-0002"]
    prefix, rest = compact_stems(stems)
    assert prefix == "08492_0"
    assert [prefix + tail for tail in rest] == stems


def test_read_loc_sheets_orders_sheets_by_seq(tmp_path: Path):
    storage = "gmd/gmd409m/g4094m/g4094sm/g4094sm_g025021917"
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text(
        HEADER
        + mapping_row("sanborn02502_005", 2, "02502_1917-0000", storage)
        + mapping_row("sanborn02502_005", 1, "02502_1917-titl", storage)
        + mapping_row("sanborn02502_005", 3, "02502_1917-0001", storage)
        # Sheets under two storage directories are two resources: no index.
        + mapping_row("sanborn00965_002", 1, "00965_1913-0001", "gmd/a/b/g009651913")
        + mapping_row(
            "sanborn00965_002", 2, "00965_1920-0001", "gmd/a/b/g009651913/g009651920"
        )
    )
    sheets = read_loc_sheets(mapping)
    assert set(sheets) == {"sanborn02502_005"}
    south_bend = sheets["sanborn02502_005"]
    assert south_bend.resource == "g4094sm.g4094sm_g025021917"
    assert [south_bend.prefix + tail for tail in south_bend.sheets] == [
        "02502_1917-titl",
        "02502_1917-0000",
        "02502_1917-0001",
    ]


def catalogue_record(item: str, city: str, coordinates: list[float] | None) -> dict:
    """A catalogue record for a Queens volume, geocoded or not."""
    location = [{"Coordinates": coordinates}] if coordinates else []
    return {
        "Id": f"http://www.loc.gov/item/{item}/",
        "Date": "1913",
        "Title": f"Sanborn Fire Insurance Map from {city}, Queens County, New York.",
        "City_text": [city],
        "County_text": ["Queens County"],
        "State_text": ["New York"],
        "Location": location,
    }


def test_read_record_places_an_ungeocoded_town_by_the_gazetteer():
    gazetteer = Gazetteer(
        places={}, subdivisions={}, counties={("NY", "queens"): (40.654658, -73.841209)}
    )
    lookups = Lookups(prefixes={}, mirror_sheets={}, gazetteer=gazetteer)
    places: dict[tuple[str, str], Place] = {}
    skipped: dict[str, int] = defaultdict(int)
    read_record(
        catalogue_record("sanborn06185_002", "Queens", None), places, skipped, lookups
    )
    queens = places[("new-york", "queens")]
    assert (queens.lat, queens.lon, queens.approximate) == (40.654658, -73.841209, True)
    assert skipped["gazetteer_county"] == 1

    # A record the catalogue did geocode moves the town to its own point.
    read_record(
        catalogue_record("sanborn06185_003", "Queens", [40.7, -73.8]),
        places,
        skipped,
        lookups,
    )
    assert (queens.lat, queens.lon, queens.approximate) == (40.7, -73.8, False)
    assert len(queens.volumes) == 2


def test_read_record_drops_an_ungeocoded_town_without_a_gazetteer():
    lookups = Lookups(prefixes={}, mirror_sheets={}, gazetteer=None)
    places: dict[tuple[str, str], Place] = {}
    skipped: dict[str, int] = defaultdict(int)
    read_record(
        catalogue_record("sanborn06185_002", "Queens", None), places, skipped, lookups
    )
    assert places == {} and skipped["no_coordinates"] == 1
