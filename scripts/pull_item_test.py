"""Tests for pull_item.py."""

from pathlib import Path

from pull_item import directory_name, state_abbreviation, sync_command


def volume(item: str, state: str, city: str, year: str) -> dict[str, str]:
    """A volumes TSV row with only the fields directory_name reads."""
    return {"item": item, "state": state, "city": city, "year": year}


def test_state_abbreviation_uses_the_postal_code():
    assert state_abbreviation("pennsylvania") == "pa"
    assert state_abbreviation("new-york") == "ny"
    assert state_abbreviation("multiple-states-cuba") == "multiple-states-cuba"


def test_directory_name_is_town_state_year():
    wernersville = volume("sanborn08035_001", "pennsylvania", "wernersville", "1914")
    assert directory_name(wernersville, [wernersville]) == "wernersville_pa_1914"


def test_directory_name_adds_the_item_when_a_town_year_has_several():
    rows = [
        volume("sanborn01790_085", "illinois", "chicago", "1950"),
        volume("sanborn01790_086", "illinois", "chicago", "1950"),
        volume("sanborn01790_010", "illinois", "chicago", "1906"),
    ]
    assert directory_name(rows[0], rows) == "chicago_il_1950_sanborn01790_085"
    assert directory_name(rows[2], rows) == "chicago_il_1906"


def test_directory_name_slugs_the_city_and_falls_back_to_the_item():
    row = volume("sanborn06185_002", "new-york", "Staten Island (Borough)", "1917")
    assert directory_name(row, [row]) == "staten_island_borough_ny_1917"
    toledo = volume("2016586562", "ohio", "", "1868")
    assert directory_name(toledo, [toledo]) == "2016586562"


def test_sync_command_mirrors_the_prefix():
    command = sync_command(
        "s3://mapsnap-sanborn/by-state/pennsylvania/1914/sanborn08035_001",
        Path("data/wernersville_pa_1914"),
    )
    assert command[:5] == [
        "aws",
        "s3",
        "sync",
        "s3://mapsnap-sanborn/by-state/pennsylvania/1914/sanborn08035_001/",
        "data/wernersville_pa_1914",
    ]
