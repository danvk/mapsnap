"""Tests for loc_item_locations.py."""

from loc_item_locations import item_location


def test_item_location_reads_the_first_coordinate():
    record = {
        "Id": "http://www.loc.gov/item/sanborn05939_001/",
        "Location": [{"Coordinates": [41.6851, -74.1543]}],
    }
    assert item_location(record) == ("sanborn05939_001", 41.6851, -74.1543)


def test_item_location_needs_an_item_and_a_coordinate():
    assert item_location({"Id": "http://www.loc.gov/item/sanborn05939_001/"}) is None
    assert (
        item_location({"Id": "not-an-item", "Location": [{"Coordinates": [1, 2]}]})
        is None
    )
    assert (
        item_location({"Id": "sanborn1_1", "Location": [{"Coordinates": []}]}) is None
    )
