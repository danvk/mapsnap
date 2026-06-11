"""Tests for mapsnap.streets."""

from mapsnap.streets import (
    _ORDINALS,
    _num_to_ordinal_word,
    canonical_street_match,
    canonical_street_matches,
    normalize_street,
)

# --- _num_to_ordinal_word ---


def test_ordinal_ones():
    assert _num_to_ordinal_word(1) == "FIRST"
    assert _num_to_ordinal_word(5) == "FIFTH"
    assert _num_to_ordinal_word(12) == "TWELFTH"
    assert _num_to_ordinal_word(19) == "NINETEENTH"


def test_ordinal_exact_tens():
    assert _num_to_ordinal_word(20) == "TWENTIETH"
    assert _num_to_ordinal_word(30) == "THIRTIETH"
    assert _num_to_ordinal_word(90) == "NINETIETH"


def test_ordinal_compound():
    assert _num_to_ordinal_word(21) == "TWENTY FIRST"
    assert _num_to_ordinal_word(42) == "FORTY SECOND"
    assert _num_to_ordinal_word(99) == "NINETY NINTH"


# --- _ORDINALS dict ---


def test_ordinals_dict_canonical_suffixes():
    assert _ORDINALS["4TH"] == "FOURTH"
    assert _ORDINALS["1ST"] == "FIRST"
    assert _ORDINALS["2ND"] == "SECOND"
    assert _ORDINALS["3RD"] == "THIRD"
    assert _ORDINALS["21ST"] == "TWENTY FIRST"


def test_ordinals_dict_all_suffixes_covered():
    # All four suffix variants should map to the same word.
    assert _ORDINALS["4ST"] == _ORDINALS["4ND"] == _ORDINALS["4RD"] == _ORDINALS["4TH"]


# --- normalize_street with ordinals ---


def test_normalize_ordinal_simple():
    # "S. 4TH ST" → "SOUTH FOURTH STREET"
    assert normalize_street("S. 4TH ST") == "SOUTH FOURTH STREET"


def test_normalize_ordinal_matches_word_form():
    # Both representations of the same street name should normalize identically.
    assert normalize_street("S. 4TH ST") == normalize_street("South Fourth Street")
    assert normalize_street("S. 5TH ST") == normalize_street("South Fifth Street")
    assert normalize_street("S 6TH ST") == normalize_street("South Sixth Street")


def test_normalize_ordinal_compound():
    # "21ST ST" → "TWENTY FIRST STREET"
    assert normalize_street("21ST ST") == "TWENTY FIRST STREET"


def test_normalize_ordinal_direction_prefix():
    assert normalize_street("N 21ST AVE") == "NORTH TWENTY FIRST AVENUE"


def test_normalize_no_ordinal_unchanged():
    # Non-ordinal tokens should pass through unchanged.
    assert normalize_street("Main Street") == "MAIN STREET"
    assert normalize_street("N RAMPART ST") == "NORTH RAMPART STREET"


def test_normalize_ordinal_does_not_expand_non_ordinal_numbers():
    # Bare numbers (no suffix) should not be altered.
    assert normalize_street("Route 66") == "ROUTE 66"


# --- canonical_street_matches / canonical_street_match: direction-suffixed type keys ---

_DC_STREETS = {
    "NORTH STREET NORTHEAST",
    "NORTH STREET NORTHWEST",
    "NORTH STREET SOUTHEAST",
    "NORTH STREET SOUTHWEST",
    "NORTH PLACE",
    "NORTH PLACE SOUTHEAST",
    "NORTH ROAD",
    "NORTH CAROLINA AVENUE NORTHEAST",
    "NORTH DAKOTA AVENUE NORTHWEST",
    "NORTH CAPITOL STREET NORTHWEST",
    "EAST STREET NORTHEAST",
    "EAST GRAND AVENUE",
}


def test_direction_word_matches_suffixed_type_key():
    # "N" → "NORTH": should match "NORTH STREET NORTHWEST" (first word of remainder is STREET).
    matches = canonical_street_matches("N", _DC_STREETS)
    assert "NORTH STREET NORTHEAST" in matches
    assert "NORTH STREET NORTHWEST" in matches
    assert "NORTH STREET SOUTHEAST" in matches
    assert "NORTH STREET SOUTHWEST" in matches


def test_direction_word_still_matches_plain_type_key():
    # Two-word "NORTH PLACE" still works after the fix.
    matches = canonical_street_matches("N", _DC_STREETS)
    assert "NORTH PLACE" in matches
    assert "NORTH PLACE SOUTHEAST" in matches
    assert "NORTH ROAD" in matches


def test_direction_word_does_not_match_named_streets():
    # "N" must NOT match streets where the word after the direction is not a type word.
    matches = canonical_street_matches("N", _DC_STREETS)
    assert "NORTH CAROLINA AVENUE NORTHEAST" not in matches
    assert "NORTH DAKOTA AVENUE NORTHWEST" not in matches
    assert "NORTH CAPITOL STREET NORTHWEST" not in matches


def test_direction_word_does_not_match_east_grand_avenue():
    # "E" must NOT match "EAST GRAND AVENUE" (GRAND is not a street type).
    matches = canonical_street_matches("E", _DC_STREETS)
    assert "EAST GRAND AVENUE" not in matches
    assert "EAST STREET NORTHEAST" in matches


def test_canonical_street_match_direction_suffixed():
    # canonical_street_match (singular) is consistent with canonical_street_matches.
    result = canonical_street_match("N", _DC_STREETS)
    assert result in {
        "NORTH STREET NORTHEAST",
        "NORTH STREET NORTHWEST",
        "NORTH STREET SOUTHEAST",
        "NORTH STREET SOUTHWEST",
        "NORTH PLACE",
        "NORTH PLACE SOUTHEAST",
        "NORTH ROAD",
    }
