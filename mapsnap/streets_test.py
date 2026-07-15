"""Tests for mapsnap.streets."""

from mapsnap.streets import (
    _ORDINALS,
    _num_to_ordinal_word,
    canonical_street_match,
    canonical_street_matches,
    is_bare_letter,
    matches_any_street,
    normalize_street,
)


def test_is_bare_letter():
    assert is_bare_letter("M")
    assert is_bare_letter("W")
    assert is_bare_letter(" M ")  # surrounding whitespace ignored
    assert not is_bare_letter("M.")  # has punctuation -> not bare
    assert not is_bare_letter("MAIN")
    assert not is_bare_letter("5")  # a digit is not a letter
    assert not is_bare_letter("")


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


def test_normalize_spaced_ordinal_suffix():
    # "21 ST" (CTC gap between digit and suffix) should match "21ST" form.
    assert normalize_street("21 ST") == normalize_street("21ST")
    assert normalize_street("5 TH") == normalize_street("5TH")
    assert normalize_street("2 ND") == normalize_street("2ND")
    assert normalize_street("3 RD") == normalize_street("3RD")


def test_normalize_spaced_ordinal_with_street_type():
    # "21 ST ST" → "TWENTY FIRST STREET" (ordinal merged, then trailing ST→STREET)
    assert normalize_street("21 ST ST") == "TWENTY FIRST STREET"
    assert normalize_street("S 4 TH ST") == "SOUTH FOURTH STREET"


def test_normalize_spaced_ordinal_does_not_merge_non_digit():
    # Only bare digit tokens trigger the merge; letters must be left alone.
    assert normalize_street("N ST") == normalize_street("N STREET")  # "N STREET"
    assert normalize_street("K ST") == normalize_street("K STREET")


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


# --- a bare label must name the whole street, not one word of a multi-word name ---

_VAN_STREETS = {
    "VAN BRUNT STREET",
    "VAN BUREN STREET",
    "VAN DYKE STREET",
    "MAGAZINE STREET",
    "PRINTERS ALLEY",
    "REP JOHN LEWIS WAY SOUTH",
}


def test_bare_label_omitting_only_the_type_still_matches():
    # The prefix rule's purpose: a label may leave off the street type.
    assert canonical_street_matches("MAGAZINE", _VAN_STREETS) == ["MAGAZINE STREET"]


def test_bare_label_omitting_an_unabbreviated_type_still_matches():
    # ALLEY is a type too (it just has no abbreviation we expand).
    assert canonical_street_matches("PRINTERS", _VAN_STREETS) == ["PRINTERS ALLEY"]


def test_one_word_of_a_multi_word_name_matches_nothing():
    # A lone "VAN" would otherwise claim VAN BRUNT, VAN BUREN and VAN DYKE alike; it must
    # earn a match by assembling with its sibling word instead.
    assert canonical_street_matches("VAN", _VAN_STREETS) == []
    assert canonical_street_match("VAN", _VAN_STREETS) is None
    assert not matches_any_street("VAN", _VAN_STREETS)


def test_assembled_multi_word_name_matches():
    assert canonical_street_matches("VAN BRUNT", _VAN_STREETS) == ["VAN BRUNT STREET"]


def test_name_fragment_is_rejected_even_when_unambiguous():
    # Only one REP* street exists here, yet "REP" is still only half a name: a building
    # abbreviation must not claim the street. This is not an ambiguity test.
    assert canonical_street_matches("REP", _VAN_STREETS) == []


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


def test_covers_whole_name_rejects_short_fragments():
    # A short fragment is an abbreviation or particle: it recurs city-wide and attaches to
    # anything, so it may not claim a street whose name continues past it.
    streets = {"VAN BRUNT STREET", "VAN DYKE STREET", "REP JOHN LEWIS WAY", "PIER C"}
    assert not matches_any_street("VAN", streets)
    assert not matches_any_street("REP", streets)
    assert not matches_any_street("PIER", streets)


def test_covers_whole_name_admits_a_long_distinctive_fragment():
    # Detroit's sheet prints "HARBOR" for what OSM now calls Harbor Island Street, and Grand
    # Rapids' prints "CLYDE" for Clyde Park Avenue. Neither has a sibling word to assemble with.
    assert matches_any_street("HARBOR", {"HARBOR ISLAND STREET"})
    assert matches_any_street("CLYDE", {"CLYDE PARK AVENUE SOUTHWEST"})


def test_covers_whole_name_measures_the_printed_label_not_the_expansion():
    # "ST" is two letters on the sheet even though it normalizes to SAINT, and "AV" two even
    # though it normalizes to AVENUE. Counting the expansion would hand the length allowance to
    # exactly the abbreviations this rule exists to stop.
    assert not matches_any_street("ST", {"SAINT CLAIR HEIGHTS DRIVE"})
    assert not matches_any_street("ST.", {"SAINT CLAIR HEIGHTS DRIVE"})
    assert not matches_any_street("AV", {"AVENUE EBERLY DRIVE"})
    # The type-only remainder path is unaffected: ST. CLAIR still names SAINT CLAIR DRIVE.
    assert matches_any_street("ST. CLAIR", {"SAINT CLAIR DRIVE"})


def test_covers_whole_name_still_allows_a_type_only_remainder_at_any_length():
    # Length only gates the *fragment* path; omitting the type and quadrant is always sound.
    assert matches_any_street("OAK", {"OAK STREET"})
    assert matches_any_street("ELM", {"NORTH ELM AVENUE SOUTHWEST"})


def test_printed_letters_ignores_punctuation_and_spaces():
    from mapsnap.streets import printed_letters

    assert printed_letters("ST.") == 2
    assert printed_letters("CLYDE") == 5
    assert printed_letters("VAN BRUNT") == 8
    assert printed_letters("3RD") == 3
