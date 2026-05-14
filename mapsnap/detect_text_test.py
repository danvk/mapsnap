"""Unit tests for detect_text.py helpers."""

from mapsnap.detect_text import (
    canonical_street_match,
    matches_any_street,
    normalize_street,
)

# ---------------------------------------------------------------------------
# normalize_street
# ---------------------------------------------------------------------------


def test_normalize_uppercase():
    assert normalize_street("Magazine St") == "MAGAZINE STREET"


def test_normalize_direction_prefix():
    assert normalize_street("N Rampart St") == "NORTH RAMPART STREET"


def test_normalize_direction_not_expanded_mid_word():
    # "S" in the middle of a name is not a direction prefix
    assert normalize_street("Canal St") == "CANAL STREET"


def test_normalize_strips_punctuation():
    assert normalize_street("St. Charles Ave.") == "SAINT CHARLES AVENUE"


def test_normalize_empty():
    assert normalize_street("") == ""


def test_normalize_multi_word_direction():
    assert normalize_street("NE Broad Ave") == "NORTHEAST BROAD AVENUE"


def test_normalize_saint_prefix():
    assert normalize_street("St. Charles Ave") == "SAINT CHARLES AVENUE"


def test_normalize_saint_prefix_without_period():
    assert normalize_street("St Charles Ave") == "SAINT CHARLES AVENUE"


def test_normalize_st_suffix_still_street():
    # "ST" at the end (not first word) is still "STREET"
    assert normalize_street("Magazine St") == "MAGAZINE STREET"


# ---------------------------------------------------------------------------
# matches_any_street
# ---------------------------------------------------------------------------

STREETS = {
    "MAGAZINE STREET",
    "COLUMBIA PLACE",
    "CANAL STREET",
    "NORTH RAMPART STREET",
    "ESPLANADE AVENUE",
}


def test_exact_match_full_name():
    assert matches_any_street("Magazine Street", STREETS)


def test_exact_match_abbreviated():
    # "Magazine St" normalizes to "MAGAZINE STREET" — exact match
    assert matches_any_street("Magazine St", STREETS)


def test_bare_name_matches_full():
    # "COLUMBIA" is a prefix of "COLUMBIA PLACE "
    assert matches_any_street("Columbia", STREETS)


def test_full_name_matches_bare_label():
    # Detection says "Columbia Place" but street set has "COLUMBIA PLACE"
    assert matches_any_street("Columbia Place", STREETS)


def test_no_match_unknown_street():
    assert not matches_any_street("Bourbon Street", STREETS)


def test_no_match_partial_word():
    # "CANAL" should not match "CANAL STREET" as a suffix test — but it does
    # because "CANAL" is a prefix of "CANAL STREET "
    assert matches_any_street("Canal", STREETS)


def test_no_match_random_word():
    assert not matches_any_street("HELLO", STREETS)


def test_direction_expanded_before_matching():
    # "N Rampart St" → "NORTH RAMPART STREET" which is in the set
    assert matches_any_street("N Rampart St", STREETS)


def test_empty_streets_set():
    assert not matches_any_street("Canal Street", set())


def test_empty_text():
    assert not matches_any_street("", STREETS)


def test_number_only_does_not_match():
    # Block numbers should not match any street
    assert not matches_any_street("1200", STREETS)


def test_prefix_of_detection_longer_than_street():
    # Detection reads "Esplanade Ave Ext" — "ESPLANADE AVENUE" is a prefix of the normalized text
    assert matches_any_street("Esplanade Ave Ext", STREETS)


def test_ignores_north_south():
    assert matches_any_street("rampart", STREETS)


def test_saint_prefix_stripped():
    # "CHARLES" should match "SAINT CHARLES AVENUE" once "SAINT" is stripped from the key
    assert matches_any_street("Charles", {"SAINT CHARLES AVENUE"})


def test_saint_prefix_stripped_with_type():
    assert matches_any_street("Charles Ave", {"SAINT CHARLES AVENUE"})


def test_saint_prefix_no_false_positive():
    assert not matches_any_street("Joseph", {"SAINT CHARLES AVENUE"})


def test_no_match_short_street_prefix():
    # "W A R E" normalizes to "WEST A R E" which starts with "WEST " — but a 1-word
    # street name ("WEST") should not match any text that happens to start with it.
    # This prevents WARE HOUSE waterfront labels from matching the street "WEST".
    assert not matches_any_street("W A R E", {"WEST", "WEST ROAD"})


def test_multi_word_street_prefix_still_matches():
    # "Esplanade Ave Ext" → "ESPLANADE AVENUE EXT"; "ESPLANADE AVENUE" (2 words) is
    # a valid prefix of the longer detection.
    assert matches_any_street("Esplanade Ave Ext", STREETS)


# ---------------------------------------------------------------------------
# canonical_street_match
# ---------------------------------------------------------------------------

# Use only the suffixed (full) forms — real block_index keys never have both "X" and
# "X STREET" as separate entries, and having both makes fuzzy tie-breaking non-deterministic.
FUZZY_STREETS = {
    "TCHOUPITOULAS STREET",
    "FRONT STREET",
    "CANAL STREET",
}


def test_canonical_exact_returns_key():
    # Exact match via normalize_street — returned value is the actual block_index key.
    result = canonical_street_match("Tchoupitoulas St", FUZZY_STREETS)
    assert result == "TCHOUPITOULAS STREET"


def test_canonical_fuzzy_one_edit():
    # TCHUUPITOULAS → TCHOUPITOULAS (stripped from key): edit dist 1, norm ~0.077
    result = canonical_street_match(
        "TCHUUPITOULAS", FUZZY_STREETS, fuzzy_threshold=0.20
    )
    assert result == "TCHOUPITOULAS STREET"


def test_canonical_fuzzy_two_edits():
    # TCHOUATOULAS → TCHOUPITOULAS (stripped): edit dist 2, norm ~0.15
    result = canonical_street_match("TCHOUATOULAS", FUZZY_STREETS, fuzzy_threshold=0.20)
    assert result == "TCHOUPITOULAS STREET"


def test_canonical_fuzzy_prefix():
    # SFRONT → FRONT (stripped from FRONT STREET): edit dist 1, norm ~0.17
    result = canonical_street_match("SFRONT", FUZZY_STREETS, fuzzy_threshold=0.20)
    assert result == "FRONT STREET"


def test_canonical_saint_prefix_returns_key():
    # "CHARLES" (no SAINT prefix) matches "SAINT CHARLES AVENUE" via prefix stripping.
    # The returned value is the actual key, not the bare normalized text.
    result = canonical_street_match("Charles", {"SAINT CHARLES AVENUE"})
    assert result == "SAINT CHARLES AVENUE"


def test_canonical_direction_prefix_returns_key():
    # "RAMPART" matches "NORTH RAMPART STREET" via direction prefix stripping.
    # The returned key is the full name so block_index lookups succeed.
    result = canonical_street_match("RAMPART", {"NORTH RAMPART STREET"})
    assert result == "NORTH RAMPART STREET"


def test_canonical_fuzzy_type_stripped():
    # JOSEBH → JOSEPH (stripped from JOSEPH STREET): edit dist 1, norm 1/6 ≈ 0.17
    result = canonical_street_match("JOSEBH", {"JOSEPH STREET"}, fuzzy_threshold=0.20)
    assert result == "JOSEPH STREET"


def test_canonical_fuzzy_disabled_by_default():
    assert canonical_street_match("TCHUUPITOULAS", FUZZY_STREETS) is None


def test_canonical_fuzzy_above_threshold():
    # Normalized dist ~0.15, but threshold set too low to match
    assert (
        canonical_street_match("TCHOUATOULAS", FUZZY_STREETS, fuzzy_threshold=0.10)
        is None
    )


def test_canonical_below_min_length():
    # "OAK" (3 chars) is below min_fuzzy_len=5 — no fuzzy match
    assert (
        canonical_street_match("OAX", {"OAK STREET", "OAK"}, fuzzy_threshold=0.20)
        is None
    )


def test_canonical_no_match():
    assert (
        canonical_street_match("ZZZZZZZZZ", FUZZY_STREETS, fuzzy_threshold=0.20) is None
    )
