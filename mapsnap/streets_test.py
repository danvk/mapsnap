"""Tests for mapsnap.streets."""

import pytest

from mapsnap.streets import normalize_street, _ORDINALS, _num_to_ordinal_word


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
