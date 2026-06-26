"""Tests for ctc_vocab_decode helpers."""

import numpy as np

from mapsnap.ctc_vocab_decode import (
    HINT_STRINGS,
    build_trie,
    generate_vocab_strings,
    prefix_constrained_ctc,
)

# ---------------------------------------------------------------------------
# generate_vocab_strings
# ---------------------------------------------------------------------------

SIMPLE_STREETS = {"EAST GRAND AVENUE", "NORTH RAMPART STREET", "MAGAZINE STREET"}


def test_generate_includes_full_name():
    vocab = generate_vocab_strings(SIMPLE_STREETS)
    assert "EAST GRAND AVENUE" in vocab


def test_generate_includes_abbreviated_direction():
    vocab = generate_vocab_strings(SIMPLE_STREETS)
    assert "E GRAND" in vocab
    assert "E. GRAND" in vocab
    assert "N RAMPART" in vocab


def test_generate_includes_abbreviated_type():
    vocab = generate_vocab_strings(SIMPLE_STREETS)
    assert "EAST GRAND AVE" in vocab
    assert "EAST GRAND AV" in vocab
    assert "NORTH RAMPART ST" in vocab


def test_generate_includes_bare_name():
    # No direction prefix, no type suffix
    vocab = generate_vocab_strings(SIMPLE_STREETS)
    assert "GRAND" in vocab
    assert "RAMPART" in vocab
    assert "MAGAZINE" in vocab


def test_generate_no_direction_bare_name_with_type():
    # MAGAZINE STREET has no direction prefix; bare + with-type forms
    vocab = generate_vocab_strings({"MAGAZINE STREET"})
    assert "MAGAZINE" in vocab
    assert "MAGAZINE STREET" in vocab
    assert "MAGAZINE ST" in vocab


def test_generate_includes_individual_words_of_multiword_name():
    # CRAFT splits "VAN BRUNT" into separate boxes; each word must be recognizable on its
    # own (not just the combined form) so a split box does not snap to an unrelated street.
    vocab = generate_vocab_strings({"VAN BRUNT STREET", "VAN DYKE STREET"})
    assert "VAN BRUNT" in vocab  # combined form still present
    assert "VAN" in vocab
    assert "BRUNT" in vocab
    assert "DYKE" in vocab


def test_generate_single_word_name_adds_no_fragments():
    # A single-word base name has no parts to split, so nothing extra is added.
    vocab = generate_vocab_strings({"MAGAZINE STREET"})
    assert "MAGAZIN" not in vocab  # no spurious sub-word fragments


def test_generate_saint_prefix():
    vocab = generate_vocab_strings({"SAINT CHARLES AVENUE"})
    assert "SAINT CHARLES" in vocab
    assert "ST CHARLES" in vocab
    assert "CHARLES" in vocab


def test_generate_returns_sorted_list():
    vocab = generate_vocab_strings(SIMPLE_STREETS)
    assert vocab == sorted(vocab)


def test_generate_no_duplicates():
    vocab = generate_vocab_strings(SIMPLE_STREETS)
    assert len(vocab) == len(set(vocab))


def test_generate_numeric_ordinal_forms():
    # "SOUTH FOURTH STREET" → also generates "4TH" variants so the constrained
    # CTC decoder can match an image showing "S. 4TH ST".
    vocab = generate_vocab_strings({"SOUTH FOURTH STREET"})
    assert "S. 4TH ST" in vocab
    assert "S 4TH ST" in vocab
    assert "4TH STREET" in vocab
    assert "4TH ST" in vocab
    assert "4TH" in vocab
    # Word-form variants should still be present.
    assert "S. FOURTH ST" in vocab
    assert "SOUTH FOURTH STREET" in vocab


def test_generate_numeric_ordinal_compound():
    # "NORTH TWENTY FIRST AVENUE" → also generates "21ST" variants.
    vocab = generate_vocab_strings({"NORTH TWENTY FIRST AVENUE"})
    assert "21ST" in vocab
    assert "N 21ST AVE" in vocab
    assert "N. 21ST AV" in vocab


def test_generate_spaced_ordinal_forms():
    # Sanborn maps print "5 TH" (space between digit and suffix). The CTC model
    # sees the typographic gap as a space, so "5TH" has near-zero path probability.
    # Both the compact form ("5TH") and the spaced form ("5 TH") must be in vocab.
    vocab = generate_vocab_strings({"SOUTH FIFTH STREET"})
    assert "5TH" in vocab
    assert "5 TH" in vocab
    assert "S. 5 TH ST" in vocab
    assert "S. 5TH ST" in vocab


def test_generate_empty_input():
    # AVENUE- and STREET-family hint strings are always included even with no streets.
    vocab = set(generate_vocab_strings(set()))
    assert "AVENUE" in vocab
    assert "AVE" in vocab
    assert "AV" in vocab
    assert "STREET" in vocab
    assert "ST" in vocab
    # Other type words and direction words are no longer hints.
    assert "COURT" not in vocab
    assert "NORTH" not in vocab


def test_generate_leading_type_street():
    # "AVENUE X": type word is a prefix, not a suffix.
    vocab = generate_vocab_strings({"AVENUE X"})
    assert "AVENUE X" in vocab
    assert "AV X" in vocab
    assert "AVE X" in vocab
    assert "X" in vocab  # bare base name


def test_generate_west_street():
    # "WEST STREET": direction word is the street name, not a prefix (regression test).
    vocab = generate_vocab_strings({"WEST STREET"})
    assert "WEST" in vocab
    assert "W" in vocab
    assert "WEST ST" in vocab
    assert "W ST" in vocab


def test_generate_direction_suffix_numbered():
    # DC-style names like "EIGHTH STREET NORTHEAST": quadrant suffix omitted on map labels.
    vocab = set(generate_vocab_strings({"EIGHTH STREET NORTHEAST"}))
    assert "EIGHTH" in vocab
    assert "8TH" in vocab
    assert "8 TH" in vocab
    assert "EIGHTH ST" in vocab
    assert "EIGHTH STREET" in vocab
    # Full form with quadrant suffix is NOT required (map labels never show it).


def test_generate_direction_suffix_lettered():
    # "A STREET NORTHEAST": map label is just "A" or "A ST".
    vocab = set(generate_vocab_strings({"A STREET NORTHEAST"}))
    assert "A" in vocab
    assert "A ST" in vocab
    assert "A STREET" in vocab


def test_generate_direction_suffix_avenue():
    # "MASSACHUSETTS AVENUE NORTHEAST": labels omit quadrant.
    vocab = set(generate_vocab_strings({"MASSACHUSETTS AVENUE NORTHEAST"}))
    assert "MASSACHUSETTS" in vocab
    assert "MASSACHUSETTS AVE" in vocab
    assert "MASSACHUSETTS AVENUE" in vocab


def test_generate_includes_hint_strings():
    # AVENUE- and STREET-family hints appear in every vocab regardless of streets present.
    vocab = set(generate_vocab_strings({"MAGAZINE STREET"}))
    for word in ("AVENUE", "AVE", "AV", "AV.", "A V", "STREET", "ST", "ST.", "S T"):
        assert word in vocab, f"hint word {word!r} missing from vocab"


def test_hint_strings_constant():
    # HINT_STRINGS contains AVENUE/STREET families and quadrant abbreviations.
    assert "AVENUE" in HINT_STRINGS
    assert "AVE" in HINT_STRINGS
    assert "AV" in HINT_STRINGS
    assert "AV." in HINT_STRINGS
    assert "A V" in HINT_STRINGS
    assert "STREET" in HINT_STRINGS
    assert "ST" in HINT_STRINGS
    assert "ST." in HINT_STRINGS
    assert "S T" in HINT_STRINGS
    # Quadrant abbreviations in all four forms.
    for q in ("NW", "NE", "SE", "SW"):
        assert q in HINT_STRINGS, f"{q} missing"
        assert " ".join(q) in HINT_STRINGS, f"{' '.join(q)} missing"
        assert f"{q[0]}.{q[1]}." in HINT_STRINGS, f"{q[0]}.{q[1]}. missing"
        assert f"{q[0]}. {q[1]}." in HINT_STRINGS, f"{q[0]}. {q[1]}. missing"
    # Other type words and full direction words remain non-hints.
    assert "COURT" not in HINT_STRINGS
    assert "NORTH" not in HINT_STRINGS
    assert "N." not in HINT_STRINGS
    assert "SAINT" not in HINT_STRINGS
    # Should NOT contain multi-word forms.
    assert "EAST GRAND" not in HINT_STRINGS


# ---------------------------------------------------------------------------
# prefix_constrained_ctc
# ---------------------------------------------------------------------------

# Minimal character list: blank + A-Z + space.
_CHARS = ["[blank]"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]
_CHAR_IDX = {c: i for i, c in enumerate(_CHARS)}
_T = len(_CHARS)  # number of CTC classes


def _one_hot_seq(text: str) -> np.ndarray:
    """Build a (len(text), T) CTC prob matrix that decodes greedily to text."""
    mat = np.zeros((len(text), _T))
    for t, ch in enumerate(text.upper()):
        mat[t, _CHAR_IDX[ch]] = 1.0
    return mat


def _uniform_seq(length: int) -> np.ndarray:
    """Uniform distribution over all characters for each time step."""
    mat = np.ones((length, _T)) / _T
    return mat


def test_ctc_exact_match():
    vocab = ["GRAND"]
    trie = build_trie(vocab)
    mat = _one_hot_seq("GRAND")
    text, prob = prefix_constrained_ctc(mat, trie, _CHARS, beam_width=10)
    assert text == "GRAND"
    assert prob > 0.9


def test_ctc_selects_vocabulary_word_over_non_vocab():
    # CTC emissions spell "GRAMD" but "GRAND" is in vocab; beam should prefer "GRAND"
    # by keeping it alive from partial overlap even if "GRAMD" isn't in the vocab.
    vocab = ["GRAND"]
    trie = build_trie(vocab)
    mat = _one_hot_seq("GRAND")
    # Slightly corrupt the 4th char (N→M) but keep N nonzero
    mat[3, _CHAR_IDX["M"]] = 0.6
    mat[3, _CHAR_IDX["N"]] = 0.4
    text, prob = prefix_constrained_ctc(mat, trie, _CHARS, beam_width=10)
    assert text == "GRAND"
    assert prob > 0


def test_ctc_no_match_returns_empty():
    # Emissions strongly spell "HELLO" which is not in vocab
    vocab = ["GRAND"]
    trie = build_trie(vocab)
    mat = _one_hot_seq("HELLO")
    text, prob = prefix_constrained_ctc(mat, trie, _CHARS, beam_width=10)
    assert text == ""
    assert prob == 0.0


def test_ctc_selects_among_multiple_vocab_words():
    vocab = ["GRAND", "GRANT"]
    trie = build_trie(vocab)
    mat = _one_hot_seq("GRAND")
    text, prob = prefix_constrained_ctc(mat, trie, _CHARS, beam_width=10)
    assert text == "GRAND"
    assert prob > 0


def test_ctc_empty_sequence_returns_empty():
    vocab = ["GRAND"]
    trie = build_trie(vocab)
    mat = np.zeros((0, _T))
    text, prob = prefix_constrained_ctc(mat, trie, _CHARS, beam_width=10)
    assert text == ""
    assert prob == 0.0
