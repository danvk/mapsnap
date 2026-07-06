from mapsnap.keymap.detect_numbers_crnn import (
    choose_reread,
    levenshtein,
    snap_to_pages,
)

# A reread result triple is (polygon, text, confidence); the polygon is irrelevant to the gate.
_POLY = [[0, 0], [1, 0], [1, 1], [0, 1]]


def _reread(text: str, confidence: float = 0.9):
    return (_POLY, text, confidence)


def test_levenshtein():
    assert levenshtein("21", "21") == 0
    assert levenshtein("2I", "21") == 1
    assert levenshtein("105", "10") == 1


def test_snap_to_pages_repairs_within_distance():
    assert snap_to_pages("21", ["1", "21", "64"]) == "21"  # exact stays
    # "521" is one substitution from "520" and >=2 from the others, so it snaps to "520".
    assert snap_to_pages("521", ["518", "520", "530"], max_distance=1) == "520"


def test_snap_to_pages_keeps_far_text():
    pages = [str(n) for n in range(1, 65)]
    # "999" is far from any 1-64 page -> unchanged.
    assert snap_to_pages("999", pages) == "999"


def test_snap_to_pages_no_pages_is_identity():
    assert snap_to_pages("42", []) == "42"


# choose_reread — the narrow-re-read agreement gate ------------------------------------------

VALID = {
    str(n) for n in range(1, 113)
}  # a normal 1-112 volume, so "69"/"61" are valid pages


def test_choose_reread_accepts_when_all_widths_agree():
    # Both tight crops resolve the squished "1" to the same valid multi-digit page.
    chosen = choose_reread("1", [_reread("105", 0.8), _reread("105", 0.9)], VALID)
    assert chosen is not None
    assert chosen[1] == "105"
    assert chosen[2] == 0.9  # highest-confidence instance of the agreed page


def test_choose_reread_rejects_disagreement():
    # A hallucinated second digit is unstable across crop widths ("69" vs "61"): keep "9".
    assert choose_reread("9", [_reread("69"), _reread("61")], VALID) is None


def test_choose_reread_rejects_lone_width():
    # Only one width resolves a longer page; without corroboration, keep the single digit.
    assert choose_reread("9", [_reread("69"), None], VALID) is None


def test_choose_reread_rejects_non_valid_page():
    # Both widths agree, but the number is not a valid page in this volume -> keep "1".
    assert choose_reread("1", [_reread("999"), _reread("999")], VALID) is None


def test_choose_reread_rejects_when_not_longer():
    # Re-reads that are not strictly longer than the original never upgrade it.
    assert choose_reread("5", [_reread("5"), _reread("5")], VALID) is None
