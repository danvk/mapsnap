from mapsnap.keymap.detect_numbers_crnn import levenshtein, snap_to_pages


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
