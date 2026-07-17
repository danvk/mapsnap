from pathlib import Path

from mapsnap.keymap.identify import (
    candidate_keys,
    detection_plan,
    is_keymap,
    page_zero_stems,
    volume_valid_pages,
)


def make_volume(tmp_path: Path, names: list[str]) -> Path:
    volume = tmp_path / "vol"
    volume.mkdir()
    for name in names:
        (volume / name).write_bytes(b"")
    return volume


def test_candidate_keys_page_zero_family(tmp_path: Path):
    # p0 is the key map; taking the two smallest numbers also nominates the page-1 page.
    volume = make_volume(
        tmp_path, ["p0.jpg", "p1N.jpg", "p5.jpg", "p112N.jpg", "covr.jpg"]
    )
    assert candidate_keys(volume) == ["p0", "p1N"]


def test_candidate_keys_lettered_page_one_family(tmp_path: Path):
    # No page 0 (washington-style): the page-1 family p1a-d are the candidates.
    volume = make_volume(
        tmp_path, ["p1a.jpg", "p1b.jpg", "p1c.jpg", "p1d.jpg", "p125.jpg"]
    )
    assert candidate_keys(volume) == ["p1a", "p1b", "p1c", "p1d"]


def test_candidate_keys_skips_split_panels(tmp_path: Path):
    volume = make_volume(tmp_path, ["p0.jpg", "p1N.jpg", "p1N__2.jpg", "p1N__3.jpg"])
    assert candidate_keys(volume) == ["p0", "p1N"]


def test_volume_valid_pages_positive_only(tmp_path: Path):
    # p0's "page 0" and non-numeric covr are excluded; split panels collapse to their number.
    volume = make_volume(
        tmp_path, ["p0.jpg", "p1.jpg", "p2.jpg", "p2__2.jpg", "covr.jpg"]
    )
    assert volume_valid_pages(volume) == ["1", "2"]


def test_is_keymap_high_coverage():
    assert is_keymap(98, 112)  # chicago key map: 0.88
    assert is_keymap(23, 24)  # champaign key map: 0.96


def test_is_keymap_rejects_regular_page():
    assert not is_keymap(1, 112)  # a coincidental valid read
    assert not is_keymap(0, 101)  # detroit p66: many candidates, zero valid coverage


def test_is_keymap_accepts_split_halfmap():
    # A split key map covers roughly half the volume — still far above any regular page.
    assert is_keymap(50, 112)


def test_is_keymap_absolute_floor_guards_tiny_volume():
    # 5 valid reads out of 12 is 0.42 coverage but below the absolute distinct floor.
    assert not is_keymap(5, 12, min_coverage=0.3, min_distinct=6)


def test_is_keymap_empty_volume():
    assert not is_keymap(0, 0)


def test_page_zero_stems_splits_and_variants(tmp_path: Path):
    volume = make_volume(
        tmp_path, ["p0.jpg", "p0b.jpg", "p0__1.jpg", "p0__2.jpg", "p1.jpg"]
    )
    unsplit, splits = page_zero_stems(volume)
    assert unsplit == ["p0", "p0b"]
    assert splits == ["p0__1", "p0__2"]


def test_detection_plan_unsplit_page_zero_short_circuits(tmp_path: Path):
    # Nashville/Grand Rapids case: p0.jpg with no splits is the key map, no
    # model confirmation needed.
    volume = make_volume(tmp_path, ["p0.jpg", "p0b.jpg", "p1.jpg", "p5.jpg"])
    assumed, to_test = detection_plan(volume)
    assert assumed == ["p0", "p0b"]
    assert to_test == []


def test_detection_plan_split_panels_tested_individually(tmp_path: Path):
    # Kansas City case: the page-0 sheet mixes the key map with a volume-index
    # map, so its panels are confirmed one by one and the composite parent is
    # dropped; the page-1 family fallback stays.
    volume = make_volume(
        tmp_path, ["p0.jpg", "p0__1.jpg", "p0__2.jpg", "p1N.jpg", "p5.jpg"]
    )
    assumed, to_test = detection_plan(volume)
    assert assumed == []
    assert to_test == ["p0__1", "p0__2", "p1N"]


def test_detection_plan_no_page_zero_uses_candidates(tmp_path: Path):
    volume = make_volume(tmp_path, ["p1a.jpg", "p1b.jpg", "p125.jpg"])
    assumed, to_test = detection_plan(volume)
    assert assumed == []
    assert to_test == ["p1a", "p1b"]
