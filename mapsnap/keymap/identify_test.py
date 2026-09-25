import json
from pathlib import Path

from mapsnap.keymap.identify import (
    MIN_DISTINCT,
    candidate_keys,
    detection_plan,
    is_keymap,
    legitimate_keymap_split,
    log_plan,
    page_zero_stems,
    panel_flush_edges,
    volume_valid_pages,
)


def make_volume(tmp_path: Path, names: list[str]) -> Path:
    volume = tmp_path / "vol"
    volume.mkdir()
    for name in names:
        (volume / name).write_bytes(b"")
    return volume


# A key map indexes the volume's other sheets, so detection_plan only nominates
# one for a volume with at least MIN_DISTINCT of them (its min_distinct floor).
# These are that volume, in the tests that are about which sheet gets nominated.
INDEXED = [f"p{number}.jpg" for number in range(20, 28)]


def make_indexed_volume(tmp_path: Path, names: list[str]) -> Path:
    """``names`` plus enough ordinary map pages for a key map to be worth having."""
    return make_volume(tmp_path, [*names, *INDEXED])


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
    volume = make_indexed_volume(tmp_path, ["p0.jpg", "p0b.jpg", "p1.jpg", "p5.jpg"])
    assumed, to_test = detection_plan(volume)
    assert assumed == ["p0", "p0b"]
    assert to_test == []


def test_detection_plan_split_panels_tested_individually(tmp_path: Path):
    # Kansas City case: the page-0 sheet mixes the key map with a volume-index
    # map, so its panels are confirmed one by one and the composite parent is
    # dropped; the page-1 family fallback stays.
    volume = make_indexed_volume(
        tmp_path, ["p0.jpg", "p0__1.jpg", "p0__2.jpg", "p1N.jpg", "p5.jpg"]
    )
    assumed, to_test = detection_plan(volume)
    assert assumed == []
    assert to_test == ["p0__1", "p0__2", "p1N"]


def _write_panels(volume: Path, parent: str, rings: list[list[list[float]]]) -> None:
    (volume / f"{parent}.panels.json").write_text(
        json.dumps(
            {"image": f"{parent}.jpg", "width": 1000, "height": 2000, "panels": rings}
        )
    )


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def test_panel_flush_edges_counts_touched_sheet_edges():
    full = _rect(0, 0, 1000, 2000)
    assert panel_flush_edges(full, 1000, 2000) == 4
    corner = _rect(0, 1400, 300, 2000)  # bottom-left inset: left + bottom
    assert panel_flush_edges(corner, 1000, 2000) == 2
    notch = _rect(150, 0, 320, 600)  # hangs off the top edge only
    assert panel_flush_edges(notch, 1000, 2000) == 1
    floating = _rect(300, 300, 600, 900)
    assert panel_flush_edges(floating, 1000, 2000) == 0
    # Within 2% of an edge counts as flush (scan borders, polygon rounding).
    nearly = _rect(15, 1400, 300, 1990)
    assert panel_flush_edges(nearly, 1000, 2000) == 2


def test_detection_plan_rejects_a_split_with_a_one_edge_notch(tmp_path: Path):
    """Chicago: a 4% notch flush with one edge is the splitter chasing linework,
    and testing the panels would lose the ten page numbers inside it. The
    parent sheet is tested whole instead; the KEY box (two edges) is fine."""
    volume = make_indexed_volume(
        tmp_path, ["p0.jpg", "p0__1.jpg", "p0__2.jpg", "p0__3.jpg", "p1N.jpg", "p5.jpg"]
    )
    _write_panels(
        volume,
        "p0",
        [
            _rect(0, 0, 1000, 2000),
            _rect(140, 0, 320, 580),
            _rect(770, 1640, 1000, 2000),
        ],
    )
    assert legitimate_keymap_split(volume, "p0") is False
    assumed, to_test = detection_plan(volume)
    assert assumed == []
    assert to_test == ["p0", "p1N"]


def test_detection_plan_keeps_a_split_of_edge_boxes(tmp_path: Path):
    # Kansas City: the volume-index inset is a boxed corner region flush with
    # two edges, so the panels are still confirmed one by one.
    volume = make_indexed_volume(
        tmp_path, ["p0.jpg", "p0__1.jpg", "p0__2.jpg", "p1N.jpg"]
    )
    _write_panels(volume, "p0", [_rect(0, 0, 1000, 2000), _rect(0, 1360, 320, 2000)])
    assert legitimate_keymap_split(volume, "p0") is True
    assert detection_plan(volume) == ([], ["p0__1", "p0__2", "p1N"])


def test_legitimate_keymap_split_without_panels_json_stands(tmp_path: Path):
    volume = make_volume(tmp_path, ["p0.jpg", "p0__1.jpg", "p0__2.jpg"])
    assert legitimate_keymap_split(volume, "p0") is True


def test_log_plan_records_the_split_verdict_per_sheet(tmp_path: Path):
    from mapsnap.keymap.log import read_section

    volume = make_indexed_volume(
        tmp_path, ["p0.jpg", "p0__1.jpg", "p0__2.jpg", "p1N.jpg"]
    )
    _write_panels(volume, "p0", [_rect(0, 0, 1000, 2000), _rect(140, 0, 320, 580)])
    assumed, to_test = detection_plan(volume)
    log_plan(volume, assumed, to_test)
    assert read_section(volume / "p0.jpg", "keymap-plan") == [
        "split rejected (a cut-away is flush with <2 sheet edges): testing the parent sheet p0 whole"
    ]
    # A sheet with no panels and no panels.json has nothing to record.
    assert read_section(volume / "p1N.jpg", "keymap-plan") is None
    # An unsplit page-0 family is recorded as the convention it is.
    (tmp_path / "plain").mkdir()
    plain = make_indexed_volume(tmp_path / "plain", ["p0.jpg", "p5.jpg"])
    log_plan(plain, *detection_plan(plain))
    assert read_section(plain / "p0.jpg", "keymap-detect") == [
        "page-0 sheet with no split panels: key map by convention"
    ]


def test_detection_plan_no_page_zero_uses_candidates(tmp_path: Path):
    volume = make_indexed_volume(tmp_path, ["p1a.jpg", "p1b.jpg", "p125.jpg"])
    assumed, to_test = detection_plan(volume)
    assert assumed == []
    assert to_test == ["p1a", "p1b"]


def test_candidate_keys_letter_pages(tmp_path: Path):
    # Los Angeles-style: un-numbered index sheets pa/pb are the key maps.
    volume = make_volume(tmp_path, ["pa.jpg", "pb.jpg", "p1401.jpg", "p1402.jpg"])
    assert candidate_keys(volume) == ["pa", "pb"]


def test_detection_plan_letter_page_splits_tested_as_panels(tmp_path: Path):
    # A split letter-page candidate is tested panel-by-panel; the parent is dropped.
    volume = make_indexed_volume(
        tmp_path,
        ["pa.jpg", "pa__1.jpg", "pa__2.jpg", "pb.jpg", "p1401.jpg"],
    )
    assumed, confirm = detection_plan(volume)
    assert assumed == []
    assert confirm == ["pa__1", "pa__2", "pb"]


def test_letter_page_panels_do_not_pollute_valid_pages(tmp_path: Path):
    # pa__1's panel index must not be read as a volume page number.
    volume = make_volume(tmp_path, ["pa.jpg", "pa__1.jpg", "p1401.jpg", "p1402.jpg"])
    assert volume_valid_pages(volume) == ["1401", "1402"]


def test_page_globs_ignore_the_roadprob_sidecars(tmp_path: Path):
    """#412's P(road) sidecars are ``.jpg`` beside the pages and ``image_stem``
    strips them to the same key, so a bare ``p*.jpg`` glob names every page
    twice. The corpus writes them before the key-map chain runs, and Gardiner
    NY 1913 came back as ``{"keys": ["p0", "p0"]}``."""
    volume = make_indexed_volume(
        tmp_path, ["p0.jpg", "p0.roadprob.jpg", "p1.jpg", "p1.roadprob.jpg"]
    )
    for name in list(INDEXED):
        (volume / name.replace(".jpg", ".roadprob.jpg")).write_bytes(b"")
    assert page_zero_stems(volume) == (["p0"], [])
    assert candidate_keys(volume) == ["p0", "p1"]
    assert detection_plan(volume) == (["p0"], [])


def test_panel_globs_ignore_the_roadprob_sidecars(tmp_path: Path):
    # The same duplication one level down: New Orleans 1951 recorded
    # {"keys": ["p0__1", "p0__1"]} once its panels had P(road) maps.
    volume = make_indexed_volume(
        tmp_path,
        [
            "p0.jpg",
            "p0__1.jpg",
            "p0__1.roadprob.jpg",
            "p0__2.jpg",
            "p0__2.roadprob.jpg",
        ],
    )
    _write_panels(volume, "p0", [_rect(0, 0, 1000, 2000), _rect(0, 1360, 320, 2000)])
    assert detection_plan(volume) == ([], ["p0__1", "p0__2"])


def test_detection_plan_declines_a_volume_with_nothing_to_index(tmp_path: Path):
    """Gardiner NY 1913 is one sheet of town: page 0 with no other page to
    index. ``is_keymap`` could never confirm a candidate there (it wants
    MIN_DISTINCT distinct valid reads and there are not that many pages to
    read), so the convention must not assert one either -- calling that sheet a
    key map left the volume with no scannable page at all."""
    assert detection_plan(make_volume(tmp_path, ["p0.jpg"])) == ([], [])
    (tmp_path / "small").mkdir()
    small = make_volume(tmp_path / "small", ["p0.jpg", "p1.jpg", "p2.jpg", "p3.jpg"])
    assert detection_plan(small) == ([], [])


def test_detection_plan_floor_is_the_one_is_keymap_applies(tmp_path: Path):
    # Exactly MIN_DISTINCT pages to index is enough: a candidate could clear
    # the floor by reading all of them, so the convention may speak.
    names = [f"p{number}.jpg" for number in range(1, MIN_DISTINCT + 1)]
    volume = make_volume(tmp_path, ["p0.jpg", *names])
    assert len(volume_valid_pages(volume)) == MIN_DISTINCT
    assert detection_plan(volume) == (["p0"], [])
    (tmp_path / "short").mkdir()
    one_short = make_volume(tmp_path / "short", ["p0.jpg", *names[:-1]])
    assert detection_plan(one_short) == ([], [])


def test_volume_valid_pages_names_half_scanned_sheets_by_number(tmp_path: Path):
    # sanborn06116_006: each sheet scanned as a left and a right half, while
    # the key map prints the sheet's number. Checked against the halves, the
    # 16 of 18 numbers it read matched nothing and it was rejected.
    names = ["p0L.jpg", "p0R.jpg"] + [
        f"p{number}{half}.jpg" for number in range(85, 88) for half in "LR"
    ]
    volume = make_volume(tmp_path, names)
    assert volume_valid_pages(volume) == ["85", "86", "87"]
