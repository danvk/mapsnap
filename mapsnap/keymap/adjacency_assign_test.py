import json

from mapsnap.keymap.adjacency_assign import (
    Repair,
    SheetPanels,
    adjacency_graphs,
    digit_family,
    gap_placement,
    panel_scale,
    plan_cross_sheet_repairs,
    plan_gap_repairs,
    plan_sheet_repairs,
    proximity_graph,
    split_multiplicity,
    support_for,
)


def graph(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for first, second in pairs:
        out.setdefault(first, set()).add(second)
        out.setdefault(second, set()).add(first)
    return out


def sheet(labels: list[str], contacts: list[tuple[int, int]]) -> SheetPanels:
    return SheetPanels("p0", labels, {frozenset(pair) for pair in contacts})


def test_digit_family_covers_lost_digits():
    keys = {"2", "20", "22", "29", "65", "6"}
    # A "2" read could be any longer key containing 2 (a lost digit either side).
    assert digit_family("2", keys) == {"20", "22", "29"}
    assert digit_family("6", keys) == {"65"}
    # Same-length or shorter keys are never candidates, and a letters-only read
    # has no digits to extend.
    assert digit_family("22", {"2", "22"}) == set()
    assert digit_family("", keys) == set()


def test_detroit_duplicate_relabels_the_unsupported_instance():
    # Detroit: two panels read "2". The real p2 has no mutual edges at all;
    # the other panel sits among 22's neighbours p21, p28, p29.
    panels = sheet(
        labels=["2", "2", "21", "28", "29", "3"],
        contacts=[(0, 5), (1, 2), (1, 3), (1, 4)],
    )
    mutual = graph([("22", "21"), ("22", "28"), ("22", "29")])
    repairs = plan_sheet_repairs(panels, mutual, {}, {"2", "3", "21", "22", "28", "29"})
    assert [(r.index, r.old, r.new) for r in repairs] == [(1, "2", "22")]
    assert repairs[0].evidence == ("21", "28", "29")
    assert repairs[0].support == 3.0


def test_split_twins_with_support_both_keep_their_labels():
    # Champaign draws p4 on four panels; each touches a different neighbour of
    # p4, so every instance is vouched for and none is a suspect.
    panels = sheet(
        labels=["4", "4", "4", "4", "3", "5", "6", "7"],
        contacts=[(0, 4), (1, 5), (2, 6), (3, 7)],
    )
    mutual = graph([("4", "3"), ("4", "5"), ("4", "6"), ("4", "7")])
    assert plan_sheet_repairs(panels, mutual, {}, {"3", "4", "5", "6", "7", "44"}) == []


def test_duplicate_with_no_support_anywhere_is_left_alone():
    # Neither instance is vouched for AND no candidate key is either: the pass
    # must not guess. (When a candidate IS vouched for, the sibling's score is
    # irrelevant -- see the Detroit case above, where p2 has no edges at all.)
    panels = sheet(labels=["9", "9", "3"], contacts=[(0, 2), (1, 2)])
    assert plan_sheet_repairs(panels, graph([]), {}, {"9", "3", "19"}) == []


def test_relabel_requires_two_mutual_neighbours():
    # One vouching neighbour is below RELABEL_MIN_MUTUAL: no action.
    panels = sheet(labels=["2", "2", "21", "9"], contacts=[(0, 3), (1, 2)])
    mutual = graph([("22", "21"), ("2", "9")])
    assert plan_sheet_repairs(panels, mutual, {}, {"2", "9", "21", "22"}) == []


def test_one_sided_claims_alone_never_relabel():
    # Four one-sided claims score 1.0, still under the bar that two mutual
    # edges clear -- they are 32-54% precise and may not decide anything.
    panels = sheet(
        labels=["2", "2", "21", "28", "29", "31"],
        contacts=[(0, 5), (1, 2), (1, 3), (1, 4), (1, 5)],
    )
    one_sided = graph([("22", "21"), ("22", "28"), ("22", "29"), ("22", "31")])
    mutual = graph([("2", "31")])
    assert (
        plan_sheet_repairs(
            panels, mutual, one_sided, {"2", "22", "21", "28", "29", "31"}
        )
        == []
    )


def test_ambiguous_candidates_change_nothing():
    # The zero-support "2" could be 22 or 29 on the evidence: refuse both.
    panels = sheet(
        labels=["2", "2", "21", "28", "3", "9"],
        contacts=[(0, 4), (1, 2), (1, 3)],
    )
    mutual = graph([("22", "21"), ("22", "28"), ("29", "21"), ("29", "28"), ("2", "3")])
    keys = {"2", "3", "9", "21", "22", "28", "29"}
    assert plan_sheet_repairs(panels, mutual, {}, keys) == []


def test_support_for_counts_mutual_and_weights_one_sided():
    panels = sheet(labels=["7", "8", "9"], contacts=[(0, 1), (0, 2)])
    mutual = graph([("7", "8")])
    one_sided = graph([("7", "9")])
    score, vouchers = support_for(0, "7", panels, mutual, one_sided)
    assert score == 1.25 and vouchers == ("8", "9")


def test_cross_sheet_strips_only_the_unsupported_copy():
    # Brooklyn: key 8 drawn on both sheets; only p0b's copy touches 8's
    # neighbours, so p0's copy is stripped.
    first = SheetPanels("p0", ["8", "40"], {frozenset((0, 1))})
    second = SheetPanels("p0b", ["8", "7", "9"], {frozenset((0, 1)), frozenset((0, 2))})
    mutual = graph([("8", "7"), ("8", "9")])
    repairs = plan_cross_sheet_repairs([first, second], mutual, {})
    assert [(r.sheet, r.index, r.old, r.new) for r in repairs] == [("p0", 0, "8", None)]


def test_cross_sheet_respects_split_multiplicity():
    # A page split into two panels is drawn twice on purpose.
    first = SheetPanels("p0", ["8", "7"], {frozenset((0, 1))})
    second = SheetPanels("p0b", ["8", "40"], set())
    mutual = graph([("8", "7")])
    assert plan_cross_sheet_repairs([first, second], mutual, {}, {"8": 2}) == []
    # Beyond the expected multiplicity the asymmetry applies again.
    assert plan_cross_sheet_repairs([first, second], mutual, {}, {"8": 1}) != []


def test_gap_recovery_places_a_missing_key_from_its_citations():
    # Detroit p59: no detection anywhere, but p27 cites it mutually and
    # p57/p61 one-sidedly, so those three regions bracket where it belongs.
    panels = sheet(labels=["27", "57", "61", "40"], contacts=[])
    mutual = graph([("59", "27"), ("59", "58")])
    one_sided = graph([("59", "57"), ("59", "61")])
    placed = plan_gap_repairs(panels, mutual, one_sided, {"27", "57", "61", "40", "59"})
    assert [(r.new, r.old) for r in placed] == [("59", None)]
    assert placed[0].support == 1.5  # 1 mutual + 2 one-sided at 0.25
    assert placed[0].evidence_indices == (0, 1, 2)


def test_gap_recovery_needs_more_than_a_lone_mutual_citation():
    panels = sheet(labels=["27", "40"], contacts=[])
    assert plan_gap_repairs(panels, graph([("59", "27")]), {}, {"27", "40", "59"}) == []


def test_gap_recovery_never_places_on_one_sided_claims_alone():
    # Eight one-sided claims reach 2.0 but include no mutual edge: refused,
    # because one-sided claims are only 32-54% precise.
    labels = [str(n) for n in range(70, 78)]
    panels = sheet(labels=labels, contacts=[])
    one_sided = graph([("59", label) for label in labels])
    assert plan_gap_repairs(panels, {}, one_sided, {*labels, "59"}) == []


def test_gap_placement_averages_the_mutual_citations():
    repair = Repair(
        sheet="p0",
        index=None,
        old=None,
        new="59",
        reason="gap",
        support=2.0,
        evidence=("27", "28"),
        evidence_indices=(0, 1),
        mutual_indices=(0, 1),
    )
    centroids: list[tuple[float, float] | None] = [(100.0, 100.0), (300.0, 100.0)]
    assert gap_placement(repair, centroids, scale=200.0) == (200.0, 100.0)
    # Mutual citations that disagree on a location mean one is misassigned.
    assert gap_placement(repair, [(0.0, 0.0), (5000.0, 0.0)], scale=200.0) is None
    # No usable centroid, and no mutual citation at all.
    assert gap_placement(repair, [None, None], scale=200.0) is None
    weak_only = Repair(
        sheet="p0",
        index=None,
        old=None,
        new="59",
        reason="gap",
        evidence_indices=(0, 1),
        mutual_indices=(),
    )
    assert gap_placement(weak_only, centroids, scale=200.0) is None


def test_gap_placement_discards_a_far_flung_one_sided_citation():
    # Detroit p59: p27 cites it mutually; p57 and p61 agree nearby; the p1
    # claim is junk (#213 names it) and sits across the sheet. Averaging p1 in
    # would throw the placement, so it is dropped and the rest still count.
    repair = Repair(
        sheet="p0",
        index=None,
        old=None,
        new="59",
        reason="gap",
        support=1.75,
        evidence=("27", "57", "61", "1"),
        evidence_indices=(0, 1, 2, 3),
        mutual_indices=(0,),
    )
    centroids: list[tuple[float, float] | None] = [
        (1000.0, 1000.0),  # 27, the mutual anchor
        (1100.0, 1000.0),  # 57
        (900.0, 1000.0),  # 61
        (9000.0, 9000.0),  # 1, junk
    ]
    assert gap_placement(repair, centroids, scale=200.0) == (1000.0, 1000.0)


def test_proximity_graph_joins_islands_within_a_few_region_widths():
    # 100px squares: neighbours 150px apart are joined (Detroit's regions are
    # islands separated by blank paper), one 900px away is not.
    squares: list[list[list[float]]] = [
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]],
        [[150.0, 0.0], [250.0, 0.0], [250.0, 100.0], [150.0, 100.0], [150.0, 0.0]],
        [[900.0, 0.0], [1000.0, 0.0], [1000.0, 100.0], [900.0, 100.0], [900.0, 0.0]],
    ]
    assert panel_scale(squares) == 100.0
    assert proximity_graph(squares) == {frozenset((0, 1))}
    # Degenerate rings are skipped rather than crashing.
    assert proximity_graph([[[0.0, 0.0], [1.0, 1.0]]]) == set()
    assert panel_scale([]) == 0.0


def test_adjacency_graphs_and_split_multiplicity(tmp_path):
    (tmp_path / "adjacency.json").write_text(
        json.dumps(
            {
                "adjacency": [["p21", "p22"], ["p22", "p28"]],
                "one_sided": [["p24", "p22"]],
            }
        )
    )
    mutual, one_sided = adjacency_graphs(tmp_path)
    assert mutual["22"] == {"21", "28"} and one_sided["22"] == {"24"}
    # Missing file is an empty graph, not an error.
    assert adjacency_graphs(tmp_path / "nowhere") == ({}, {})

    (tmp_path / "p4.panels.json").write_text(json.dumps({"panels": [[], [], [], []]}))
    (tmp_path / "p9.panels.json").write_text(json.dumps({"panels": []}))
    assert split_multiplicity(tmp_path) == {"4": 4}
