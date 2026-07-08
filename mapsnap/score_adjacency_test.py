from mapsnap.score_adjacency import (
    score_adjacency,
    truth_adjacent_pairs,
    truth_shapes,
)


def square(lon: float, lat: float, size_deg: float = 0.003) -> list[list[float]]:
    return [
        [lon, lat],
        [lon + size_deg, lat],
        [lon + size_deg, lat + size_deg],
        [lon, lat + size_deg],
        [lon, lat],
    ]


def test_truth_adjacent_pairs_touching_vs_far():
    # Pages 1 and 2 share an edge; page 3 is ~1.1 km away.
    shapes = truth_shapes(
        {
            1: [square(0.0, 0.0)],
            2: [square(0.003, 0.0)],
            3: [square(0.02, 0.0)],
        }
    )
    pairs = truth_adjacent_pairs(shapes, max_gap_m=30.0)
    assert pairs == {frozenset((1, 2))}


def test_truth_shapes_unions_split_footprints():
    # A split page's two footprints act as one geometry: page 2 touches only the
    # second footprint of page 1.
    shapes = truth_shapes(
        {
            1: [square(0.0, 0.0), square(0.006, 0.0)],
            2: [square(0.009, 0.0)],
        }
    )
    pairs = truth_adjacent_pairs(shapes, max_gap_m=30.0)
    assert pairs == {frozenset((1, 2))}


def make_doc() -> dict:
    return {
        "pages": {
            "p1": {"number": 1},
            "p2": {"number": 2},
            "p3": {"number": 3},
            "p9": {"number": 9},
        },
        "adjacency": [["p1", "p2"], ["p1", "p3"], ["p2", "p9"]],
    }


def test_score_adjacency_counts():
    truth_pairs = {frozenset((1, 2)), frozenset((2, 3))}
    truth_numbers = {1, 2, 3}
    score = score_adjacency(make_doc(), truth_pairs, truth_numbers)
    assert score.edges == 3
    assert score.known == 2  # p2-p9 lacks truth
    assert score.correct == 1  # p1-p2
    assert score.wrong == [("p1", "p3")]
    assert score.unknown == [("p2", "p9")]
    assert score.pages == 4
    assert score.pages_covered == 4


def test_score_adjacency_recall_restricted_to_scanned_pages():
    # Truth pair (2, 3) was not recovered; pair (3, 7) involves an unscanned page
    # and must not count against recall.
    truth_pairs = {frozenset((1, 2)), frozenset((2, 3)), frozenset((3, 7))}
    score = score_adjacency(make_doc(), truth_pairs, {1, 2, 3, 7})
    assert score.truth_pairs == 2
    assert score.recovered == 1


def test_score_adjacency_single_digit_stats():
    truth_pairs: set[frozenset[int]] = {frozenset((1, 2))}
    score = score_adjacency(make_doc(), truth_pairs, {1, 2, 3})
    # Both scored edges involve single-digit pages here.
    assert score.single_digit_known == 2
    assert score.single_digit_correct == 1
