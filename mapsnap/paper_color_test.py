"""Tests for the off-paper background test."""

from mapsnap.paper_color import (
    BACKGROUND_CHROMA_MARGIN,
    BACKGROUND_HUE_MARGIN,
    BACKGROUND_HUE_MIN_CHROMA,
    hue_difference,
    is_off_paper,
)


def color(hue: float, chroma: float) -> dict:
    """A recorded {color, hue, chroma} background."""
    return {"color": "#000000", "hue": hue, "chroma": chroma}


def test_hue_difference_goes_the_short_way_round():
    assert hue_difference(10, 350) == 20
    assert hue_difference(97, 194) == 97
    assert hue_difference(0, 180) == 180
    assert hue_difference(720, 0) == 0


def test_more_saturated_than_the_paper_is_off_it_whatever_the_hue():
    paper = color(95, 15)  # yellowed paper
    assert is_off_paper(color(95, 15 + BACKGROUND_CHROMA_MARGIN + 0.1), paper)
    assert not is_off_paper(color(95, 15 + BACKGROUND_CHROMA_MARGIN), paper)


def test_a_far_hue_is_off_the_paper_even_when_paler():
    # Miami p25's '2 ND': a pale blue fill (hue 194, chroma 4.1) on chroma-8 paper at hue 97.
    paper = color(97, 8)
    assert is_off_paper(color(194, 4.1), paper)
    assert not is_off_paper(color(97 + BACKGROUND_HUE_MARGIN - 1, 4.1), paper)


def test_a_nearly_neutral_box_has_no_meaningful_hue():
    # Near-white paper: a street label's box can measure any hue at chroma ~1-2.
    paper = color(90, 2)
    assert not is_off_paper(color(270, BACKGROUND_HUE_MIN_CHROMA - 0.1), paper)
    assert is_off_paper(color(270, BACKGROUND_HUE_MIN_CHROMA), paper)


def test_without_a_paper_a_recorded_background_is_off_it():
    # Files written before the paper colour was recorded only kept off-paper backgrounds.
    assert is_off_paper(color(95, 2), None)
