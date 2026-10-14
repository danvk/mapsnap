"""Tests for the plausible-scale band."""

from mapsnap.scale_bounds import (
    CORPUS_M_PER_PX,
    MAX_M_PER_PX,
    MIN_M_PER_PX,
    corpus_px_per_ft,
    m_per_px_is_plausible,
    px_per_ft_is_plausible,
)


def test_m_per_px_band_keeps_every_real_sheet_rung():
    for rung in (
        0.1,
        0.205,
        0.41,
        0.82,
        0.0375,
    ):  # 25/50/100/200 ft/in, Champaign's inset
        assert m_per_px_is_plausible(rung)
    assert m_per_px_is_plausible(MIN_M_PER_PX) and m_per_px_is_plausible(MAX_M_PER_PX)


def test_m_per_px_band_rejects_498s_cases():
    assert not m_per_px_is_plausible(0.0117)  # sanborn05857_001
    assert not m_per_px_is_plausible(14.15)  # sanborn01711_002
    assert not m_per_px_is_plausible(38.1)  # sanborn09046_002


def test_px_per_ft_round_trips_through_the_band():
    assert abs(corpus_px_per_ft() - 1.0 / (CORPUS_M_PER_PX * 3.280839895)) < 1e-12
    assert px_per_ft_is_plausible(corpus_px_per_ft())
    assert not px_per_ft_is_plausible(0.0080)  # sanborn09046_002's 38 m/px reference
    assert not px_per_ft_is_plausible(0.0)
