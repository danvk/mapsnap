"""Tests for mapsnap.page_units' volume-level helpers."""

import math

import numpy as np
import pytest

from mapsnap.page_units import FALLBACK_M_PER_PX, PageUnit, volume_median_scale


def _unit(stem: str, fit_state: str, m_per_px: float | None) -> PageUnit:
    """A page unit whose affine, if any, has the given metres-per-pixel scale."""
    affine = None
    if m_per_px is not None:
        lat = 40.0
        deg_per_m_lat = 1.0 / 110_570.0
        deg_per_m_lon = 1.0 / (111_320.0 * math.cos(math.radians(lat)))
        affine = np.array(
            [
                [m_per_px * deg_per_m_lon, 0.0, -74.0],
                [0.0, -m_per_px * deg_per_m_lat, lat],
            ]
        )
    return PageUnit(
        stem=stem,
        number=1,
        width=1000,
        height=800,
        fit_state=fit_state,
        truth=None,
        split_truth=False,
        gen_affine=affine,
        inlier_intersections=0,
        inlier_streets=0,
        keymap_centers=[],
        keymap_radius_m=600.0,
    )


def test_volume_median_scale_prefers_base_pages() -> None:
    """Panels must not shift a volume that has fitted base pages of its own."""
    base = [_unit("p1", "fitted", 0.20), _unit("p2", "fitted", 0.20)]
    panels = [_unit("p3__1", "fitted", 0.80)]
    assert volume_median_scale(base, panels) == pytest.approx(0.20, abs=1e-3)


def test_volume_median_scale_falls_back_to_panels() -> None:
    """Gardiner NY 1913: one sheet, split, so every base page is 'split'."""
    base = [_unit("p0", "split", None)]
    panels = [_unit("p0__1", "fitted", 0.504), _unit("p0__2", "nofit", None)]
    assert volume_median_scale(base, panels) == pytest.approx(0.504, abs=1e-3)


def test_volume_median_scale_survives_a_volume_that_fits_nothing(capsys) -> None:
    """It must still run: the point of such a volume is to record its abstentions."""
    got = volume_median_scale(
        [_unit("p1", "nofit", None)], [_unit("p1__1", "nofit", None)]
    )
    assert got == FALLBACK_M_PER_PX
    assert "No plausibly-scaled fitted page" in capsys.readouterr().err
    assert volume_median_scale([]) == FALLBACK_M_PER_PX


def test_volume_median_scale_ignores_unfitted_units() -> None:
    """A pose only counts once the pipeline has accepted it."""
    units = [
        _unit("p1", "fitted", 0.20),
        _unit("p2", "nofit", 5.0),
        _unit("p3", "outlier", 5.0),
    ]
    assert volume_median_scale(units) == pytest.approx(0.20, abs=1e-3)


def test_volume_median_scale_ignores_an_implausible_fit(capsys) -> None:
    """#498: sanborn09046_002's 38 m/px page must not drag the median off its 0.21 page."""
    got = volume_median_scale(
        [_unit("p1", "fitted", 38.1), _unit("p2", "fitted", 0.21)]
    )
    assert got == pytest.approx(0.21, abs=1e-3)
    assert (
        "Ignoring 1 fitted page(s) at an implausible scale" in capsys.readouterr().err
    )


def test_volume_median_scale_falls_back_when_no_fit_is_plausible() -> None:
    """sanborn01711_002: its only sheet fitted at 14.2 m/px."""
    assert volume_median_scale([_unit("p1", "fitted", 14.15)]) == FALLBACK_M_PER_PX
