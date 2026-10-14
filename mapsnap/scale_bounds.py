"""The scales a Sanborn sheet's fit can plausibly have (#498).

A volume's scale used to be whatever its fitted pages said, with no bound, so a
volume with one or two fitted pages took a fit wrong by orders of magnitude as a
population parameter: sanborn05857_001 published its sheet at 0.0117 m/px (a
sheet ~30 m across), and in sanborn09046_002 a 38 m/px page became the reference
that dropped the volume's plausible 0.21 m/px page as an outlier.

Scales here are metres per pixel in the pipeline's page frame: the mirror's 25%
scans that every page and panel image is cut from. Key maps are georeferenced at
full resolution and draw a whole city, so callers exempt them (see #598).

Measured over OIM truth for 415 volumes (29,097 pages): p0.1 0.10 m/px, median
0.205, p99.9 0.82 -- the 50, 100 and 200 ft-to-the-inch rungs. Only five truth
pages fall outside the band below: four key maps or overview sheets (4-41 m/px)
and nothing else; Champaign 1915 p4's detail inset, the finest real page, sits
at 0.0375. corpus-v1's 10,868 published poses of those volumes all fall inside.
"""

# Metres per page pixel: about 0.15x to 10x the corpus median.
MIN_M_PER_PX = 0.03
MAX_M_PER_PX = 2.0
# Median over the 2,540 fitted pages of the truth volumes (p10 0.174, p90 0.312):
# the 50 ft-to-the-inch rung at a typical 25% scan.
CORPUS_M_PER_PX = 0.203
FT_PER_M = 3.280839895


def m_per_px_is_plausible(m_per_px: float) -> bool:
    """Whether a sheet could be drawn at this many metres per page pixel."""
    return MIN_M_PER_PX <= m_per_px <= MAX_M_PER_PX


def px_per_ft_is_plausible(px_per_ft: float) -> bool:
    """:func:`m_per_px_is_plausible` for a scale in page pixels per foot."""
    return px_per_ft > 0 and m_per_px_is_plausible(1.0 / (px_per_ft * FT_PER_M))


def corpus_px_per_ft() -> float:
    """The corpus-median scale in page pixels per foot."""
    return 1.0 / (CORPUS_M_PER_PX * FT_PER_M)
