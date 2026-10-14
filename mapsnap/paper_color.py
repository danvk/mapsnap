"""Whether a detection's background differs from its page's paper (#500/#510).

OCR records every detection's background color and the page's own paper color in
<stem>.streets.json, as CIELAB {color, hue, chroma}. Sanborn street names are printed on the
paper, so a read whose background is off the paper sits on a building fill; this module holds
that one shared test. Which off-paper colors actually disqualify a street label is decided
downstream (``georef_from_labels.drop_labels_on_fill``).
"""

# How much a detection's box must exceed the page's own paper chroma (CIELAB) before its
# background counts as off the paper. 4.0 separates Nashville's building labels (p51 "REP" at
# 9-14 chroma over 2.2 paper) from its street labels (1.0-3.6).
BACKGROUND_CHROMA_MARGIN = 4.0
# A background whose hue is this far from the paper's (degrees) is off the paper even when it is
# no more saturated than the paper (#500): Miami's pale blue building fills sit at chroma 4-7 on
# chroma-8 yellowed paper, ~100 degrees round the hue circle from it.
BACKGROUND_HUE_MARGIN = 60.0
# ...but only at this chroma or above: a nearly neutral box's hue is noise, and a street label on
# near-white paper (Nashville, chroma 1-5) can measure any hue at all.
BACKGROUND_HUE_MIN_CHROMA = 3.0
# Olive to grey-green: the tint of a grey or aged scan, which is no Sanborn fill colour (those are
# pink/red, yellow and blue). The hue rule ignores a box in this band. Every on-paper read in a
# 422-read review that the hue rule would have dropped was here (hue 116-124, chroma 3.6-5.8):
# grey margins on near-white paper (Fargo, Grand Rapids), and Nashville p41__2's "2ND" on grey
# against a page whose measured paper is pink. Blue water and stone fills sit at 160-215.
BACKGROUND_HUE_TINT_BAND = (100.0, 150.0)


def hue_difference(first: float, second: float) -> float:
    """The angle between two hues in degrees, going the short way round (0-180)."""
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


def is_off_paper(
    background: dict,
    paper: dict | None,
    *,
    chroma_margin: float = BACKGROUND_CHROMA_MARGIN,
    hue_margin: float = BACKGROUND_HUE_MARGIN,
    hue_min_chroma: float = BACKGROUND_HUE_MIN_CHROMA,
    tint_band: tuple[float, float] = BACKGROUND_HUE_TINT_BAND,
) -> bool:
    """Whether a detection's recorded background differs from the page's paper.

    Off the paper means more saturated than it by ``chroma_margin`` -- a red brick or
    saturated yellow fill -- or, at ``hue_min_chroma`` or above, a hue ``hue_margin``
    degrees or more from the paper's: a pale blue fill on yellowed paper (#500). A box
    in ``tint_band`` is a grey scan tint, not a fill, so only the chroma rule applies to it.

    ``paper`` is None for a streets.json written before pages recorded their paper color;
    such files recorded a background only when it was off the paper, so its presence alone
    is the answer.
    """
    if paper is None:
        return True
    if background["chroma"] > paper["chroma"] + chroma_margin:
        return True
    low, high = tint_band
    return (
        background["chroma"] >= hue_min_chroma
        and hue_difference(background["hue"], paper["hue"]) >= hue_margin
        and not low <= background["hue"] <= high
    )
