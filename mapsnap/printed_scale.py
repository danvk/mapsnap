"""Read the printed scale note from a page's OCR output, and calibrate it.

Most Sanborn sheets print their scale ("Scale 50 Ft. to One Inch."), and #196
taught the constrained OCR decoder to emit it (173/173 parsed numbers correct
against truth across the twelve-volume corpus). This module turns those reads
into a per-page scale *authority*: parse the note, then calibrate the volume's
pixels-per-paper-inch from pages that have both a fit and a note, so a printed
"N ft to one inch" converts to an absolute expected scale for any page.

The calibration constant is the scan resolution in disguise (LOC scans are
uniform within a volume), measured rather than assumed: fitted pages with notes
give ``px_per_paper_inch = px_per_ft * printed_ft`` directly.
"""

import json
import re
import statistics
from pathlib import Path

NOTE_PATTERN = re.compile(r"\b([0-9IO]{2,3})\s*FT\.?\s*TO\s*ONE\s*INCH", re.IGNORECASE)
MIN_NOTE_CONFIDENCE = 0.2
"""Reads below this are the decoder emitting a vocab string over noise. The
corpus sweep found full-phrase reads at >=0.2 were correct 173/173 times while
sub-0.1 reads are junk-box conversions."""

MIN_CALIBRATION_PAGES = 3
"""Fewest (fit, note) pages that anchor a volume's px-per-paper-inch."""

DEFAULT_PX_PER_PAPER_INCH = 62.5
"""Last-resort calibration for volumes with no fitted pages at all.

Measured self-calibrations at the standard 25% working scale: Brooklyn 62.3,
DC 63, KC ~62 -- but Columbus measures 76.2 (~305 DPI raw vs ~250), so scan
resolution is NOT uniform across LOC volumes and this constant can run 20%
hot or cold. Any volume with fitted pages gets the median-rung estimate
instead. Only valid for the 25% working scale."""

MEDIAN_RUNG_FT = 50
"""Sanborn's standard detail scale. A volume's median fitted scale is assumed
to sit on this rung, which calibrates px-per-paper-inch without any note+fit
pair: Columbus's median-rung estimate lands 0.8% from p297's truth scale
where the corpus default was 21% off."""

PLAUSIBLE_PX_PER_PAPER_INCH = (45.0, 100.0)
"""Working-scale calibrations implying ~180-400 DPI raw scans. A median-rung
estimate outside this band means the volume's median rung is not 50 ft (or
its fits are junk), so the estimate is discarded rather than trusted."""


def median_rung_px_per_paper_inch(median_px_per_ft: float | None) -> float | None:
    """Calibration assuming the volume's median fitted scale is the 50 ft rung."""
    if median_px_per_ft is None or median_px_per_ft <= 0:
        return None
    estimate = median_px_per_ft * MEDIAN_RUNG_FT
    low, high = PLAUSIBLE_PX_PER_PAPER_INCH
    if low <= estimate <= high:
        return estimate
    return None


def resolve_px_per_paper_inch(
    pairs: list[tuple[float, int]],
    median_px_per_ft: float | None = None,
) -> tuple[float, str]:
    """(px-per-paper-inch, source), by decreasing trust.

    Self-calibration (>=3 note+fit pairs) measures the scan directly;
    the median-rung estimate assumes the volume's median fitted scale is
    the 50 ft rung; the corpus default is a last resort for volumes with
    no fitted pages.
    """
    measured = volume_px_per_paper_inch(pairs)
    if measured is not None:
        return measured, "self-calibrated"
    rung = median_rung_px_per_paper_inch(median_px_per_ft)
    if rung is not None:
        return rung, "median-rung"
    return DEFAULT_PX_PER_PAPER_INCH, "corpus-default"


def note_m_per_px(printed_ft: int, px_per_paper_inch: float) -> float:
    """The working-scale metres-per-pixel a printed note implies."""
    return printed_ft * 0.3048 / px_per_paper_inch


VALID_PRINTED_FT = (25, 50, 100, 200, 300, 400, 600)
"""Scales Sanborn actually printed; a parse outside this list is a misread."""


def printed_scale_ft(streets_path: Path) -> tuple[int, float] | None:
    """(printed ft-per-inch, confidence) from a page's OCR sidecar, or None.

    The best-confidence full-phrase read wins; single tokens ("SCALE", "50")
    are never trusted alone — "50" is also a street number and a house number.
    """
    if not streets_path.exists():
        return None
    best: tuple[int, float] | None = None
    for street in json.loads(streets_path.read_text()).get("streets", []):
        text = street.get("text") or ""
        match = NOTE_PATTERN.search(text)
        if not match:
            continue
        confidence = float(street.get("confidence") or 0.0)
        if confidence < MIN_NOTE_CONFIDENCE:
            continue
        digits = match.group(1).upper().replace("I", "1").replace("O", "0")
        try:
            feet = int(digits)
        except ValueError:
            continue
        if feet not in VALID_PRINTED_FT:
            continue
        if best is None or confidence > best[1]:
            best = (feet, confidence)
    return best


def volume_px_per_paper_inch(pairs: list[tuple[float, int]]) -> float | None:
    """Median px-per-paper-inch from (fitted px_per_ft, printed ft) pairs.

    Each fitted page with a note measures the scan resolution directly:
    ``px_per_ft * printed_ft`` pixels rendered one paper inch. The median over
    at least MIN_CALIBRATION_PAGES pages rejects the occasional wrong fit.
    """
    samples = [px_per_ft * feet for px_per_ft, feet in pairs if px_per_ft > 0]
    if len(samples) < MIN_CALIBRATION_PAGES:
        return None
    return float(statistics.median(samples))


def expected_px_per_ft(px_per_paper_inch: float, printed_ft: int) -> float:
    """The absolute scale a printed note implies, in the fitter's px/ft units."""
    return px_per_paper_inch / printed_ft
