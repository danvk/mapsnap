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
