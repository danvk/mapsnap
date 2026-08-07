"""Golden-image tests for `_erase_underlines`.

Each case in ``testdata/erase_underlines`` is a real label crop: ``<name>.before.png``
is the pixels CRAFT handed the recognizer, ``<name>.after.png`` is what the eraser
should produce. Synthetic images cannot exercise this — the failure mode the
rewrite fixes (issue #250) is a rule sharing rows with the glyphs above it, and a
digit crossbar that looks exactly like a rule until you check what lies beneath.

Cases are drawn from Fargo p60__1 (the page that motivated the rewrite) and from
Queens vol 1, the volume the original underline work regressed (cf8817f). Each is
an ordinal street label chosen by what it *says*, not by whether the rule fires --
selecting on "the code did something" only finds cases the code already agrees
with, and an earlier version of this file froze three such false positives
(dark map blobs, >90% ink) as expected output.

Three kinds of case, and the distinction matters:

* ``*_rule`` -- a rule the detector finds and removes. Golden output.
* ``*_smallrule`` -- a Queens ordinal that **is** underlined, with a rule too
  small or faint for the current detector. These pin *current* behaviour, not
  desired behaviour: catching them is future work (#250).
* ``*_plain`` / ``*_blob`` -- true controls that must never fire. The blobs are
  CRAFT boxes on dense map content at >90% ink; an earlier version of this file
  mistakenly froze the detector erasing runs inside them as expected output.
* ``eraser: "legacy"`` -- cases for :func:`legacy_erase_rows`, the pre-#250
  row-painting eraser kept as a third arbitration vote (#263). Two artifact
  classes motivate it, from the 2026-08-06 baseline: ``*_smallrule`` ordinals
  whose rules sit under the precise detector's min_run (nashville 2ND
  0.965 legacy vs 0.127 raw), and ``*_dashline`` labels printed on a dashed
  street line -- not underlines at all; the dash row crosses the bottom of the
  box and the row-paint strips it (champaign S NEIL 0.476 vs 0.265). The
  precise detector correctly refuses the dash-line class (ink below the run),
  so only the legacy vote reads these.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mapsnap.detect_text import _erase_underlines, legacy_erase_rows

TESTDATA = Path(__file__).parent.parent / "testdata" / "erase_underlines"
ALL_CASES = json.loads((TESTDATA / "cases.json").read_text())
CASES = [c for c in ALL_CASES if c.get("eraser") != "legacy"]
LEGACY_CASES = [c for c in ALL_CASES if c.get("eraser") == "legacy"]


def load(name: str, suffix: str) -> np.ndarray:
    """A fixture PNG as RGB, matching what the pipeline passes in."""
    image = cv2.imread(str(TESTDATA / f"{name}.{suffix}.png"))
    assert image is not None, f"missing fixture {name}.{suffix}.png"
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_matches_golden_output(case):
    before = load(case["name"], "before")
    expected = load(case["name"], "after")
    assert np.array_equal(_erase_underlines(before, [case["box"]]), expected)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_only_whitens_never_darkens(case):
    """The eraser may only remove ink. Any pixel it darkens is a bug."""
    before = load(case["name"], "before")
    after = _erase_underlines(before, [case["box"]])
    assert (after.astype(int) >= before.astype(int)).all()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_does_not_mutate_input(case):
    before = load(case["name"], "before")
    original = before.copy()
    _erase_underlines(before, [case["box"]])
    assert np.array_equal(before, original)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_changed_pixel_count_recorded(case):
    """The recorded count is what distinguishes a rule case from a control."""
    before = load(case["name"], "before")
    after = _erase_underlines(before, [case["box"]])
    assert int((before != after).any(axis=2).sum()) == case["changed_px"]


def test_negative_controls_are_untouched():
    """A digit crossbar is long, thin, horizontal and low -- and must survive.

    Erasing the crossbar of the "4" in 14TH turned the read into "6TH" and
    dropped it below the acceptance floor; that regression is what the
    ink-beneath check exists to prevent.
    """
    controls = [c for c in CASES if c["changed_px"] == 0]
    assert {c["name"] for c in controls} >= {
        "fargo_14th_crossbar",
        "fargo_12th_plain",
        "queens_blob_dense",
        "queens_blob_hatched",
    }
    for case in controls:
        before = load(case["name"], "before")
        assert np.array_equal(_erase_underlines(before, [case["box"]]), before)


def test_underlined_cases_lose_ink_low_in_the_box():
    """Every positive case removes ink, and only from the bottom of the box."""
    for case in [c for c in CASES if c["changed_px"] > 0]:
        before = load(case["name"], "before")
        after = _erase_underlines(before, [case["box"]])
        rows = np.where((before != after).any(axis=2).any(axis=1))[0]
        assert len(rows), case["name"]
        _x0, _x1, y0, y1 = case["box"]
        assert rows.min() >= y0 + (y1 - y0) * 0.4, case["name"]
        assert rows.max() <= y1, case["name"]


def test_repaint_uses_paper_colour_not_pure_white():
    """255 is a pathological fill: the recognizer scores it far worse.

    On Fargo p60__1's 8TH, filling the rule with 255 reads at 0.733 while the
    label's own paper (254, 254, 251) reads at 0.949 -- as does every other
    non-255 fill tried. So the rule is repainted in the median colour of the
    box's non-ink pixels.
    """
    image = np.full((20, 60, 3), 250, np.uint8)
    image[16:18, :, :] = 0
    result = _erase_underlines(image, [[0, 60, 0, 20]])
    assert tuple(int(v) for v in result[16, 30]) == (250, 250, 250)
    assert result[16, 30].max() != 255


def test_mostly_ink_box_is_skipped():
    """A CRAFT box on hatching is not a label, and every bottom row looks like a rule.

    Three fixtures in an earlier version of this file were exactly this: dark map
    blobs at >90% ink where the rule happily erased an arbitrary horizontal run.
    """
    image = np.full((20, 60, 3), 0, np.uint8)
    image[0:2, :, :] = 255
    assert np.array_equal(_erase_underlines(image, [[0, 60, 0, 20]]), image)


def test_box_smaller_than_min_run_is_skipped():
    """A box narrower than the shortest credible rule cannot contain one."""
    image = np.full((20, 8, 3), 255, np.uint8)
    image[15:18, :, :] = 0
    assert np.array_equal(_erase_underlines(image, [[0, 8, 0, 20]]), image)


@pytest.mark.parametrize("case", LEGACY_CASES, ids=[c["name"] for c in LEGACY_CASES])
def test_legacy_matches_golden_output(case):
    before = load(case["name"], "before")
    expected = load(case["name"], "after")
    after, fired = legacy_erase_rows(before, [case["box"]])
    assert fired == [0]
    assert np.array_equal(after, expected)


@pytest.mark.parametrize("case", LEGACY_CASES, ids=[c["name"] for c in LEGACY_CASES])
def test_legacy_only_whitens_never_darkens(case):
    before = load(case["name"], "before")
    after, _fired = legacy_erase_rows(before, [case["box"]])
    assert (after.astype(int) >= before.astype(int)).all()


@pytest.mark.parametrize("case", LEGACY_CASES, ids=[c["name"] for c in LEGACY_CASES])
def test_legacy_changed_pixel_count_recorded(case):
    before = load(case["name"], "before")
    after, _fired = legacy_erase_rows(before, [case["box"]])
    assert int((before != after).any(axis=2).sum()) == case["changed_px"]


@pytest.mark.parametrize("case", LEGACY_CASES, ids=[c["name"] for c in LEGACY_CASES])
def test_legacy_output_differs_from_the_precise_eraser(case):
    """Each legacy fixture must be a read the precise eraser cannot produce.

    On most (dash-line labels, sub-min_run rules) the precise detector does not
    fire at all; on nashville_4th_smallrule's taller box it fires but erases
    differently, and the legacy variant still won the arbitration (0.690 vs
    0.010 raw). If the two erasers converge on one of these, the legacy vote
    has stopped mattering for it and the fixture should be revisited.
    """
    before = load(case["name"], "before")
    precise = _erase_underlines(before, [case["box"]])
    legacy, _fired = legacy_erase_rows(before, [case["box"]])
    assert not np.array_equal(precise, legacy)
