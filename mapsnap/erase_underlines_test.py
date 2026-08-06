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

Only 2 of 41 Queens ordinals carry a rule at all, which is the likely reason the
original underline work regressed that volume: there is almost nothing to remove,
so over-eager removal is pure damage. Hence the Queens cases here are mostly
CONTROLS -- high-confidence reads that must come back byte-identical.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mapsnap.detect_text import _erase_underlines

TESTDATA = Path(__file__).parent.parent / "testdata" / "erase_underlines"
CASES = json.loads((TESTDATA / "cases.json").read_text())


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
        "queens_34th_plain",
        "queens_9th_plain",
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
        x0, x1, y0, y1 = case["box"]
        assert rows.min() >= y0 + (y1 - y0) * 0.4, case["name"]
        assert rows.max() <= y1, case["name"]


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
