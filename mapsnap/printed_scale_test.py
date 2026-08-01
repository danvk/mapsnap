import json

from mapsnap.printed_scale import (
    DEFAULT_PX_PER_PAPER_INCH,
    expected_px_per_ft,
    median_rung_px_per_paper_inch,
    note_m_per_px,
    printed_scale_ft,
    resolve_px_per_paper_inch,
    volume_px_per_paper_inch,
)


def sidecar(tmp_path, streets):
    path = tmp_path / "p1.streets.json"
    path.write_text(json.dumps({"streets": streets}))
    return path


def test_printed_scale_parses_the_note_with_ocr_confusions(tmp_path):
    path = sidecar(
        tmp_path,
        [
            {"text": "SCALE IOO FT. TO ONE INCH", "confidence": 0.86},
            {"text": "MAIN ST", "confidence": 0.9},
        ],
    )
    assert printed_scale_ft(path) == (100, 0.86)


def test_printed_scale_best_confidence_wins(tmp_path):
    path = sidecar(
        tmp_path,
        [
            {"text": "SCALE 50 FT TO ONE INCH", "confidence": 0.3},
            {"text": "SCALE 100 FT. TO ONE INCH", "confidence": 0.7},
        ],
    )
    assert printed_scale_ft(path) == (100, 0.7)


def test_printed_scale_rejects_junk(tmp_path):
    # Low confidence, lone tokens, and non-Sanborn numbers never parse.
    path = sidecar(
        tmp_path,
        [
            {"text": "SCALE 50 FT TO ONE INCH", "confidence": 0.05},
            {"text": "SCALE", "confidence": 1.0},
            {"text": "50", "confidence": 1.0},
            {"text": "SCALE 70 FT TO ONE INCH", "confidence": 0.9},
        ],
    )
    assert printed_scale_ft(path) is None
    assert printed_scale_ft(tmp_path / "absent.streets.json") is None


def test_calibration_and_expected_scale():
    # Three fitted 50ft pages at ~6.1 px/ft -> ~305 px per paper inch.
    pairs = [(6.1, 50), (6.05, 50), (6.2, 50), (3.1, 100)]
    px_per_inch = volume_px_per_paper_inch(pairs)
    assert px_per_inch is not None and 305 <= px_per_inch <= 312
    # A 100ft note then implies half the px/ft of a 50ft page.
    assert abs(expected_px_per_ft(px_per_inch, 100) - px_per_inch / 100) < 1e-9
    assert volume_px_per_paper_inch([(6.1, 50)]) is None  # too few to calibrate


def test_cold_start_calibration_falls_back_to_the_corpus_default():
    measured, source = resolve_px_per_paper_inch([(6.1, 50), (6.05, 50), (6.2, 50)])
    assert source == "self-calibrated" and 300 <= measured <= 312
    fallback, source = resolve_px_per_paper_inch([(6.1, 50)])  # too few pairs
    assert source == "corpus-default" and fallback == DEFAULT_PX_PER_PAPER_INCH


def test_median_rung_estimate_beats_the_corpus_default():
    # Columbus: median fitted scale 0.2001 m/px -> 1.523 px/ft -> 76.2 px/inch,
    # 22% denser than the corpus default and 0.8% from p297's truth scale.
    estimate, source = resolve_px_per_paper_inch([], median_px_per_ft=1.523)
    assert source == "median-rung" and 76.0 <= estimate <= 76.3
    assert abs(note_m_per_px(200, estimate) - 0.8005) < 0.002
    # Self-calibration still outranks it.
    _, source = resolve_px_per_paper_inch(
        [(6.1, 50), (6.05, 50), (6.2, 50)], median_px_per_ft=1.523
    )
    assert source == "self-calibrated"


def test_median_rung_estimate_rejects_implausible_scan_resolutions():
    # A volume whose median rung is 100 ft would imply a ~150 DPI scan: the
    # 50 ft assumption is wrong there, so the estimate is discarded.
    assert median_rung_px_per_paper_inch(0.762) is None
    assert median_rung_px_per_paper_inch(None) is None
    assert median_rung_px_per_paper_inch(-1.0) is None
    _, source = resolve_px_per_paper_inch([], median_px_per_ft=0.762)
    assert source == "corpus-default"


def test_note_m_per_px_conversion():
    # A 200 ft note at the corpus default: ~0.98 m/px, ~5x a 50 ft page.
    assert abs(note_m_per_px(200, 62.5) - 0.9754) < 0.001
    assert abs(note_m_per_px(50, 62.5) / note_m_per_px(200, 62.5) - 0.25) < 1e-9
