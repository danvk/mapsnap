import json

from mapsnap.printed_scale import (
    expected_px_per_ft,
    printed_scale_ft,
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
