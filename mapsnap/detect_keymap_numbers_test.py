import math

from mapsnap.detect_keymap_numbers import detection_record, filter_args


def test_detection_record_horizontal_box():
    # A 40-wide, 20-tall upright box (EasyOCR corner order: TL, TR, BR, BL).
    record = detection_record([[10, 5], [50, 5], [50, 25], [10, 25]], "21", 0.873)
    assert record["polygon"] == [[10, 5], [50, 5], [50, 25], [10, 25]]
    assert record["text"] == "21"
    assert record["confidence"] == 0.873
    assert record["angle"] == 0
    assert record["long_side"] == 40.0
    assert record["short_side"] == 20.0
    assert record["dir_pix"] == 0.0  # longer side runs horizontally


def test_detection_record_rounds_confidence():
    record = detection_record([[0, 0], [10, 0], [10, 10], [0, 10]], "7", 0.123456)
    assert record["confidence"] == 0.1235
    assert record["long_side"] == 10.0
    assert record["short_side"] == 10.0


def test_detection_record_dir_pix_in_unit_range():
    record = detection_record([[0, 0], [30, 10], [25, 25], [-5, 15]], "13", 0.5)
    assert 0.0 <= record["dir_pix"] < math.pi


def test_filter_args_keeps_only_the_named_image():
    argv = ["detect", "--min-size", "60", "a.jpg", "b.jpg", "c.jpg"]
    assert filter_args(argv, "b.jpg") == ["detect", "--min-size", "60", "b.jpg"]
