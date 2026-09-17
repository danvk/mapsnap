"""Tests for the key-map pipeline's flag handling."""

import subprocess
import sys


def run_help() -> str:
    out = subprocess.run(
        [sys.executable, "-m", "mapsnap.keymap.pipeline", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_repair_assignments_flag_exists_and_defaults_off():
    """The repairs are opt-in (#239): applying them by default cost 7 points."""
    import argparse

    from mapsnap.keymap.pipeline import build_parser

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args(["raw/p0.jpg"])
    assert args.repair_assignments is False
    assert parser.parse_args(["--repair-assignments", "raw/p0.jpg"]).repair_assignments


def test_dry_run_still_available():
    from mapsnap.keymap.pipeline import build_parser

    args = build_parser().parse_args(["--dry-run", "raw/p0.jpg"])
    assert args.dry_run is True
    assert args.repair_assignments is False


def _reads(tmp_path, stem, detections):
    import json

    (tmp_path / f"{stem}.streets.json").write_text(
        json.dumps({"width": 100, "height": 100, "streets": detections})
    )


def _det(conf, short=40.0):
    return {
        "text": "MAIN STREET",
        "confidence": conf,
        "short_side": short,
        "long_side": short * 2,
        "polygon": [[0, 0], [short * 2, 0], [short * 2, short], [0, short]],
    }


def test_georef_log_reports_a_missing_pose(tmp_path):
    """georef writes nothing and exits 0 on failure; the log must say so.

    This is the Los Angeles 1949 vol 14 shape: three confident reads, all
    cartouche words, and no pose. It cost the volume 25 points and left no
    trace in the sheet's own log.
    """
    from mapsnap.keymap.pipeline import georef_log_lines

    _reads(tmp_path, "p0a", [_det(0.1)] * 2384 + [_det(0.98, 150.0)] * 3)
    lines = georef_log_lines(tmp_path / "p0a.jpg")
    text = "\n".join(lines)
    assert "NO POSE WRITTEN" in text
    assert "2387 detection(s), 3 at confidence >= 0.5" in text
    assert "only 3 confident read(s)" in text


def test_georef_log_reports_a_pose(tmp_path):
    """A sheet that fitted says where, and on how much evidence."""
    import json

    from mapsnap.keymap.pipeline import georef_log_lines

    _reads(tmp_path, "p0b", [_det(0.9)] * 40)
    (tmp_path / "p0b.georef.json").write_text(
        json.dumps(
            {
                "corners": [
                    [-118.3, 34.1],
                    [-118.1, 34.1],
                    [-118.1, 33.9],
                    [-118.3, 33.9],
                ],
                "intersections": [{"inlier": True}] * 167,
                "streets": [{"inlier": True}] * 99,
            }
        )
    )
    text = "\n".join(georef_log_lines(tmp_path / "p0b.jpg"))
    assert "pose: 34.00000,-118.20000" in text
    assert "167 inlier intersection(s)" in text
    assert "99 inlier street(s)" in text
    assert "NO POSE" not in text


def test_georef_log_reports_missing_reads(tmp_path):
    """No streets.json at all is a different failure from a failed fit."""
    from mapsnap.keymap.pipeline import georef_log_lines

    text = "\n".join(georef_log_lines(tmp_path / "p0c.jpg"))
    assert "no reads file" in text
    assert "NO POSE WRITTEN" in text
