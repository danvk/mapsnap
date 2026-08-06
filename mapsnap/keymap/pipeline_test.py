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
