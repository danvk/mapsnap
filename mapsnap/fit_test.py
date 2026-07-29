"""Tests for the `mapsnap fit` driver's flag handling."""

from mapsnap.fit import worker_flag


def test_worker_flag_extracts_both_spellings():
    assert worker_flag(["--num-workers", "4"]) == ["--num-workers", "4"]
    assert worker_flag(["--num-workers=4"]) == ["--num-workers", "4"]
    # Found among other georef-only passthrough flags.
    assert worker_flag(["--debug", "--num-workers", "2", "--affine"]) == [
        "--num-workers",
        "2",
    ]


def test_worker_flag_absent_or_malformed_yields_nothing():
    assert worker_flag([]) == []
    assert worker_flag(["--debug", "--affine"]) == []
    # A trailing --num-workers with no value: leave it to georef's parser to
    # reject rather than forwarding half a flag.
    assert worker_flag(["--debug", "--num-workers"]) == []
