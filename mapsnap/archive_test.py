"""Tests for `mapsnap archive`."""

import json

from mapsnap.archive import archived_stems, is_complete, run_outputs


def test_run_outputs_finds_both(tmp_path):
    (tmp_path / "run-a.iiif.json").write_text("{}")
    (tmp_path / "run-a.txt").write_text("table")
    iiif, compare = run_outputs(tmp_path, "run-a")
    assert iiif == tmp_path / "run-a.iiif.json"
    assert compare == tmp_path / "run-a.txt"


def test_run_outputs_tolerates_a_missing_compare_table(tmp_path):
    """A volume with no truth has no compare output; the run is still archivable."""
    (tmp_path / "run-a.iiif.json").write_text("{}")
    iiif, compare = run_outputs(tmp_path, "run-a")
    assert iiif is not None
    assert compare is None


def test_run_outputs_returns_none_for_an_unknown_tag(tmp_path):
    assert run_outputs(tmp_path, "nope") == (None, None)


def test_is_complete_requires_the_manifest(tmp_path):
    """A directory alone is not a finished archive.

    archive_run creates the directory before copying into it, so an interrupted
    run leaves one behind. The manifest is written last, which is what makes it
    the honest completion marker -- and why `fit` checks for it rather than for
    the directory, which would skip the run's computation for good.
    """
    run_dir = tmp_path / "artifacts" / "run-a"
    run_dir.mkdir(parents=True)
    assert not is_complete(run_dir)
    (run_dir / "p1.georef.json").write_text("{}")
    assert not is_complete(run_dir)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run-a"}))
    assert is_complete(run_dir)


def test_is_complete_on_a_missing_directory(tmp_path):
    assert not is_complete(tmp_path / "artifacts" / "never-ran")


def test_archived_stems_lists_canonical_georefs(tmp_path):
    for name in ("p1.georef.json", "p2.georef.json", "p33A.georef.json"):
        (tmp_path / name).write_text("{}")
    # Failure variants describe the same page and would double-count it.
    (tmp_path / "p9.georef-nofit.json").write_text("{}")
    assert archived_stems(tmp_path) == {"p1", "p2", "p33A"}
