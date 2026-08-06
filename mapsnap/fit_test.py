"""Tests for the `mapsnap fit` driver's flag handling and run hygiene."""

from pathlib import Path

from mapsnap.fit import clear_derived_sidecars, worker_flag


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


def test_clear_derived_sidecars_removes_every_variant(tmp_path: Path):
    """All georef sidecar variants go, and nothing else does."""
    for name in (
        "p1.georef.json",
        "p2.georef-nofit.json",
        "p3.georef-osm.json",
        "p4.georef-streets.json",
        "p5.georef-contradicted.json",
        "p6.georef-keymap-outlier.json",
        "p7__2.georef.json",
    ):
        (tmp_path / name).write_text("{}")
    # Inputs and unrelated outputs must survive: a run regenerates sidecars from
    # these, so deleting them would destroy work no stage recreates.
    keep = (
        "p1.streets.json",
        "p1.boxes.json",
        "p1.jpg",
        "centerlines.geojson",
        "main.iiif.json",
        "adjacency.json",
    )
    for name in keep:
        (tmp_path / name).write_text("{}")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "p0.georef.json").write_text("{}")  # key-map georef, not ours

    removed = clear_derived_sidecars(tmp_path)

    assert removed == 7
    assert not list(tmp_path.glob("p*.georef*.json"))
    for name in keep:
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "raw" / "p0.georef.json").exists(), (
        "key-map sidecar must survive"
    )


def test_clear_derived_sidecars_on_empty_dir(tmp_path: Path):
    """A first run has nothing to clear and must not fail."""
    assert clear_derived_sidecars(tmp_path) == 0
