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
        "p3.georef-snap.json",
        "p4.georef-street.json",
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


def test_channel_writers_and_arbiter_agree_on_names(tmp_path):
    """Every channel the pipeline WRITES is one the arbiter LOOKS FOR.

    These names live in the writer, in reconcile's CHANNEL_ORDER and in fit's
    glob, and a mismatch is silent: the sidecar is still globbed as a
    hypothesis, so the arbiter sees the pose, but the channel stops counting as
    the incumbent and every keep_prior/entry decision on that page shifts.

    Renaming the channels produced exactly this twice. The first miss was an
    argparse DEFAULT ("streets"); the second was a bare positional literal at
    the one call site `mapsnap fit` actually uses -- so a test that only read
    the CLI default passed while production still wrote the old name. This
    calls the writer the way cmd_select does, taking every default.
    """
    from mapsnap.osm_snap_experiment import osm_variant_path
    from mapsnap.reconcile import CHANNEL_ORDER
    from mapsnap.street_solve_experiment import (
        PriorLocation,
        StreetGates,
        write_georef_streets,
    )

    assert osm_variant_path(tmp_path, "p1").name == "p1.georef-snap.json"

    from mapsnap.reconcile_test import make_unit

    unit = make_unit("p1")
    written = write_georef_streets(
        tmp_path,
        unit,
        PriorLocation(
            center=(-77.0, 38.9),
            radius_m=500.0,
            centers=[(-77.0, 38.9)],
            source="keymap-exact",
        ),
        [],
        (0.0, 1.0, -77.0, 38.9),
        StreetGates(),
        (unit.width, unit.height),
        {},
    )
    channels = {"georef", "georef-snap", written.name[len("p1.") : -len(".json")]}
    assert channels == set(CHANNEL_ORDER), (
        f"channels written {channels} != channels arbitrated {set(CHANNEL_ORDER)}"
    )


def test_clear_derived_sidecars_clears_snap_caches(tmp_path: Path):
    # #342: a fit must never inherit a previous run's snap candidate or
    # selection records -- cache temperature alone flipped KC pages 7.4 <-> 282 ft.
    (tmp_path / "p1.georef.json").write_text("{}")
    snap = tmp_path / "artifacts" / "osm_snap"
    street = tmp_path / "artifacts" / "street_solve"
    snap.mkdir(parents=True)
    street.mkdir(parents=True)
    (snap / "candidates.jsonl").write_text("{}\n")
    (snap / "selection_volume.jsonl").write_text("{}\n")
    (street / "candidates.jsonl").write_text("{}\n")
    removed = clear_derived_sidecars(tmp_path)
    assert removed == 4
    assert not (snap / "candidates.jsonl").exists()
    assert not (snap / "selection_volume.jsonl").exists()
    assert not (street / "candidates.jsonl").exists()
