"""Tests for the corpus key-map identification driver (mapsnap.loc_keymaps)."""

from pathlib import Path

from mapsnap.loc_craft import Item
from mapsnap.loc_keymaps import (
    KEYMAPS_NAME,
    page_keys_of,
    placeholder_volume,
    plan_keymaps,
)

ITEM = Item(item="sanborn00001_003", state="alabama", year="1924")


def test_page_keys_ignores_sidecars_and_raw_sheets() -> None:
    present = [
        "metadata.json",
        "p1.jpg",
        "p1.boxes.json",
        "p1.roadprob.jpg",  # a sidecar written by the corpus pass
        "p0b.jpg",
        "raw/p0b.jpg",  # full-resolution, not a page image
        KEYMAPS_NAME,
    ]
    assert page_keys_of(present) == ["p0b", "p1"]


def test_placeholder_volume_is_empty_files_named_for_the_pages(tmp_path: Path) -> None:
    volume = placeholder_volume(tmp_path, ITEM, ["p0", "p1", "p2"])
    assert volume == tmp_path / ITEM.item
    assert sorted(path.name for path in volume.iterdir()) == [
        "p0.jpg",
        "p1.jpg",
        "p2.jpg",
    ]
    assert all(path.stat().st_size == 0 for path in volume.iterdir())
    # Rebuilding drops whatever the last item left behind.
    (volume / "stale.jpg").touch()
    assert "stale.jpg" not in {
        p.name for p in placeholder_volume(tmp_path, ITEM, ["p1"]).iterdir()
    }


def test_an_unsplit_page_zero_is_a_key_map_without_the_model(tmp_path: Path) -> None:
    """The expensive path is skipped for the 3,220 items that have a page 0."""
    keys = ["p0", "p1", "p2", "p3", "p4", "p5", "p6"]
    volume = placeholder_volume(tmp_path, ITEM, keys)
    work = plan_keymaps(ITEM, [f"{key}.jpg" for key in keys], volume)
    assert work.assumed == ["p0"]
    assert work.to_test == []
    assert not work.needs_model


def test_a_volume_without_page_zero_tests_its_page_one_family(tmp_path: Path) -> None:
    """These are the items whose key map has no raw copy in the mirror."""
    keys = ["p1", "p1b", "p2", "p3", "p4", "p5", "p6"]
    volume = placeholder_volume(tmp_path, ITEM, keys)
    work = plan_keymaps(ITEM, [f"{key}.jpg" for key in keys], volume)
    assert work.assumed == []
    assert work.needs_model
    assert set(work.to_test) <= {"p1", "p1b"}
    assert "p1" in work.to_test


def test_planning_reads_names_only_so_placeholders_suffice(tmp_path: Path) -> None:
    """No page is opened during planning; that is what lets the shard skip downloads."""
    keys = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
    volume = placeholder_volume(tmp_path, ITEM, keys)
    assert all(path.stat().st_size == 0 for path in volume.glob("*.jpg"))
    work = plan_keymaps(ITEM, [f"{key}.jpg" for key in keys], volume)
    assert work.page_keys == keys
