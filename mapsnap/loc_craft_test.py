"""Tests for the corpus GPU pass driver (mapsnap.loc_craft)."""

import subprocess
from collections import Counter
from pathlib import Path

import pytest

from mapsnap.loc_craft import (
    Item,
    format_duration,
    key_prefix,
    list_prefix,
    plan_item,
    read_manifest,
    resolve_manifest,
    select_shard,
    shard_of,
)

MANIFEST = """item\tstate\tyear\tcity\tseq\tstem\tpage_key\tsource\tbytes\tstorage_dir
sanborn00001_003\talabama\t1924\tabbeville\t1\t00001_1924-0001\tp1\tjp2\t9\tgmd/x
sanborn00001_003\talabama\t1924\tabbeville\t2\t00001_1924-0002\tp2\tjp2\t9\tgmd/x
sanborn05791_007\tnew-york\t1906\tbrooklyn\t1\t05791_06_1906-0001\tp1\tjp2\t9\tgmd/y
"""


def write_manifest(tmp_path: Path, text: str = MANIFEST) -> Path:
    path = tmp_path / "mapping.tsv"
    path.write_text(text)
    return path


def test_read_manifest_keeps_one_row_per_item(tmp_path: Path) -> None:
    items = read_manifest(write_manifest(tmp_path))
    assert [item.item for item in items] == ["sanborn00001_003", "sanborn05791_007"]
    assert items[0].prefix == "by-state/alabama/1924/sanborn00001_003"
    assert items[1].prefix == "by-state/new-york/1906/sanborn05791_007"


def test_read_manifest_rejects_a_file_without_the_columns(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, "item\tcity\nx\ty\n")
    with pytest.raises(SystemExit, match="'state'"):
        read_manifest(path)


def test_shards_are_stable_disjoint_and_roughly_even() -> None:
    items = [f"sanborn{index:05d}_001" for index in range(4000)]
    counts = Counter(shard_of(item, 8) for item in items)
    assert set(counts) == set(range(8))
    assert max(counts.values()) < 2 * min(counts.values())
    # Stable across calls (and so across processes: sha1, not hash()).
    assert [shard_of(item, 8) for item in items[:20]] == [
        shard_of(item, 8) for item in items[:20]
    ]
    assert all(0 <= shard_of(item, 1) < 1 for item in items)


def test_select_shard_partitions_every_item_exactly_once(tmp_path: Path) -> None:
    items = read_manifest(write_manifest(tmp_path))
    selected = [
        item.item for shard in range(4) for item in select_shard(items, shard, 4)
    ]
    assert sorted(selected) == sorted(item.item for item in items)


ITEM = Item(item="sanborn00001_003", state="alabama", year="1924")


def test_plan_item_finds_the_images_and_the_missing_sidecars() -> None:
    work = plan_item(ITEM, ["metadata.json", "p1.jpg", "p2.jpg", "raw/p1.jpg"])
    assert work.pages == ["p1.jpg", "p2.jpg"]
    assert work.raw_sheets == ["raw/p1.jpg"]
    assert work.missing == [
        "p1.boxes.json",
        "p1.roadprob.jpg",
        "p2.boxes.json",
        "p2.roadprob.jpg",
        "raw/p1.boxes.json",
    ]
    assert not work.complete


def test_plan_item_calls_a_finished_item_complete() -> None:
    work = plan_item(
        ITEM,
        [
            "metadata.json",
            "p1.jpg",
            "p1.boxes.json",
            "p1.roadprob.jpg",
            "raw/p1.jpg",
            "raw/p1.boxes.json",
        ],
    )
    assert work.complete
    assert work.missing == []


def test_plan_item_does_not_mistake_a_sidecar_for_a_page() -> None:
    work = plan_item(ITEM, ["p1.jpg", "p1.roadprob.jpg", "p1.boxes.json"])
    assert work.pages == ["p1.jpg"]
    assert work.complete


def test_plan_item_wants_no_road_map_for_a_raw_key_map_sheet() -> None:
    work = plan_item(ITEM, ["raw/p0.jpg", "raw/p0.boxes.json"])
    assert work.pages == []
    assert work.complete


def test_list_prefix_returns_keys_relative_to_the_prefix(monkeypatch) -> None:
    prefix = "by-state/alabama/1924/sanborn00001_003"
    listing = (
        f"2026-09-11 11:26:18       1155 {prefix}/metadata.json\n"
        f"2026-09-11 11:26:19    1061353 {prefix}/p1.jpg\n"
        f"2026-09-11 11:26:19     920685 {prefix}/raw/p1.jpg\n"
    )
    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout=listing, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert list_prefix("s3://bucket/", prefix) == [
        "metadata.json",
        "p1.jpg",
        "raw/p1.jpg",
    ]
    assert captured["command"][:4] == ["aws", "s3", "ls", f"s3://bucket/{prefix}/"]


def test_list_prefix_strips_a_path_in_the_bucket_url(monkeypatch) -> None:
    """A bucket URL with a path once made every item look complete (no work done)."""
    prefix = "by-state/alabama/1924/sanborn00001_003"
    listing = f"2026-09-11 11:26:19    1061353 _craft/selftest/{prefix}/p1.jpg\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=listing, stderr=""
        ),
    )
    assert (
        key_prefix("s3://bucket/_craft/selftest", prefix) == f"_craft/selftest/{prefix}"
    )
    assert key_prefix("s3://bucket", prefix) == prefix
    assert list_prefix("s3://bucket/_craft/selftest", prefix) == ["p1.jpg"]


def test_list_prefix_raises_when_the_cli_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Access Denied\n"
        ),
    )
    with pytest.raises(OSError, match="Access Denied"):
        list_prefix("s3://bucket", "by-state/x")


def test_resolve_manifest_prefers_a_local_path_and_downloads_otherwise(
    tmp_path: Path, monkeypatch
) -> None:
    local = write_manifest(tmp_path)
    assert resolve_manifest(str(local), "s3://bucket", tmp_path) == local

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        (tmp_path / "work" / "loc-sanborn-maps.mapping.tsv").write_text(MANIFEST)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    path = resolve_manifest(None, "s3://bucket/", tmp_path / "work")
    assert path == tmp_path / "work" / "loc-sanborn-maps.mapping.tsv"
    assert calls[0][:3] == ["aws", "s3", "cp"]
    assert calls[0][3] == "s3://bucket/loc-sanborn-maps.mapping.tsv"

    # Already downloaded: no second fetch.
    calls.clear()
    resolve_manifest(None, "s3://bucket/", tmp_path / "work")
    assert calls == []


def test_format_duration_reads_as_hours_and_minutes() -> None:
    assert format_duration(0) == "0:00"
    assert format_duration(3600) == "1:00"
    assert format_duration(3660) == "1:01"
    assert format_duration(258000) == "71:40"


def test_limit_counts_work_done_not_items_skipped(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A pilot re-run must process --limit fresh items, not stop after N skips."""
    from mapsnap import loc_craft

    manifest = write_manifest(
        tmp_path,
        MANIFEST + "sanborn00009_004\talabama\t1930\tdothan\t1\ts\tp1\tjp2\t9\tgmd/z\n",
    )
    # The first two items are finished; the third still needs both sidecars.
    listings = {
        "sanborn00001_003": ["p1.jpg", "p1.boxes.json", "p1.roadprob.jpg"],
        "sanborn05791_007": ["p1.jpg", "p1.boxes.json", "p1.roadprob.jpg"],
        "sanborn00009_004": ["p1.jpg"],
    }
    monkeypatch.setattr(
        loc_craft, "list_prefix", lambda bucket, prefix: listings[prefix.split("/")[-1]]
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mapsnap loc-craft",
            "--manifest",
            str(manifest),
            "--work-dir",
            str(tmp_path / "work"),
            "--dry-run",
            "--limit",
            "1",
        ],
    )
    loc_craft.main()
    out = capsys.readouterr()
    assert "sanborn00009_004" in out.out
    assert "1 items processed, 2 already complete" in out.err
