"""Tests for the key-map raw-sheet fetcher (mapsnap.loc_raw)."""

from pathlib import Path

import pytest

from mapsnap.loc_raw import (
    Wanted,
    keymap_keys,
    missing_raw,
    read_list,
    resolve_list,
    write_list,
)


def test_missing_raw_wants_only_the_sheets_with_no_copy() -> None:
    present = ["p1.jpg", "p2.jpg", "raw/p0.jpg", "keymaps.json"]
    assert missing_raw(present, ["p1"]) == ["p1"]
    assert missing_raw(present, ["p0"]) == []  # already mirrored
    assert missing_raw(present, ["p0", "p1"]) == ["p1"]
    assert missing_raw(present, []) == []


def test_missing_raw_asks_for_a_panel_s_parent_sheet() -> None:
    """A split panel is cut locally from the parent, so the parent is fetched."""
    assert missing_raw(["p1.jpg"], ["p1__1", "p1__2"]) == ["p1"]
    assert missing_raw(["raw/p1.jpg"], ["p1__1", "p1__2"]) == []


def test_keymap_keys_reads_the_record_and_survives_rubbish() -> None:
    assert keymap_keys(["keymaps.json"], '{"keys": ["p1", "p1b"]}') == ["p1", "p1b"]
    assert keymap_keys(["keymaps.json"], '{"keys": []}') == []
    assert keymap_keys(["keymaps.json"], "not json") == []
    assert keymap_keys(["p1.jpg"], '{"keys": ["p1"]}') == []  # no record present


def test_list_round_trips_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "wanted.tsv"
    write_list(
        path,
        [Wanted("sanborn2", "p1"), Wanted("sanborn1", "p3"), Wanted("sanborn1", "p1")],
    )
    assert path.read_text().splitlines()[0] == "item\tkey"
    assert read_list(path) == [
        Wanted("sanborn1", "p1"),
        Wanted("sanborn1", "p3"),
        Wanted("sanborn2", "p1"),
    ]


def test_read_list_rejects_a_file_without_the_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("item\tstate\nx\ty\n")
    with pytest.raises(SystemExit, match="'key'"):
        read_list(path)


def test_resolve_list_takes_a_local_path_or_downloads_from_s3(
    tmp_path: Path, monkeypatch
) -> None:
    """The fleet reads the list from the bucket; a single machine can use a file."""
    import subprocess

    local = tmp_path / "wanted.tsv"
    local.write_text("item\tkey\n")
    assert resolve_list(str(local), tmp_path) == local

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        (tmp_path / "work" / "fetch-list.tsv").write_text("item\tkey\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    got = resolve_list("s3://bucket/_craft/list.tsv", tmp_path / "work")
    assert got == tmp_path / "work" / "fetch-list.tsv"
    assert calls[0][:4] == ["aws", "s3", "cp", "s3://bucket/_craft/list.tsv"]

    calls.clear()
    resolve_list("s3://bucket/_craft/list.tsv", tmp_path / "work")
    assert calls == []  # already downloaded


def test_fetch_one_picks_the_decoder_from_the_source_kind(
    tmp_path: Path, monkeypatch
) -> None:
    """A TIFF master fed to the JPEG-2000 decoder killed all four shards once."""
    from dataclasses import dataclass

    from mapsnap import loc_raw

    @dataclass
    class FakeSheet:
        key: str
        stem: str
        source: str
        bytes: int
        storage_dir: str

    @dataclass
    class FakePlan:
        item: str
        state: str
        year: str
        sheets: list

    calls: list[str] = []
    monkeypatch.setattr(loc_raw, "run_aws", lambda command, **kw: None)

    def fake_fetch(url, dest, expected=0):
        calls.append(f"fetch {Path(url).suffix}")
        dest.write_bytes(b"x")
        return 1

    def make(kind):
        sheet = FakeSheet("p1", "00001_1900-0001", kind, 10, "gmd/x")
        return FakePlan("sanborn1", "alabama", "1900", [sheet])

    import mapsnap.loc_mirror as mirror

    monkeypatch.setattr(mirror, "fetch", fake_fetch)
    monkeypatch.setattr(
        mirror,
        "decode_jp2",
        lambda src, out, r: (calls.append("jp2"), out.write_bytes(b"j"))[1],
    )
    monkeypatch.setattr(
        mirror,
        "save_jpeg",
        lambda image, out: (calls.append("pillow"), out.write_bytes(b"p"))[1],
    )
    monkeypatch.setattr("PIL.Image.open", lambda path: object())

    loc_raw.fetch_one(
        Wanted("sanborn1", "p1"), make("torrent-jp2"), "s3://b", "http://m", tmp_path
    )
    assert calls == ["fetch .jp2", "jp2"]

    calls.clear()
    loc_raw.fetch_one(
        Wanted("sanborn1", "p1"),
        make("torrent-master-tif"),
        "s3://b",
        "http://m",
        tmp_path,
    )
    assert calls == ["fetch .tif", "pillow"]
