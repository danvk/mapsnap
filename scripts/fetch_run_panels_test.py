"""Tests for fetch_run_panels.py."""

from pathlib import Path

import fetch_run_panels
from fetch_run_panels import benchmark_pages, fetch_item, run_outputs

from mapsnap.loc_craft import Item


def test_benchmark_pages_groups_pages_by_item(tmp_path: Path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("image\titem\tpage\na\tx\tp1\nb\ty\tp2\nc\tx\tp3\n")
    assert benchmark_pages(manifest) == {"x": ["p1", "p3"], "y": ["p2"]}


def test_run_outputs_needs_the_done_marker_and_finds_cut_pages():
    present = ["p1.panels.json", "p1.georef.json", "mapsnap.iiif.json"]
    assert run_outputs(present, ["p1", "p2"]) == (True, ["p1"])
    assert run_outputs(["p1.panels.json"], ["p1"]) == (False, ["p1"])


def test_fetch_item_names_files_by_item_and_page(tmp_path: Path, monkeypatch):
    item = Item("sanborn00001_001", "ohio", "1900")
    run = f"{item.prefix}/runs/corpus-v1"
    monkeypatch.setattr(
        fetch_run_panels,
        "list_prefix",
        lambda bucket, prefix: (
            ["mapsnap.iiif.json", "p2.panels.json"] if prefix == run else []
        ),
    )

    def fake_sync(source: str, destination: str, *filters: str) -> None:
        assert source == f"s3://bucket/{run}"
        assert filters == ("--exclude", "*", "--include", "p2.panels.json")
        Path(destination).mkdir(parents=True)
        (Path(destination) / "p2.panels.json").write_text("{}")

    monkeypatch.setattr(fetch_run_panels, "sync", fake_sync)
    assert fetch_item(item, ["p1", "p2"], "s3://bucket", "corpus-v1", tmp_path)
    assert (tmp_path / "sanborn00001_001__p2.panels.json").exists()
    assert not fetch_item(item, ["p1"], "s3://bucket", "other-run", tmp_path)
