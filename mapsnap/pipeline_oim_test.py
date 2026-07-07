from pathlib import Path

from mapsnap.pipeline_oim import delete_other_raw


def make_raw(volume: Path, names: list[str]) -> None:
    (volume / "raw").mkdir(parents=True, exist_ok=True)
    for name in names:
        (volume / "raw" / name).write_bytes(b"")


def test_delete_other_raw_keeps_only_key_maps(tmp_path: Path):
    make_raw(tmp_path, ["p0.jpg", "p1.jpg", "p2.jpg", "p50.jpg"])
    delete_other_raw(tmp_path, ["p0"])
    remaining = sorted(p.name for p in (tmp_path / "raw").glob("*.jpg"))
    assert remaining == ["p0.jpg"]


def test_delete_other_raw_keeps_multiple_split_key_maps(tmp_path: Path):
    make_raw(tmp_path, ["p0L.jpg", "p0R.jpg", "p1.jpg", "p2.jpg"])
    delete_other_raw(tmp_path, ["p0L", "p0R"])
    remaining = sorted(p.name for p in (tmp_path / "raw").glob("*.jpg"))
    assert remaining == ["p0L.jpg", "p0R.jpg"]


def test_delete_other_raw_leaves_non_jpg_untouched(tmp_path: Path):
    make_raw(tmp_path, ["p0.jpg", "p1.jpg"])
    (tmp_path / "raw" / "p0.keymap.json").write_text("{}")
    delete_other_raw(tmp_path, ["p0"])
    assert (tmp_path / "raw" / "p0.keymap.json").exists()
    assert not (tmp_path / "raw" / "p1.jpg").exists()
