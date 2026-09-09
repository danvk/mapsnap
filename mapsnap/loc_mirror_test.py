"""Tests for the LoC JP2 mirror builder: planning, layout, decode, resume, broken sheets."""

from pathlib import Path

import pytest
from PIL import Image, features

from mapsnap.loc_mirror import (
    DONE,
    ItemPlan,
    Settings,
    Sheet,
    broken_log_path,
    decode_jp2,
    is_candidate,
    item_dir,
    jp2_path,
    keep_sheet,
    load_mapping,
    process_item,
    s3_prefix,
    sheet_outputs,
    source_url,
)

HEADER = "item\tstate\tyear\tcity\tseq\tstem\tpage_key\tsource\tbytes\tstorage_dir\n"


def test_keep_and_candidate_rules():
    assert keep_sheet("p101s") and keep_sheet("p0") and keep_sheet("pa")
    assert (
        not keep_sheet("pcovr") and not keep_sheet("pind1") and not keep_sheet("ptitl")
    )
    assert (
        is_candidate("p0")
        and is_candidate("p0b")
        and is_candidate("p1")
        and is_candidate("pa")
    )
    assert is_candidate("p1a") and is_candidate("p0L")
    assert (
        not is_candidate("p2") and not is_candidate("p101s") and not is_candidate("p10")
    )


def test_load_mapping_keeps_numbered_and_letter_sheets(tmp_path: Path):
    tsv = tmp_path / "map.tsv"
    tsv.write_text(
        HEADER
        + "sanborn00081_001\talabama\t1922\tmobile\t1\t00081_1922-0000\tp0\ttorrent-jp2\t100\tgmd/x/g1\n"
        + "sanborn00081_001\talabama\t1922\tmobile\t2\t00081_1922-covr\tpcovr\ttorrent-jp2\t50\tgmd/x/g1\n"
        + "sanborn00081_001\talabama\t1922\tmobile\t3\t00081_1922-0001s\tp1s\ttorrent-jp2\t70\tgmd/x/g1\n"
        + "sanborn00081_001\talabama\t1922\tmobile\t4\t\tp\tghost\t0\t\n"
    )
    plans = load_mapping(tsv)
    plan = plans["sanborn00081_001"]
    assert [s.key for s in plan.sheets] == ["p0", "p1s"]
    assert plan.state == "alabama" and plan.year == "1922"


def test_layout_matches_the_s3_prefix_shape(tmp_path: Path):
    plan = ItemPlan("sanborn00081_001", "alabama", "1922", "mobile")
    sheet = Sheet(
        1,
        "00081_1922-0123",
        "p123",
        "torrent-jp2",
        10,
        "gmd/gmd390m/g3904m/g3904mm/g000811922",
    )
    assert (
        item_dir(tmp_path, plan) == tmp_path / "by-state/alabama/1922/sanborn00081_001"
    )
    assert (
        s3_prefix("s3://mapsnap-sanborn", plan)
        == "s3://mapsnap-sanborn/by-state/alabama/1922/sanborn00081_001"
    )
    assert (
        jp2_path(tmp_path, sheet)
        == tmp_path
        / "storage-services/service"
        / sheet.storage_dir
        / "00081_1922-0123.jp2"
    )
    assert (
        source_url("http://m", sheet)
        == "http://m/storage-services/service/gmd/gmd390m/g3904m/g3904mm/g000811922/00081_1922-0123.jp2"
    )
    iiif = Sheet(
        1,
        "00081_1922-0124",
        "p124",
        "loc-iiif",
        0,
        "gmd/gmd390m/g3904m/g3904mm/g000811922",
    )
    assert source_url("http://m", iiif).endswith(
        ":00081_1922-0124/full/pct:25/0/default.jpg"
    )
    assert source_url("http://m", iiif, full=True).endswith("/full/full/0/default.jpg")
    quarter, raw = sheet_outputs(
        item_dir(tmp_path, plan), Sheet(1, "x-0000", "p0", "torrent-jp2", 1, "d")
    )
    assert quarter.name == "p0.jpg" and raw is not None and raw.parent.name == "raw"
    assert sheet_outputs(item_dir(tmp_path, plan), sheet)[1] is None


def write_jp2(path: Path, width: int = 200, height: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), (240, 230, 200))
    for x in range(0, width, 20):
        for y in range(height):
            image.putpixel((x, y), (30, 30, 30))
    image.save(
        path, "JPEG2000", quality_mode="rates", quality_layers=[10], num_resolutions=4
    )


@pytest.mark.skipif(not features.check("jpg_2000"), reason="Pillow lacks JPEG 2000")
def test_decode_jp2_quarter_and_full(tmp_path: Path):
    jp2 = tmp_path / "s.jp2"
    write_jp2(jp2)
    assert decode_jp2(jp2, tmp_path / "q.jpg", 2) == (50, 40)
    assert decode_jp2(jp2, tmp_path / "f.jpg", 0) == (200, 160)
    assert Image.open(tmp_path / "q.jpg").size == (50, 40)


@pytest.mark.skipif(not features.check("jpg_2000"), reason="Pillow lacks JPEG 2000")
def test_process_item_is_resumable_and_logs_broken_sheets(tmp_path: Path):
    # A file:// "mirror" holding one good JP2 and one corrupt one.
    mirror = tmp_path / "mirror"
    good = mirror / "storage-services/service/gmd/x/g1/00081_1922-0000.jp2"
    bad = mirror / "storage-services/service/gmd/x/g1/00081_1922-0001.jp2"
    write_jp2(good)
    bad.write_bytes(b"not a jp2")
    plan = ItemPlan(
        "sanborn00081_001",
        "alabama",
        "1922",
        "mobile",
        [
            Sheet(
                1,
                "00081_1922-0000",
                "p0",
                "torrent-jp2",
                good.stat().st_size,
                "gmd/x/g1",
            ),
            Sheet(
                2,
                "00081_1922-0001",
                "p1",
                "torrent-jp2",
                bad.stat().st_size,
                "gmd/x/g1",
            ),
        ],
    )
    settings = Settings(
        jp2_dir=tmp_path / "jp2",
        out_dir=tmp_path / "out",
        mirror=mirror.as_uri(),
        streams=2,
    )
    record = process_item((plan, settings))
    dest = item_dir(settings.out_dir, plan)
    assert record["sheets"] == 2 and (dest / DONE).exists()
    assert (dest / "p0.jpg").exists() and (dest / "raw" / "p0.jpg").exists()
    assert Image.open(dest / "p0.jpg").size == (50, 40)
    assert not (dest / "p1.jpg").exists()
    assert (
        settings.jp2_dir / "storage-services/service/gmd/x/g1/00081_1922-0000.jp2"
    ).exists()
    log = broken_log_path(settings.out_dir).read_text()
    assert log.startswith("sanborn00081_001\t00081_1922-0001\ttorrent-jp2\t")
    import json

    meta = json.loads((dest / "metadata.json").read_text())
    assert [s["key"] for s in meta["sheets"]] == ["p0"] and meta["broken"] == [
        "00081_1922-0001"
    ]
    assert meta["sheets"][0]["width"] == 50 and meta["sheets"][0]["raw"] is True
    # Second run: nothing re-fetched or re-decoded (the mirror can even be gone).
    (dest / "p0.jpg").write_bytes(b"sentinel")
    settings_offline = Settings(
        jp2_dir=settings.jp2_dir,
        out_dir=settings.out_dir,
        mirror="http://127.0.0.1:9",
        streams=1,
    )
    process_item((plan, settings_offline))
    assert (dest / "p0.jpg").read_bytes() == b"sentinel"
