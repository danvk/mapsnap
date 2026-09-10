"""Tests for the LoC JP2 mirror builder: rules, layout, decode, the pipeline, resume, upload."""

import json
from pathlib import Path

import pytest
from PIL import Image, features

from mapsnap import loc_mirror
from mapsnap.loc_mirror import (
    DONE,
    UPLOADED,
    ItemPlan,
    Settings,
    Sheet,
    _connections,
    broken_log_path,
    decode_jp2,
    fetch,
    is_candidate,
    item_relative,
    jp2_path,
    keep_sheet,
    load_mapping,
    prune_empty_parents,
    run_pipeline,
    s3_prefix,
    select_items,
    sheet_outputs,
    source_url,
)

HEADER = "item\tstate\tyear\tcity\tseq\tstem\tpage_key\tsource\tbytes\tstorage_dir\n"
DIR = "gmd/x/g1"


def test_keep_and_candidate_rules():
    assert keep_sheet("p101s") and keep_sheet("p0") and keep_sheet("pa")
    assert (
        not keep_sheet("pcovr") and not keep_sheet("pind1") and not keep_sheet("ptitl")
    )
    # Raw copies: the page-0 family and letter pages only; page 1 is not a candidate here.
    assert (
        is_candidate("p0")
        and is_candidate("p0b")
        and is_candidate("p0L")
        and is_candidate("pa")
    )
    assert (
        not is_candidate("p1") and not is_candidate("p1a") and not is_candidate("p101s")
    )


def test_load_mapping_keeps_numbered_and_letter_sheets(tmp_path: Path):
    tsv = tmp_path / "map.tsv"
    tsv.write_text(
        HEADER
        + f"sanborn00081_001\talabama\t1922\tmobile\t1\t00081_1922-0000\tp0\ttorrent-jp2\t100\t{DIR}\n"
        + f"sanborn00081_001\talabama\t1922\tmobile\t2\t00081_1922-covr\tpcovr\ttorrent-jp2\t50\t{DIR}\n"
        + f"sanborn00081_001\talabama\t1922\tmobile\t3\t00081_1922-0001s\tp1s\ttorrent-jp2\t70\t{DIR}\n"
        + "sanborn00081_001\talabama\t1922\tmobile\t4\t\tp\tghost\t0\t\n"
    )
    plan = load_mapping(tsv)["sanborn00081_001"]
    assert [s.key for s in plan.sheets] == ["p0", "p1s"]
    assert (plan.state, plan.year, plan.city) == ("alabama", "1922", "mobile")


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
    assert item_relative(plan) == Path("by-state/alabama/1922/sanborn00081_001")
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
    tif = Sheet(
        1, "00081_1922-0124", "p124", "torrent-master-tif", 0, sheet.storage_dir
    )
    assert (
        jp2_path(tmp_path, tif)
        .as_posix()
        .endswith(
            "storage-services/master/" + sheet.storage_dir + "/00081_1922-0124.tif"
        )
    )
    iiif = Sheet(1, "00081_1922-0125", "p125", "loc-iiif", 0, sheet.storage_dir)
    assert source_url("http://m", iiif).endswith(
        ":00081_1922-0125/full/pct:25/0/default.jpg"
    )
    assert source_url("http://m", iiif, full=True).endswith("/full/full/0/default.jpg")
    quarter, raw = sheet_outputs(
        tmp_path, Sheet(1, "x-0000", "p0", "torrent-jp2", 1, "d")
    )
    assert quarter.name == "p0.jpg" and raw is not None and raw.parent.name == "raw"
    assert sheet_outputs(tmp_path, sheet)[1] is None


def write_jp2(path: Path, width: int = 200, height: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), (240, 230, 200))
    for x in range(0, width, 20):
        for y in range(height):
            image.putpixel((x, y), (30, 30, 30))
    image.save(
        path, "JPEG2000", quality_mode="rates", quality_layers=[10], num_resolutions=4
    )


needs_jp2 = pytest.mark.skipif(
    not features.check("jpg_2000"), reason="Pillow lacks JPEG 2000"
)


@needs_jp2
def test_decode_jp2_quarter_and_full(tmp_path: Path):
    jp2 = tmp_path / "s.jp2"
    write_jp2(jp2)
    assert decode_jp2(jp2, tmp_path / "q.jpg", 2) == (50, 40)
    assert decode_jp2(jp2, tmp_path / "f.jpg", 0) == (200, 160)
    assert Image.open(tmp_path / "q.jpg").size == (50, 40)


def make_volume(tmp_path: Path) -> tuple[list[ItemPlan], Settings, Path]:
    """A file:// mirror with two items: one good sheet, one page-0 sheet, one corrupt sheet."""
    mirror = tmp_path / "mirror"
    files = {
        "00081_1922-0000": True,
        "00081_1922-0001": False,  # corrupt
        "00082_1922-0005": True,
    }
    for stem, good in files.items():
        path = mirror / "storage-services/service" / DIR / f"{stem}.jp2"
        if good:
            write_jp2(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a jp2")
    size = lambda stem: (
        (mirror / "storage-services/service" / DIR / f"{stem}.jp2").stat().st_size
    )
    items = [
        ItemPlan(
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
                    size("00081_1922-0000"),
                    DIR,
                ),
                Sheet(
                    2,
                    "00081_1922-0001",
                    "p1",
                    "torrent-jp2",
                    size("00081_1922-0001"),
                    DIR,
                ),
            ],
        ),
        ItemPlan(
            "sanborn00082_001",
            "alabama",
            "1922",
            "selma",
            [
                Sheet(
                    1,
                    "00082_1922-0005",
                    "p5",
                    "torrent-jp2",
                    size("00082_1922-0005"),
                    DIR,
                )
            ],
        ),
    ]
    settings = Settings(
        jp2_dir=tmp_path / "jp2",
        out_dir=tmp_path / "state",
        staging_dir=tmp_path / "staging",
        mirror=mirror.as_uri(),
    )
    return items, settings, mirror


@needs_jp2
def test_pipeline_decodes_logs_broken_and_resumes(tmp_path: Path):
    items, settings, _ = make_volume(tmp_path)
    totals = run_pipeline(items, settings, streams=2, decode_workers=1, progress=False)
    assert totals == {
        "items": 2,
        "sheets": 2,
        "broken": 1,
        "uploaded": 0,
        "bytes": pytest.approx(totals["bytes"]),
        "errors": 0,
        "retries": 0,
    }
    staged = settings.staging_dir / item_relative(items[0])
    assert Image.open(staged / "p0.jpg").size == (50, 40)
    assert Image.open(staged / "raw" / "p0.jpg").size == (200, 160)
    assert not (staged / "p1.jpg").exists()
    state = settings.out_dir / item_relative(items[0])
    assert (state / DONE).exists() and not (state / UPLOADED).exists()
    meta = json.loads((state / "metadata.json").read_text())
    assert [s["key"] for s in meta["sheets"]] == ["p0"] and meta["broken"] == [
        "00081_1922-0001"
    ]
    assert meta["sheets"][0]["width"] == 50 and meta["sheets"][0]["raw"] is True
    assert (staged / "metadata.json").read_text() == (
        state / "metadata.json"
    ).read_text()
    assert (
        broken_log_path(settings.out_dir)
        .read_text()
        .startswith("sanborn00081_001\t00081_1922-0001\ttorrent-jp2\t")
    )
    assert (
        settings.jp2_dir / "storage-services/service" / DIR / "00081_1922-0000.jp2"
    ).exists()
    # Resume with the mirror gone: both items are complete from local markers alone.
    offline = Settings(
        jp2_dir=settings.jp2_dir,
        out_dir=settings.out_dir,
        staging_dir=settings.staging_dir,
        mirror="http://127.0.0.1:9",
    )
    assert (
        run_pipeline(items, offline, streams=1, decode_workers=1, progress=False)[
            "items"
        ]
        == 0
    )


@needs_jp2
def test_pipeline_uploads_then_drops_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    items, settings, _ = make_volume(tmp_path)
    settings.bucket, settings.upload = "s3://bucket", True
    synced: list[tuple[str, str]] = []

    def fake_run(cmd, check):
        synced.append((cmd[3], cmd[4]))

    monkeypatch.setattr(loc_mirror.subprocess, "run", fake_run)
    totals = run_pipeline(items, settings, streams=2, decode_workers=1, progress=False)
    assert totals["uploaded"] == 2
    assert sorted(dest for _, dest in synced) == [
        "s3://bucket/by-state/alabama/1922/sanborn00081_001",
        "s3://bucket/by-state/alabama/1922/sanborn00082_001",
    ]
    assert (settings.out_dir / item_relative(items[0]) / UPLOADED).exists()
    assert not (settings.staging_dir / item_relative(items[0])).exists()
    assert not (settings.staging_dir / "by-state").exists()  # empty parents pruned
    # A second run finds every item uploaded and does nothing.
    assert (
        run_pipeline(items, settings, streams=1, decode_workers=1, progress=False)[
            "items"
        ]
        == 0
    )


def test_select_items_random_order_is_seeded_and_optional():
    plans = {
        f"sanborn{i:05d}_001": ItemPlan(
            f"sanborn{i:05d}_001", "alabama" if i % 2 else "ohio", "1900", ""
        )
        for i in range(40)
    }
    first = [p.item for p in select_items(plans, seed=0)]
    assert first == [
        p.item for p in select_items(plans, seed=0)
    ]  # a restart replays it
    assert first != [p.item for p in select_items(plans, seed=1)]
    assert sorted(first) == sorted(plans)
    ordered = [(p.state, p.item) for p in select_items(plans, seed=None)]
    assert ordered == sorted(ordered)
    assert [p.state for p in select_items(plans, states="ohio", seed=0)] == [
        "ohio"
    ] * 20
    assert [p.item for p in select_items(plans, limit=5, seed=0)] == first[:5]


def test_fetch_reuses_one_keep_alive_connection_per_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import http.server
    import threading

    root = tmp_path / "srv"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a" * 1000)
    (root / "b.bin").write_bytes(b"b" * 2000)

    class Handler(http.server.SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, as nginx does

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, format: str, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(loc_mirror, "RETRY_DELAY", 0.0)
    try:
        fetch(f"{base}/a.bin", tmp_path / "a.bin", 1000)
        first = _connections.conn
        fetch(f"{base}/b.bin", tmp_path / "b.bin", 2000)
        assert _connections.conn is first  # same socket, not a new connection per file
        assert (tmp_path / "b.bin").read_bytes() == b"b" * 2000
        with pytest.raises(OSError):
            fetch(f"{base}/missing.bin", tmp_path / "m.bin")
        assert (
            not (tmp_path / "m.bin").exists() and not (tmp_path / "m.bin.part").exists()
        )
        with pytest.raises(OSError):
            fetch(f"{base}/a.bin", tmp_path / "a2.bin", expected_bytes=999)
    finally:
        server.shutdown()


def test_prune_stops_at_the_root_and_survives_races(tmp_path: Path):
    root = tmp_path / "staging"
    leaf = root / "by-state" / "colorado" / "1905"
    leaf.mkdir(parents=True)
    (root / "by-state" / "ohio").mkdir()
    prune_empty_parents(leaf, root)
    assert not (root / "by-state" / "colorado").exists()
    assert (
        root / "by-state" / "ohio"
    ).exists() and root.exists()  # by-state kept: not empty
    prune_empty_parents(
        root / "by-state" / "gone" / "1900", root
    )  # already removed elsewhere: no error


@needs_jp2
def test_failed_uploads_are_retried_in_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # aws fails twice per item (an expired session), then works: the running
    # process retries with backoff, keeps the staging copy meanwhile, and
    # never needs a restart. Every failure is still logged.
    items, settings, _ = make_volume(tmp_path)
    settings.bucket, settings.upload = "s3://bucket", True
    monkeypatch.setattr(loc_mirror, "UPLOAD_RETRY_DELAY", 0.05)
    calls: dict[str, int] = {}

    def flaky_run(cmd, check):
        calls[cmd[4]] = calls.get(cmd[4], 0) + 1
        if calls[cmd[4]] <= 2:
            raise RuntimeError("Your session has expired")

    monkeypatch.setattr(loc_mirror.subprocess, "run", flaky_run)
    totals = run_pipeline(items, settings, streams=2, decode_workers=1, progress=False)
    assert totals["uploaded"] == 2 and totals["errors"] == 4 and totals["retries"] == 4
    assert all(n == 3 for n in calls.values())
    state = settings.out_dir / item_relative(items[0])
    assert (state / DONE).exists() and (state / UPLOADED).exists()
    assert not (settings.staging_dir / item_relative(items[0])).exists()
    assert (settings.out_dir / "errors.log").read_text().count("\tupload\t") == 4


@needs_jp2
def test_upload_failure_survives_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # If the run ends while uploads are still failing, the item is done but
    # not uploaded, its staging copy is kept, and the next run goes straight
    # to the upload stage without re-decoding.
    items, settings, _ = make_volume(tmp_path)
    settings.bucket, settings.upload = "s3://bucket", True
    monkeypatch.setattr(loc_mirror, "UPLOAD_RETRY_DELAY", 0.05)
    monkeypatch.setattr(loc_mirror, "UPLOAD_RETRY_MAX", 0.05)
    attempts = {"n": 0}

    def failing_then_stop(cmd, check):
        attempts["n"] += 1
        if attempts["n"] > 6:
            raise KeyboardInterrupt  # the operator gives up on this run
        raise RuntimeError("aws exploded")

    monkeypatch.setattr(loc_mirror.subprocess, "run", failing_then_stop)
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(items, settings, streams=2, decode_workers=1, progress=False)
    state = settings.out_dir / item_relative(items[0])
    assert (state / DONE).exists() and not (state / UPLOADED).exists()
    assert (settings.staging_dir / item_relative(items[0]) / "p0.jpg").exists()
    synced: list[str] = []
    monkeypatch.setattr(
        loc_mirror.subprocess, "run", lambda cmd, check: synced.append(cmd[4])
    )
    again = run_pipeline(items, settings, streams=1, decode_workers=1, progress=False)
    assert again["uploaded"] == 2 and again["items"] == 0 and len(synced) == 2
