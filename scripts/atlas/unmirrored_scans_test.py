"""Tests for unmirrored_scans.py."""

import json
from pathlib import Path

import unmirrored_scans
from unmirrored_scans import (
    Scan,
    cache_path,
    cached_size,
    fetch_info,
    fill_cache,
    info_url,
    read_unmirrored,
    scan_role,
)

HEADER = "item\tstate\tyear\tcity\tseq\tstem\tpage_key\tsource\tbytes\tstorage_dir\n"
STORAGE = "gmd/gmd380m/g3804m/g3804bm/g3804bm_g05791193901"


def row(seq: int, key: str, stem: str) -> str:
    """One mapping TSV row of Brooklyn 1939 vol. 1."""
    return (
        f"sanborn05791_053\tnew-york\t1939\tbrooklyn\t{seq}\t{stem}\t{key}"
        f"\ttorrent-jp2\t0\t{STORAGE}\n"
    )


def test_read_unmirrored_keeps_only_what_the_mirror_skips(tmp_path: Path):
    mapping = tmp_path / "mapping.tsv"
    mapping.write_text(
        HEADER
        + row(2, "pind1", "05791_01_1939-ind1")
        + row(1, "ptitl", "05791_01_1939-titl")
        + row(5, "p0", "05791_01_1939-0000")
        + row(6, "", "")
    )
    scans = read_unmirrored(mapping)
    assert [(s.seq, s.key) for s in scans["sanborn05791_053"]] == [
        (1, "ptitl"),
        (2, "pind1"),
    ]


def test_info_url_is_the_cdn_service():
    scan = Scan("sanborn05791_053", 1, "ptitl", "05791_01_1939-titl", STORAGE)
    assert info_url(scan) == (
        "https://cdn.chronoscope.io/mapsnap/service:gmd:gmd380m:g3804m:g3804bm:"
        "g3804bm_g05791193901:05791_01_1939-titl/info.json"
    )


def test_scan_role():
    roles = {
        "ptitl": "title",
        "pind2": "index",
        "pIndx": "index",
        "pin2": "index",
        "pcovr": "cover",
        "pcovra": "cover",
        "pnote": "notes",
        "pcorr": "corrections",
        "pcbd1": "map",
        "pCBDa": "map",
        "pkey2": "map",
        "pM010": "map",
        "pierL": "map",
        "phvda": "map",
        "psb0000B2": "other",
        "p": "other",
    }
    assert {key: scan_role(key) for key in roles} == roles


def test_fill_cache_keeps_sizes_and_misses_but_retries_failures(
    tmp_path: Path, monkeypatch
):
    scans = [
        Scan("item", seq, f"pind{seq}", f"stem-{seq}", STORAGE) for seq in (1, 2, 3)
    ]
    answers = {
        "stem-1": (200, json.dumps({"width": 1380, "height": 2067}).encode()),
        "stem-2": (404, b""),
        "stem-3": (503, b""),
    }
    monkeypatch.setattr(
        unmirrored_scans,
        "fetch_info",
        lambda url: answers[url.split(":")[-1].removesuffix("/info.json")],
    )
    counts = fill_cache(scans, tmp_path, rate_per_s=1000)
    assert counts == {"cached": 0, "fetched": 1, "missing": 1, "failed": 1}
    assert cached_size(tmp_path, scans[0]) == (1380, 2067)
    assert cached_size(tmp_path, scans[1]) is None
    assert cache_path(tmp_path, scans[1]).exists()
    assert not cache_path(tmp_path, scans[2]).exists()
    # A rerun fetches only the failure.
    assert fill_cache(scans, tmp_path, rate_per_s=1000)["cached"] == 2


def test_fetch_info_reports_a_reset_connection_as_a_failure(monkeypatch):
    def reset(*args, **kwargs):
        raise ConnectionResetError(54, "Connection reset by peer")

    monkeypatch.setattr(unmirrored_scans.urllib.request, "urlopen", reset)
    assert fetch_info("https://cdn.example/info.json") == (0, b"")
