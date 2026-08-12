"""Tests for mapsnap.download_osm retry behavior."""

import io
import json
import urllib.error

import pytest

from mapsnap import download_osm as dl


class _FakeResp:
    """Minimal context-manager stand-in for urlopen's response."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return io.BytesIO(self._body)

    def __exit__(self, *exc):
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(dl.OVERPASS_URL, code, "boom", {}, None)  # type: ignore[arg-type]


def test_retries_then_succeeds_on_transient_error(monkeypatch):
    # 429, then 504, then a successful JSON response.
    attempts: list[None] = []
    responses = [_http_error(429), _http_error(504), _FakeResp(b'{"elements": [1, 2]}')]

    def fake_urlopen(req, timeout=0):
        result = responses[len(attempts)]
        attempts.append(None)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)

    result = dl.download_osm("query", max_attempts=5, initial_delay=0.01)
    assert result == {"elements": [1, 2]}
    assert len(attempts) == 3


def test_exits_after_exhausting_retries(monkeypatch):
    def always_429(req, timeout=0):
        raise _http_error(429)

    monkeypatch.setattr(dl.urllib.request, "urlopen", always_429)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit):
        dl.download_osm("query", max_attempts=3, initial_delay=0.01)


def test_exits_immediately_on_non_transient_error(monkeypatch):
    calls: list[None] = []

    def http_400(req, timeout=0):
        calls.append(None)
        raise _http_error(400)

    monkeypatch.setattr(dl.urllib.request, "urlopen", http_400)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit):
        dl.download_osm("query", max_attempts=5, initial_delay=0.01)
    # A client error (400) is not retried.
    assert len(calls) == 1


# --- output directory + relation boundary ---


def run_main(monkeypatch, argv, responses):
    """Run main() with Overpass stubbed, returning the queries it issued."""
    import sys

    queries = []

    def fake_download(query, **_):
        queries.append(query)
        return responses[len(queries) - 1]

    monkeypatch.setattr(dl, "download_osm", fake_download)
    monkeypatch.setattr(sys, "argv", ["mapsnap download-osm", *argv])
    dl.main()
    return queries


STREETS = {"elements": [{"type": "way", "id": 1}]}
BOUNDARY = {
    "elements": [
        {"type": "relation", "tags": {"name": "Richmond"}, "members": [{"type": "way"}]}
    ]
}


def test_relation_mode_writes_streets_and_the_boundary(monkeypatch, tmp_path):
    queries = run_main(monkeypatch, ["r3864712", str(tmp_path)], [STREETS, BOUNDARY])
    assert json.loads((tmp_path / "streets.osm.json").read_text()) == STREETS
    # The boundary the viewer draws, named for the relation it came from.
    assert json.loads((tmp_path / "r3864712.json").read_text()) == BOUNDARY
    assert "out geom" in queries[1] and "rel(3864712)" in queries[1]


def test_bbox_mode_writes_no_boundary(monkeypatch, tmp_path):
    # A bbox download has no relation; drawing no ring beats drawing a wrong one.
    queries = run_main(
        monkeypatch, ["1.0", "2.0", "3.0", "4.0", str(tmp_path)], [STREETS]
    )
    assert (tmp_path / "streets.osm.json").exists()
    assert list(tmp_path.glob("r*.json")) == []
    assert len(queries) == 1


def test_a_new_relation_clears_the_previous_boundary(monkeypatch, tmp_path):
    # Re-downloading a volume against a different relation must not leave the
    # old ring behind: it would trace an extent the streets did not come from.
    (tmp_path / "r999.json").write_text("{}")
    run_main(monkeypatch, ["r3864712", str(tmp_path)], [STREETS, BOUNDARY])
    assert not (tmp_path / "r999.json").exists()
    assert (tmp_path / "r3864712.json").exists()
