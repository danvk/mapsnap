"""Tests for mapsnap.download_osm retry behavior."""

import io
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
