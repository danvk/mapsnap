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


# --- buffering ---


def square_relation(size_deg=0.01, lat=37.55):
    """An `out geom` relation whose member ways form a closed square."""
    c = [(0.0, lat), (size_deg, lat), (size_deg, lat + size_deg), (0.0, lat + size_deg)]
    ways = [[c[i], c[(i + 1) % 4]] for i in range(4)]
    return {
        "elements": [
            {
                "type": "relation",
                "tags": {"name": "Square"},
                "members": [
                    {
                        "type": "way",
                        "role": "outer",
                        "geometry": [{"lat": la, "lon": lo} for lo, la in w],
                    }
                    for w in ways
                ],
            }
        ]
    }


def test_buffer_grows_the_boundary_by_roughly_the_requested_metres():
    from shapely.geometry import Point, Polygon

    rings = dl.relation_rings(square_relation())
    assert len(rings) == 4  # four member ways, not yet a ring
    grown = dl.buffered_rings(rings, 1000.0)
    assert len(grown) == 1

    poly = Polygon(grown[0])
    original = Polygon([(0.0, 37.55), (0.01, 37.55), (0.01, 37.56), (0.0, 37.56)])
    assert poly.contains(original)
    # A point 900 m west of the original edge is inside the buffer; 2.5 km is not.
    deg_per_m = 1 / (111_320.0 * 0.79)
    assert poly.contains(Point(-900 * deg_per_m, 37.555))
    assert not poly.contains(Point(-2500 * deg_per_m, 37.555))


def test_boundary_document_describes_the_buffered_extent():
    # The ring the viewer draws must claim the area actually downloaded, not
    # the administrative line it was grown from.
    rings = dl.relation_rings(square_relation())
    grown = dl.buffered_rings(rings, 1000.0)
    doc = dl.boundary_document(42, square_relation(), grown, 1000.0)
    element = doc["elements"][0]
    assert element["tags"]["mapsnap:buffer_m"] == "1000"
    assert element["tags"]["name"] == "Square"
    assert len(element["members"]) == len(grown)
    # Same schema the viewer already reads: way members carrying geometry.
    assert all(m["type"] == "way" and m["geometry"] for m in element["members"])


def test_poly_query_lists_lat_lon_pairs_per_ring():
    query = dl.form_osm_query_polygons([[(1.5, 2.5), (3.5, 4.5), (5.5, 6.5)]])
    assert 'poly:"2.500000 1.500000 4.500000 3.500000 6.500000 5.500000"' in query
    assert '["highway"]["name"]' in query
