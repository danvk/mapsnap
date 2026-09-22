#!/usr/bin/env python
"""Serve the static IIIF tree build.py writes.

    scripts/local_iiif/serve.py [--root DIR] [--port 8183]

Everything under `--root` is served at `/iiif/...`, which is what the info.json
`id` build.py bakes in points at. Two things a plain static server would not do
and a IIIF viewer needs:

  * CORS. A viewer fetches info.json with XHR from another origin -- the
    debugger on :5173 or :8182, Allmaps on the web -- and the browser drops
    the response without `Access-Control-Allow-Origin`.
  * `info.json` as `application/json`, and the long-lived immutable caching a
    real host would set, so the browser stops re-asking.

This is the whole server. In the arrangement #501 proposes there is no server
at all: the same tree sits in an object store behind a CDN, and these headers
come from its configuration.
"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".cache/mapsnap/local-iiif"
# A tile of a 1910 sheet never changes, which is the whole reason this is
# cheap to host: every object is immutable and cacheable forever.
CACHE_CONTROL = "public, max-age=31536000, immutable"


class Handler(SimpleHTTPRequestHandler):
    """Static files under /iiif, with the headers a IIIF client needs."""

    def translate_path(self, path: str) -> str:
        # /iiif/<item>/<page>/... -> <root>/<item>/<page>/...
        cleaned = path.split("?", 1)[0].split("#", 1)[0]
        if cleaned.startswith("/iiif/"):
            path = cleaned[len("/iiif") :]
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", CACHE_CONTROL)
        super().end_headers()

    def guess_type(self, path):  # type: ignore[override]
        if str(path).endswith("info.json"):
            return "application/json"
        return super().guess_type(path)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        if "404" in (format % args):  # a miss is worth seeing; the rest is noise
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--port", type=int, default=8183)
    args = parser.parse_args()
    if not args.root.exists():
        raise SystemExit(f"{args.root} does not exist; run build.py first")
    handler = partial(Handler, directory=str(args.root))
    items = sorted(p.name for p in args.root.iterdir() if p.is_dir())
    print(
        f"serving {len(items)} items from {args.root} at http://localhost:{args.port}/iiif/"
    )
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
