#!/usr/bin/env python
"""The scans the mirror skips, and their sizes from the Chronoscope CDN.

    PYTHONPATH=.:scripts/atlas uv run python scripts/atlas/unmirrored_scans.py

The mirror takes only the sheets the pipeline can use (``keep_sheet`` in
mapsnap/loc_mirror.py): numbered and lettered pages. A volume's cover, title
page and street index are left out, and so are a few map sheets the pipeline
cannot name yet (central business district sheets, ``pkey1``, Morrisville's
``pM002`` in Trenton 1927 vol. 3; #467). That is 8,864 of LoC's 441,178
scans. The Chronoscope CDN serves them all, so the item pages
(build_item_pages.py) can show every scan LoC has.

An item page needs each image's size: the CDN is a level-0 image service,
which serves only the sizes its info.json lists, and a IIIF manifest's canvas
needs one too. The mirror's metadata.json gives the size of every sheet it
took; for the rest, this fetches each scan's info.json once, at RATE_PER_S,
into a cache that a rerun resumes from. A scan the CDN does not have is
cached as such; any other failure is not, so the next run tries it again.
"""

import argparse
import csv
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mapsnap.loc_mirror import keep_sheet

DOCUMENTS = Path.home() / "Documents/mapsnap"
DEFAULT_MAPPING = Path.home() / "Downloads/loc-sanborn-maps.mapping.tsv"
DEFAULT_CACHE = DOCUMENTS / "cdn-info"
CDN_BASE = "https://cdn.chronoscope.io/mapsnap"
RATE_PER_S = 2.0
# A skipped scan's role, by the start of its page key after the "p"
# (scan_role). "in2" is one volume's misspelled second index page.
ROLE_PATTERNS = [
    ("cover", re.compile(r"cov")),
    ("title", re.compile(r"titl")),
    ("index", re.compile(r"ind|in\d")),
    ("notes", re.compile(r"not")),
    ("corrections", re.compile(r"cor")),
    (
        "map",
        re.compile(r"cbd|key|m\d|ier|dock|airp|watr|fire|topo|resv|dry|hvd"),
    ),
]
# Cloudflare answers Python's default User-Agent with a 403.
USER_AGENT = "mapsnap-item-pages/1.0 (+https://mapsnap.org)"


@dataclass
class Scan:
    """One scan of a volume that the mirror did not take, from the mirror's mapping TSV."""

    item: str
    seq: int
    key: str
    stem: str
    storage_dir: str
    source: str = ""


def service_id(storage_dir: str, stem: str) -> str:
    """LoC's image-service id for a sheet: its storage path with colons, then its stem."""
    return "service:" + storage_dir.replace("/", ":") + ":" + stem


def info_url(scan: Scan) -> str:
    """A scan's info.json on the Chronoscope CDN."""
    return f"{CDN_BASE}/{service_id(scan.storage_dir, scan.stem)}/info.json"


def read_unmirrored(mapping: Path) -> dict[str, list[Scan]]:
    """Item -> the scans the mirror left out, in LoC's order.

    A row with no stem names no image (one item has an empty page key) and is
    skipped.
    """
    scans: dict[str, list[Scan]] = {}
    with mapping.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row["stem"] or keep_sheet(row["page_key"]):
                continue
            scans.setdefault(row["item"], []).append(
                Scan(
                    row["item"],
                    int(row["seq"]),
                    row["page_key"],
                    row["stem"],
                    row["storage_dir"],
                    row.get("source", ""),
                )
            )
    for item_scans in scans.values():
        item_scans.sort(key=lambda scan: scan.seq)
    return scans


def cache_path(cache: Path, scan: Scan) -> Path:
    """Where a scan's info.json is cached."""
    return cache / scan.item / f"{scan.stem}.json"


def cached_size(cache: Path, scan: Scan) -> tuple[int, int] | None:
    """A scan's (width, height) on the CDN, from the cache; None if not fetched or missing."""
    path = cache_path(cache, scan)
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    if "width" not in info or "height" not in info:
        return None
    return int(info["width"]), int(info["height"])


def fetch_info(url: str) -> tuple[int, bytes]:
    """GET a URL; (status, body), with status 0 for a network failure of any kind."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""
    except (OSError, http.client.HTTPException):
        # Timeouts, refused or reset connections, truncated responses.
        return 0, b""


def fill_cache(scans: list[Scan], cache: Path, rate_per_s: float = RATE_PER_S) -> dict:
    """Fetch the info.json of each scan not yet cached, one at a time at ``rate_per_s``."""
    counts = {"cached": 0, "fetched": 0, "missing": 0, "failed": 0}
    todo = [scan for scan in scans if not cache_path(cache, scan).exists()]
    counts["cached"] = len(scans) - len(todo)
    print(f"{len(todo):,} of {len(scans):,} scans to fetch", file=sys.stderr)
    interval = 1.0 / rate_per_s
    for n, scan in enumerate(todo, start=1):
        started = time.monotonic()
        status, body = fetch_info(info_url(scan))
        path = cache_path(cache, scan)
        if status == 200:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            counts["fetched"] += 1
        elif status == 404:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"status": 404}))
            counts["missing"] += 1
        else:
            counts["failed"] += 1
            print(f"  {scan.item} {scan.stem}: HTTP {status}", file=sys.stderr)
        if n % 500 == 0:
            print(f"  {n:,}/{len(todo):,} {counts}", file=sys.stderr)
        time.sleep(max(0.0, interval - (time.monotonic() - started)))
    return counts


def scan_role(key: str) -> str:
    """What a scan the mirror skipped is, from its page key.

    "cover", "title", "index", "notes" and "corrections" are a volume's front
    and back matter. "map" is a map sheet the pipeline cannot name yet (#467):
    a central business district (pcbd1, pCBDa), a key map (pkey1), the piers
    or a dock, or Morrisville's sheets in Trenton 1927 vol. 3 (pM002).
    Anything else is "other".
    """
    name = key.lower().removeprefix("p")
    for role, pattern in ROLE_PATTERNS:
        if pattern.match(name):
            return role
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--rate", type=float, default=RATE_PER_S, help="Requests per second"
    )
    args = parser.parse_args()
    scans = [scan for item in read_unmirrored(args.mapping).values() for scan in item]
    counts = fill_cache(scans, args.cache, args.rate)
    print(counts, file=sys.stderr)


if __name__ == "__main__":
    main()
