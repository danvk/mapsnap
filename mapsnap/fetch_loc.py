#!/usr/bin/env python3
"""Fetch all Sanborn map metadata pages from the Library of Congress."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "loc"
DELAY_SECONDS = 10
BASE_URL = (
    "https://www.loc.gov/collections/sanborn-maps/?fo=json&c=100&at=results,pagination"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}


MAX_RETRIES = 5
RETRY_DELAYS = [30, 60, 120, 300, 600]  # seconds between retries


def fetch_page(page_num: int) -> dict | None:
    url = BASE_URL if page_num == 1 else f"{BASE_URL}&sp={page_num}"
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} on page {page_num} (attempt {attempt + 1})")
        except Exception as e:
            print(f"  Error on page {page_num} (attempt {attempt + 1}): {e}")
        if attempt < MAX_RETRIES - 1:
            delay = RETRY_DELAYS[attempt]
            print(f"  Retrying in {delay}s...", flush=True)
            time.sleep(delay)
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    page_num = 1

    while True:
        out_path = OUTPUT_DIR / f"page_{page_num:04d}.json"

        if out_path.exists():
            print(f"Page {page_num}: already exists, skipping")
            page_num += 1
            continue

        print(f"Page {page_num}: fetching...", flush=True)
        data = fetch_page(page_num)

        if data is None:
            print(f"Page {page_num}: all retries failed, stopping")
            break

        results = data.get("results", [])
        if not results:
            print(f"Page {page_num}: no results, stopping")
            break

        pagination = data.get("pagination", {})
        total_pages = pagination.get("total", "?")

        out_path.write_text(json.dumps(data, indent=2))
        print(
            f"Page {page_num}/{total_pages}: saved {len(results)} results to {out_path.name}"
        )

        page_num += 1

        if page_num > 1:
            print(f"  Waiting {DELAY_SECONDS}s...", flush=True)
            time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
