#!/usr/bin/env python
"""Fetch a corpus run's split outputs for the pages of a split benchmark.

The splitter is deterministic and unchanged since a corpus run, so scoring it
on a benchmark of mirror pages need not re-run it: the run already wrote a
``pN.panels.json`` beside every page it cut, under ``<item>/runs/<tag>/``. A
page with no such file was left whole -- but only if the run finished the item,
which its ``mapsnap.iiif.json`` marks.

Writes ``<out>/<item>__<page>.panels.json`` for every benchmark page the run
cut, and ``<out>/unfinished.json`` listing the benchmark images whose item the
run never finished (score_splits_oim.py skips those). Run from the project root:

    uv run python scripts/fetch_run_panels.py \\
        ~/Documents/mapsnap/cutline-training/manifest.tsv \\
        ~/Downloads/loc-sanborn-maps.mapping.tsv corpus-v1-panels --run-tag corpus-v1
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapsnap.loc_craft import Item, list_prefix, read_manifest, sync
from mapsnap.loc_fit import ARCHIVE_TAG, RUNS_DIRNAME

DONE_MARKER = f"{ARCHIVE_TAG}.iiif.json"


def benchmark_pages(manifest: Path) -> dict[str, list[str]]:
    """Page keys per item in a benchmark manifest (columns item, page)."""
    pages: dict[str, list[str]] = {}
    with manifest.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pages.setdefault(row["item"], []).append(row["page"])
    return pages


def run_outputs(present: list[str], pages: list[str]) -> tuple[bool, list[str]]:
    """(whether the run finished the item, which of these pages it cut)."""
    cut = [page for page in pages if f"{page}.panels.json" in present]
    return DONE_MARKER in present, cut


def fetch_item(
    item: Item, pages: list[str], bucket: str, run_tag: str, out: Path
) -> bool:
    """Copy one item's panels.json files for these pages into out; False if unfinished."""
    run = f"{item.prefix}/{RUNS_DIRNAME}/{run_tag}"
    done, cut = run_outputs(list_prefix(bucket, run), pages)
    if not done:
        return False
    if cut:
        staging = out / ".staging" / item.item
        includes = [arg for page in cut for arg in ("--include", f"{page}.panels.json")]
        sync(f"{bucket.rstrip('/')}/{run}", str(staging), "--exclude", "*", *includes)
        for page in cut:
            (staging / f"{page}.panels.json").replace(
                out / f"{item.item}__{page}.panels.json"
            )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("manifest", type=Path, help="The benchmark's manifest.tsv")
    parser.add_argument(
        "mapping", type=Path, help="The mirror's loc-sanborn-maps.mapping.tsv"
    )
    parser.add_argument("out", type=Path, help="Directory for the panels.json files")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--bucket", default="s3://mapsnap-sanborn")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    pages = benchmark_pages(args.manifest)
    items = {item.item: item for item in read_manifest(args.mapping)}
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        finished = dict(
            zip(
                pages,
                pool.map(
                    lambda name: fetch_item(
                        items[name], pages[name], args.bucket, args.run_tag, args.out
                    ),
                    pages,
                ),
            )
        )
    unfinished = sorted(
        f"{name}__{page}"
        for name, done in finished.items()
        if not done
        for page in pages[name]
    )
    (args.out / "unfinished.json").write_text(json.dumps(unfinished, indent=1))
    cut = len(list(args.out.glob("*.panels.json")))
    print(
        f"{len(pages)} items: {cut} pages cut, {len(unfinished)} pages in "
        f"{sum(not done for done in finished.values())} unfinished item(s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
