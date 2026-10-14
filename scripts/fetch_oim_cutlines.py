#!/usr/bin/env python
"""Download OIM's cutlines and region boundaries for every split page, politely.

Training data for a cut-line detector: the dividing polylines volunteers drew
on OldInsuranceMaps.net, with the regions they produced. Unlike
fetch_oim_truth.py, which only needs the split pages a 100%-complete volume
georeferenced, this takes every volume with split pages, finished or not.

Per volume, written to ``<out>/<identifier>/``:

- ``documents.json``: the volume's document listing from its map page (id,
  title, page key, image size, LoC IIIF service, OIM file) -- what a trainer
  needs to fetch each sheet;
- ``oim/pN.panels.json`` and ``oim/pN.cutlines.json`` for every document OIM
  split into >= 2 regions (see oim_panels.write_page_files);
- ``volume.json``: what was fetched, and any problems. Its presence marks the
  volume done, so a rerun skips it.

Requests per volume: the map page, one document page (which lists every region
in the volume, so it says which documents are split), then one page per other
split document. Every request goes through one throttle.

    uv run python scripts/fetch_oim_cutlines.py ~/Documents/mapsnap/oim-truth-volumes.tsv \\
        ~/Documents/mapsnap/oim-cutlines --have ~/Documents/mapsnap/oim --delay 1.5
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_oim_truth import Throttle

from mapsnap.oim_panels import (
    OIM_BASE,
    embedded_json,
    parse_document_regions,
    title_page_key,
    volume_region_counts,
    write_page_files,
)

DOCUMENT_FIELDS = ("id", "title", "page_number", "image_size", "iiif_info", "file")


def has_split_pages(row: dict[str, str]) -> bool:
    """Whether OIM lists more regions than documents for a volume: some page was cut."""
    return int(row["region_ct"] or 0) > int(row["document_ct"] or 0)


def page_key_for(document: dict) -> str:
    """The page key a document's title names, or doc<id> when it names none."""
    return title_page_key(str(document.get("title", ""))) or f"doc{document['id']}"


def fetch_volume(row: dict[str, str], out: Path, throttle: Throttle) -> dict:
    """Fetch one volume's split pages into out/<identifier>/ and return its record."""
    slug = row["identifier"]
    volume = out / slug
    volume.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "identifier": slug,
        "title": row["title"],
        "year": row["year"],
        "document_ct": int(row["document_ct"] or 0),
        "region_ct": int(row["region_ct"] or 0),
        "completion_pct": row.get("completion_pct"),
        "problems": [],
    }
    documents = (
        embedded_json(throttle.text(f"{OIM_BASE}/map/{slug}"), "documents") or []
    )
    listing = [
        {**{k: d.get(k) for k in DOCUMENT_FIELDS}, "page_key": page_key_for(d)}
        for d in documents
    ]
    (volume / "documents.json").write_text(json.dumps(listing, indent=1))
    if not documents:
        record["problems"].append("no document listing on the map page")
        return record

    first = documents[0]["id"]
    first_html = throttle.text(f"{OIM_BASE}/document/{first}")
    counts = volume_region_counts(first_html)
    split = [d for d in documents if counts.get(d["id"], 0) >= 2]
    record["split_documents"] = len(split)
    written = []
    for document in split:
        doc_id = document["id"]
        html = (
            first_html
            if doc_id == first
            else throttle.text(f"{OIM_BASE}/document/{doc_id}")
        )
        regions, cutlines, canvas = parse_document_regions(html, doc_id)
        canvas = canvas or document.get("image_size")
        key = page_key_for(document)
        if canvas is None:
            record["problems"].append(f"{key}: no canvas size")
            continue
        if not cutlines:
            record["problems"].append(
                f"{key}: split into {len(regions)} with no cutlines"
            )
        if write_page_files(volume, key, regions, cutlines, canvas):
            written.append(key)
        else:
            record["problems"].append(f"{key}: fewer than 2 region boundaries")
    record["pages_written"] = written
    record["cutlines"] = sum(
        len(json.loads(p.read_text())["cutlines"])
        for p in (volume / "oim").glob("*.cutlines.json")
    )
    record["fetched"] = datetime.now(UTC).isoformat(timespec="seconds")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OIM cutlines and region boundaries for every split page."
    )
    parser.add_argument("tsv", type=Path, help="The OIM volume listing")
    parser.add_argument("out", type=Path, help="One subdirectory per volume")
    parser.add_argument(
        "--have",
        type=Path,
        action="append",
        default=[],
        help="A directory of volumes already fetched (fetch_oim_truth's); skipped",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5, help="Seconds between requests"
    )
    parser.add_argument("--limit", type=int, help="Only the first N volumes")
    parser.add_argument("--only", nargs="*", help="Only these identifiers")
    args = parser.parse_args()

    with args.tsv.open() as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if has_split_pages(r)]
    if args.only:
        rows = [r for r in rows if r["identifier"] in set(args.only)]
    have = {p.name for d in args.have for p in d.iterdir() if (p / "oim").is_dir()}
    todo = [
        r
        for r in rows
        if r["identifier"] not in have
        and not (args.out / r["identifier"] / "volume.json").exists()
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"{len(rows)} volumes with split pages; {len(rows) - len(todo)} already "
        f"fetched; {len(todo)} to fetch",
        file=sys.stderr,
    )
    throttle = Throttle(args.delay)
    for index, row in enumerate(todo, start=1):
        try:
            record = fetch_volume(row, args.out, throttle)
        except OSError as error:
            print(
                f"[{index}/{len(todo)}] {row['identifier']}: FAILED {error}",
                file=sys.stderr,
            )
            continue
        (args.out / row["identifier"] / "volume.json").write_text(
            json.dumps(record, indent=1)
        )
        print(
            f"[{index}/{len(todo)}] {row['identifier']} {row['title']}: "
            f"{len(record.get('pages_written', []))} split pages, "
            f"{record.get('cutlines', 0)} cutlines, {throttle.requests} requests so far"
            + (f"; problems: {len(record['problems'])}" if record["problems"] else ""),
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
