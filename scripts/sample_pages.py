#!/usr/bin/env python
"""Draw a random sample of corpus pages into a labelable "pseudo-volume".

Truth labelled on three volumes (Los Angeles, Hudson, Nashville) says little
about the corpus as a whole, which is mostly small towns. This draws N sheets
uniformly at random from every sheet of the items a corpus run finished, keeps
the ones that run left whole (not split) and that are not key maps, and copies
each one's mirror image into ``OUT/p1.jpg``, ``OUT/p2.jpg``, …. Pages from
different volumes would collide on their own names (every volume has a p1), so
they are renumbered; ``OUT/sources.json`` ties each back to its source:

    {"description": ..., "seed": 0, "run_tag": "corpus-v1", ...,
     "pages": {"p1": {"item": "sanborn03297_002", "page": "p3",
                      "city": "covington", "state": "louisiana", "year": "1909",
                      "loc_url": "https://www.loc.gov/item/sanborn03297_002/",
                      "loc_iiif": "https://tile.loc.gov/image-services/iiif/service:…",
                      "mirror": "s3://mapsnap-sanborn/by-state/…/p3.jpg",
                      "width": 1613, "height": 1913}}}

Under ``data/`` the directory is an ordinary volume to the labelers (the
adjacency labeler lists it and shows each page's source), and labels made on it
are keyed by the synthetic names. Run from the project root:

    uv run python scripts/sample_pages.py ~/Downloads/loc-sanborn-maps.mapping.tsv \\
        --items ~/Documents/mapsnap/corpus-run/corpus-v1-done.txt \\
        --count 100 --seed 0 --out data/samples/adjacency-corpus-100
"""

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mapsnap.aws_cli import run_aws
from mapsnap.loc_craft import Item, list_prefix
from mapsnap.make_iiif_georef import loc_service_id

BUCKET = "s3://mapsnap-sanborn"


@dataclass(frozen=True)
class Sheet:
    """One sheet of one LoC item, as the mirror's mapping lists it."""

    item: str
    state: str
    year: str
    city: str
    page: str  # the page key, e.g. "p3", "p33B"

    @property
    def location(self) -> Item:
        return Item(item=self.item, state=self.state, year=self.year)


def read_sheets(mapping: Path, items: set[str] | None = None) -> list[Sheet]:
    """Every sheet in the mirror mapping (one row per sheet), optionally only of some items."""
    with mapping.open() as handle:
        return [
            Sheet(row["item"], row["state"], row["year"], row["city"], row["page_key"])
            for row in csv.DictReader(handle, delimiter="\t")
            if items is None or row["item"] in items
        ]


def draw(sheets: list[Sheet], seed: int) -> list[Sheet]:
    """The sheets in a seeded random order: each sheet equally likely, so big volumes weigh more."""
    order = list(sheets)
    random.Random(seed).shuffle(order)
    return order


def rejection(
    sheet: Sheet, keys: list[str], run_tag: str, keymaps: list[str]
) -> str | None:
    """Why a drawn sheet can't join the sample, or None if it can.

    ``keys`` is the item's key listing in the bucket (relative to the item).
    """
    if f"runs/{run_tag}/mapsnap.iiif.json" not in keys:
        return f"{run_tag} did not finish the item"
    if f"{sheet.page}.jpg" not in keys:
        return "no mirror image"
    if f"runs/{run_tag}/{sheet.page}.panels.json" in keys:
        return f"split by {run_tag}"
    if sheet.page in keymaps:
        return "key map"
    return None


def source_record(sheet: Sheet, metadata: dict, bucket: str) -> dict:
    """Where a sampled page came from: item, page, place, LoC links, mirror image, size."""
    entry = next(
        (s for s in metadata.get("sheets") or [] if s.get("key") == sheet.page), {}
    )
    storage_dir = entry.get("storage_dir") or metadata.get("storage_dir")
    return {
        "item": sheet.item,
        "page": sheet.page,
        "city": sheet.city,
        "state": sheet.state,
        "year": sheet.year,
        "loc_url": metadata.get("loc_url") or f"https://www.loc.gov/item/{sheet.item}/",
        "loc_iiif": loc_service_id(storage_dir, entry["stem"])
        if storage_dir and entry.get("stem")
        else None,
        "mirror": f"{bucket}/{sheet.location.prefix}/{sheet.page}.jpg",
        "width": entry.get("width"),
        "height": entry.get("height"),
    }


def fetch_json(url: str) -> dict:
    """An S3 object parsed as JSON, or {} when it is absent or unreadable."""
    try:
        return json.loads(run_aws(["aws", "s3", "cp", url, "-"], capture=True).stdout)
    except (OSError, ValueError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("mapping", type=Path, help="loc-sanborn-maps.mapping.tsv")
    parser.add_argument(
        "--items", type=Path, default=None, help="Only these items (one id per line)."
    )
    parser.add_argument("--run-tag", default="corpus-v1")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = (
        {line.strip() for line in args.items.read_text().splitlines() if line.strip()}
        if args.items
        else None
    )
    sheets = read_sheets(args.mapping, items)
    print(f"{len(sheets)} sheets in the pool", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    listings: dict[str, list[str]] = {}
    rejected: dict[str, int] = {}
    pages: dict[str, dict] = {}
    for sheet in draw(sheets, args.seed):
        if len(pages) == args.count:
            break
        prefix = f"{args.bucket}/{sheet.location.prefix}"
        if sheet.item not in listings:
            listings[sheet.item] = list_prefix(args.bucket, sheet.location.prefix)
        keymaps = (
            fetch_json(f"{prefix}/keymaps.json").get("keys") or []
            if "keymaps.json" in listings[sheet.item]
            else []
        )
        reason = rejection(sheet, listings[sheet.item], args.run_tag, keymaps)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        name = f"p{len(pages) + 1}"
        run_aws(
            [
                "aws",
                "s3",
                "cp",
                f"{prefix}/{sheet.page}.jpg",
                str(args.out / f"{name}.jpg"),
                "--only-show-errors",
            ]
        )
        pages[name] = source_record(
            sheet, fetch_json(f"{prefix}/metadata.json"), args.bucket
        )
        print(
            f"{name}: {sheet.city}, {sheet.state} {sheet.year} {sheet.item} {sheet.page}",
            file=sys.stderr,
        )
    (args.out / "sources.json").write_text(
        json.dumps(
            {
                "description": f"{len(pages)} sheets drawn uniformly from the items "
                f"{args.run_tag} finished, excluding split sheets and key maps",
                "created": datetime.now(UTC).strftime("%Y-%m-%d"),
                "seed": args.seed,
                "run_tag": args.run_tag,
                "pool_sheets": len(sheets),
                "rejected": rejected,
                "pages": pages,
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"wrote {len(pages)} pages to {args.out}; rejected {rejected}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
