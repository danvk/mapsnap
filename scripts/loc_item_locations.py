#!/usr/bin/env python
"""Write the catalogue's coordinate for every Sanborn item, for `loc-fit --locations`.

Reads the Library of Congress Sanborn data package (one JSON record per line,
``Id`` naming the item and ``Location[0].Coordinates`` its ``[lat, lon]``) and
writes ``item, lat, lon`` for every record that has one: 45,790 of 50,600.

    uv run python scripts/loc_item_locations.py ~/Documents/mapsnap/metadata.jsonl \\
        ~/Documents/mapsnap/loc-counties/item-locations.tsv
"""

import argparse
import csv
import json
import re
from pathlib import Path


def item_location(record: dict) -> tuple[str, float, float] | None:
    """(item, lat, lon) for a data-package record, or None when it has no coordinate."""
    match = re.search(r"(sanborn\d+_\d+)", str(record.get("Id", "")))
    location = next(iter(record.get("Location") or []), None) or {}
    coordinates = location.get("Coordinates")
    if not match or not coordinates or len(coordinates) < 2:
        return None
    return match.group(1), float(coordinates[0]), float(coordinates[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write item, lat, lon from the LoC Sanborn data package."
    )
    parser.add_argument("metadata", type=Path, help="The data package's metadata.jsonl")
    parser.add_argument("out", type=Path, help="Where to write the TSV")
    args = parser.parse_args()

    rows = []
    with args.metadata.open() as handle:
        for line in handle:
            if line.strip() and (row := item_location(json.loads(line))):
                rows.append(row)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["item", "lat", "lon"])
        writer.writerows(sorted(rows))
    print(f"{len(rows):,} items with a coordinate -> {args.out}")


if __name__ == "__main__":
    main()
