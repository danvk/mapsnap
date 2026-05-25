#!/usr/bin/env python3
"""Convert the Sanborn LOC JSON file to CSV, optionally filtering by image count."""

import argparse
import csv
import json
import sys
from pathlib import Path


def first(value: list | str | None) -> str:
    """Return the first element if a list, the value itself if a string, or empty string."""
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def image_count(item: dict) -> int:
    return sum(res.get("files", 0) for res in item.get("resources", []))


def to_row(item: dict) -> dict[str, str]:
    return {
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "date": item.get("date", ""),
        "description": first(item.get("description")),
        "num_images": str(image_count(item)),
        "location_city": first(item.get("location_city")),
        "location_county": first(item.get("location_county")),
        "location_state": first(item.get("location_state")),
        "location_country": first(item.get("location_country")),
    }


COLUMNS = [
    "id",
    "title",
    "date",
    "description",
    "num_images",
    "location_city",
    "location_county",
    "location_state",
    "location_country",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="data/loc/sanborn.json",
        help="Path to merged Sanborn JSON file (default: data/loc/sanborn.json)",
    )
    parser.add_argument(
        "-n",
        "--min-images",
        type=int,
        default=10,
        metavar="N",
        help="Only include items with >= N images (default: 10)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output CSV path (default: stdout)",
    )
    args = parser.parse_args()

    items: list[dict] = json.loads(Path(args.input).read_text())

    filtered = [item for item in items if image_count(item) >= args.min_images]
    filtered.sort(key=lambda r: (-image_count(r), r.get("date", "")))

    out = open(args.output, "w", newline="") if args.output != "-" else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        for item in filtered:
            writer.writerow(to_row(item))
    finally:
        if args.output != "-":
            out.close()

    print(
        f"Wrote {len(filtered):,} items (of {len(items):,} total) to "
        f"{'stdout' if args.output == '-' else args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
