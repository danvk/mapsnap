#!/usr/bin/env python3
"""Convert data/oim/all.json to CSV with one row per volume."""

import csv
import json
import sys
from pathlib import Path

INPUT = Path("data/oim/all.json")
COLUMNS = [
    "identifier",
    "year",
    "title",
    "load_date",
    "document_ct",
    "multimask_ct",
    "completion_pct",
    "region_ct",
    "main_layer_ct",
    "featured",
]


def main() -> None:
    data = json.loads(INPUT.read_text())
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    writer.writeheader()
    for item in data["items"]:
        writer.writerow({col: item.get(col, "") for col in COLUMNS})


if __name__ == "__main__":
    main()
