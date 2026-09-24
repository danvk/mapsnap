#!/usr/bin/env python
"""Build the atlas app's place index from the LoC catalogue and the mirror mapping.

    scripts/atlas/build_places.py --run-tag corpus-v1

The atlas opens on a map of every town the Sanborn collection covers, so the
first thing it loads must be small. That file is `places.json`: one row per
town, carrying only what the map and the autocomplete need. Everything about
individual volumes -- which years exist, how many sheets, where the annotation
lives in the bucket -- is split per state and fetched when a town is clicked.

Two inputs, and the join between them is the point:

* ``metadata.jsonl`` is the LoC catalogue: title, date, and the geocoded
  coordinates that put a town on the map. It covers all 50,600 volumes,
  digitized or not.
* the mirror's ``*.mapping.tsv`` says which volumes were actually mirrored and,
  for each, the ``by-state/<state>/<year>/<item>`` prefix its run published
  under.

The year comes from the TSV, never from the catalogue's ``Date``: they disagree
for 310 items (sanborn00518_001 is catalogued 1890 and mirrored under 1899),
and a derived year sends the app to a key that does not exist.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_METADATA = Path.home() / "Documents/mapsnap/metadata.jsonl"
DEFAULT_MAPPING = Path.home() / "Downloads/loc-sanborn-maps.mapping.tsv"


@dataclass
class LocSheets:
    """Where a mirrored volume's sheets sit at loc.gov, for linking to one of them.

    ``prefix + sheets[i]`` is the LoC stem of the sheet at ``?sp=i+1`` of
    ``https://www.loc.gov/resource/<resource>/``. The prefix is shared by every
    stem of the volume and written once. Even so the per-state files grow by
    about two thirds (New York 577 -> 949 KB), which is 50 -> 93 KB gzipped:
    small for a file fetched once when a town in that state is picked.
    """

    resource: str
    prefix: str
    sheets: list[str]


@dataclass
class Volume:
    """One catalogued volume, with the mirror prefix when it has one."""

    item: str
    date: str
    year: int | None
    sheets: int
    title: str
    mirror_state: str | None = None
    mirror_year: str | None = None
    loc: LocSheets | None = None


@dataclass
class Place:
    """A town: everything catalogued for it, and where it sits."""

    name: str
    state: str
    lon: float
    lat: float
    volumes: list[Volume] = field(default_factory=list)


def slugify(text: str) -> str:
    """Lowercase, hyphen-separated, as the mirror names its state directories."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def item_id(url: str) -> str | None:
    """`sanborn01790_085` out of `http://www.loc.gov/item/sanborn01790_085/`."""
    match = re.search(r"/item/([^/]+)/?$", url.rstrip("/") + "/")
    return match.group(1) if match else None


def sheet_count(record: dict) -> int:
    """Sheets in a volume: the file count, or the number the notes state.

    ``Number_of_files`` is absent for everything not digitized -- 15,479 of
    50,600 -- and those volumes still deserve their size on the map, so the
    "N sheet(s)." note carries them.
    """
    files = record.get("Number_of_files")
    if isinstance(files, int) and files > 0:
        return files
    notes = " ".join(record.get("Notes") or [])
    match = re.search(r"(\d+)\s+sheet", notes)
    return int(match.group(1)) if match else 0


def read_mirror(path: Path) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    """Each mirrored item's (state, year) prefix and its mirrored sheet count."""
    prefixes: dict[str, tuple[str, str]] = {}
    sheets: dict[str, int] = defaultdict(int)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item = row["item"]
            prefixes.setdefault(item, (row["state"], row["year"]))
            sheets[item] += 1
    return prefixes, dict(sheets)


def loc_resource(storage_dir: str) -> str | None:
    """The loc.gov resource id of a mirror storage directory, if it names one.

    ``gmd/gmd409m/g4094m/g4094sm/g4094sm_g025021917`` is served at
    ``https://www.loc.gov/resource/g4094sm.g4094sm_g025021917/``, and
    ``.../g4124pm/g096701899`` at ``.../resource/g4124pm.g096701899/``. None
    for a path too short to name one: the mapping's lone "ghost" row has none.
    """
    parts = [part for part in storage_dir.split("/") if part]
    return f"{parts[-2]}.{parts[-1]}" if len(parts) >= 2 else None


def compact_stems(stems: list[str]) -> tuple[str, list[str]]:
    """Factor out the stems' common prefix: (prefix, what each stem adds to it)."""
    prefix = os.path.commonprefix(stems)
    return prefix, [stem[len(prefix) :] for stem in stems]


def read_loc_sheets(path: Path) -> dict[str, LocSheets]:
    """Each mirrored item's loc.gov resource and its sheets in ``?sp=`` order.

    The mapping's ``seq`` is a sheet's position in its LoC resource: ``?sp=33``
    of sanborn02502_005 shows 02502_1917-0028, its p28, and 12 random sheets
    from 12 other items agreed. Two items are left out because their sheets
    span two storage directories, and so two resources; so is any item whose
    seq does not run 1..N, which no item does today, or whose storage path names
    no resource.
    """
    rows: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["item"]].append((int(row["seq"]), row["stem"], row["storage_dir"]))
    sheets_by_item: dict[str, LocSheets] = {}
    for item, sheets in rows.items():
        sheets.sort()
        storage_dirs = {storage_dir for _, _, storage_dir in sheets}
        if len(storage_dirs) != 1:
            continue
        if [seq for seq, _, _ in sheets] != list(range(1, len(sheets) + 1)):
            continue
        resource = loc_resource(storage_dirs.pop())
        if resource is None:
            continue
        prefix, rest = compact_stems([stem for _, stem, _ in sheets])
        sheets_by_item[item] = LocSheets(resource, prefix, rest)
    return sheets_by_item


def year_of(date: str) -> int | None:
    """The four-digit year a catalogue date starts with, when it has one."""
    match = re.match(r"(\d{4})", date or "")
    return int(match.group(1)) if match else None


def collect_places(
    metadata: Path, prefixes: dict[str, tuple[str, str]], mirror_sheets: dict[str, int]
) -> tuple[dict[tuple[str, str], Place], dict[str, int]]:
    """Group every catalogued volume under its town; count what had to be dropped."""
    places: dict[tuple[str, str], Place] = {}
    skipped = {"no_coordinates": 0, "no_name": 0, "no_item": 0, "year_differs": 0}
    with open(metadata) as handle:
        for line in handle:
            read_record(json.loads(line), places, skipped, prefixes, mirror_sheets)
    return places, skipped


def read_record(
    record: dict,
    places: dict[tuple[str, str], Place],
    skipped: dict[str, int],
    prefixes: dict[str, tuple[str, str]],
    mirror_sheets: dict[str, int],
) -> None:
    """Fold one catalogue record into its town, or count why it cannot be placed."""
    item = item_id(record.get("Id") or "")
    if not item:
        skipped["no_item"] += 1
        return
    location = next(iter(record.get("Location") or []), None)
    coordinates = (location or {}).get("Coordinates")
    if not coordinates or len(coordinates) != 2:
        skipped["no_coordinates"] += 1
        return
    latitude, longitude = float(coordinates[0]), float(coordinates[1])
    state = next(iter(record.get("State_text") or []), None)
    # The town's own name, not the geocoder's full string: "Abbeville", not
    # "Abbeville, Henry County, Alabama, 36310, United States".
    name = next(iter(record.get("City_text") or []), None) or (location or {}).get(
        "Short_name"
    )
    if not state or not name:
        skipped["no_name"] += 1
        return

    key = (slugify(state), slugify(name))
    place = places.get(key)
    if place is None:
        place = Place(name=name, state=state, lon=longitude, lat=latitude)
        places[key] = place
    prefix = prefixes.get(item)
    date = record.get("Date") or ""
    # The mirror's year wins where the two disagree. The mirror takes its year
    # from LoC's own storage path, and the catalogue's Date can be plain wrong:
    # sanborn01790_085 is catalogued 1906 and filed under g01790195001N, and
    # its sheets are 1950 maps. Grouping by the catalogue year would file a
    # city's newest survey half a century early.
    mirror_year = int(prefix[1]) if prefix and prefix[1].isdigit() else None
    if mirror_year is not None and mirror_year != year_of(date):
        skipped["year_differs"] += 1
    place.volumes.append(
        Volume(
            item=item,
            date=date,
            year=mirror_year if mirror_year is not None else year_of(date),
            # A mirrored item's sheet count is the mirror's own, which is what
            # the annotation will actually hold.
            sheets=mirror_sheets.get(item) or sheet_count(record),
            title=record.get("Title") or "",
            mirror_state=prefix[0] if prefix else None,
            mirror_year=prefix[1] if prefix else None,
        )
    )


def write_outputs(
    places: dict[tuple[str, str], Place], out_dir: Path, run_tag: str, bucket: str
) -> tuple[int, int]:
    """Write places.json and one volumes file per state; return their counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "volumes").mkdir(exist_ok=True)

    by_state: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    rows = []
    for (state_slug, name_slug), place in sorted(places.items()):
        place_id = f"{state_slug}/{name_slug}"
        years = sorted({v.year for v in place.volumes if v.year is not None})
        mirrored = sum(1 for v in place.volumes if v.mirror_state)
        rows.append(
            {
                "id": place_id,
                "name": place.name,
                "state": place.state,
                "lon": round(place.lon, 5),
                "lat": round(place.lat, 5),
                "volumes": len(place.volumes),
                "sheets": sum(v.sheets for v in place.volumes),
                "mirrored": mirrored,
                "firstYear": years[0] if years else None,
                "lastYear": years[-1] if years else None,
            }
        )
        # Newest first: opening a town on its most recent year is the default,
        # so the app should not have to sort before it can render.
        by_state[state_slug][place_id] = [
            {
                "item": volume.item,
                "date": volume.date,
                "year": volume.year,
                "sheets": volume.sheets,
                "title": volume.title,
                **(
                    {"state": volume.mirror_state, "mirrorYear": volume.mirror_year}
                    if volume.mirror_state
                    else {}
                ),
                **(
                    {
                        "loc": {
                            "resource": volume.loc.resource,
                            "prefix": volume.loc.prefix,
                            "sheets": volume.loc.sheets,
                        }
                    }
                    if volume.loc
                    else {}
                ),
            }
            for volume in sorted(
                place.volumes,
                key=lambda v: (v.year is None, -(v.year or 0), v.item),
            )
        ]

    index = {
        "runTag": run_tag,
        "bucket": bucket,
        "places": rows,
    }
    (out_dir / "places.json").write_text(json.dumps(index, separators=(",", ":")))
    for state_slug, place_volumes in by_state.items():
        (out_dir / "volumes" / f"{state_slug}.json").write_text(
            json.dumps(place_volumes, separators=(",", ":"))
        )
    return len(rows), len(by_state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "app/public/atlas",
    )
    parser.add_argument(
        "--run-tag",
        default="corpus-v1",
        help="Run whose annotations the app reads (default: %(default)s)",
    )
    parser.add_argument("--bucket", default="mapsnap-sanborn")
    args = parser.parse_args()

    prefixes, mirror_sheets = read_mirror(args.mapping)
    print(f"{len(prefixes):,} mirrored items", file=sys.stderr)
    places, skipped = collect_places(args.metadata, prefixes, mirror_sheets)
    loc_sheets = read_loc_sheets(args.mapping)
    for place in places.values():
        for volume in place.volumes:
            volume.loc = loc_sheets.get(volume.item)
    n_places, n_states = write_outputs(places, args.out_dir, args.run_tag, args.bucket)

    volumes = sum(len(p.volumes) for p in places.values())
    mirrored = sum(1 for p in places.values() for v in p.volumes if v.mirror_state)
    print(
        f"{n_places:,} places, {volumes:,} volumes ({mirrored:,} mirrored) "
        f"across {n_states} states",
        file=sys.stderr,
    )
    print(f"dropped: {skipped}", file=sys.stderr)
    index_kb = (args.out_dir / "places.json").stat().st_size / 1024
    print(f"places.json: {index_kb:,.0f} KB", file=sys.stderr)


if __name__ == "__main__":
    main()
