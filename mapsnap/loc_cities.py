"""Map the independent-city Sanborn items to OSM boundary relations (#407).

``mapsnap loc-counties`` keys every item to a Natural Earth county, which is the
unit ``mapsnap osm-counties`` cuts street extracts for. 428 items covering
13,665 sheets fall through that mapping because their LoC county reads
"independent cities": Virginia's cities belong to no county at all, and neither
do Baltimore or St. Louis, so Natural Earth has nothing to key them on.

OSM does carry them, as ``admin_level=6`` relations tagged ``border_type=city``
-- the level normally used for counties, because that is the tier they sit in.
Most also carry ``nist:fips_code``, which is the same identifier
``loc-counties`` writes, so a matched city slots into the existing pipeline
without a second naming scheme.

Five of the 38 cities have no such relation, all for historical reasons rather
than tagging gaps, and are redirected to the boundary that covers their streets
today (see ``SUCCESSORS``). A redirect is recorded as ``successor`` in the
``match`` column so it can be told apart from a name match.

    mapsnap loc-cities --skipped skipped.tsv --items items.tsv \\
        --pbf us-counties.osm.pbf --out-tsv cities-items.tsv \\
        --out-pbf independent-cities.osm.pbf
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import osmium
from osmium.filter import EntityFilter
from osmium.osm import RELATION

# Cities whose Sanborn volumes predate a boundary change, mapped to the FIPS of
# whatever covers that ground now. Without these, 22 items and 134 sheets have
# nowhere to go; with them the mapping is complete.
SUCCESSORS: dict[tuple[str, str], str] = {
    ("virginia", "bedford"): "51019",  # a town again since 2013, in Bedford County
    ("virginia", "clifton forge"): "51005",  # a town again since 2001, Alleghany County
    ("virginia", "colonial beach"): "51193",  # a Westmoreland County town, never a city
    ("virginia", "fortress monroe"): "51650",  # Fort Monroe, absorbed by Hampton
    ("virginia", "warwick"): "51700",  # merged into Newport News in 1958
}

TSV_COLUMNS = [
    "item",
    "state",
    "county",
    "city",
    "sheets",
    "match",
    "fips",
    "osm_relation",
    "osm_name",
]


@dataclass(frozen=True)
class Item:
    """One Sanborn volume whose county is an independent city."""

    item: str
    state: str
    county: str
    city: str
    sheets: int


@dataclass(frozen=True)
class Boundary:
    """An OSM administrative relation that a Sanborn city can be cut from."""

    relation_id: int
    name: str
    fips: str
    border_type: str

    @property
    def county_fips(self) -> str:
        """The identifier `loc-counties` uses, so both mappings share a key."""
        return f"US{self.fips}"


def normalize_place(name: str) -> str:
    """Fold a place name to its matchable form: lowercase, unpunctuated, St->Saint."""
    folded = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    if folded.startswith("st "):
        folded = "saint " + folded[3:]
    return " ".join(folded.split())


def read_independent_city_items(path: Path) -> list[Item]:
    """Read the rows of `loc-counties`' skipped.tsv whose county is an independent city.

    The LoC spells it "independent cities", "independent city", "independent
    cities (prior to 2001)" and "hampton (independent city)", so the test is a
    substring rather than an equality.
    """
    items: list[Item] = []
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if "independent cit" not in row["county"].lower():
                continue
            items.append(
                Item(
                    item=row["item"],
                    state=row["state"].lower(),
                    county=row["county"],
                    city=row["city"],
                    sheets=int(row["sheets"] or 0),
                )
            )
    return items


def state_by_fips_prefix(path: Path) -> dict[str, str]:
    """Learn which state each two-digit FIPS prefix names, from the matched items.

    Deriving it from `loc-counties`' own output rather than hardcoding a table
    keeps the two files spelling states the same way.
    """
    prefixes: dict[str, str] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            fips = row["fips"]
            if fips.startswith("US") and len(fips) == 7:
                prefixes.setdefault(fips[2:4], row["state"].lower())
    return prefixes


def boundary_from_tags(relation_id: int, tags: dict[str, str]) -> Boundary | None:
    """The county-tier boundary a relation describes, or None if it is not one.

    A FIPS code is required: it is the key the rest of the pipeline joins on,
    and a boundary without one cannot be matched to `loc-counties`' output.
    """
    if tags.get("boundary") != "administrative" or tags.get("admin_level") != "6":
        return None
    fips = tags.get("nist:fips_code", "")
    if not fips:
        return None
    return Boundary(
        relation_id=relation_id,
        name=tags.get("name", ""),
        fips=fips,
        border_type=tags.get("border_type", ""),
    )


def read_boundaries(pbf: Path) -> list[Boundary]:
    """Every county-tier boundary relation in the dump, cities included."""
    boundaries: list[Boundary] = []
    for relation in osmium.FileProcessor(str(pbf)).with_filter(EntityFilter(RELATION)):
        boundary = boundary_from_tags(relation.id, dict(relation.tags))
        if boundary is not None:
            boundaries.append(boundary)
    return boundaries


def city_index(
    boundaries: list[Boundary], states: dict[str, str]
) -> dict[tuple[str, str], Boundary]:
    """Index the city boundaries by (state, normalized name), the way items name them.

    ``border_type`` is matched as a substring because a few places are tagged
    ``county;city`` or ``borough;city``.
    """
    index: dict[tuple[str, str], Boundary] = {}
    for boundary in boundaries:
        if "city" not in boundary.border_type:
            continue
        state = states.get(boundary.fips[:2], "")
        if state:
            index[(state, normalize_place(boundary.name))] = boundary
    return index


@dataclass(frozen=True)
class Match:
    """One item resolved to a boundary, and how it got there."""

    item: Item
    boundary: Boundary
    kind: str  # "exact" or "successor"


def match_items(
    items: list[Item],
    index: dict[tuple[str, str], Boundary],
    by_fips: dict[str, Boundary],
) -> tuple[list[Match], list[Item]]:
    """Resolve each item to a city relation by name, else to its successor boundary."""
    matched: list[Match] = []
    unmatched: list[Item] = []
    for item in items:
        key = (item.state, normalize_place(item.city))
        boundary = index.get(key)
        if boundary is not None:
            matched.append(Match(item, boundary, "exact"))
            continue
        successor = SUCCESSORS.get(key)
        boundary = by_fips.get(successor) if successor else None
        if boundary is not None:
            matched.append(Match(item, boundary, "successor"))
        else:
            unmatched.append(item)
    return matched, unmatched


def write_items_tsv(path: Path, matches: list[Match]) -> None:
    """Write the per-item mapping, shaped like `loc-counties`' items.tsv."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for match in sorted(matches, key=lambda m: m.item.item):
            writer.writerow(
                [
                    match.item.item,
                    match.item.state,
                    match.item.county,
                    match.item.city,
                    match.item.sheets,
                    match.kind,
                    match.boundary.county_fips,
                    match.boundary.relation_id,
                    match.boundary.name,
                ]
            )


def write_boundary_extract(pbf: Path, relation_ids: list[int], out_path: Path) -> None:
    """Cut the matched relations, with their member ways and nodes, into one file.

    The ids go through a file rather than the command line: the list is short
    today but `osmium getid` reads either, and a file cannot overflow.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        for relation_id in sorted(relation_ids):
            handle.write(f"r{relation_id}\n")
        id_file = handle.name
    try:
        subprocess.run(
            [
                "osmium",
                "getid",
                "--add-referenced",
                "--id-file",
                id_file,
                "--overwrite",
                "-o",
                str(out_path),
                str(pbf),
            ],
            check=True,
        )
    finally:
        Path(id_file).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skipped", type=Path, required=True, help="loc-counties' skipped.tsv"
    )
    parser.add_argument(
        "--items", type=Path, required=True, help="loc-counties' items.tsv"
    )
    parser.add_argument(
        "--pbf", type=Path, required=True, help="OSM dump of US county boundaries"
    )
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument(
        "--out-pbf", type=Path, help="write the matched boundaries to this extract"
    )
    args = parser.parse_args(argv)

    items = read_independent_city_items(args.skipped)
    states = state_by_fips_prefix(args.items)
    boundaries = read_boundaries(args.pbf)
    by_fips = {boundary.fips: boundary for boundary in boundaries}
    index = city_index(boundaries, states)
    print(
        f"{len(items)} items, {len(boundaries)} county-tier relations, "
        f"{len(index)} of them cities"
    )

    matched, unmatched = match_items(items, index, by_fips)
    exact = sum(1 for match in matched if match.kind == "exact")
    sheets = sum(match.item.sheets for match in matched)
    print(
        f"matched {len(matched)} of {len(items)} items "
        f"({exact} by name, {len(matched) - exact} by successor), {sheets:,} sheets"
    )
    for item in unmatched:
        print(f"  unmatched: {item.item} {item.state}/{item.city}", file=sys.stderr)

    write_items_tsv(args.out_tsv, matched)
    print(f"wrote {args.out_tsv}")

    if args.out_pbf:
        relation_ids = sorted({match.boundary.relation_id for match in matched})
        write_boundary_extract(args.pbf, relation_ids, args.out_pbf)
        print(f"wrote {args.out_pbf} with {len(relation_ids)} relations")
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
