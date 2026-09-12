"""Map every Library of Congress Sanborn item to its Natural Earth county.

The county is the unit the corpus run extracts OpenStreetMap streets for
(#354, stage 2): every LoC item names exactly one county, and a county's
named roads are the vocabulary and centerline set its volumes fit against.
This joins the catalog's free-text ``location_county`` to Natural Earth's
admin-2 counties, which carry a FIPS code and a polygon.

Inputs:

- the loc.gov catalog records, one ``<item>.json`` (the ``?fo=json``
  document) per item under ``<state>/<year>/``, as the mirror keeps them;
- ``ne_10m_admin_2_counties.geojson`` from Natural Earth (US only: counties,
  parishes, boroughs, census areas, independent cities, DC, Puerto Rico);
- optionally the mirror's mapping TSV, to weight the report by sheet count.

Outputs, in ``--out-dir``:

- ``items.tsv``: one row per item that maps to at least one county --
  ``item, state, county, city, sheets, match, fips, ne_name`` (``fips`` and
  ``ne_name`` are ``;``-joined when a catalog string names several counties);
- ``counties.tsv``: one row per Natural Earth county with items -- the
  extraction list -- ``fips, state, ne_name, ne_type, items, sheets,
  loc_names``;
- ``skipped.tsv``: every item that got no county, with the reason.

How a catalog string is resolved, in order:

1. the whole normalized name against the state's Natural Earth names
   (``lewis and clark county`` must not be split);
2. a fixed alias table, for the catalog's known typos and renames
   (``dade`` -> Miami-Dade, ``latab`` -> Latah);
3. the string split on ``and``, ``,``, ``/`` and ``&``, each part resolved
   on its own (``douglas and sarpy county`` names two counties);
4. a same-state fuzzy match above FUZZY_CUTOFF, which the report lists so
   every one can be audited.

Skipped on purpose: items outside the United States (Mexico, Canada, Cuba),
items whose county is Virginia's or Maryland's ``independent cities``
pseudo-county (a later pass will join those on the city name), and the few
catalog strings that are not a county at all (``new jersey coast``).

    mapsnap loc-counties /Volumes/fivetera/loc-sanborn-maps/metadata-http/metadata \\
        ~/Downloads/ne_10m_admin_2_counties.geojson --out-dir counties \\
        --mapping ~/Downloads/loc-sanborn-maps.mapping.tsv
"""

import argparse
import csv
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from mapsnap.loc_mirror import keep_sheet

# Below this SequenceMatcher ratio a same-state near-miss is left unmatched.
# The catalog's genuine typos (tebama/Tehama, kiokuk/Keokuk, nemaba/Nemaha)
# all sit at 0.83 or above; anything lower goes in the alias table instead.
FUZZY_CUTOFF = 0.8

STATE_POSTAL = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "puerto rico": "PR", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virgin islands": "VI", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}  # fmt: skip

# Catalog spellings that no normalization reaches, keyed by (postal, normalized
# LoC name) -> normalized Natural Earth name. Renames first, then the typos the
# fuzzy pass cannot see (a missing space, a swapped consonant cluster).
ALIASES = {
    ("NY", "manhattan"): "new york",
    ("FL", "dade"): "miami-dade",
    ("HI", "oahu"): "honolulu",
    ("DC", "district of columbia"): "washington dc",
    ("DC", "washington"): "washington dc",
    ("ID", "latab"): "latah",
    ("ID", "lembi"): "lemhi",
    ("AL", "valher"): "walker",
    ("MS", "yolabusha"): "yalobusha",
    ("OH", "luking"): "licking",
    ("KY", "gallatm"): "gallatin",
    ("KY", "logon"): "logan",
    ("KS", "neosbe"): "neosho",
    ("CA", "stalslious"): "stanislaus",
    ("NE", "thayercounty"): "thayer",
}

# Trailing words the catalog appends to a county name; Natural Earth carries the
# bare name and the type in a separate field. A state code in parentheses may
# follow the suffix (``washington county (va)``) and is kept for resolve().
SUFFIX = re.compile(
    r"\s+(county|counties|parish|parishes|borough|boroughs|census area|"
    r"census division|municipality|municipio|city and borough|district)"
    r"(?=(\s*\([a-z]{2}\))?$)"
)
SPLIT = re.compile(r"\s*(?:,|/|&|\band\b)\s*")
PAREN_STATE = re.compile(r"\(([a-z]{2})\)")
INDEPENDENT = re.compile(r"independent cit")

# A token of a multi-county string that names a state rather than a county
# (``sussex county, del., and wicomico county, md``): postal codes, full names
# and the catalog's abbreviations. It sets the state of the county before it.
STATE_TOKENS = {
    **{postal.lower(): postal for postal in STATE_POSTAL.values()},
    **STATE_POSTAL,
    "ala": "AL", "ariz": "AZ", "ark": "AR", "calif": "CA", "cal": "CA",
    "colo": "CO", "conn": "CT", "del": "DE", "fla": "FL", "ill": "IL",
    "ind": "IN", "kans": "KS", "kan": "KS", "mass": "MA", "mich": "MI",
    "minn": "MN", "miss": "MS", "mont": "MT", "nebr": "NE", "neb": "NE",
    "nev": "NV", "okla": "OK", "ore": "OR", "oreg": "OR", "penn": "PA",
    "penna": "PA", "tenn": "TN", "tenna": "TN", "tex": "TX", "wash": "WA",
    "wis": "WI", "wyo": "WY",
}  # fmt: skip


@dataclass
class County:
    """One Natural Earth admin-2 feature."""

    fips: str
    postal: str
    name: str  # as printed
    kind: str  # TYPE: County, Parish, City, ...
    key: str  # normalized name, the join key


@dataclass
class Match:
    """How one catalog county string resolved."""

    kind: str  # exact | alias | multi | fuzzy | unmatched
    counties: list[County] = field(default_factory=list)
    note: str = ""


@dataclass
class Item:
    """The location fields of one catalog record."""

    item: str
    state: str
    county: str
    city: str
    country: str
    sheets: int = 0


def normalize(name: str) -> str:
    """The join key for a county name: lowercase, no suffix, saint spelled out.

    ``St. Louis County`` and ``saint louis county`` both become ``saint louis``;
    ``Miami-Dade`` keeps its hyphen. Parentheticals other than a state code
    (``(prior to 2001)``) are dropped.
    """
    text = name.lower().strip()
    text = re.sub(r"\((?![a-z]{2}\))[^)]*\)", " ", text)
    text = text.replace("&", " and ").replace(".", "")
    text = re.sub(r"\bste\b", "sainte", text)
    text = re.sub(r"\bst\b", "saint", text)
    text = re.sub(r"[^a-z0-9 \-()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    while True:  # ``orleans parish county`` carries two suffixes
        stripped = SUFFIX.sub("", text).strip()
        if stripped == text:
            return text
        text = stripped


def load_counties(geojson_path: Path) -> dict[str, dict[str, County]]:
    """Natural Earth counties indexed ``postal -> normalized name -> County``.

    Independent cities (Natural Earth ``TYPE`` City: Virginia's, Baltimore,
    Saint Louis, Carson City) are always indexed under ``<name> city`` and take
    the bare name only when no county of that name exists in the state, so
    ``richmond county`` finds Virginia's rural Richmond County, ``richmond
    city`` the city, and ``carson city`` (Natural Earth's ``Carson``) resolves.
    """
    index: dict[str, dict[str, County]] = defaultdict(dict)
    with open(geojson_path) as handle:
        features = json.load(handle)["features"]
    counties = [
        County(
            fips=f["properties"]["FIPS"],
            postal=f["properties"]["REGION"],
            name=f["properties"]["NAME"],
            kind=f["properties"]["TYPE"],
            key=normalize(f["properties"]["NAME"]),
        )
        for f in features
    ]
    for county in counties:
        if county.kind != "City":
            index[county.postal][county.key] = county
    for county in counties:
        if county.kind == "City":
            index[county.postal][f"{county.key} city"] = county
            index[county.postal].setdefault(county.key, county)
    return index


def load_items(metadata_root: Path) -> list[Item]:
    """Location fields of every catalog record under ``<state>/<year>/<item>.json``."""

    def first(value) -> str:
        if isinstance(value, list):
            return str(value[0]).strip() if value else ""
        return str(value).strip() if value else ""

    items: list[Item] = []
    for path in sorted(metadata_root.glob("*/*/*.json")):
        try:
            with open(path) as handle:
                record = json.load(handle).get("item") or {}
        except (OSError, ValueError) as error:
            print(f"skip {path}: {error.__class__.__name__}", file=sys.stderr)
            continue
        items.append(
            Item(
                item=path.stem,
                state=first(record.get("location_state")).lower(),
                county=first(record.get("location_county")).lower(),
                city=first(record.get("location_city")).lower(),
                country=first(record.get("location_country")).lower(),
            )
        )
    return items


def load_sheet_counts(mapping_path: Path) -> Counter:
    """Kept sheets per item from the mirror's mapping TSV."""
    counts: Counter = Counter()
    with open(mapping_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["stem"] and keep_sheet(row["page_key"]):
                counts[row["item"]] += 1
    return counts


def resolve_one(
    key: str, postal: str, index: dict[str, dict[str, County]]
) -> tuple[str, County | None]:
    """A single normalized name against one state: exact, alias, then fuzzy."""
    names = index.get(postal, {})
    if key in names:
        return "exact", names[key]
    alias = ALIASES.get((postal, key))
    if alias and alias in names:
        return "alias", names[alias]
    close = difflib.get_close_matches(key, list(names), n=1, cutoff=FUZZY_CUTOFF)
    if close:
        return "fuzzy", names[close[0]]
    return "unmatched", None


def resolve(county: str, postal: str, index: dict[str, dict[str, County]]) -> Match:
    """Resolve one catalog county string for an item in ``postal``.

    The whole string is tried first so ``lewis and clark county`` stays one
    county; only then is it split into tokens. A token that names a state,
    bare (``sussex county, del., and wicomico county, md``) or in parentheses
    (``washington county (va) and sullivan county (tn)``), sets the state of
    the county token before it; the rest are counties.
    """
    whole = normalize(county)
    kind, found = resolve_one(whole, postal, index)
    if found is not None:
        return Match(kind, [found])
    tokens = [token for token in SPLIT.split(county.lower()) if token.strip()]
    parts: list[tuple[str, str]] = []  # (normalized county name, postal)
    for token in tokens:
        bare = re.sub(r"[.\s]+", " ", token).strip().replace(" ", "")
        if bare in STATE_TOKENS and parts:
            parts[-1] = (parts[-1][0], STATE_TOKENS[bare])
            continue
        override = PAREN_STATE.search(token)
        part_postal = override.group(1).upper() if override else postal
        parts.append((normalize(PAREN_STATE.sub(" ", token)), part_postal))
    if len(parts) < 2:
        return Match("unmatched", note=whole)
    counties: list[County] = []
    kinds: list[str] = []
    for part_key, part_postal in parts:
        part_kind, part_found = resolve_one(part_key, part_postal, index)
        if part_found is None:
            return Match("unmatched", note=f"{whole} (part {part_key!r})")
        counties.append(part_found)
        kinds.append(part_kind)
    kind = "multi-fuzzy" if "fuzzy" in kinds else "multi"
    return Match(kind, counties)


def skip_reason(item: Item) -> str | None:
    """Why an item is left out before matching, or None to match it."""
    if item.country and "united states" not in item.country:
        return "foreign"
    if item.state not in STATE_POSTAL:
        return "no-state"
    if not item.county:
        return "no-county"
    if INDEPENDENT.search(item.county):
        return "independent-city"
    return None


def match_items(
    items: list[Item], index: dict[str, dict[str, County]]
) -> tuple[list[tuple[Item, Match]], list[tuple[Item, str]]]:
    """(matched item/match pairs, skipped item/reason pairs)."""
    matched: list[tuple[Item, Match]] = []
    skipped: list[tuple[Item, str]] = []
    for item in items:
        reason = skip_reason(item)
        if reason is not None:
            skipped.append((item, reason))
            continue
        match = resolve(item.county, STATE_POSTAL[item.state], index)
        if match.kind == "unmatched":
            skipped.append((item, "unmatched"))
        else:
            matched.append((item, match))
    return matched, skipped


def write_outputs(
    out_dir: Path,
    matched: list[tuple[Item, Match]],
    skipped: list[tuple[Item, str]],
) -> None:
    """items.tsv, counties.tsv and skipped.tsv under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "items.tsv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["item", "state", "county", "city", "sheets", "match", "fips", "ne_name"]
        )
        for item, match in matched:
            writer.writerow(
                [
                    item.item,
                    item.state,
                    item.county,
                    item.city,
                    item.sheets,
                    match.kind,
                    ";".join(c.fips for c in match.counties),
                    ";".join(c.name for c in match.counties),
                ]
            )
    per_county: dict[str, dict] = {}
    for item, match in matched:
        for county in match.counties:
            row = per_county.setdefault(
                county.fips,
                {"county": county, "items": 0, "sheets": 0, "loc_names": set()},
            )
            row["items"] += 1
            row["sheets"] += item.sheets
            row["loc_names"].add(item.county)
    with open(out_dir / "counties.tsv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["fips", "state", "ne_name", "ne_type", "items", "sheets", "loc_names"]
        )
        for fips, row in sorted(
            per_county.items(), key=lambda kv: (-kv[1]["sheets"], kv[0])
        ):
            county = row["county"]
            writer.writerow(
                [
                    fips,
                    county.postal,
                    county.name,
                    county.kind,
                    row["items"],
                    row["sheets"],
                    ";".join(sorted(row["loc_names"])),
                ]
            )
    with open(out_dir / "skipped.tsv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["item", "state", "county", "city", "country", "sheets", "reason"]
        )
        for item, reason in skipped:
            writer.writerow(
                [
                    item.item,
                    item.state,
                    item.county,
                    item.city,
                    item.country,
                    item.sheets,
                    reason,
                ]
            )


def report(matched: list[tuple[Item, Match]], skipped: list[tuple[Item, str]]) -> str:
    """A summary by match kind, weighted by items and sheets, plus the audit lists."""
    items_by: Counter = Counter()
    sheets_by: Counter = Counter()
    for item, match in matched:
        items_by[match.kind] += 1
        sheets_by[match.kind] += item.sheets
    for item, reason in skipped:
        items_by[f"skipped: {reason}"] += 1
        sheets_by[f"skipped: {reason}"] += item.sheets
    total_items = len(matched) + len(skipped)
    total_sheets = sum(sheets_by.values()) or 1
    lines = [f"{total_items:,} items, {total_sheets:,} sheets"]
    for kind, n in items_by.most_common():
        lines.append(
            f"  {kind:26s} {n:6,} items  {sheets_by[kind]:8,} sheets "
            f"({100 * sheets_by[kind] / total_sheets:5.1f}%)"
        )
    fuzzy = sorted(
        {
            (item.state, item.county, ";".join(c.name for c in match.counties))
            for item, match in matched
            if "fuzzy" in match.kind
        }
    )
    if fuzzy:
        lines.append(f"fuzzy matches to audit ({len(fuzzy)} distinct):")
        lines.extend(f"  {state}: {county!r} -> {ne}" for state, county, ne in fuzzy)
    unmatched = Counter(
        (item.state, item.county) for item, reason in skipped if reason == "unmatched"
    )
    if unmatched:
        lines.append(f"unmatched county strings ({len(unmatched)} distinct):")
        lines.extend(
            f"  {state}: {county!r} ({n} items)"
            for (state, county), n in unmatched.most_common()
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "metadata", type=Path, help="Catalog root: <state>/<year>/<item>.json"
    )
    parser.add_argument(
        "counties", type=Path, help="Natural Earth ne_10m_admin_2_counties.geojson"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Mirror mapping TSV, to count each item's kept sheets.",
    )
    args = parser.parse_args()

    index = load_counties(args.counties)
    items = load_items(args.metadata)
    if args.mapping:
        counts = load_sheet_counts(args.mapping)
        for item in items:
            item.sheets = counts.get(item.item, 0)
    matched, skipped = match_items(items, index)
    write_outputs(args.out_dir, matched, skipped)
    print(report(matched, skipped), file=sys.stderr)
    print(
        f"wrote items.tsv, counties.tsv, skipped.tsv to {args.out_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
