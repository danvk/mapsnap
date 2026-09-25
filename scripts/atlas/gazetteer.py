"""Locate a catalogue town the LoC catalogue never geocoded, from the Census Gazetteer.

4,729 of the catalogue's 50,600 records carry no coordinates, and for 996
towns none of their records do -- among them Manhattan, Queens, Saint Louis,
Baltimore, San Francisco and Denver. The atlas can only draw a town it can put
somewhere, so those towns were missing from it entirely.

The fallback is the Census Bureau's Gazetteer files, which give an internal
point for every incorporated place and census-designated place, every county
subdivision, and every county. Unzip these three into one directory:

    https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/
        2023_Gaz_place_national.zip
        2023_Gaz_cousubs_national.zip
        2023_Gaz_counties_national.zip

A town is looked up as a place, then as a county subdivision, then as a county.
New England towns are subdivisions rather than places (Westport, Connecticut is
"Westport town"), and Connecticut has had no counties since 2022. Many catalogue
"towns" are counties or county-wide atlases ("Hudson Co.", "Nassau"), and
villages since absorbed into a city ("Jamaica", "Flushing") are no longer places
at all, so their record's county is the nearest thing the Gazetteer has.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

PLACE_FILE = "2023_Gaz_place_national.txt"
SUBDIVISION_FILE = "2023_Gaz_cousubs_national.txt"
COUNTY_FILE = "2023_Gaz_counties_national.txt"

STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Puerto Rico": "PR", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}  # fmt: skip

STATE_CODES_BY_LOWERCASE = {name.lower(): code for name, code in STATE_CODES.items()}

# Towns whose Gazetteer point is the wrong answer. San Francisco's internal
# point is in the Pacific, since the city takes in the Farallon Islands; this
# is its City Hall.
POINT_OVERRIDES = {("CA", "sanfrancisco"): (37.7793, -122.4193)}

# Towns the Gazetteer names differently.
PLACE_ALIASES = {("HI", "honolulu"): "urbanhonolulu"}

# Towns whose Gazetteer place is the wrong answer. New York City is one place
# whose internal point is in Brooklyn, while the catalogue files Manhattan and
# the Bronx as "New York" and each other borough by its own name -- and a
# borough is a county. (Brooklyn itself is geocoded in the catalogue.)
COUNTY_OVERRIDES = {
    ("NY", "newyork"): "newyork",
    ("NY", "manhattan"): "newyork",
    ("NY", "bronx"): "bronx",
    ("NY", "queens"): "queens",
    ("NY", "statenisland"): "richmond",
}

# The legal-status word(s) the Gazetteer appends to a place name, longest first
# so "city and borough" is stripped whole.
PLACE_SUFFIXES = sorted(
    [
        "city and borough", "consolidated government", "unified government",
        "metro government", "metropolitan government", "urban county",
        "city", "town", "village", "borough", "CDP", "municipality",
        "comunidad", "zona urbana", "plantation", "township", "CCD",
    ],
    key=len,
    reverse=True,
)  # fmt: skip

# Place kinds in order of preference when two share a name in one state: an
# incorporated city over the CDP of the same name next to it.
INCORPORATED = ("city", "town", "village", "borough")


def normalize_name(name: str) -> str:
    """A town, place or county name reduced to what two spellings share.

    Lowercase, "Saint" and "St." both "st", no parenthetical ("Staten Island
    (Borough Of Richmond)"), and nothing but letters and digits: the catalogue
    writes "De Kalb" and "La Salle" where the Census writes DeKalb and LaSalle.
    """
    name = re.sub(r"\([^)]*\)", " ", name.lower())
    name = re.sub(r"\bsaint\b|\bst\b\.?", "st", name)
    name = re.sub(r"\bste\b\.?|\bsainte\b", "ste", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def strip_place_suffix(gazetteer_name: str) -> tuple[str, str]:
    """Split a Gazetteer place name into (name, legal status): "St. Louis city"."""
    for suffix in PLACE_SUFFIXES:
        if gazetteer_name.endswith(" " + suffix):
            return gazetteer_name[: -len(suffix) - 1], suffix
    return gazetteer_name, ""


def county_name(text: str) -> str | None:
    """The county a town or county text names, if it names one: "Hudson Co."."""
    match = re.match(
        r"(.+?)\s+(?:co\.?|county|parish|borough)$", text.strip(), re.IGNORECASE
    )
    return normalize_name(match.group(1)) if match else None


@dataclass
class Located:
    """Where the Gazetteer puts a town, and which of its tables said so."""

    lat: float
    lon: float
    source: str  # "place", "subdivision" or "county"


@dataclass
class Gazetteer:
    """Internal points of Census places, subdivisions and counties, by (state, name)."""

    places: dict[tuple[str, str], tuple[float, float]]
    subdivisions: dict[tuple[str, str], tuple[float, float]]
    counties: dict[tuple[str, str], tuple[float, float]]

    def locate(self, town: str, state: str, counties: list[str]) -> Located | None:
        """Where to put a catalogue town that has no coordinates of its own.

        ``counties`` is the record's County_text, which the county fallback
        reads when the town itself is not a place.
        """
        code = STATE_CODES_BY_LOWERCASE.get(state.lower())
        if code is None:
            return None
        name = normalize_name(town)
        if (code, name) in POINT_OVERRIDES:
            return Located(*POINT_OVERRIDES[(code, name)], "place")
        override = COUNTY_OVERRIDES.get((code, name))
        if override and (code, override) in self.counties:
            return Located(*self.counties[(code, override)], "county")
        place = PLACE_ALIASES.get((code, name), name)
        if (code, place) in self.places:
            return Located(*self.places[(code, place)], "place")
        if (code, name) in self.subdivisions:
            return Located(*self.subdivisions[(code, name)], "subdivision")
        for candidate in [county_name(town), name, *map(county_name, counties)]:
            if candidate and (code, candidate) in self.counties:
                return Located(*self.counties[(code, candidate)], "county")
        return None


def read_gazetteer(directory: Path) -> Gazetteer:
    """Load the three national Gazetteer files from the directory they were unzipped to."""
    counties: dict[tuple[str, str], tuple[float, float]] = {}
    for row in read_gazetteer_rows(directory / COUNTY_FILE):
        name = county_name(row["NAME"]) or normalize_name(row["NAME"])
        counties[(row["USPS"], name)] = (
            float(row["INTPTLAT"]),
            float(row["INTPTLONG"]),
        )
    return Gazetteer(
        places=read_named_points(directory / PLACE_FILE),
        subdivisions=read_named_points(directory / SUBDIVISION_FILE),
        counties=counties,
    )


def read_named_points(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """A place or subdivision file's internal points, by (state, name less its status).

    Where two share a name in one state, an incorporated city, town, village or
    borough beats anything else, then the larger by land area wins: Richmond,
    Virginia is the city, not a CDP of the same name.
    """
    ranked: dict[tuple[str, str], tuple[int, int, tuple[float, float]]] = {}
    for row in read_gazetteer_rows(path):
        name, status = strip_place_suffix(row["NAME"])
        key = (row["USPS"], normalize_name(name))
        rank = (0 if status in INCORPORATED else 1, -int(row["ALAND"] or 0))
        point = (float(row["INTPTLAT"]), float(row["INTPTLONG"]))
        if key not in ranked or rank < ranked[key][:2]:
            ranked[key] = (*rank, point)
    return {key: value[2] for key, value in ranked.items()}


def read_gazetteer_rows(path: Path) -> list[dict[str, str]]:
    """A Gazetteer file's rows, with the trailing whitespace of its last column trimmed."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            {key.strip(): (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]
