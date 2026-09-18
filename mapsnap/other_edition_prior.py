"""Another edition's placements as a location prior for snap's rescue search.

Sanborn kept sheet numbers across re-issues, so sheet N of one edition covers
the ground of sheet N in another: over the four pairings under `data/`, 2-10 m
apart at the median and every one within 300 m, against a random-page null of
over a kilometre.

The other edition is named by a IIIF AnnotationPage -- a volunteer file or one
`mapsnap iiif` wrote -- and nothing else about it is read, so a half-run volume
can serve. Which pages it serves, and how wide they search, is decided by
`other_edition_plan`; that rule and the window width are both measured, and
both alternatives cost placements (docs/fit-pipeline.md stage 3).

`snap --other-edition` writes artifacts/osm_snap/other_edition_prior.json;
reconcile reads the same file, and `fit` clears it like any derived sidecar.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    extract_gcps,
    fit_transform,
)
from mapsnap.utils import haversine_m, source_id_to_page_key

OTHER_EDITION_RADIUS_M = 50.0
"""The search window around the other edition's center, where it replaces a
page's key map.

Wider was measured and costs placements: at 100 m a one-block alias entered
two of Chicago 1950 vol 1's three rescues and the margin rule refused both.
Deriving it per pairing from how well the two editions agree would read only
the sheets both already placed, which are the easy ones."""

OTHER_EDITION_CONTRADICTION_M = 200.0
"""How far a key-map center may sit from the same sheet in the other edition
before the key map counts as wrong rather than imprecise, and is replaced.

The measured distances are not a continuum: Chicago 1950 vol 1 tops out at
38.8 m from its 1906 issue, and Queens 1898/1915 drop from 249 m straight to
64 m, so every threshold in 65-249 m selects the same nine sheets."""

SECTION_SUFFIX = re.compile(r"[NSW]$")
SHEET_KEY = re.compile(r"P\d+[A-Z]?")
COMPOUND_SUFFIX = re.compile(r"p\d+[a-z]s(?:__\d+)?", re.IGNORECASE)


def section_key(stem: str) -> str:
    """The sheet key shared across editions: 'p10n' -> 'P10', 'p3a' -> 'P3A'.

    A trailing N/S/W division letter is dropped (Chicago's 1950 issue added
    them to the 1906 numbering); any other letter names a distinct sheet and is
    kept. Anything that is not a numbered sheet gets no key at all: a split
    panel ('p12n__1'), a cover or index page, and the path-shaped junk
    `source_id_to_page_key` returns for an image URL it cannot parse.

    A compound suffix ('p6ns') is ambiguous — the trailing 's' may be a
    skeleton twin of p6n or a sequence letter on p6 — and gets no key either,
    the same call compare_iiif_georef.redundant_skeleton_keys makes. The sheet
    is left to its own key map rather than seeded from a guess.
    """
    if "__" in stem or COMPOUND_SUFFIX.fullmatch(stem):
        return ""
    key = SECTION_SUFFIX.sub("", stem.upper())
    return key if SHEET_KEY.fullmatch(key) else ""


def item_center(item: dict) -> tuple[float, float] | None:
    """(lon, lat) at the middle of one annotation's page, or None when unusable.

    Uses the full page rectangle, not the annotated content region: that is a
    sub-polygon whose centroid sits tens of metres off.
    """
    source = item.get("target", {}).get("source") or {}
    width, height = source.get("width"), source.get("height")
    if not width or not height:
        return None
    gcps = extract_gcps(item)
    transform_type = annotation_transform_type(item)
    if len(gcps) < (2 if transform_type == "helmert" else 3):
        return None
    affine = fit_transform(gcps, transform_type)
    lon, lat = affine @ np.array([width / 2.0, height / 2.0, 1.0])
    return float(lon), float(lat)


def annotation_pages(annotation: Path) -> list[tuple[str, dict]]:
    """(page stem, annotation item) for every item of a IIIF AnnotationPage."""
    doc = json.loads(annotation.read_text())
    return [
        (
            source_id_to_page_key(
                (item.get("target", {}).get("source") or {}).get("id"),
                str(item.get("label", "")),
            ),
            item,
        )
        for item in doc.get("items") or []
    ]


def annotation_centers(annotation: Path) -> dict[str, tuple[float, float]]:
    """Section key -> sheet center, over every page a IIIF annotation places.

    Two labels can collapse to one key ('p83' and a skeleton 'p83s'). The stem
    that IS the key wins; where neither is, the key is dropped rather than
    decided by file order, which would seed a 50 m search half a sheet away.
    """
    by_key: dict[str, list[tuple[str, tuple[float, float]]]] = {}
    for stem, item in annotation_pages(annotation):
        key = section_key(stem)
        center = item_center(item) if key else None
        if center is not None:
            by_key.setdefault(key, []).append((stem, center))

    centers: dict[str, tuple[float, float]] = {}
    for key, placed in by_key.items():
        if len(placed) > 1:
            placed = [entry for entry in placed if entry[0].upper() == key]
        if len(placed) != 1:
            stems = ", ".join(stem for stem, _ in by_key[key])
            print(
                f"{annotation.name}: {stems} all key as {key}; "
                "no other-edition center for it",
                file=sys.stderr,
            )
            continue
        centers[key] = placed[0][1]
    return centers


def annotation_signature(annotation: Path) -> str:
    """Short content hash of the annotation, for candidate-cache freshness.

    Bytes, not mtime: a copied or re-downloaded annotation would otherwise cost
    the volume a half-hour re-search.
    """
    return hashlib.sha256(annotation.read_bytes()).hexdigest()[:16]


@dataclass
class OtherEditionPrior:
    """Per-sheet centers read from another edition's IIIF annotation."""

    source: str
    centers: dict[str, tuple[float, float]]
    # annotation_signature of the file these came from. A cached candidates
    # record carries it, so re-pointing the flag or re-publishing the edition
    # does not serve back the old search.
    signature: str

    def center_for(self, stem: str) -> tuple[float, float] | None:
        """This sheet's center in the other edition, or None when it has none.

        A split panel never has one: a sheet is what gets a key.
        """
        return self.centers.get(section_key(stem))

    def describe(self) -> str:
        """One operator line: what was paired, and how many sheets it places."""
        return f"other edition {self.source}: {len(self.centers)} sheets"

    def to_json(self) -> dict[str, Any]:
        """The sidecar's contents."""
        doc = asdict(self)
        doc["centers"] = {key: list(value) for key, value in self.centers.items()}
        return doc

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> OtherEditionPrior:
        """Load a prior sidecar written by an earlier `snap --other-edition`."""
        return cls(
            source=doc["source"],
            centers={
                key: (float(v[0]), float(v[1])) for key, v in doc["centers"].items()
            },
            signature=doc["signature"],
        )


def prior_path(volume: Path) -> Path:
    """Where the volume's other-edition prior sidecar lives."""
    return volume / "artifacts" / "osm_snap" / "other_edition_prior.json"


def load_prior(volume: Path) -> OtherEditionPrior | None:
    """The prior a previous `snap --other-edition` wrote for this volume, if any."""
    path = prior_path(volume)
    if not path.exists():
        return None
    return OtherEditionPrior.from_json(json.loads(path.read_text()))


def ensure_prior(volume: Path, annotation: Path) -> OtherEditionPrior:
    """Read the annotation into a prior and cache it under artifacts/osm_snap.

    Returns what was written, which every later stage loads back with
    ``load_prior`` so they all search the same centers.
    """
    try:
        centers = annotation_centers(annotation)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"{annotation}: cannot read as a IIIF annotation ({error!r})")
    if not centers:
        stems = [stem for stem, _ in annotation_pages(annotation)[:3]]
        sys.exit(
            f"{annotation}: no sheet centers; its pages key as "
            f"{', '.join(repr(s) for s in stems) or '(no items)'}. "
            "`mapsnap iiif --image-base-url` writes target.source.id as "
            "<base>/<stem>.jpg, which carries no page number — publish it "
            "without that flag, or pass the LoC or volunteer annotation."
        )
    prior = OtherEditionPrior(
        source=str(annotation),
        centers=centers,
        signature=annotation_signature(annotation),
    )
    path = prior_path(volume)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prior.to_json(), indent=1))
    return prior


@dataclass
class OtherEditionPlan:
    """What the other edition does to one page's search. See other_edition_plan."""

    sheet_center: tuple[float, float] | None
    centers: list[tuple[float, float]]  # the key map's, with the edition applied
    regions: list[list[list[float]]] | None
    radius_m: float
    # Whether the other edition's center stood in for the key map's. An
    # unserved page is the one with no ``sheet_center`` at all.
    replaced_key_map: bool = False


def other_edition_plan(
    prior: OtherEditionPrior | None,
    stem: str,
    *,
    centers: list[tuple[float, float]],
    regions: list[list[list[float]]] | None,
    radius_m: float,
    rescued: bool = False,
) -> OtherEditionPlan:
    """Resolve what the other-edition prior does to one page's key-map search.

    A rescue-state whole sheet more than OTHER_EDITION_CONTRADICTION_M from its
    key map searches from the other edition alone, inside
    OTHER_EDITION_RADIUS_M; nearer, the key map is left alone. Panels and pages
    that already have a pose keep ``centers`` and ``radius_m`` untouched.

    ``radius_m`` is the window to fall back to where the prior does not
    replace the key map; callers differ on which one that is.
    """
    plain = OtherEditionPlan(
        sheet_center=None,
        centers=centers,
        regions=regions,
        radius_m=radius_m,
    )
    if prior is None or not rescued:
        return plain
    sheet_center = prior.center_for(stem)
    if sheet_center is None:
        return plain
    nearest = min(
        (
            haversine_m(sheet_center[1], sheet_center[0], lat, lon)
            for lon, lat in centers
        ),
        default=None,
    )
    if nearest is not None and nearest <= OTHER_EDITION_CONTRADICTION_M:
        # They agree: the ordinary radius gate already judges candidates
        # against the page's own centers.
        return replace(plain, sheet_center=sheet_center)
    return replace(
        plain,
        sheet_center=sheet_center,
        centers=[sheet_center],
        regions=None,
        radius_m=OTHER_EDITION_RADIUS_M,
        replaced_key_map=True,
    )
