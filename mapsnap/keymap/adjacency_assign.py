"""Repair key-map page-number assignments using the printed adjacency graph (#213).

The key map's page numbers are the coarse-location source for OCR restriction,
georeferencing and snap: placing each page at its region centroid alone puts
Detroit's pages a median 158 ft from truth, which is why a missing or misread
number is expensive. Two failure classes survive the CRNN's own repair pass
(#172), because that pass arbitrates by CTC margin and these cells have no
margin to arbitrate:

  * a number read as a shorter one, where BOTH reads are confident and valid
    (Detroit prints 22 and 65; the sheet carries two "2"s and three "6"s at
    confidence 1.0, so the duplicate keeper logic cannot tell which is which);
  * a number with no detection at all (Detroit p59 — nothing to repair).

The printed adjacency graph settles both, and is now precise enough to do it:
mutual edges measure 96-100% against hand-labelled truth (#209), letter-aware
with roughly twice the recall on 4-digit volumes (#206). The evidence is
strikingly asymmetric on exactly the cells that matter — Detroit's *missing*
p22 carries three mutual edges (p21, p28, p29) while the *false* duplicate p2
carries none.

The rule in one line: a page's number should be printed among the numbers of
the pages that cite it in their margins. Only the DETECTIONS are consulted --
never the segmented regions -- so a poor segmentation cannot corrupt a page
number, and the repair can run before page_regions and seed it.

Three cell types, in decreasing order of evidence:

  duplicates    Instances of one label are scored by how many of that page's
                MUTUAL neighbours are nearby. Every mutually-vouched instance
                keeps its label -- split pages legitimately appear several
                times on the key map (Champaign draws p4 four times), and
                undersplit pages may too, so multiplicity is never itself a
                trigger. Only an unvouched instance is a suspect, and it is
                relabelled solely to a MISSING key its own neighbours name.
  cross-sheet   The same key drawn on two key-map SHEETS beyond its expected
                multiplicity: keep the vouched sheet's copies, drop the
                other's (Brooklyn's p8/p1/p5 each sit correctly on one sheet
                and 1-1.8 km wrong on the other). Duplicates within one sheet
                are the case above, and are relabelled rather than stripped.
  gaps          A missing key is placed at the centroid of the numbers whose
                pages cite it, when enough of them do.

Constraints inherited from #183's failed global-optimization experiments:
adjacency never overrides a read that has its own support (its negative signal
is weak); spatial coherence is a gate, never an energy; repairs only ever
target keys the sheet is missing; ambiguity means no action.

One-sided claims are corroboration only, never decisive: they measure 32-54%
precise against label truth (#207) versus 96-100% for mutual edges.
"""

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

PROXIMITY_FACTOR = 1.5
"""How many page-pitches apart two printed numbers may be and still count as
neighbours (see proximity_graph)."""

ONE_SIDED_WEIGHT = 0.25
"""Weight of an unreciprocated claim relative to a mutual edge.

One-sided claims are 32-54% precise (LA/Hudson/Nashville label truth) against
96-100% for mutual edges, so four of them are worth less than one mutual edge
and cannot on their own reach any adoption threshold below."""

RELABEL_MIN_MUTUAL = 2
"""Mutual neighbours that must vouch for a key before it takes over a panel.

A relabel OVERWRITES an existing assignment, so it can regress a page that was
right: two independent mutual edges is the bar. Detroit's 22 clears it with
three (p21, p28, p29) and its 65 with two (p34, p40)."""

GAP_MIN_MUTUAL = 1
GAP_MIN_SUPPORT = 1.5
"""Bar for placing a missing key: at least one mutual citation, plus
corroboration reaching GAP_MIN_SUPPORT in total.

Deliberately weaker than the relabel bar, because the two decisions have
different downsides. A relabel can take a correct assignment away; a gap
placement fills a void -- the page has no key-map location at all today, so
it cannot be snapped or vocabulary-restricted, and the worst case is that it
stays unusable. Detroit p59 is exactly this shape: one mutual edge (p27) plus
one-sided claims from p57 and p61, which is 1.5. A mutual edge is still
required, since one-sided claims are only 32-54% precise and no pile of them
should place a page on its own."""


@dataclass(frozen=True)
class Repair:
    """One proposed change to a key map's page-number assignment."""

    sheet: str
    index: int | None  # panel/detection index; None for a synthesized gap panel
    old: str | None  # previous label; None when nothing was there
    new: str | None  # new label; None strips the assignment
    reason: str
    support: float = 0.0
    evidence: tuple[str, ...] = ()
    evidence_indices: tuple[int, ...] = ()
    mutual_indices: tuple[int, ...] = ()

    def describe(self) -> str:
        vouchers = ", ".join(self.evidence) if self.evidence else "-"
        old = self.old if self.old is not None else "(none)"
        new = self.new if self.new is not None else "(strip)"
        return (
            f"{self.sheet}[{self.index if self.index is not None else 'new'}] "
            f"{old} -> {new} [{self.reason}; support {self.support:.2f}; {vouchers}]"
        )


@dataclass
class SheetNumbers:
    """One key-map sheet's page-number detections and which are near each other."""

    sheet: str
    labels: list[str]
    contacts: set[frozenset[int]] = field(default_factory=set)

    def touching(self, index: int) -> set[int]:
        return {
            other
            for pair in self.contacts
            if index in pair
            for other in pair
            if other != index
        }


def digit_family(text: str, keys: set[str]) -> set[str]:
    """Keys a read could be, had the recogniser dropped digits.

    The key-map failure is digit LOSS: a printed "22" reads "2", "65" reads
    "6". So a candidate is any key whose digits contain the read's digits in
    order (a subsequence), which covers both a lost leading digit and a lost
    trailing one. Mirrors the spirit of page_adjacency.resolve_page_key (#206)
    without its uniqueness requirement, since adjacency -- not the string --
    does the disambiguating here.
    """
    read = "".join(ch for ch in text if ch.isdigit())
    if not read:
        return set()
    family = set()
    for key in keys:
        digits = "".join(ch for ch in key if ch.isdigit())
        if len(digits) <= len(read):
            continue
        position = 0
        for char in digits:
            if position < len(read) and char == read[position]:
                position += 1
        if position == len(read):
            family.add(key)
    return family


def support_for(
    index: int,
    key: str,
    sheet: SheetNumbers,
    mutual: dict[str, set[str]],
    one_sided: dict[str, set[str]],
) -> tuple[float, tuple[str, ...]]:
    """How strongly the printed graph vouches for ``key`` sitting at ``index``.

    Counts the page's mutual neighbours printed near this number,
    plus a fractional credit for one-sided claims. Returns the score and the
    vouching page keys, which are carried into the repair record so a human
    can check the reasoning.
    """
    neighbours = {sheet.labels[other] for other in sheet.touching(index)}
    return score_neighbours(neighbours, key, mutual, one_sided)


def score_neighbours(
    neighbours: set[str],
    key: str,
    mutual: dict[str, set[str]],
    one_sided: dict[str, set[str]],
) -> tuple[float, tuple[str, ...]]:
    """(support, vouching keys) for ``key`` given the page keys around it."""
    vouchers = sorted(neighbours & mutual.get(key, set()))
    weak = sorted(neighbours & (one_sided.get(key, set()) - mutual.get(key, set())))
    return len(vouchers) + ONE_SIDED_WEIGHT * len(weak), tuple(vouchers + weak)


def mutual_count(neighbours: set[str], key: str, mutual: dict[str, set[str]]) -> int:
    """How many of ``neighbours`` are mutual-adjacency neighbours of ``key``."""
    return len(neighbours & mutual.get(key, set()))


def plan_sheet_repairs(
    sheet: SheetNumbers,
    mutual: dict[str, set[str]],
    one_sided: dict[str, set[str]],
    volume_keys: set[str],
) -> list[Repair]:
    """Duplicate-label arbitration for one sheet (the pure decision core).

    A label with several instances is only interesting when some instance has
    NO adjacency support: split pages appear several times legitimately, and
    every supported instance keeps its label. A zero-support instance adopts a
    missing key only when exactly one candidate in its digit family is vouched
    for by RELABEL_MIN_MUTUAL neighbours, so a tie or a shortage of evidence
    changes nothing.
    """
    assigned = {label for label in sheet.labels}
    missing = {key for key in volume_keys if key not in assigned}
    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(sheet.labels):
        by_label.setdefault(label, []).append(index)

    repairs: list[Repair] = []
    claimed: set[str] = set()
    for label, indices in sorted(by_label.items()):
        if len(indices) < 2:
            continue
        for index in indices:
            neighbours = {sheet.labels[other] for other in sheet.touching(index)}
            if mutual_count(neighbours, label, mutual) > 0:
                continue  # a mutually-vouched instance always keeps its label
            # Only MUTUAL edges protect a label. Detroit's misread "2" panel
            # sits next to one page that one-sidedly claims p2 -- 0.25 of
            # support, and at 32-54% precision nowhere near enough to shield
            # it from three mutual edges naming 22.
            # The sibling instances are deliberately NOT consulted: Detroit's
            # p2 has no mutual edges at all, so neither of its "2" panels can
            # be vouched for. What identifies the impostor is the positive
            # evidence for 22 at one of them, not a comparison between them.
            candidates = []
            for key in sorted(digit_family(label, missing - claimed)):
                if mutual_count(neighbours, key, mutual) < RELABEL_MIN_MUTUAL:
                    continue
                key_score, vouchers = score_neighbours(
                    neighbours, key, mutual, one_sided
                )
                candidates.append((key_score, key, vouchers))
            if len(candidates) != 1:
                continue  # no evidence, or ambiguous: leave it alone
            key_score, key, vouchers = candidates[0]
            claimed.add(key)
            repairs.append(
                Repair(
                    sheet=sheet.sheet,
                    index=index,
                    old=label,
                    new=key,
                    reason="duplicate-no-support",
                    support=key_score,
                    evidence=vouchers,
                )
            )
    return repairs


def plan_cross_sheet_repairs(
    sheets: list[SheetNumbers],
    mutual: dict[str, set[str]],
    one_sided: dict[str, set[str]],
    split_counts: dict[str, int] | None = None,
) -> list[Repair]:
    """Strip a key from the SHEET whose regions nothing vouches for.

    Only applies to keys drawn on more than one key-map sheet: a volume with
    two index sheets can place the same number on both, and one of them is
    wrong (Brooklyn's p8 sits 72 m from truth on p0b and 1827 m away on p0).
    Duplicates WITHIN one sheet are not this case -- they are usually split
    panels, and plan_sheet_repairs relabels rather than strips them.

    Fires only on a clean asymmetry: some sheet's copies are mutually vouched
    for and another sheet's are not, beyond the key's split multiplicity.
    """
    split_counts = split_counts or {}
    placements: dict[str, dict[str, list[int]]] = {}
    by_name = {sheet.sheet: sheet for sheet in sheets}
    for sheet in sheets:
        for index, label in enumerate(sheet.labels):
            placements.setdefault(label, {}).setdefault(sheet.sheet, []).append(index)

    repairs: list[Repair] = []
    for label, per_sheet in sorted(placements.items()):
        if len(per_sheet) < 2:
            continue  # one sheet only: not a cross-sheet conflict
        total = sum(len(indices) for indices in per_sheet.values())
        if total <= max(1, split_counts.get(label, 1)):
            continue
        vouched: dict[str, int] = {}
        for name, indices in per_sheet.items():
            sheet = by_name[name]
            vouched[name] = max(
                mutual_count(
                    {sheet.labels[other] for other in sheet.touching(index)},
                    label,
                    mutual,
                )
                for index in indices
            )
        supported = [name for name, count in vouched.items() if count > 0]
        unsupported = [name for name, count in vouched.items() if count == 0]
        if not supported or not unsupported:
            continue  # all or nothing vouched for: no asymmetry to exploit
        for name in unsupported:
            sheet = by_name[name]
            for index in per_sheet[name]:
                score, _vouchers = support_for(index, label, sheet, mutual, one_sided)
                repairs.append(
                    Repair(
                        sheet=name,
                        index=index,
                        old=label,
                        new=None,
                        reason="cross-sheet-no-support",
                        support=score,
                        evidence=tuple(sorted(supported)),
                    )
                )
    return repairs


def plan_gap_repairs(
    sheet: SheetNumbers,
    mutual: dict[str, set[str]],
    one_sided: dict[str, set[str]],
    volume_keys: set[str],
    missing_keys: set[str] | None = None,
) -> list[Repair]:
    """Propose a location for each missing key from the pages that cite it.

    A page absent from the key map has no region to score, so the evidence is
    the other direction: the panels of the pages whose margins print its
    number. Those panels surround where it belongs, and their centroid is the
    estimate. Requires GAP_MIN_MUTUAL mutual citations and GAP_MIN_SUPPORT in
    total, so a page vouched for only by one-sided claims is never placed.

    ``missing_keys`` narrows the candidates to the volume-wide shortfall. A
    volume's key maps split the city between them, so a page already drawn on
    another sheet is not missing at all: filling it here anyway put Brooklyn's
    9 and 22 on the wrong half, 6970 ft and 1079 ft from truth, while correct
    copies sat on the other sheet. Absent the argument every key the sheet
    lacks is fair game, which is right for a single-sheet volume.

    The vouching panel indices ride along in ``evidence_indices`` so the caller
    can compute the position and apply its own spatial sanity check.
    """
    assigned = set(sheet.labels)
    wanted = volume_keys if missing_keys is None else volume_keys & missing_keys
    panels_by_label: dict[str, list[int]] = {}
    for index, label in enumerate(sheet.labels):
        panels_by_label.setdefault(label, []).append(index)

    repairs: list[Repair] = []
    for key in sorted(key for key in wanted if key not in assigned):
        strong = sorted(mutual.get(key, set()) & assigned)
        weak = sorted((one_sided.get(key, set()) - mutual.get(key, set())) & assigned)
        score = len(strong) + ONE_SIDED_WEIGHT * len(weak)
        if len(strong) < GAP_MIN_MUTUAL or score < GAP_MIN_SUPPORT:
            continue
        strong_indices = tuple(
            index for label in strong for index in panels_by_label.get(label, [])
        )
        weak_indices = tuple(
            index for label in weak for index in panels_by_label.get(label, [])
        )
        repairs.append(
            Repair(
                sheet=sheet.sheet,
                index=None,
                old=None,
                new=key,
                reason="gap",
                support=score,
                evidence=tuple(strong + weak),
                evidence_indices=strong_indices + weak_indices,
                mutual_indices=strong_indices,
            )
        )
    return repairs


def volume_shortfall(
    sheets: list[SheetNumbers], volume_keys: set[str], splits: dict[str, int]
) -> set[str]:
    """Keys the volume's key maps draw fewer times than the page deserves.

    Counted across every sheet, because a volume's key maps divide the city
    between them and a page drawn on one is not missing from the volume. A
    split page is drawn once per panel, so its expected count is its panel
    count rather than one.
    """
    placed: dict[str, int] = {}
    for sheet in sheets:
        for label in sheet.labels:
            placed[label] = placed.get(label, 0) + 1
    return {key for key in volume_keys if placed.get(key, 0) < splits.get(key, 1)}


# --- volume-level inputs ----------------------------------------------------


def page_key_of(stem: str) -> str | None:
    """The page key a stem or printed text denotes ('p22' -> '22', '1499A')."""
    match = re.search(r"\d+[A-Za-z]{0,2}", str(stem))
    return match.group().upper() if match else None


def adjacency_graphs(volume: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(mutual, one_sided) neighbour maps keyed by page key, from adjacency.json."""
    path = volume / "adjacency.json"
    mutual: dict[str, set[str]] = {}
    one_sided: dict[str, set[str]] = {}
    if not path.exists():
        return mutual, one_sided
    doc = json.loads(path.read_text())
    for field_name, target in (("adjacency", mutual), ("one_sided", one_sided)):
        for pair in doc.get(field_name, []):
            if len(pair) != 2:
                continue
            first, second = (page_key_of(pair[0]), page_key_of(pair[1]))
            if not first or not second or first == second:
                continue
            target.setdefault(first, set()).add(second)
            target.setdefault(second, set()).add(first)
    return mutual, one_sided


def volume_page_keys(volume: Path) -> set[str]:
    """Page keys of the volume's real page images (split parents included)."""
    keys = set()
    for image in volume.glob("p*.jpg"):
        if "__" in image.stem:
            continue
        key = page_key_of(image.stem)
        if key:
            keys.add(key)
    return keys


def split_multiplicity(volume: Path) -> dict[str, int]:
    """Page key -> number of split panels, for pages the splitter divided.

    A split page is legitimately drawn once per panel on the key map, so its
    duplicate regions are expected rather than suspicious.
    """
    counts: dict[str, int] = {}
    for path in volume.glob("p*.panels.json"):
        stem = path.name[: -len(".panels.json")]
        key = page_key_of(stem)
        if not key:
            continue
        try:
            panels = json.loads(path.read_text()).get("panels", [])
        except (OSError, ValueError):
            continue
        if panels:
            counts[key] = len(panels)
    return counts


def detection_centers(streets: list[dict]) -> list[tuple[float, float] | None]:
    """Each page-number detection's centre, from its CRNN polygon."""
    centers: list[tuple[float, float] | None] = []
    for street in streets:
        polygon = street.get("polygon") or []
        if len(polygon) < 3:
            centers.append(None)
            continue
        centers.append(
            (
                sum(float(point[0]) for point in polygon) / len(polygon),
                sum(float(point[1]) for point in polygon) / len(polygon),
            )
        )
    return centers


def page_pitch(centers: list[tuple[float, float] | None]) -> float:
    """Typical spacing between neighbouring page numbers, in key-map pixels.

    The median nearest-neighbour distance over the printed numbers: one page
    step on this sheet, measured from the drawing rather than assumed (Detroit
    320 px, Champaign 351, Brooklyn 580).
    """
    points = [point for point in centers if point is not None]
    if len(points) < 3:
        return 0.0
    nearest = sorted(
        min(math.dist(point, other) for j, other in enumerate(points) if j != i)
        for i, point in enumerate(points)
    )
    return nearest[len(nearest) // 2]


def proximity_graph(
    centers: list[tuple[float, float] | None], factor: float = PROXIMITY_FACTOR
) -> set[frozenset[int]]:
    """Detection pairs close enough to be plausible neighbours on the sheet.

    Distance between printed numbers, scaled by the sheet's own page pitch.
    Deliberately permissive: the *printed* graph does the discriminating (a
    candidate still needs two mutual edges among these neighbours), so this
    only has to avoid missing true neighbours.

    Region contact was tried first and is unusable -- segmented regions are
    islands, not a mosaic (Detroit's cover 18% of the sheet), so p65's block
    sits ~180 px of blank paper from p34's and p40's and no contact tolerance
    that also respects real separations joins them.
    """
    pitch = page_pitch(centers)
    if pitch <= 0:
        return set()
    radius = factor * pitch
    pairs: set[frozenset[int]] = set()
    for i, first in enumerate(centers):
        if first is None:
            continue
        for j in range(i + 1, len(centers)):
            second = centers[j]
            if second is None:
                continue
            if math.dist(first, second) <= radius:
                pairs.add(frozenset((i, j)))
    return pairs


def load_sheets(volume: Path) -> list[tuple[str, dict]]:
    """(stem, keymap doc) for every key map carrying page-number detections.

    Only ``<stem>.keymap.json`` is read. The segmented regions are NOT an
    input: their quality varies and their failures are cheap to fix, while a
    wrong page number poisons every downstream coarse location. This also
    lets the repair run before page_regions, so corrected numbers seed the
    segmentation rather than depending on it.
    """
    sheets = []
    for keymap_path in sorted((volume / "raw").glob("*.keymap.json")):
        if ".truth." in keymap_path.name:
            continue
        stem = keymap_path.name[: -len(".keymap.json")]
        try:
            doc = json.loads(keymap_path.read_text())
        except (OSError, ValueError):
            continue
        if doc.get("streets"):
            sheets.append((stem, doc))
    return sheets


GAP_SPREAD_FACTOR = 3.0
"""How far apart the numbers citing a missing page may be, in page pitches.

The gap placement is the centroid of the citing numbers, which is only
meaningful if they actually surround one spot. Scattered citations mean at
least one of them is itself misassigned, so the placement is refused rather
than dropped in the middle of the sheet."""


def gap_placement(
    repair: Repair, centroids: list[tuple[float, float] | None], scale: float
) -> tuple[float, float] | None:
    """Where a gap repair's page belongs: the centroid of its citing numbers.

    Anchored on the MUTUAL citations, which are 96-100% precise, then refined
    with whichever one-sided citations agree with them. Detroit's p59 is cited
    by p27 mutually and by p57, p61 and p1 one-sidedly, and the p1 claim is
    junk (issue #213 names it) -- averaging it in would drag the placement
    across the sheet, so a one-sided citation more than GAP_SPREAD_FACTOR
    page pitches from the mutual anchor is discarded.

    Returns None when no mutual citation has a usable position, or when the
    mutual citations themselves disagree on a location (which means one of
    them is misassigned).
    """

    def point_of(index: int) -> tuple[float, float] | None:
        if 0 <= index < len(centroids):
            return centroids[index]
        return None

    anchors = [
        point for point in map(point_of, repair.mutual_indices) if point is not None
    ]
    if not anchors:
        return None
    anchor = (
        sum(point[0] for point in anchors) / len(anchors),
        sum(point[1] for point in anchors) / len(anchors),
    )
    limit = GAP_SPREAD_FACTOR * scale if scale > 0 else math.inf
    if len(anchors) > 1 and max(math.dist(anchor, p) for p in anchors) > limit:
        return None
    weak = [
        point
        for index in repair.evidence_indices
        if index not in repair.mutual_indices
        and (point := point_of(index)) is not None
        and math.dist(anchor, point) <= limit
    ]
    points = anchors + weak
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def plan_volume_repairs(
    volume: Path,
) -> tuple[list[Repair], dict[str, dict[str, tuple[float, float]]]]:
    """All proposed repairs for a volume, plus the placement of each gap fill.

    Reads only ``raw/*.keymap.json`` and ``adjacency.json`` -- never the
    segmented regions. Returns (repairs, placements) where
    ``placements[sheet][key]`` is the key-map pixel position for a gap
    recovery, so the caller can synthesize a detection there. Gap repairs whose citing
    regions disagree on a location are dropped here rather than proposed.
    """
    mutual, one_sided = adjacency_graphs(volume)
    if not mutual:
        return [], {}
    volume_keys = volume_page_keys(volume)
    splits = split_multiplicity(volume)

    sheet_panels: list[SheetNumbers] = []
    geometry: dict[str, tuple[list[tuple[float, float] | None], float]] = {}
    for stem, doc in load_sheets(volume):
        streets = doc.get("streets", [])
        labels = [
            page_key_of(street.get("text", "")) or str(street.get("text", ""))
            for street in streets
        ]
        centers = detection_centers(streets)
        sheet_panels.append(SheetNumbers(stem, labels, proximity_graph(centers)))
        geometry[stem] = (centers, page_pitch(centers))

    repairs: list[Repair] = []
    placements: dict[str, dict[str, tuple[float, float]]] = {}
    # Gap recovery must see the relabels: a key just recovered from a misread
    # panel is no longer missing, and proposing it again would place the same
    # page twice (Detroit's 65 was found both ways). So relabel every sheet
    # first, then take stock of what the volume is still short of.
    repaired_sheets: list[SheetNumbers] = []
    for sheet in sheet_panels:
        relabels = plan_sheet_repairs(sheet, mutual, one_sided, volume_keys)
        repairs.extend(relabels)
        repaired = SheetNumbers(sheet.sheet, list(sheet.labels), sheet.contacts)
        for relabel in relabels:
            if relabel.index is not None and relabel.new is not None:
                repaired.labels[relabel.index] = relabel.new
        repaired_sheets.append(repaired)

    missing = volume_shortfall(repaired_sheets, volume_keys, splits)
    for repaired in repaired_sheets:
        centers, pitch = geometry[repaired.sheet]
        for repair in plan_gap_repairs(
            repaired, mutual, one_sided, volume_keys, missing
        ):
            point = gap_placement(repair, centers, pitch)
            if point is None or repair.new is None:
                continue
            repairs.append(repair)
            placements.setdefault(repaired.sheet, {})[repair.new] = point
            # One fill per shortfall: two sheets that both border the page
            # must not each grow a copy of it.
            missing.discard(repair.new)
    repairs.extend(plan_cross_sheet_repairs(sheet_panels, mutual, one_sided, splits))
    return repairs, placements


RAW_SUFFIX = ".keymap-raw.json"
"""Where the pre-repair detections are preserved, once per sheet.

The repaired file BECOMES the key map (that is the point -- better numbers
must drive region segmentation and OCR restriction downstream), so the
original is kept beside it for inspection. Written only if absent, so
re-running the repair never overwrites the true original with a repaired one.
"""


GAP_CONFIDENCE = 0.2
"""``confidence`` on a synthesized gap detection.

``confidence`` means the recogniser's posterior over the text it read, and a
gap detection read nothing -- there is no posterior to report. Claiming a high
one would let it pass for a confident read anywhere confidence is thresholded
or ranked, and the adjacency support that DID place it is a different quantity
on a different scale (``support``, roughly 1.5-5), so it is not a stand-in.
What is left is a deliberately low placeholder: low enough to sort below every
real read, non-zero so the debugger draws it (a 0 renders as invisible)."""

MIN_GAP_SIDE = 20.0
"""Smallest a synthesized gap detection's box may be, in key-map pixels.

A floor under the sheet's median detection box, matching the debugger's default
minimum-short-side filter so a gap detection is never hidden by it. Only affects
display and the colour-block sample page_regions takes underneath -- both
indifferent to a few pixels on a sheet thousands wide."""


def median_detection_box(streets: list[dict]) -> tuple[float, float]:
    """(width, height) of the sheet's typical page-number box, in key-map pixels.

    Measured from the polygons rather than the long_side/short_side fields, so
    it holds whatever their orientation convention is. Previously-synthesized
    boxes are excluded -- otherwise re-running the repair would let them drag
    the median toward themselves. Falls back to a MIN_GAP_SIDE square when the
    sheet has no real detections to measure.
    """
    widths: list[float] = []
    heights: list[float] = []
    for street in streets:
        polygon = street.get("polygon") or []
        if street.get("via") == "adjacency-gap" or len(polygon) < 3:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        widths.append(max(xs) - min(xs))
        heights.append(max(ys) - min(ys))
    if not widths:
        return (MIN_GAP_SIDE, MIN_GAP_SIDE)
    return (
        max(MIN_GAP_SIDE, statistics.median(widths)),
        max(MIN_GAP_SIDE, statistics.median(heights)),
    )


def synthetic_detection(
    key: str, point: tuple[float, float], box: tuple[float, float], repair: Repair
) -> dict:
    """A detection record for a page the sheet never printed a readable number for.

    Shaped like a CRNN detection so every existing reader (load_seeds,
    load_detections, the debugger) treats it as one, with ``via`` marking its
    provenance. ``box`` is the (width, height) to draw it at -- the sheet's
    median real detection, so a gap reads at a glance as the same kind of thing
    as its neighbours rather than as a speck. page_regions only uses a seed's
    box to pick the colour block under it, and the world position downstream
    comes from the box centre, so the size is a display choice either way.
    """
    # Integer half-extents about an integer centre, so the polygon's centre is
    # exactly the estimate rather than half a pixel off it: downstream this box
    # IS the page's position, and the sides are reported from what was drawn.
    half_width = max(1, round(box[0] / 2))
    half_height = max(1, round(box[1] / 2))
    x, y = round(point[0]), round(point[1])
    return {
        "polygon": [
            [x - half_width, y - half_height],
            [x + half_width, y - half_height],
            [x + half_width, y + half_height],
            [x - half_width, y + half_height],
        ],
        "text": key,
        "confidence": GAP_CONFIDENCE,
        "angle": 0,
        "long_side": float(max(2 * half_width, 2 * half_height)),
        "short_side": float(min(2 * half_width, 2 * half_height)),
        "dir_pix": 0.0,
        "via": "adjacency-gap",
        "support": round(repair.support, 2),
        "cited_by": list(repair.evidence),
    }


def apply_repairs(
    volume: Path,
    repairs: list[Repair],
    placements: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, int]:
    """Rewrite each sheet's keymap.json with its repairs; return per-sheet counts.

    The repaired detections become the key map. The pre-repair file is copied
    to ``<stem>.keymap-raw.json`` once, and a ``assignment_repairs`` array
    records every change with its evidence so the reasoning survives in the
    artifact.
    """
    by_sheet: dict[str, list[Repair]] = {}
    for repair in repairs:
        by_sheet.setdefault(repair.sheet, []).append(repair)

    counts: dict[str, int] = {}
    for stem, sheet_repairs in sorted(by_sheet.items()):
        path = volume / "raw" / f"{stem}.keymap.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        streets = doc.get("streets", [])
        raw_path = volume / "raw" / f"{stem}{RAW_SUFFIX}"
        if not raw_path.exists():
            raw_path.write_text(json.dumps(doc, indent=2))

        box = median_detection_box(streets)
        stripped: list[int] = []
        for repair in sheet_repairs:
            if repair.index is None:
                point = placements.get(stem, {}).get(repair.new or "")
                if point is None or repair.new is None:
                    continue
                streets.append(synthetic_detection(repair.new, point, box, repair))
            elif repair.new is None:
                stripped.append(repair.index)
            elif 0 <= repair.index < len(streets):
                streets[repair.index]["text"] = repair.new
                streets[repair.index]["via"] = "adjacency-relabel"
                streets[repair.index]["cited_by"] = list(repair.evidence)
        for index in sorted(stripped, reverse=True):
            if 0 <= index < len(streets):
                del streets[index]

        doc["streets"] = streets
        doc.setdefault("assignment_repairs", []).extend(
            {
                "index": repair.index,
                "old": repair.old,
                "new": repair.new,
                "reason": repair.reason,
                "support": round(repair.support, 2),
                "evidence": list(repair.evidence),
            }
            for repair in sheet_repairs
        )
        path.write_text(json.dumps(doc, indent=2))
        counts[stem] = len(sheet_repairs)
    return counts


def repair_volume(volume: Path, dry_run: bool = False) -> list[Repair]:
    """Plan (and unless ``dry_run``, apply) a volume's assignment repairs."""
    repairs, placements = plan_volume_repairs(volume)
    if repairs and not dry_run:
        apply_repairs(volume, repairs, placements)
    return repairs
