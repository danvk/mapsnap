"""Demote fitted pages that contradict their own printed mutual-adjacency claims.

Mutual adjacent-sheet references are ~100% precise (measured against hand-labelled
truth on LA and Hudson), and both sides of an edge print their reference at the
same physical spot on the shared boundary road. Mapping each claim's pixel through
its own page's fit, correct pairs land a median 35 m apart (p95 63 m) — so a pair
whose stamps land > GATE_STAMP_M apart is a contradiction: one page, the other, or
the edge itself is wrong. Arbitration, in order of trust:

  1. multi-contradiction   a page contradicting >= 2 different neighbors is the
                           liar; its partners each contradict only it
  2. corroboration         a page with other COMPATIBLE mutual edges is vouched
                           for; a page with none is the suspect
  3. edge verdict          both sides corroborated -> the EDGE is junk (false
                           reciprocation, or a long boundary printed at two
                           spots); demote nobody
  4. tie-breaks            fewer effective GCPs, then rotation/scale deviation

A named suspect is only demoted when a hard signal corroborates the verdict —
effective GCPs <= GATE_MAX_GCPS, or rotation/scale deviation from the volume
median beyond GATE_ROT_DEV_DEG / GATE_SCALE_DEV_LOG. On the 14 truth volumes
this combination names the wrong page on 18 of 19 decidable contradictions and
demotes 10 true disasters against one false demotion (Nashville p8, a genuine
odd-scale sheet whose five mutual edges are all junk single-digit reciprocation).

Demotion records ``status: contradicted`` on every channel sidecar the page has
-- the pose stays where it is, the channel simply stops claiming it -- and writes
pN.contradiction.json carrying the partner-side stamp positions — the trusted
neighbors' printed claims of this page, each a world point on the shared seam.
Snap's rescue reads those as extra search centers (see build_page_context), so a
demoted page gets re-searched where its neighbors say it belongs; the usual
rescue verification gates decide whether anything better is published.

Runs in ``mapsnap fit`` between georef and snap. A volume without adjacency.json
is a no-op.

    uv run mapsnap adjacency-gate data/kansas_city_mo_1951_vol_4 [--dry-run]
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mapsnap import sidecar
from mapsnap.keymap.locate import page_key
from mapsnap.road_model import effective_gcp_count, page_world_affine
from mapsnap.utils import haversine_m

GATE_STAMP_M = 100.0
"""Stamp-pair distance beyond which a mutual edge is a contradiction.

Healthy pairs across the 14 truth volumes: p50 35 m, p95 63 m, p99 126 m.
100 m keeps the flag above the healthy p95 while catching 200 ft-class errors
(a 200 ft misplacement moves its stamp ~61 m relative to the neighbor's)."""

GATE_MAX_GCPS = 2
"""A suspect supported by <= this many effective GCPs may be demoted: two
points exactly determine a similarity, so nothing over-determines the pose."""

GATE_ROT_DEV_DEG = 15.0
GATE_SCALE_DEV_LOG = 0.2
"""Suspects whose rotation/scale deviate this far from the volume median may
be demoted even with many GCPs (NO-1896 p125: 12 GCPs, 46 deg and 0.76 off —
a confidently wrong fit). Both thresholds clear every healthy suspect measured
across the corpus."""

CHANNELS = ("georef-street", "georef-snap", "georef")


@dataclass
class FittedPage:
    """One parent page's published fit, with the signals arbitration needs."""

    stem: str
    affine: np.ndarray  # 2x3 page px -> (lon, lat)
    width: int
    height: int
    channel_paths: list[Path]
    gcps: int
    theta_deg: float
    log_scale: float


@dataclass
class Contradiction:
    """One flagged mutual edge."""

    a: str
    b: str
    distance_m: float


@dataclass
class Verdict:
    """Arbitration outcome for one gated suspect."""

    stem: str
    reason: str  # arbitration rule that named it
    signal: str  # hard signal that licensed demotion
    partners: list[dict] = field(default_factory=list)  # re-search hints


def _accepted(path: Path) -> bool:
    """Whether a channel sidecar holds a pose its channel stands behind."""
    try:
        return sidecar.accepted(json.loads(path.read_text()))
    except (OSError, ValueError):
        return False


def load_fitted_pages(volume: Path, adjacency: dict) -> dict[str, FittedPage]:
    """Parent pages with a published fit, keyed by stem.

    Effective GCPs are taken as the max over channel sidecars: a snap-rescued
    page keeps its RANSAC doc's intersections when one exists, and a
    rescue-only page (no intersections anywhere) counts 0 — appropriately
    fragile for the hard-signal gate.
    """
    pages: dict[str, FittedPage] = {}
    for stem in adjacency.get("pages", {}):
        # A channel sidecar exists whether or not the channel stands behind its
        # pose, so "published" is the recorded verdict, not the file's presence.
        paths = [
            path
            for channel in CHANNELS
            if (path := volume / f"{stem}.{channel}.json").exists() and _accepted(path)
        ]
        if not paths:
            continue
        try:
            doc = json.loads(paths[0].read_text())
            affine = page_world_affine(doc)
        except (ValueError, KeyError, TypeError):
            continue
        gcps = 0
        for path in paths:
            try:
                gcps = max(gcps, effective_gcp_count(json.loads(path.read_text())))
            except (ValueError, KeyError, TypeError):
                pass
        pages[stem] = FittedPage(
            stem=stem,
            affine=affine,
            width=int(doc["width"]),
            height=int(doc["height"]),
            channel_paths=paths,
            gcps=gcps,
            theta_deg=math.degrees(math.atan2(-affine[1, 0], affine[0, 0])),
            log_scale=math.log(math.hypot(affine[1, 0], affine[1, 1])),
        )
    return pages


def stamp_worlds(
    adjacency: dict, page: FittedPage, other_stem: str
) -> list[tuple[float, float]]:
    """World (lon, lat) of ``page``'s printed claims of ``other_stem``'s key."""
    want = (page_key(other_stem) or "").upper()
    out: list[tuple[float, float]] = []
    for det in adjacency["pages"].get(page.stem, {}).get("detections", []):
        key = str(det.get("key") or det.get("number") or "").upper()
        if not det.get("claim") or key != want:
            continue
        px = det["x_frac"] * page.width
        py = det["y_frac"] * page.height
        a = page.affine
        out.append(
            (
                a[0, 0] * px + a[0, 1] * py + a[0, 2],
                a[1, 0] * px + a[1, 1] * py + a[1, 2],
            )
        )
    return out


def edge_scale_factor(a: FittedPage, b: FittedPage, median_log_scale: float) -> float:
    """How much to widen the stamp bar for an edge between coarse sheets.

    The stamp separation measures *pixel* registration pushed through each
    page's fit, so a fixed metric bar demands twice the pixel accuracy of a
    sheet drawn at twice the ground scale. Widen the bar by the coarser
    page's scale relative to the volume median (Fargo's two large-format
    sheets p57/p64 sit at ~2x the median and healthy edges between them
    measure ~115 m). Never narrowed below 1: the 100 m floor is calibrated
    on pooled healthy pairs, and fine sheets keep it.
    """
    return max(1.0, math.exp(max(a.log_scale, b.log_scale) - median_log_scale))


def edge_contradictions(
    adjacency: dict, pages: dict[str, FittedPage], stamp_m: float = GATE_STAMP_M
) -> tuple[list[Contradiction], dict[str, list[bool]]]:
    """Flagged edges plus, per page, the compatible/contradicted flags of its edges."""
    contradictions: list[Contradiction] = []
    edge_flags: dict[str, list[bool]] = defaultdict(list)
    log_scales = sorted(p.log_scale for p in pages.values())
    median_log_scale = log_scales[len(log_scales) // 2] if log_scales else 0.0
    for a, b in adjacency.get("adjacency", []):
        if a not in pages or b not in pages:
            continue
        wa = stamp_worlds(adjacency, pages[a], b)
        wb = stamp_worlds(adjacency, pages[b], a)
        if not wa or not wb:
            continue
        distance = min(haversine_m(p[1], p[0], q[1], q[0]) for p in wa for q in wb)
        bad = distance > stamp_m * edge_scale_factor(
            pages[a], pages[b], median_log_scale
        )
        edge_flags[a].append(bad)
        edge_flags[b].append(bad)
        if bad:
            contradictions.append(Contradiction(a, b, distance))
    return contradictions, edge_flags


def hard_signal(
    page: FittedPage, median_theta_deg: float, median_log_scale: float
) -> str | None:
    """The demotion-licensing signal, or None when the fit looks structurally sound."""
    if page.gcps <= GATE_MAX_GCPS:
        return f"gcps={page.gcps}"
    rot = abs((page.theta_deg - median_theta_deg + 180.0) % 360.0 - 180.0)
    if rot > GATE_ROT_DEV_DEG:
        return f"rot_dev={rot:.0f}deg"
    scale = abs(page.log_scale - median_log_scale)
    if scale > GATE_SCALE_DEV_LOG:
        return f"scale_dev={scale:.2f}"
    return None


def arbitrate_suspects(adjacency: dict, pages: dict[str, FittedPage]) -> list[Verdict]:
    """Gated suspects across the volume's contradicted mutual edges."""
    contradictions, edge_flags = edge_contradictions(adjacency, pages)
    if not contradictions:
        return []
    thetas = sorted(p.theta_deg for p in pages.values())
    scales = sorted(p.log_scale for p in pages.values())
    median_theta = thetas[len(thetas) // 2]
    median_scale = scales[len(scales) // 2]

    def n_bad(stem: str) -> int:
        return sum(1 for bad in edge_flags[stem] if bad)

    def n_good(stem: str) -> int:
        return sum(1 for bad in edge_flags[stem] if not bad)

    suspects: dict[str, Verdict] = {}
    for edge in contradictions:
        a, b = edge.a, edge.b
        if n_bad(a) >= 2 and n_bad(b) < 2:
            suspect, reason = a, "multi-contradiction"
        elif n_bad(b) >= 2 and n_bad(a) < 2:
            suspect, reason = b, "multi-contradiction"
        elif n_good(a) > 0 and n_good(b) == 0:
            suspect, reason = b, "uncorroborated"
        elif n_good(b) > 0 and n_good(a) == 0:
            suspect, reason = a, "uncorroborated"
        elif n_good(a) > 0 and n_good(b) > 0:
            continue  # both vouched for: the edge is the liar, demote nobody
        elif pages[a].gcps != pages[b].gcps:
            suspect = a if pages[a].gcps < pages[b].gcps else b
            reason = "fewer-gcps"
        else:
            continue  # undecidable pair: leave both alone
        signal = hard_signal(pages[suspect], median_theta, median_scale)
        if signal is None:
            continue  # structurally sound fit: never demote on the stamp alone
        partner = b if suspect == a else a
        verdict = suspects.setdefault(suspect, Verdict(suspect, reason, signal))
        for lon, lat in stamp_worlds(adjacency, pages[partner], suspect):
            verdict.partners.append(
                {
                    "stem": partner,
                    "stamp": [round(lon, 7), round(lat, 7)],
                    "distance_m": round(edge.distance_m, 1),
                }
            )
    return sorted(suspects.values(), key=lambda v: v.stem)


def demote(volume: Path, page: FittedPage, verdict: Verdict) -> None:
    """Record the contradiction on the page's channel sidecars, write the hint file."""
    for path in page.channel_paths:
        sidecar.demote(
            path,
            sidecar.CONTRADICTED,
            {"reason": verdict.reason, "signal": verdict.signal},
        )
    hint = {
        "timestamp": datetime.now(UTC).isoformat(),
        "reason": verdict.reason,
        "signal": verdict.signal,
        "partners": verdict.partners,
    }
    (volume / f"{page.stem}.contradiction.json").write_text(json.dumps(hint, indent=2))


STAMP_REHOME_M = GATE_STAMP_M
"""A rescue candidate for a contradiction-demoted page must satisfy the same
invariant that demoted it: its own printed claim of a hinting neighbor, mapped
through the CANDIDATE pose, must land within this distance of the neighbor's
stamp. KC p551's re-adopted alias sits ~150 m from p545's stamp and dies here;
a true pose passes by construction. Where the page never read that neighbor's
number, the neighbor's stamp (printed on the shared seam) must at least touch
the candidate footprint."""


@dataclass
class StampGate:
    """Re-adoption consistency check for one contradiction-demoted page."""

    width: int
    height: int
    # Parallel lists: each hinting partner's stamp (lon, lat) and this page's
    # own claim pixels of that partner ([] when its read never resolved).
    partner_stamps: list[tuple[float, float]]
    own_claims: list[list[tuple[float, float]]]

    def partner_separations_m(self, affine: np.ndarray) -> list[float]:
        """Per-partner stamp separation under a candidate pose."""
        corners = [
            (0.0, 0.0),
            (float(self.width), 0.0),
            (float(self.width), float(self.height)),
            (0.0, float(self.height)),
        ]

        def world(px: float, py: float) -> tuple[float, float]:
            return (
                affine[0, 0] * px + affine[0, 1] * py + affine[0, 2],
                affine[1, 0] * px + affine[1, 1] * py + affine[1, 2],
            )

        separations: list[float] = []
        for (lon, lat), claims in zip(self.partner_stamps, self.own_claims):
            if claims:
                distance = min(
                    haversine_m(lat, lon, w[1], w[0])
                    for w in (world(px, py) for px, py in claims)
                )
            else:
                # No read of this neighbor's number: the neighbor's stamp sits
                # on the shared seam, so it must at least touch the candidate
                # footprint (corner distance is a conservative proxy).
                distance = max(
                    0.0,
                    min(
                        haversine_m(lat, lon, w[1], w[0])
                        for w in (world(px, py) for px, py in corners)
                    )
                    - self.footprint_diag_m(affine) / 2.0,
                )
            separations.append(distance)
        return separations

    def separation_m(self, affine: np.ndarray) -> float | None:
        """Best-partner stamp separation under a candidate pose, or None.

        The minimum over partners: the permissive HARD-gate statistic. A pose
        inconsistent with every partner is certainly not where the neighbors
        say the page belongs, and a single junk partner cannot veto the true
        pose.
        """
        separations = self.partner_separations_m(affine)
        return min(separations) if separations else None

    def median_separation_m(self, affine: np.ndarray) -> float | None:
        """Median-partner stamp separation: the strict CORROBORATION statistic.

        Genuine hints agree — a true pose satisfies essentially all of them —
        while junk hints (single-digit reciprocation) scatter across the
        volume, so no wrong pose can satisfy most. Nashville p8's 325 ft
        candidate matched ONE of its four junk stamps at 66 m (min) but its
        median was far out; requiring the median keeps relaxed adoption from
        publishing it.
        """
        separations = self.partner_separations_m(affine)
        if not separations:
            return None
        return float(statistics.median(separations))

    def footprint_diag_m(self, affine: np.ndarray) -> float:
        tl = (affine[0, 2], affine[1, 2])
        br = (
            affine[0, 0] * self.width + affine[0, 1] * self.height + affine[0, 2],
            affine[1, 0] * self.width + affine[1, 1] * self.height + affine[1, 2],
        )
        return haversine_m(tl[1], tl[0], br[1], br[0])


def load_stamp_gate(
    volume: Path, stem: str, width: int, height: int
) -> StampGate | None:
    """The stamp-consistency gate for a demoted page, or None without hints."""
    hint_path = volume / f"{stem}.contradiction.json"
    adjacency_path = volume / "adjacency.json"
    if not hint_path.exists() or not adjacency_path.exists():
        return None
    try:
        hint = json.loads(hint_path.read_text())
        adjacency = json.loads(adjacency_path.read_text())
    except (OSError, ValueError):
        return None
    detections = adjacency.get("pages", {}).get(stem, {}).get("detections", [])
    partner_stamps: list[tuple[float, float]] = []
    own_claims: list[list[tuple[float, float]]] = []
    seen: set[str] = set()
    for partner in hint.get("partners", []):
        partner_stem = partner.get("stem")
        stamp = partner.get("stamp")
        if not isinstance(stamp, list) or len(stamp) != 2:
            continue
        want = (page_key(partner_stem or "") or "").upper()
        claims = [
            (det["x_frac"] * width, det["y_frac"] * height)
            for det in detections
            if det.get("claim")
            and str(det.get("key") or det.get("number") or "").upper() == want
        ]
        marker = f"{partner_stem}:{stamp[0]}:{stamp[1]}"
        if marker in seen:
            continue
        seen.add(marker)
        partner_stamps.append((float(stamp[0]), float(stamp[1])))
        own_claims.append(claims)
    if not partner_stamps:
        return None
    return StampGate(width, height, partner_stamps, own_claims)


def contradiction_centers(volume: Path, stem: str) -> list[tuple[float, float]]:
    """Partner-stamp search centers for a demoted page, or [] (snap rescue hook).

    Each stamp is the trusted neighbor's printed claim of this page, mapped
    through the NEIGHBOR's fit — a world point on the shared seam the page
    must sit against.
    """
    path = volume / f"{stem}.contradiction.json"
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [
        (float(p["stamp"][0]), float(p["stamp"][1]))
        for p in doc.get("partners", [])
        if isinstance(p.get("stamp"), list) and len(p["stamp"]) == 2
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demote fits that contradict their printed mutual-adjacency claims."
    )
    parser.add_argument("volume", type=Path, help="Volume directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report verdicts without renaming files."
    )
    args = parser.parse_args()

    adjacency_path = args.volume / "adjacency.json"
    if not adjacency_path.exists():
        print(f"No {adjacency_path}; nothing to gate.", file=sys.stderr)
        return
    adjacency = json.loads(adjacency_path.read_text())
    pages = load_fitted_pages(args.volume, adjacency)
    verdicts = arbitrate_suspects(adjacency, pages)
    if not verdicts:
        print("No gated contradictions.", file=sys.stderr)
        return
    for verdict in verdicts:
        partners = sorted({p["stem"] for p in verdict.partners})
        print(
            f"{'Would demote' if args.dry_run else 'Demoted'} {verdict.stem}: "
            f"contradicts {', '.join(partners)} "
            f"[{verdict.reason}; {verdict.signal}]",
            file=sys.stderr,
        )
        if not args.dry_run:
            demote(args.volume, pages[verdict.stem], verdict)


if __name__ == "__main__":
    main()
