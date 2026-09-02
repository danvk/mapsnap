"""Find the volume-index insets on a key-map sheet and mask them (#276).

Many key-map sheets carry a "GRAPHIC MAP OF VOLUMES" (Richmond: "GENERAL INDEX
TO VOLUMES"): a second map of the whole city, at a coarser scale, whose
numerals are volume numbers. The page-number detector reads them like any
other number, and the damage is downstream -- Richmond p311 was published
14,713 ft from truth because the inset's "1" was snapped to the sole valid
page within edit distance and became a third search center.

The tell is *distance*, on the reads themselves. Measured over the 23
truth-labeled key-map sheets, clustering reads by single linkage at
EPS_SPACING times the sheet's median nearest-neighbour spacing and flagging a
cluster with at least MIN_READS reads, at most DOMINANCE of the largest
cluster, and a majority of small numbers caught 38 junk reads and masked zero
true page numbers, robustly across the parameter sweep. Ink connectivity was
tried first and does worse: the faint base street grid or the inset's own
border bridges the ink on columbus and los_angeles, but the reads stay
isolated. Ellenville-shaped sheets -- two real key maps side by side (the
village and Napanoch), all small page numbers -- are not flagged because
neither cluster is dominated; they are reported as separate key maps.

Isolation carries the rule; small numbers only corroborate, because
Richmond's inset reads ``2 3 4 311`` and miami's inset numerals 1-5 are also
real pages of that volume. Two corroborations, either of which confirms:

- re-reading the cluster's numerals against a 1..SMALL_MAX vocabulary with no
  edit-distance repair (the "311" glyph reads as "1"), and
- a *specific* cartouche word (GRAPHIC, VOLUMES, INDEX; see cartouche.py)
  inside the cluster's hull.

Output is ``raw/<stem>.inset.panels.json`` in the panels.json shape the
debugger renders (one ring per inset, labelled), plus an ``inset`` section in
the sheet's decision log. ``inset_rings`` is the consumer's entry point.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon, box

from mapsnap.keymap.cartouche import cartouche_sidecar_path, is_specific
from mapsnap.keymap.log import append_keymap_log
from mapsnap.keymap.records import INSET_FLAG, keymap_path, page_key_sort
from mapsnap.utils import image_stem

SMALL_MAX = 15  # a "small number": plausible volume index, 1..15
EPS_SPACING = (
    3.0  # cluster radius, as a multiple of the median nearest-neighbour spacing
)
DOMINANCE = 1 / 3  # a cluster larger than this share of the biggest is part of the map
MIN_READS = 2
SMALL_MAJORITY = 0.5
MARGIN_SPACING = 1.0  # mask margin around the cluster hull, in spacings
# How far from the cluster hull a cartouche title may sit and still belong to
# it: the title is printed above or beside the inset map, not among its
# numerals -- richmond 1.3, detroit 1.3, columbus 1.7, los_angeles 2.2 spacings
# (miami's 4.4 is confirmed by the re-read instead). Only *specific* words
# count, and those never appear elsewhere on a key-map sheet.
CARTOUCHE_SPACINGS = 3.0
SEPARATE_MIN_READS = 5  # a non-dominant cluster this big is a key map of its own
REREAD_MIN_CONFIDENCE = 0.2


@dataclass
class Read:
    """One page-number read on the sheet, in its own pixel frame."""

    text: str
    confidence: float
    polygon: list[list[float]]
    center: tuple[float, float]


@dataclass
class Inset:
    """One flagged cluster and the evidence behind its verdict."""

    indices: list[int]
    texts: list[str]
    n_small: int
    ring: list[list[float]]
    reread: tuple[int, int] | None = None  # (raw decodes that are small, decodes)
    cartouche: list[str] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        """Whether a corroboration backs the distance rule."""
        by_reread = (
            self.reread is not None
            and self.reread[1] > 0
            and (self.reread[0] >= SMALL_MAJORITY * self.reread[1])
        )
        return by_reread or bool(self.cartouche)

    def label(self) -> str:
        """The panels.json label: what was read and what confirmed it."""
        evidence = []
        if self.reread is not None:
            evidence.append(f"re-read small {self.reread[0]}/{self.reread[1]}")
        if self.cartouche:
            evidence.append("cartouche " + ", ".join(self.cartouche))
        return f"volumes inset: {', '.join(self.texts)}" + (
            f" ({'; '.join(evidence)})" if evidence else " (unconfirmed)"
        )


@dataclass
class InsetResult:
    """Everything the detector decided about one sheet."""

    width: int
    height: int
    spacing: float
    insets: list[Inset]
    candidates: list[Inset]  # rule clusters, confirmed or not
    separate_keymaps: bool
    n_reads: int


def is_small(text: str) -> bool:
    """A read that could be a volume index rather than a page number."""
    return text.isdigit() and 1 <= int(text) <= SMALL_MAX


def polygon_center(polygon: list[list[float]]) -> tuple[float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def load_reads(image_path: str | Path) -> tuple[list[Read], int, int]:
    """(reads, width, height) from the sheet's <stem>.keymap.json."""
    doc = json.loads(Path(keymap_path(str(image_path))).read_text())
    reads = [
        Read(
            text=str(d["text"]),
            confidence=float(d.get("confidence", 0.0)),
            polygon=[[float(x), float(y)] for x, y in d["polygon"]],
            center=polygon_center(d["polygon"]),
        )
        for d in doc.get("streets", [])
    ]
    return reads, int(doc["width"]), int(doc["height"])


def median_spacing(centers: list[tuple[float, float]]) -> float:
    """Median nearest-neighbour distance between reads (0 for fewer than two)."""
    if len(centers) < 2:
        return 0.0
    pts = np.asarray(centers, dtype=float)
    dists = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    return float(np.median(dists.min(axis=1)))


def cluster_reads(centers: list[tuple[float, float]], eps: float) -> list[list[int]]:
    """Single-linkage clusters of read indices at radius ``eps``, largest first."""
    n = len(centers)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pts = np.asarray(centers, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.hypot(*(pts[i] - pts[j]))) <= eps:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def cluster_ring(
    reads: list[Read], indices: list[int], margin: float, width: int, height: int
) -> list[list[float]]:
    """Convex hull of the cluster's read polygons, buffered by ``margin`` and clipped."""
    points = [tuple(p) for i in indices for p in reads[i].polygon]
    hull = (
        MultiPoint(points)
        .convex_hull.buffer(margin)
        .intersection(box(0, 0, width, height))
    )
    if not isinstance(hull, Polygon):  # clipping can split a sliver off
        hull = hull.convex_hull
    assert isinstance(hull, Polygon)
    return [[round(x, 1), round(y, 1)] for x, y in hull.exterior.coords]


def cartouche_words_near(
    image_path: str | Path, ring: list[list[float]], radius: float
) -> list[str]:
    """Specific cartouche words within ``radius`` of ``ring`` (from <stem>.cartouche.json)."""
    path = cartouche_sidecar_path(image_path)
    if not path.exists():
        return []
    hull = Polygon(ring)
    words = []
    for read in json.loads(path.read_text()).get("streets", []):
        if not is_specific(read["text"]):
            continue
        if hull.distance(Point(polygon_center(read["polygon"]))) <= radius:
            words.append(read["text"])
    return words


def reread_small_count(
    image: np.ndarray, reads: list[Read], indices: list[int], crnn, device
) -> tuple[int, int]:
    """(raw decodes that are small numbers, decodes) for the cluster's numerals.

    The main pass snaps every read to the nearest valid page number, which is
    exactly how an inset's "1" became a valid "311". Here the CRNN's raw
    greedy decode is kept as is: no vocabulary, no repair.
    """
    from mapsnap.keymap.crnn_model import central_group, ctc_greedy_decode
    from mapsnap.keymap.detect_numbers_crnn import read_candidates
    from mapsnap.keymap.keymap_patches import working_scale

    height, width = image.shape[:2]
    factor = working_scale(width, height)
    centers = [reads[i].center for i in indices]
    results, _ = read_candidates(image, centers, factor, crnn, device)
    n_small = n_decoded = 0
    for confidence, path in results:
        group = central_group(path)
        if group is None or confidence < REREAD_MIN_CONFIDENCE:
            continue
        text = ctc_greedy_decode(path[group[0] : group[1] + 1])
        if not text:
            continue
        n_decoded += 1
        n_small += is_small(text)
    return n_small, n_decoded


def detect_insets(
    image_path: str | Path,
    *,
    reread: bool = False,
    crnn=None,
    device=None,
) -> InsetResult:
    """Apply the distance rule and its corroborations to one sheet's reads."""
    reads, width, height = load_reads(image_path)
    centers = [r.center for r in reads]
    spacing = median_spacing(centers)
    clusters = cluster_reads(centers, EPS_SPACING * spacing) if spacing > 0 else []
    biggest = len(clusters[0]) if clusters else 0
    candidates: list[Inset] = []
    for indices in clusters:
        if len(indices) < MIN_READS or len(indices) > DOMINANCE * biggest:
            continue
        texts = [reads[i].text for i in indices]
        n_small = sum(is_small(t) for t in texts)
        if n_small < SMALL_MAJORITY * len(indices):
            continue
        ring = cluster_ring(reads, indices, MARGIN_SPACING * spacing, width, height)
        inset = Inset(
            indices=indices,
            texts=sorted(texts, key=page_key_sort),
            n_small=n_small,
            ring=ring,
            cartouche=cartouche_words_near(
                image_path, ring, CARTOUCHE_SPACINGS * spacing
            ),
        )
        candidates.append(inset)
    if reread and candidates:
        from PIL import Image

        image = np.asarray(Image.open(image_path).convert("RGB"))
        for inset in candidates:
            inset.reread = reread_small_count(image, reads, inset.indices, crnn, device)
    # Two or more sizeable clusters with no dominant one: separate key maps on
    # one sheet (Ellenville's village + Napanoch), never an inset.
    sizeable = [c for c in clusters if len(c) >= SEPARATE_MIN_READS]
    separate = len(sizeable) >= 2 and all(
        len(c) > DOMINANCE * biggest for c in sizeable
    )
    return InsetResult(
        width=width,
        height=height,
        spacing=spacing,
        insets=[c for c in candidates if c.confirmed],
        candidates=candidates,
        separate_keymaps=separate,
        n_reads=len(reads),
    )


def annotate_keymap_reads(image_path: str | Path, result: InsetResult) -> int:
    """Flag the reads inside the confirmed masks in <stem>.keymap.json; return how many.

    The flag is recomputed from scratch on every run -- set on reads inside a
    confirmed ring, removed from every other read -- so the file always says
    what the current masks say. Consumers skip flagged reads through
    records.is_inset; the reads themselves stay, so the debugger can show what
    was masked and the detector can re-judge them next time.
    """
    path = Path(keymap_path(str(image_path)))
    doc = json.loads(path.read_text())
    rings = [Polygon(inset.ring).buffer(0) for inset in result.insets]
    flagged = 0
    for record in doc.get("streets", []):
        inside = bool(rings) and any(
            ring.contains(Point(polygon_center(record["polygon"]))) for ring in rings
        )
        if inside:
            record[INSET_FLAG] = True
            flagged += 1
        else:
            record.pop(INSET_FLAG, None)
    path.write_text(json.dumps(doc, indent=2))
    return flagged


def inset_sidecar_path(image_path: str | Path) -> Path:
    """``<dir>/<stem>.inset.panels.json`` beside the key-map image."""
    image_path = Path(image_path)
    return image_path.parent / (image_stem(str(image_path)) + ".inset.panels.json")


def write_inset_sidecar(image_path: str | Path, result: InsetResult) -> Path:
    """Write the confirmed insets as a labelled panels.json; return the path."""
    path = inset_sidecar_path(image_path)
    path.write_text(
        json.dumps(
            {
                "image": Path(image_path).name,
                "width": result.width,
                "height": result.height,
                "panels": [inset.ring for inset in result.insets],
                "labels": [inset.label() for inset in result.insets],
            },
            indent=1,
        )
    )
    return path


def inset_rings(image_path: str | Path) -> list[Polygon]:
    """The sheet's confirmed inset masks as polygons, or [] with no sidecar."""
    path = inset_sidecar_path(image_path)
    if not path.exists():
        return []
    return [Polygon(ring).buffer(0) for ring in json.loads(path.read_text())["panels"]]


def log_lines(result: InsetResult) -> list[str]:
    """The decision-log section for one sheet."""
    lines = [
        (
            f"{result.n_reads} reads, median spacing {result.spacing:.0f} px, "
            f"cluster radius {EPS_SPACING * result.spacing:.0f} px"
        )
    ]
    if not result.candidates:
        lines.append("no isolated small-number cluster")
    for inset in result.candidates:
        verdict = "INSET (masked)" if inset.confirmed else "unconfirmed (not masked)"
        lines.append(
            f"cluster of {len(inset.indices)} reads, {inset.n_small} small: "
            + ", ".join(inset.texts)
        )
        if inset.reread is not None:
            lines.append(
                f"  re-read without snapping: {inset.reread[0]}/{inset.reread[1]} small"
            )
        lines.append(
            f"  specific cartouche within {CARTOUCHE_SPACINGS:g} spacings: "
            + (", ".join(inset.cartouche) if inset.cartouche else "none")
        )
        lines.append(f"  → {verdict}")
    if result.separate_keymaps:
        lines.append(
            "several sizeable clusters and no dominant one: separate key maps, nothing masked"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find and mask volume-index insets on key-map sheets (#276)."
    )
    parser.add_argument(
        "images", nargs="+", help="Key-map image(s), e.g. data/<volume>/raw/p0.jpg"
    )
    parser.add_argument(
        "--no-reread",
        action="store_true",
        help="Skip the CRNN re-read corroboration (cartouche words alone then confirm).",
    )
    parser.add_argument(
        "--crnn-weights", type=Path, default=Path("models/number_crnn.pt")
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Run the recognizer on the CPU."
    )
    args = parser.parse_args()

    crnn = device = None
    if not args.no_reread:
        import torch

        from mapsnap.keymap.crnn_model import build_crnn
        from mapsnap.keymap.number_model import select_device

        device = select_device() if not args.no_gpu else torch.device("cpu")
        crnn = build_crnn()
        crnn.load_state_dict(torch.load(args.crnn_weights, map_location=device))
        crnn.to(device)
    for image in args.images:
        result = detect_insets(
            image, reread=not args.no_reread, crnn=crnn, device=device
        )
        path = write_inset_sidecar(image, result)
        flagged = annotate_keymap_reads(image, result)
        append_keymap_log(
            image,
            "inset",
            [
                *log_lines(result),
                f"{flagged} read(s) flagged inset in the key-map reads file",
            ],
        )
        summary = "; ".join(inset.label() for inset in result.insets) or "none"
        print(
            f"{Path(image).name}: {len(result.insets)} inset(s) -> {path.name}: {summary}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
