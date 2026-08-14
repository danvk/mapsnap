"""Read key-map page numbers with the CNN localizer + CRNN recognizer (no CRAFT/EasyOCR).

The CNN localizer (mapsnap.keymap.detect_numbers_cnn) proposes page-number centers at ~99%
recall; the CRNN (mapsnap.keymap.crnn_model) reads the digit string from a crop around each
center. Because the box stays centered on the candidate (no CRAFT box-tightening to drift
off), recall tracks the localizer and recognition is the learned CRNN — which handles the
ornate / low-resolution fonts that defeated CRAFT+EasyOCR.

Writes the same ``<stem>.keymap.json`` schema as the other detectors. The valid page
set is derived from each image's own volume (its ``p*.jpg`` stems, letter suffixes
included); ``--pages`` overrides it. Each decode is snapped to the page key it denotes —
exact match, the unique letter-suffixed key sharing its digits (chicago's printed ``51``
when only ``51N`` exists), or the unique key within edit distance 1 (see snap_to_pages) —
and the narrow-detection minimum-width re-read (which recovers a multi-digit number the
wide strip squished to one digit) is enabled — the re-read only fires when it can validate
the longer number against the page set, since ungated it hallucinates a second digit onto
genuine single-digit pages. If no page set can be derived (a volume with no page images)
the raw CRNN output is kept and no re-read is done.

    uv run python -m mapsnap.keymap.detect_numbers_crnn data/chicago_il_1950_vol_1/raw/p0.jpg
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from mapsnap.keymap.crnn_model import (
    build_crnn,
    central_group,
    ctc_greedy_decode,
    ctc_log_likelihood,
    eval_transform,
    greedy_paths,
    locate_number,
    number_strip,
    strip_crop_box,
)
from mapsnap.keymap.detect_numbers_cnn import (
    DEDUP_WORKING,
    DEFAULT_NMS_DIST,
    DEFAULT_STRIDE,
    DEFAULT_THRESHOLD,
    detect_candidate_centers,
    nms_peaks,
)
from mapsnap.keymap.fit_keymap import (
    collapse_skeleton_keys,
    key_stem,
    volume_page_keys,
)
from mapsnap.keymap.number_model import build_model, select_device
from mapsnap.keymap.records import (
    detection_record,
    filter_args,
    keymap_path,
    page_key_sort,
    parse_page_spec,
)


def volume_pages_for(image_path: str) -> list[str]:
    """The valid page keys for a key-map image, derived from its volume's page images.

    The key map's volume is its parent directory (or grandparent when the image
    sits under ``raw/``); its ``p*.jpg`` stems, letter suffixes included, are the
    page keys reads snap to. Page 0 and its variants (the key map's own sheet)
    are excluded. Empty when the volume has no page images.
    """
    from mapsnap.keymap.pipeline import keymap_volume_dir

    keys = collapse_skeleton_keys(volume_page_keys(keymap_volume_dir(Path(image_path))))
    return sorted(
        (key for key in keys if page_key_sort(key)[0] >= 1), key=page_key_sort
    )


def levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def snap_to_pages(text: str, pages: list[str], max_distance: int = 1) -> str:
    """The valid page key ``text`` denotes, else ``text`` unchanged.

    Three rules, in order:
      1. an exact match wins outright;
      2. a digit-only read whose stem names exactly ONE page key adopts that
         key — the letter comes from the vocabulary (chicago's printed ``51``
         when only ``51N`` exists on disk). With several lettered variants
         (asheville's ``35A``–``35F``) the read is ambiguous and kept raw;
      3. otherwise the nearest key within ``max_distance`` repairs a misread —
         adopted when it is the sole key at that distance, or the sole one
         completing the read's missing leading digit ('14' on a 113–223 sheet
         can only be '114'). Any other tie would be a valid-but-wrong key, so
         the raw read is kept.
    """
    if not text or not pages:
        return text
    if text in pages:
        return text
    if text.isdigit():
        family = [p for p in pages if p != text and key_stem(p) == text]
        if len(family) == 1:
            return family[0]
        if family:
            return text  # ambiguous lettered family: never guess a letter
    else:
        # The mirror rule (#318 nashville): a LETTERED read whose stem names a
        # page in a volume with no lettered variant of it adopts the stem.
        # Nashville prints "4 A" (the A names the volume, 1A) but LoC's page
        # names are bare, so the letter-accurate read "4A" snapped to nothing
        # -- distance-1 repair is ambiguous (4A ~ 4/41/44) -- and p4 lost its
        # location prior, unplacing it. Digits-era truncation accidentally
        # produced the right key; this makes it deliberate.
        stem = key_stem(text)
        if stem in pages and not any(p != stem and key_stem(p) == stem for p in pages):
            return stem
    best_distance = min(levenshtein(text, p) for p in pages)
    if best_distance > max_distance:
        return text
    nearest = [p for p in pages if levenshtein(text, p) == best_distance]
    if len(nearest) == 1:
        return nearest[0]
    completions = [p for p in nearest if p.endswith(text)]
    return completions[0] if len(completions) == 1 else text


@torch.no_grad()
def read_candidates(
    image: np.ndarray,
    centers: list[tuple[float, float]],
    factor: float,
    crnn: torch.nn.Module,
    device,
    *,
    batch_size: int = 256,
) -> tuple[list[tuple[float, list[int]]], list[np.ndarray]]:
    """Per candidate: (confidence, CTC path) plus the grayscale strips.

    The path is segmented into number clusters downstream (central_group); decoding and
    boxing use only the cluster nearest the strip center, so a neighbor caught in the wide
    crop is dropped.
    """
    transform = eval_transform()
    strips = [number_strip(image, cx, cy, factor) for cx, cy in centers]
    results: list[tuple[float, list[int]]] = []
    crnn.eval()
    for start in range(0, len(strips), batch_size):
        batch = strips[start : start + batch_size]
        tensors = torch.stack(
            [cast(torch.Tensor, transform(strip)) for strip in batch]
        ).to(device)
        log_probs = crnn(tensors)  # (T, N, C)
        paths = greedy_paths(log_probs)
        confidences = log_probs.exp().max(dim=2).values.mean(dim=0).cpu().numpy()
        results.extend((float(conf), path) for conf, path in zip(confidences, paths))
    return results, strips


# Vertical offsets (working px) each candidate is read at; the best read wins.
# The recognizer reads a glyph-centered strip near-perfectly on every volume
# probed (chicago's 8N, nashville's 12A, hudson's 92) but degrades on the
# localizer's off-center candidates, which ride the CNN's patch grid and land
# up to ~17 working px from the glyph -- mostly BELOW it. Stretching training
# augmentation over that range traded suffix-letter accuracy for offset
# robustness in every combination tried (a black-bar affine fill erased the
# superscripts; a paper fill degraded everything), so the offset problem is
# solved at inference instead: deliver the model a centered strip.
READ_OFFSETS_WORKING = (0.0, -8.0, 8.0)


@torch.no_grad()
def read_best_offset(
    image: np.ndarray,
    centers: list[tuple[float, float]],
    factor: float,
    crnn: torch.nn.Module,
    device,
    *,
    pages: list[str],
    valid_pages: set[str],
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, list[int]]],
    list[np.ndarray],
]:
    """Read every candidate at each READ_OFFSETS_WORKING and keep its best read.

    Best = highest (snaps-to-valid-page, raw length, confidence); a read that
    names a real page beats a longer invalid one, so a well-centered offset
    that sees the whole number wins over one that catches a fragment or merges
    a neighbor. Ties keep the unshifted read (it is first). Returns the chosen
    (center, read, strip) triple per candidate -- the shifted center, so the
    located box and any downstream re-reads use the geometry the read came
    from.
    """
    variants = []
    for offset in READ_OFFSETS_WORKING:
        shifted = [(cx, cy + offset / factor) for cx, cy in centers]
        reads, strips = read_candidates(image, shifted, factor, crnn, device)
        variants.append((shifted, reads, strips))

    def quality(read: tuple[float, list[int]]) -> tuple[int, int, float]:
        confidence, path = read
        group = central_group(path)
        if group is None:
            return (0, 0, 0.0)
        raw = ctc_greedy_decode(path[group[0] : group[1] + 1])
        snapped = snap_to_pages(raw, pages)
        return (2 if snapped in valid_pages else 1, len(raw), confidence)

    best_centers, best_reads, best_strips = [], [], []
    for i in range(len(centers)):
        shifted, reads, strips = max(
            variants, key=lambda variant: quality(variant[1][i])
        )
        best_centers.append(shifted[i])
        best_reads.append(reads[i])
        best_strips.append(strips[i])
    return best_centers, best_reads, best_strips


# Half-widths (working px) for the narrow-detection re-read, tighter than the default
# BOX_HALF_W_WORKING=55 so a squished multi-digit number resolves into separate digits.
REREAD_HALF_WIDTHS_WORKING = [30.0, 38.0]


@torch.no_grad()
def reread_narrow(
    image: np.ndarray,
    crnn: torch.nn.Module,
    device,
    center: tuple[float, float],
    factor: float,
    pages: list[str],
    half_w_working: float,
) -> tuple[list[list[int]], str, float] | None:
    """Re-read the central number around ``center`` from a tighter, minimum-width crop.

    The default strip is wide enough that a multi-digit number is squished until the CRNN
    fires only its central digit; a tighter crop resolves the full number. Returns the same
    (polygon, text, confidence) triple as the main pass, or None if nothing decodes.
    """
    height, width = image.shape[:2]
    cx, cy = center
    strip = number_strip(image, cx, cy, factor, half_w_working=half_w_working)
    tensor = cast(torch.Tensor, eval_transform()(strip)).unsqueeze(0).to(device)
    log_probs = crnn(tensor)
    path = greedy_paths(log_probs)[0]
    confidence = float(log_probs.exp().max(dim=2).values.mean())
    group = central_group(path)
    if group is None:
        return None
    text = snap_to_pages(ctc_greedy_decode(path[group[0] : group[1] + 1]), pages)
    if not text:
        return None
    crop_box = strip_crop_box(
        width, height, cx, cy, factor, half_w_working=half_w_working
    )
    polygon = locate_number(strip, group, len(path), crop_box)
    return polygon, text, confidence


# A tight-crop re-read result: (polygon, text, confidence), matching reread_narrow's return.
RereadResult = tuple[list[list[int]], str, float]


def choose_reread(
    text: str, rereads: list[RereadResult | None], valid_pages: set[str]
) -> RereadResult | None:
    """Pick a longer page number that every tight-crop re-read agrees on, else None (keep text).

    A single-digit read is often a multi-digit number the wide strip squished until the CRNN
    fired only its central digit (e.g. the "0" of "105", the "1" of "61"). Re-reading from
    minimum-width crops resolves it — but only when the crops *agree*: a genuine multi-digit
    label reads the same from every tight crop, whereas a hallucinated second digit on a real
    single-digit page is unstable across crop widths ("9" -> "69" at one width, "61" at
    another). So a longer page is accepted only when every re-read is a strictly-longer valid
    page and they all decode the same text; a lone width or any disagreement keeps the original
    single digit. Requiring agreement (rather than a confidence threshold, which is not
    comparable across crop widths) guards against upgrading a real single-digit page to a
    valid-but-wrong multi-digit one. Returns the highest-confidence instance of the agreed page.
    """
    longer = [
        r
        for r in rereads
        if r is not None and len(r[1]) > len(text) and r[1] in valid_pages
    ]
    if len(longer) != len(rereads) or len({r[1] for r in longer}) != 1:
        return None
    return max(longer, key=lambda r: r[2])


def choose_duplicate_reread(
    text: str, rereads: list[RereadResult | None], missing_pages: set[str]
) -> RereadResult | None:
    """Pick a missing page that every tight-crop re-read of a duplicated label agrees on.

    A label appearing twice is either a genuine split page (its number is printed on both
    blocks — Champaign) or a misread of a similar number (62 and 63 both squished to "6";
    80 misread as the already-found 30). The genuine case re-reads as the same text from
    the tight crops, so it is left alone. A relabel is accepted only when every re-read
    decodes the same different page that is *missing* from the detected set — filling a
    hole in the page coverage rather than shuffling labels — mirroring choose_reread's
    agreement gate.
    """
    different = [
        r for r in rereads if r is not None and r[1] != text and r[1] in missing_pages
    ]
    if len(different) != len(rereads) or len({r[1] for r in different}) != 1:
        return None
    return max(different, key=lambda r: r[2])


# Conflict-repair gates (measured on the 23-sheet truth corpus, 2026-07-28: +19
# key-correct, zero regressions; each gate is load-bearing — see the PR).
REPAIR_FLOOR = -1.5  # min margin for adopting a missing page
ADOPT_DELTA = 1.0  # adopted margin must come within this of the read's own margin
SPATIAL_NEIGHBORS = 4  # nearest detections consulted for the plausibility gate
SPATIAL_GATE = 6.0  # adopted stem within this multiple of the median neighbor delta
WINDOW_PAD = 2  # timesteps of context around the central group when scoring


@torch.no_grad()
def stem_margins(
    image: np.ndarray,
    center: tuple[float, float],
    factor: float,
    crnn: torch.nn.Module,
    device,
    stems: list[str],
) -> dict[str, float]:
    """CTC-margin score per digit stem for the crop(s) around ``center``.

    Each stem is scored in the central-group window of the wide strip and both
    narrow re-read strips: margin = log P(stem | window) minus the window's
    best-path log-probability, so scores compare across windows; a stem's
    margin is its best across the windows. ~0 means the stem explains the ink
    as well as anything can; strongly negative means it does not fit.
    """
    cx, cy = center
    margins: dict[str, float] = {}
    for half_w in (None, *REREAD_HALF_WIDTHS_WORKING):
        kwargs = {} if half_w is None else {"half_w_working": half_w}
        strip = number_strip(image, cx, cy, factor, **kwargs)
        tensor = cast(torch.Tensor, eval_transform()(strip)).unsqueeze(0).to(device)
        log_probs = crnn(tensor)[:, 0, :].cpu().numpy().astype(np.float64)
        path = log_probs.argmax(axis=1).tolist()
        group = central_group(path)
        if group is None:
            continue
        t0 = max(0, group[0] - WINDOW_PAD)
        t1 = min(len(path) - 1, group[1] + WINDOW_PAD)
        window = log_probs[t0 : t1 + 1]
        best_path = float(window.max(axis=1).sum())
        for stem in stems:
            margin = ctc_log_likelihood(window, stem) - best_path
            if stem not in margins or margin > margins[stem]:
                margins[stem] = margin
    return margins


def plan_repairs(
    texts: list[str],
    centers: list[tuple[float, float]],
    margins: list[dict[str, float]],
    pages: list[str],
) -> list[tuple[int, str]]:
    """Which detections to relabel to which missing pages (pure decision logic).

    A detection is suspect when its text duplicates another detection's (the
    weakest-margin instances of the group) or is no page key at all. Each
    suspect adopts the best-margin *missing* page that clears every gate:
    margin above REPAIR_FLOOR, within ADOPT_DELTA of the read's own margin (a
    read that strongly IS its text stays a duplicate — one instance is right),
    a different digit stem (same-stem alternatives score identically, so there
    is no evidence to move on), and a stem plausible among the spatial
    neighbors' stems. Returns (index, new text) pairs.
    """
    valid = set(pages)
    stems_in_vocab = {key_stem(p) for p in pages}

    def own_margin(i: int) -> float:
        return margins[i].get(key_stem(texts[i]), -1e9)

    # Spatial context: median |stem delta| between nearby detections.
    def neighbor_stems(i: int) -> list[int]:
        distances = sorted(
            (
                (centers[j][0] - centers[i][0]) ** 2
                + (centers[j][1] - centers[i][1]) ** 2,
                j,
            )
            for j in range(len(texts))
            if j != i
        )
        stems = []
        for _, j in distances[:SPATIAL_NEIGHBORS]:
            stem = key_stem(texts[j])
            if stem.isdigit():
                stems.append(int(stem))
        return stems

    deltas = []
    for i in range(len(texts)):
        stem = key_stem(texts[i])
        if not stem.isdigit():
            continue
        deltas.extend(abs(int(stem) - n) for n in neighbor_stems(i))
    median_delta = max(1.0, float(np.median(deltas))) if deltas else 1.0

    def spatially_ok(i: int, page: str) -> bool:
        stems = neighbor_stems(i)
        if not stems:
            return True
        return (
            abs(int(key_stem(page)) - float(np.median(stems)))
            <= SPATIAL_GATE * median_delta
        )

    counts = Counter(texts)
    missing = [p for p in pages if counts[p] == 0]
    groups: dict[str, list[int]] = {}
    for i, text in enumerate(texts):
        groups.setdefault(text, []).append(i)

    repairs: list[tuple[int, str]] = []
    for text, members in sorted(groups.items()):
        if text not in valid and key_stem(text) not in stems_in_vocab:
            suspects = members  # not a page key: every instance is suspect
        elif len(members) > 1 and text in valid:
            keeper = max(members, key=own_margin)
            suspects = [i for i in members if i != keeper]
        else:
            continue
        for i in suspects:
            options = [
                (margins[i].get(key_stem(page), -1e9), page)
                for page in missing
                if key_stem(page) != key_stem(text)
                and margins[i].get(key_stem(page), -1e9) > REPAIR_FLOOR
                and spatially_ok(i, page)
            ]
            if not options:
                continue
            best_margin, best_page = max(options)
            if text in valid and best_margin < own_margin(i) - ADOPT_DELTA:
                continue
            repairs.append((i, best_page))
            missing.remove(best_page)
    return repairs


def repair_conflicts(
    image: np.ndarray,
    crnn: torch.nn.Module,
    device,
    found: list[tuple[list[list[int]], str, float]],
    *,
    factor: float,
    pages: list[str],
    image_name: str,
) -> list[tuple[list[list[int]], str, float]]:
    """Relabel duplicated / invalid reads to missing pages by CTC-margin evidence.

    Runs after resolve_duplicate_labels: that pass relabels only when two
    narrow re-reads agree, which misses conflicts whose evidence sits in the
    wide window (KC's 478 read as a duplicate '470': narrow crops disagree,
    but '478' is the wide window's best interpretation by a wide margin).
    This pass scores every still-missing page's stem directly against the
    CRNN's own likelihoods and adopts under plan_repairs' gates.
    """
    if not found:
        return found
    texts = [text for _, text, _ in found]
    centers = [
        (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
        for poly, _, _ in found
    ]
    counts = Counter(texts)
    valid = set(pages)
    stems_in_vocab = {key_stem(p) for p in pages}

    # Margins are only needed for detections in a conflicted group.
    def conflicted(text: str) -> bool:
        if text not in valid and key_stem(text) not in stems_in_vocab:
            return True
        return counts[text] > 1 and text in valid

    stems = sorted({key_stem(p) for p in pages})
    margins: list[dict[str, float]] = [
        stem_margins(image, centers[i], factor, crnn, device, stems)
        if conflicted(texts[i])
        else {}
        for i in range(len(found))
    ]
    repaired = list(found)
    for i, new_text in plan_repairs(texts, centers, margins, pages):
        polygon, old_text, confidence = found[i]
        print(
            f"{image_name}: conflict repair {old_text!r} -> {new_text!r}",
            file=sys.stderr,
        )
        repaired[i] = (polygon, new_text, confidence)
    return repaired


def resolve_duplicate_labels(
    image: np.ndarray,
    crnn: torch.nn.Module,
    device,
    found: list[tuple[list[list[int]], str, float]],
    *,
    factor: float,
    pages: list[str],
    image_name: str,
) -> list[tuple[list[list[int]], str, float]]:
    """Re-read every detection whose label appears more than once, relabeling misreads.

    Runs after the main pass, so the set of still-missing page numbers is known; a
    duplicate is relabeled only under choose_duplicate_reread's agreement-on-a-missing-page
    gate, which leaves genuine split-page duplicates untouched.
    """
    counts = Counter(text for _, text, _ in found)
    missing = set(pages) - set(counts)
    resolved: list[tuple[list[list[int]], str, float]] = []
    for polygon, text, confidence in found:
        if counts[text] >= 2 and missing:
            cx = sum(point[0] for point in polygon) / len(polygon)
            cy = sum(point[1] for point in polygon) / len(polygon)
            rereads = [
                reread_narrow(image, crnn, device, (cx, cy), factor, pages, hw)
                for hw in REREAD_HALF_WIDTHS_WORKING
            ]
            chosen = choose_duplicate_reread(text, rereads, missing)
            if chosen is not None:
                print(
                    f"{image_name}: duplicate {text!r} re-read -> {chosen[1]!r}",
                    file=sys.stderr,
                )
                polygon, text, confidence = chosen
                missing.discard(text)
        resolved.append((polygon, text, confidence))
    return resolved


def detect_and_read(
    image_path: str,
    cnn: torch.nn.Module,
    crnn: torch.nn.Module,
    device,
    *,
    stride: int,
    threshold: float,
    nms_dist: float,
    pages: list[str],
    disable_reread: bool = False,
    disable_repair: bool = False,
) -> list[dict]:
    """CNN-localize then CRNN-read one image; write <stem>.keymap.json.

    ``disable_reread`` turns off the tight-crop re-reads (the narrow-detection
    minimum-width re-read and the duplicate-label re-read) and ``disable_repair``
    the CTC-margin conflict repair — ablation switches for measuring those
    steps' effect; all are on by default.
    """
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    centers, factor = detect_candidate_centers(
        image, cnn, device, stride=stride, threshold=threshold, nms_dist=nms_dist
    )
    valid_pages = set(pages)
    centers, reads, strips = read_best_offset(
        image, centers, factor, crnn, device, pages=pages, valid_pages=valid_pages
    )

    # Decode and box only the central number of each crop, dropping a neighbor caught in the
    # wide window.
    found: list[tuple[list[list[int]], str, float]] = []
    for (cx, cy), (confidence, path), strip in zip(centers, reads, strips):
        group = central_group(path)
        if group is None:
            continue  # CRNN emitted nothing -> no number here
        raw = ctc_greedy_decode(path[group[0] : group[1] + 1])
        text = snap_to_pages(raw, pages)
        if not text:
            continue
        if text != raw:
            print(
                f"{Path(image_path).name}: snapped {raw!r} -> {text!r} "
                f"(edit distance {levenshtein(raw, text)})",
                file=sys.stderr,
            )
        crop_box = strip_crop_box(width, height, cx, cy, factor)
        polygon = locate_number(strip, group, len(path), crop_box)
        # A single-digit read is often a multi-digit number the wide strip squished to one
        # digit; re-read from minimum-width crops and accept a longer valid page only when the
        # crops agree (see choose_reread — the agreement gate guards against upgrading a genuine
        # single-digit page to a valid-but-wrong multi-digit one).
        if len(text) == 1 and valid_pages and not disable_reread:
            rereads = [
                reread_narrow(image, crnn, device, (cx, cy), factor, pages, hw)
                for hw in REREAD_HALF_WIDTHS_WORKING
            ]
            chosen = choose_reread(text, rereads, valid_pages)
            if chosen is not None:
                polygon, widened, confidence = chosen
                print(
                    f"{Path(image_path).name}: re-read narrow {text!r} -> {widened!r}",
                    file=sys.stderr,
                )
                text = widened
        found.append((polygon, text, confidence))

    # Trimming a between-two-numbers candidate can duplicate a neighbor's detection; keep the
    # higher-confidence box of any that now sit on the same number.
    if found:
        box_centers = [
            (sum(p[0] for p in poly) / 4, sum(p[1] for p in poly) / 4)
            for poly, _, _ in found
        ]
        scores = [confidence for _, _, confidence in found]
        keep = nms_peaks(box_centers, scores, round(DEDUP_WORKING / factor))
        found = [found[k] for k in keep]

    # A label appearing twice may be a misread of a similar still-missing number; re-read
    # duplicates from tight crops (genuine split-page duplicates re-read unchanged).
    if valid_pages and not disable_reread:
        found = resolve_duplicate_labels(
            image,
            crnn,
            device,
            found,
            factor=factor,
            pages=pages,
            image_name=Path(image_path).name,
        )

    # Remaining duplicated or invalid reads: adopt a missing page when the CRNN's
    # own likelihoods say it is the better interpretation (see repair_conflicts).
    if valid_pages and not disable_repair:
        found = repair_conflicts(
            image,
            crnn,
            device,
            found,
            factor=factor,
            pages=pages,
            image_name=Path(image_path).name,
        )

    detections = [
        detection_record(cast(list, polygon), text, confidence)
        for polygon, text, confidence in found
    ]

    doc = {
        "width": width,
        "height": height,
        "timestamp": datetime.now(UTC).isoformat(),
        "command": filter_args(sys.argv[:], image_path),
        "streets": detections,
    }
    with open(keymap_path(image_path), "w") as f:
        json.dump(doc, f, indent=2)
    print(
        f"{Path(image_path).name}: {len(centers)} candidates -> {len(detections)} read",
        file=sys.stderr,
    )
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read page numbers with the CNN localizer + CRNN recognizer."
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE")
    parser.add_argument(
        "--pages",
        metavar="SPEC",
        help=(
            "Valid page keys (e.g. '1-111' or '1-33,33A,33B'), overriding the set "
            "derived from each image's volume (its p*.jpg stems)."
        ),
    )
    parser.add_argument(
        "--cnn-weights", type=Path, default=Path("models/number_detector.pt")
    )
    parser.add_argument(
        "--crnn-weights", type=Path, default=Path("models/number_crnn.pt")
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--nms-dist", type=float, default=DEFAULT_NMS_DIST)
    parser.add_argument(
        "--disable-reread",
        action="store_true",
        help="Turn off the narrow-detection and duplicate-label re-reads (ablation).",
    )
    parser.add_argument(
        "--disable-repair",
        action="store_true",
        help="Turn off the CTC-margin conflict repair of duplicated/invalid reads (ablation).",
    )
    args = parser.parse_args()

    device = select_device()
    cnn = build_model(pretrained=False)
    cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))
    cnn.to(device)
    crnn = build_crnn()
    crnn.load_state_dict(torch.load(args.crnn_weights, map_location=device))
    crnn.to(device)

    override = parse_page_spec(args.pages) if args.pages else None
    for image_path in args.images:
        # Derived per image, so one invocation spanning volumes never mixes
        # their page sets.
        pages = override if override is not None else volume_pages_for(image_path)
        if not pages:
            print(
                f"{Path(image_path).name}: no page keys found in its volume; "
                "keeping raw decodes (pass --pages to snap them).",
                file=sys.stderr,
            )
        detect_and_read(
            image_path,
            cnn,
            crnn,
            device,
            stride=args.stride,
            threshold=args.threshold,
            nms_dist=args.nms_dist,
            pages=pages,
            disable_reread=args.disable_reread,
            disable_repair=args.disable_repair,
        )


if __name__ == "__main__":
    main()
