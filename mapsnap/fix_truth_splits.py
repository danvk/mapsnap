"""Repair OIM-exported SvgSelectors for split pages (OIM#402 / mapsnap#188).

OIM's IIIF export writes a split page's GCP resourceCoords in the FULL parent
image's pixel frame but its SvgSelector polygon in the CROPPED split image's
frame — one annotation, two coordinate systems. Projecting such a selector
through the annotation's own transform lands the split's footprint translated by
the crop offset: onto its sibling's territory for a 50/50 split. Fifty-five of
the sixty-nine split groups in the truth corpus are affected. `mapsnap compare`
is immune (it pairs and grades splits via the template-matched
``oim/pN.panels.json`` regions), but everything that projects the selector —
``mapsnap score``'s land weighting, the key-map region truth builder, the volume
viewer's truth footprints — inherits the translation.

The crop offset is recovered exactly, offline, from artifacts every affected
volume already has: ``oim-split-truth`` built each ``oim/pN.panels.json`` ring
as ``panel_polygon(crop image) + (template-matched offset)``, so re-deriving the
crop's panel polygon and subtracting it from the stored ring returns the offset
to the rounding of the file.

A shifted selector is adopted only when it contains strictly more of the
annotation's own GCPs than the original did — the GCPs are in the full frame,
so containment is the frame test. That makes the repair self-validating and
idempotent: an already-correct selector (or a rerun over a fixed file) shifts
containment down or not at all and is left untouched.

    mapsnap fix-truth-splits data/<vol>/main.iiif.json            # dry run
    mapsnap fix-truth-splits data/<vol>/main.iiif.json --write
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid

from mapsnap.compare_iiif_georef import (
    extract_gcps,
    label_split_index,
    parse_svg_polygon,
)
from mapsnap.utils import source_id_to_page_key

# A GCP within this many canvas pixels of the selector counts as inside it;
# GCPs are clicked at street corners, occasionally just outside the drawn edge.
GCP_BUFFER_PX = 50.0


def ring_offset(
    ring: list[list[float]], crop_polygon: np.ndarray
) -> tuple[float, float] | None:
    """The crop's offset in the parent canvas, recovered from a panels.json ring.

    ``oim-split-truth`` wrote the ring as ``crop_polygon + offset`` vertex for
    vertex, so when the shapes still correspond the offset falls out exactly (to
    the file's 0.1 px rounding). If the vertex counts differ — the mask tracer
    changed since the ring was written — fall back to aligning bounding-box
    corners, which is exact for identical masks and within a few pixels
    otherwise; every downstream consumer tolerates far more.
    """
    if not len(crop_polygon):
        return None
    arr = np.asarray(ring, dtype=np.float64)
    if len(arr) > 1 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]  # closed ring
    if len(arr) == len(crop_polygon):
        deltas = arr - crop_polygon
        if float(np.ptp(deltas, axis=0).max()) <= 1.0:
            mean = deltas.mean(axis=0)
            return float(mean[0]), float(mean[1])
    dx = float(arr[:, 0].min() - crop_polygon[:, 0].min())
    dy = float(arr[:, 1].min() - crop_polygon[:, 1].min())
    return dx, dy


# Panel extraction by luminance, inherited from the retired `oim-split-truth`.
# #273 showed this is NOT a sound way to build truth -- on bright scans the
# paper is whiter than OIM's JPEG-dithered mask, so the "non-white" region
# collapses onto ink. It survives here only to re-derive a CROP OFFSET from
# artifacts a volume already has, where the polygon is subtracted from a stored
# ring and small shape errors cancel. Do not use it to build panel geometry;
# `mapsnap oim-panels` reads OIM's published boundaries instead.
WHITE_THRESHOLD = 250  # pixels at or above this are masked-out (not part of the panel)
CLOSE_KERNEL_PX = 25  # close small holes/noise in the panel mask before contouring
APPROX_EPS_FRAC = 0.003  # Douglas-Peucker tolerance as a fraction of contour perimeter


def panel_polygon(split_gray: np.ndarray) -> np.ndarray | None:
    """Largest non-white region of a split image as an (N, 2) [x, y] polygon.

    Pure-white pixels are masked-out (not part of the panel), so the remaining
    shape is the panel's outline in the split image's own pixel frame. Returns
    None if the image has no non-white content.
    """
    mask = ((split_gray < WHITE_THRESHOLD).astype(np.uint8)) * 255
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL_PX, CLOSE_KERNEL_PX), np.uint8)
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    eps = APPROX_EPS_FRAC * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2).astype(float)


def gcp_containment(item: dict, points: list[tuple[float, float]]) -> float:
    """Fraction of the annotation's GCPs inside the (buffered) selector polygon."""
    gcps = extract_gcps(item)
    if not gcps or len(points) < 3:
        return 0.0
    polygon = make_valid(Polygon(points)).buffer(GCP_BUFFER_PX)
    return sum(1 for (px, py), _ in gcps if polygon.contains(Point(px, py))) / len(gcps)


def shifted_selector(
    value: str, offset: tuple[float, float]
) -> tuple[str, list[tuple[float, float]]]:
    """The SvgSelector value with every point translated by ``offset``."""
    points = [(x + offset[0], y + offset[1]) for x, y in parse_svg_polygon(value)]
    rendered = " ".join(f"{x},{y}" for x, y in points)
    return f'<svg><polygon points="{rendered}" /></svg>', points


def fix_annotation_page(doc: dict, oim_dir: Path) -> list[str]:
    """Repair frame-mixed split selectors in place; return a log of actions."""
    log: list[str] = []
    groups: dict[str, list[dict]] = {}
    for page_key, items in annotations_by_source_doc(doc).items():
        splits = [i for i in items if label_split_index(i) is not None]
        if len(splits) >= 2:
            groups[page_key] = splits
    for page_key, splits in sorted(groups.items()):
        panels_path = oim_dir / f"{page_key}.panels.json"
        rings = None
        if panels_path.exists():
            rings = json.loads(panels_path.read_text())["panels"]
        for item in splits:
            index = label_split_index(item)
            crop_path = oim_dir / f"{page_key}__{index}.jpg"
            selector = item["target"].get("selector") or {}
            if selector.get("type") != "SvgSelector":
                continue
            label = f"{page_key} [{index}]"
            if rings is None or index is None or index > len(rings):
                if gcp_containment(item, parse_svg_polygon(selector["value"])) < 0.5:
                    log.append(
                        f"{label}: LOOKS BROKEN but {panels_path.name} is "
                        "missing; cannot repair"
                    )
                continue
            if crop_path.exists():
                crop = np.asarray(Image.open(crop_path).convert("L"))
                polygon = panel_polygon(crop)
                if polygon is None:
                    log.append(
                        f"{label}: no panel polygon in {crop_path.name}; skipped"
                    )
                    continue
                offset = ring_offset(rings[index - 1], polygon)
            else:
                # No crop image (panels.json built from the OIM API rather than
                # oim-split-truth). The ring and the selector then trace the same
                # drawn region in different frames, so align their corners.
                offset = ring_offset(
                    rings[index - 1],
                    np.asarray(parse_svg_polygon(selector["value"]), dtype=np.float64),
                )
            if offset is None:
                continue
            before = gcp_containment(item, parse_svg_polygon(selector["value"]))
            new_value, new_points = shifted_selector(selector["value"], offset)
            after = gcp_containment(item, new_points)
            if after > before:
                selector["value"] = new_value
                log.append(
                    f"{label}: shifted by ({offset[0]:.0f}, {offset[1]:.0f}); "
                    f"gcps inside {before:.0%} -> {after:.0%}"
                )
            elif before < 0.5:
                log.append(
                    f"{label}: still broken (gcps inside {before:.0%}); the "
                    f"recovered offset ({offset[0]:.0f}, {offset[1]:.0f}) did not help"
                )
    return log


def annotations_by_source_doc(doc: dict) -> dict[str, list[dict]]:
    """Group items by parent page key (mirrors compare's annotations_by_source)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in doc.get("items", []):
        source_id = item["target"]["source"].get("id")
        parent_key = source_id_to_page_key(source_id, item.get("label", "")).split(
            "__"
        )[0]
        if parent_key:
            groups[parent_key].append(item)
    return dict(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth", type=Path, help="main.iiif.json to repair")
    parser.add_argument(
        "--oim-dir",
        type=Path,
        help="Directory of OIM split crops and panels.json (default: sibling oim/)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the repaired file in place (default: report only). The "
        "original is kept once as <name>.pre-oim402.json.",
    )
    args = parser.parse_args()

    oim_dir = args.oim_dir if args.oim_dir else args.truth.parent / "oim"
    doc = json.loads(args.truth.read_text())
    log = fix_annotation_page(doc, oim_dir)
    for line in log:
        print(line)
    fixed = sum(1 for line in log if "shifted by" in line)
    unfixable = len(log) - fixed
    print(f"{fixed} selector(s) repaired, {unfixable} problem(s) left", file=sys.stderr)
    if not args.write:
        if fixed:
            print("(dry run; pass --write to apply)", file=sys.stderr)
        return
    if fixed:
        backup = args.truth.with_suffix(".pre-oim402.json")
        if not backup.exists():
            backup.write_text(args.truth.read_text())
        args.truth.write_text(json.dumps(doc, indent=1))
        print(f"wrote {args.truth} (original at {backup.name})", file=sys.stderr)


if __name__ == "__main__":
    main()
