"""Derive a split panel's CRAFT boxes from its parent page's boxes (#361).

Splitting used to happen before detection, so every splitter change invalidated
the corpus's most expensive vision pass. Deriving panel boxes from the parent's
instead makes CRAFT a run-once-per-page step: detection happens only on full
pages (and key-map raws), and split panels get their ``boxes.json`` by a pure
coordinate remap — per-angle unrotate to the parent frame, majority-panel
assignment, crop offset, re-rotate into the panel's frame.

The schenectady end-to-end A/B established one hard requirement: ``free_list``
quads MUST be transformed point-by-point. Quads carry the text's long-axis
orientation (``dir_pix``), which the georef uses to extrapolate label
crossings; collapsing them to axis-aligned bboxes preserved every read yet
cost 3.0 points of volume score through degraded GCP geometry. With exact quad
transforms the remap is score-neutral (78.9 vs 78.8 on the corpus's most-split
volume), and box-level analysis favors the parent side: full pages find ~9%
more boxes inside panel regions than panel-image detection does, while only
0.13% of parent boxes straddle a seam (majority assignment covers them).
"""

import json
import math
from pathlib import Path

from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

Point = tuple[float, float]

# A box belongs to the panel holding at least this share of its area. Measured
# corpus-wide, 0 of 5,639 parent boxes on split pages failed to reach it.
MIN_PANEL_SHARE = 0.3


def unrotate_point(
    x: float, y: float, angle: int, rotated_w: int, rotated_h: int
) -> Point:
    """Rotated-frame (PIL CCW, expand=True) coordinates back to the original frame."""
    if angle == 0:
        return (x, y)
    if angle == 90:
        return (rotated_h - 1 - y, x)
    return (y, rotated_w - 1 - x)  # 270


def rotate_point(x: float, y: float, angle: int, width: int, height: int) -> Point:
    """Original-frame coordinates into the angle's rotated frame (inverse of unrotate)."""
    if angle == 0:
        return (x, y)
    if angle == 90:
        return (y, width - 1 - x)
    return (height - 1 - y, x)  # 270


def panel_frame(
    panel_polygon: Polygon, panel_size: tuple[int, int], parent_size: tuple[int, int]
) -> Point:
    """The panel image's origin in parent coordinates.

    The first panel keeps the parent's full canvas (offset zero, non-panel area
    masked); later panels are cropped to their polygon's bounding box.
    """
    if panel_size == parent_size:
        return (0.0, 0.0)
    return (math.floor(panel_polygon.bounds[0]), math.floor(panel_polygon.bounds[1]))


def derive_panel_boxes(
    parent_doc: dict,
    panel_polygons: list[Polygon],
    panel_index: int,
    panel_size: tuple[int, int],
) -> dict:
    """A panel's boxes.json document, derived from its parent's.

    ``panel_index`` is 1-based (panel ``pN__i``). Rects stay rects (axis
    alignment survives 90-degree rotations); quads transform point-by-point.
    Boxes whose majority area lies in a different panel are omitted here and
    appear in that panel's document instead.
    """
    parent_w, parent_h = parent_doc["width"], parent_doc["height"]
    polygon = panel_polygons[panel_index - 1]
    width, height = panel_size
    offset_x, offset_y = panel_frame(polygon, panel_size, (parent_w, parent_h))

    def clamp(x: float, y: float) -> Point:
        return (
            min(max(x - offset_x, 0.0), width - 1),
            min(max(y - offset_y, 0.0), height - 1),
        )

    def owner(original_points: list[Point]) -> int | None:
        xs = [p[0] for p in original_points]
        ys = [p[1] for p in original_points]
        bounds = shapely_box(min(xs), min(ys), max(xs), max(ys))
        if bounds.area <= 0:
            return None
        best, best_share = None, 0.0
        for i, poly in enumerate(panel_polygons, 1):
            share = bounds.intersection(poly).area / bounds.area
            if share > best_share:
                best, best_share = i, share
        return best if best_share >= MIN_PANEL_SHARE else None

    angle_entries = []
    for entry in parent_doc["boxes"]:
        angle = entry["angle"]
        rotated_w, rotated_h = (
            (parent_w, parent_h) if angle == 0 else (parent_h, parent_w)
        )
        horizontal: list[list[int]] = []
        free: list[list[list[float]]] = []
        for x0, x1, y0, y1 in entry["horizontal_list"]:
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            original = [
                unrotate_point(x, y, angle, rotated_w, rotated_h) for x, y in corners
            ]
            if owner(original) != panel_index:
                continue
            local = [
                rotate_point(*clamp(x, y), angle, width, height) for x, y in original
            ]
            rx0, rx1 = int(min(p[0] for p in local)), int(max(p[0] for p in local))
            ry0, ry1 = int(min(p[1] for p in local)), int(max(p[1] for p in local))
            if rx1 > rx0 and ry1 > ry0:
                horizontal.append([rx0, rx1, ry0, ry1])
        for quad in entry["free_list"]:
            original = [
                unrotate_point(x, y, angle, rotated_w, rotated_h) for x, y in quad
            ]
            if owner(original) != panel_index:
                continue
            free.append(
                [
                    list(rotate_point(*clamp(x, y), angle, width, height))
                    for x, y in original
                ]
            )
        angle_entries.append(
            {"angle": angle, "horizontal_list": horizontal, "free_list": free}
        )

    return {
        "width": width,
        "height": height,
        "timestamp": parent_doc.get("timestamp"),
        "command": ["derived_from_parent", "#361"],
        "derived_from_parent": True,
        "boxes": angle_entries,
    }


def derive_boxes_for_panel_image(panel_image: str | Path) -> bool:
    """Write ``<panel>.boxes.json`` derived from the parent, if inputs exist.

    Returns True when derivation happened; False when the image is not a panel
    or the parent's boxes / panels sidecars are missing (caller falls back to
    real detection).
    """
    panel_path = Path(panel_image)
    stem = panel_path.stem
    if "__" not in stem:
        return False
    parent_stem, _, index_text = stem.rpartition("__")
    if not index_text.isdigit():
        return False
    parent_boxes = panel_path.parent / f"{parent_stem}.boxes.json"
    panels_json = panel_path.parent / f"{parent_stem}.panels.json"
    if not parent_boxes.exists() or not panels_json.exists():
        return False
    panels_doc = json.loads(panels_json.read_text())
    polygons = [Polygon(ring) for ring in panels_doc["panels"] if len(ring) >= 3]
    index = int(index_text)
    if not 1 <= index <= len(polygons):
        return False
    with Image.open(panel_path) as img:
        panel_size = img.size
    doc = derive_panel_boxes(
        json.loads(parent_boxes.read_text()), polygons, index, panel_size
    )
    (panel_path.parent / f"{stem}.boxes.json").write_text(json.dumps(doc))
    return True
