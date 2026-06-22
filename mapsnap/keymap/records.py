"""Shared helpers for key-map page-number detections: page specs, paths, and records.

Detections are written to a ``<stem>.keymap.json`` sidecar in the same schema as
mapsnap.detect_text (so the debugger app loads them); these helpers build that schema and
parse the volume's valid page-number set.
"""

from pathlib import Path

import numpy as np

from mapsnap.streets import polygon_side_lengths
from mapsnap.utils import image_stem


def parse_page_spec(spec: str) -> list[int]:
    """Parse a page-number spec like '1-64', '451-577', or '1,3,5-8' into a sorted list."""
    numbers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            numbers.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            numbers.add(int(part))
    return sorted(numbers)


def keymap_path(image_path: str) -> str:
    """Return the path for the detections file (<stem>.keymap.json)."""
    return str(Path(image_path).parent / (image_stem(image_path) + ".keymap.json"))


def filter_args(argv: list[str], image: str) -> list[str]:
    """Compact a long argv by dropping every .jpg argument except ``image``."""
    return [arg for arg in argv if not arg.endswith(".jpg") or arg == image]


def detection_record(bbox: list[list[float]], text: str, confidence: float) -> dict:
    """Build one detection dict in the .keymap.json (.streets.json) format.

    ``bbox`` is a quadrilateral (four [x, y] corners). The record matches the fields
    mapsnap.detect_text writes: polygon, text, confidence, angle (always 0 here, since the
    numbers are upright), long_side, short_side, and dir_pix (orientation of the longer
    side, radians in [0, pi)).
    """
    polygon = [[int(x), int(y)] for x, y in bbox]
    sides = polygon_side_lengths(polygon)
    pts = np.array(polygon, dtype=float)
    edge_vecs = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    long_vec = max(edge_vecs, key=np.linalg.norm)
    return {
        "polygon": polygon,
        "text": text,
        "confidence": round(float(confidence), 4),
        "angle": 0,
        "long_side": round(max(sides), 1),
        "short_side": round(min(sides), 1),
        "dir_pix": round(float(np.arctan2(long_vec[1], long_vec[0])) % np.pi, 4),
    }
