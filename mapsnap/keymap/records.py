"""Shared helpers for key-map page-number detections: page specs, paths, and records.

Detections are written to a ``<stem>.keymap.json`` sidecar in the same schema as
mapsnap.detect_text (so the debugger app loads them); these helpers build that schema and
parse the volume's valid page-number set.
"""

import json
import re
from pathlib import Path

import numpy as np

from mapsnap.streets import polygon_side_lengths
from mapsnap.utils import image_stem


def page_key_sort(key: str) -> tuple[int, str]:
    """Sort key for page keys: numeric stem first, then letter suffix (35 < 35A < 36)."""
    match = re.match(r"(\d+)([A-Za-z]*)", key)
    if not match:
        return (1 << 31, key)
    return (int(match.group(1)), match.group(2))


def parse_page_spec(spec: str) -> list[str]:
    """Parse a page spec like '1-64', '1,3,5-8', or '1-33,33A,33B' into sorted page keys.

    Ranges expand over bare numbers; a lettered token (``33A``) names one page key.
    Keys are uppercased, so specs built from lowercase disk stems match printed text.
    """
    keys: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            keys.update(str(n) for n in range(min(lo, hi), max(lo, hi) + 1))
        else:
            if not re.fullmatch(r"\d+[A-Za-z]{0,2}", part):
                raise ValueError(f"invalid page token: {part!r}")
            keys.add(part.upper())
    return sorted(keys, key=page_key_sort)


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


KEYMAPS_FILENAME = "keymaps.json"


def write_keymaps_record(volume: Path, keys: list[str]) -> Path:
    """Record the volume's identified key-map page keys in ``keymaps.json``.

    Consumers that must skip key-map sheets used to test for
    ``<stem>.keymap.json``, which only exists once ``mapsnap keymap`` has run.
    Since ``mapsnap adjacency`` now runs BEFORE that (#132/#213), the
    identification is persisted here, right where ``keymap-detect`` computes it.
    Lives in this module rather than ``identify`` so light consumers do not
    import torch to read one JSON file.
    """
    path = volume / KEYMAPS_FILENAME
    path.write_text(json.dumps({"keys": sorted(keys)}, indent=2))
    return path


def recorded_keymap_keys(volume: Path) -> set[str]:
    """Key-map page keys from ``keymaps.json``, or empty when absent/unreadable."""
    path = volume / KEYMAPS_FILENAME
    if not path.exists():
        return set()
    try:
        return {str(key) for key in json.loads(path.read_text()).get("keys", [])}
    except (OSError, ValueError):
        return set()
