"""Detect printed adjacent-sheet numbers on page margins and build a volume adjacency graph.

Sanborn sheets print the neighboring sheet's number on the margin (the key legend calls it a
"reference to adjoining sheet"): a big numeral across the shared boundary road on the side that
sheet continues. Reading these gives page adjacency that is independent of the key map — and a
detected neighbor's edge (N/E/S/W) pins the page's orientation.

Raw digit reads are noisy (house/block/dimension numbers collide with valid page numbers), so a
detection only becomes a *claim* when it is a valid page number, near the page edge, and printed
large; and a claim only becomes an adjacency *edge* when it is reciprocated — page A claims B
AND page B claims A. Measured on Hudson County: directed claims are ~72% precise, mutual edges
~98%, covering ~76% of georeferenced-overlap adjacencies and reaching 30/33 pages that street
matching failed to georeference.

This is a whole-volume step: it scans every non-split page image (skipping key-map sheets,
whose faces are covered in page numbers) and writes a single ``<volume>/adjacency.json`` with
each page's number detections and the mutual-edge adjacency graph.

    uv run mapsnap adjacency data/hudson_co_nj_1950_vol_9
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from mapsnap.keymap.locate import page_number
from mapsnap.utils import image_stem

# A detection only counts as an adjacency claim when it sits in the outer EDGE_BAND of the
# page, is at least MIN_HEIGHT px tall (the printed sheet references are large numerals; this
# excludes house/dimension numbers), and clears a permissive confidence floor (true references
# read at high confidence, but a partly-degraded one is still rescued by reciprocity).
EDGE_BAND = 0.25
MIN_HEIGHT = 28.0
MIN_CONF = 0.05


def volume_page_images(volume: Path) -> list[Path]:
    """The whole-page images to scan: top-level ``p*.jpg``, skipping split panels and key maps.

    Split panels (``__`` stems) are cuts of a parent sheet, so the printed margin references
    live on the parent; the parent is scanned even when panels supersede it for OCR. Key-map
    sheets are excluded: their faces are covered in page numbers, which would all read as
    spurious claims.
    """
    images = []
    for image in sorted(volume.glob("p*.jpg")):
        stem = image_stem(str(image))
        if "__" in stem:
            continue
        if (volume / f"{stem}.keymap.json").exists() or (
            volume / "raw" / f"{stem}.keymap.json"
        ).exists():
            continue
        images.append(image)
    return images


def classify_edge(x_frac: float, y_frac: float, band: float = EDGE_BAND) -> str:
    """Which page edge(s) a point (as fractions of width/height) is near: "E", "NW", "center"...

    Corners use the conventional compass spelling (vertical first): "NE", "NW", "SE", "SW".
    """
    edges = ""
    if y_frac < band:
        edges += "N"
    if y_frac > 1 - band:
        edges += "S"
    if x_frac < band:
        edges += "W"
    if x_frac > 1 - band:
        edges += "E"
    return edges or "center"


def digit_detections(image_path: Path, reader, valid_numbers: set[int]) -> list[dict]:
    """All digit reads on one page that parse to a valid volume page number.

    Each detection records the parsed ``number``, the raw OCR ``text``, EasyOCR's polygon and
    confidence, the glyph ``height`` in pixels, position as width/height fractions, and the
    ``edge`` classification. Filtering into claims is left to the caller so the JSON keeps the
    full picture.
    """
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    detections = []
    for bbox, text, confidence in reader.readtext(image, allowlist="0123456789"):
        digits = "".join(c for c in text if c.isdigit())
        if not digits or int(digits) not in valid_numbers:
            continue
        x_frac = sum(p[0] for p in bbox) / 4 / width
        y_frac = sum(p[1] for p in bbox) / 4 / height
        glyph_height = float(max(p[1] for p in bbox) - min(p[1] for p in bbox))
        detections.append(
            {
                "number": int(digits),
                "text": text,
                "confidence": round(float(confidence), 4),
                "polygon": [[int(p[0]), int(p[1])] for p in bbox],
                "height": round(glyph_height, 1),
                "x_frac": round(x_frac, 4),
                "y_frac": round(y_frac, 4),
                "edge": classify_edge(x_frac, y_frac),
            }
        )
    return detections


def is_claim(
    detection: dict,
    own_number: int | None,
    *,
    min_height: float = MIN_HEIGHT,
    min_confidence: float = MIN_CONF,
) -> bool:
    """Whether a detection qualifies as an adjacency claim (edge-band, large, not the page itself)."""
    return (
        detection["number"] != own_number
        and detection["edge"] != "center"
        and detection["height"] >= min_height
        and detection["confidence"] >= min_confidence
    )


def mutual_edges(claims_by_page: dict[str, set[int]]) -> list[tuple[str, str]]:
    """Reciprocated adjacency edges: A claims B's number and B claims A's number.

    ``claims_by_page`` maps page stems to claimed page numbers. A number claimed by A resolves
    to the page stem(s) carrying that number; the edge exists only if some such stem claims A's
    number back. Returns sorted (stem, stem) pairs, each once.
    """
    stems_by_number: dict[int, list[str]] = defaultdict(list)
    for stem in claims_by_page:
        number = page_number(stem)
        if number is not None:
            stems_by_number[number].append(stem)
    edges: set[tuple[str, str]] = set()
    for stem, claimed in claims_by_page.items():
        own = page_number(stem)
        if own is None:
            continue
        for number in claimed:
            for other in stems_by_number.get(number, []):
                if other != stem and own in claims_by_page.get(other, set()):
                    first, second = sorted((stem, other))
                    edges.add((first, second))
    return sorted(edges)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect printed adjacent-sheet numbers and build a volume adjacency graph."
    )
    parser.add_argument(
        "volume", type=Path, help="Volume directory holding p*.jpg pages."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: <volume>/adjacency.json).",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=MIN_HEIGHT,
        metavar="PX",
        help="Minimum glyph height for a claim (default: %(default)s, tuned for 25%%-scale pages).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONF,
        metavar="C",
        help="Minimum OCR confidence for a claim (default: %(default)s).",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration for EasyOCR."
    )
    args = parser.parse_args()

    import easyocr

    images = volume_page_images(args.volume)
    if not images:
        sys.exit(f"No page images found in {args.volume}.")
    valid_numbers = {
        number
        for image in images
        if (number := page_number(image_stem(str(image)))) is not None
    }
    print(
        f"Scanning {len(images)} pages ({len(valid_numbers)} page numbers)...",
        file=sys.stderr,
    )
    reader = easyocr.Reader(["en"], gpu=not args.no_gpu, verbose=False)

    pages: dict[str, dict] = {}
    claims_by_page: dict[str, set[int]] = {}
    for image in tqdm(images, smoothing=0):
        stem = image_stem(str(image))
        own_number = page_number(stem)
        detections = digit_detections(image, reader, valid_numbers)
        for detection in detections:
            detection["claim"] = is_claim(
                detection,
                own_number,
                min_height=args.min_height,
                min_confidence=args.min_confidence,
            )
        claims = {d["number"] for d in detections if d["claim"]}
        # Record the scanned image's dimensions so a viewer can rescale detection
        # polygons when it loads the page at a different resolution.
        width, height = Image.open(image).size
        pages[stem] = {
            "number": own_number,
            "width": width,
            "height": height,
            "detections": detections,
        }
        claims_by_page[stem] = claims

    edges = mutual_edges(claims_by_page)
    doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv[:],
        "edge_band": EDGE_BAND,
        "min_height": args.min_height,
        "min_confidence": args.min_confidence,
        "pages": pages,
        "adjacency": [list(edge) for edge in edges],
    }
    output = args.output or (args.volume / "adjacency.json")
    output.write_text(json.dumps(doc, indent=2))
    total_claims = sum(len(c) for c in claims_by_page.values())
    print(
        f"Wrote {output}: {len(pages)} pages, {total_claims} directed claims, "
        f"{len(edges)} mutual edges.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
