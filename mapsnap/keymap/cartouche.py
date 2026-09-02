"""Read a key-map sheet's cartouche words: the titles of boxed regions that are not the key map.

A key-map sheet often carries boxed regions the key map itself does not want: a
"GRAPHIC MAP OF VOLUMES" (Richmond's "GENERAL INDEX TO VOLUMES"), the "KEY TO
SYMBOLS" legend, the "CORRECTION RECORD". Their titles are large decorative
text, and the vocabulary-constrained recognizer reads them well even in flowery
type: run over the seven volume-index insets in the truth corpus with the
stock recognizer, five read GRAPHIC / MAP / VOLUMES / INDEX at 0.58-0.99
inside the inset's corner, one read weakly, and one (Kansas City) has no title
at all. KEY reads at 0.88-1.0 on four sheets.

This pass runs `detect_text` with ONLY those words as vocabulary and writes
``raw/<stem>.cartouche.json`` next to the key map's other sidecars. The inset
detector (#276) consumes it as one corroborating signal among several -- a
cartouche inside an isolated cluster of small page numbers is as strong a tell
as there is -- and nothing else reads it. The words are kept out of the
page-level street vocabulary on purpose: GENERAL, MAP and KEY occur in real
street names (New Orleans has GENERAL TAYLOR), and a page-level "ignore" entry
would swallow them.
"""

import argparse
import json
import sys
from pathlib import Path

from mapsnap.detect_text import detect_text
from mapsnap.keymap.log import append_keymap_log
from mapsnap.utils import image_stem

CARTOUCHE_VOCAB: list[str] = [
    "GRAPHIC",
    "MAP",
    "VOLUMES",
    "GRAPHIC MAP",
    "MAP OF VOLUMES",
    "GRAPHIC MAP OF VOLUMES",
    "GENERAL",
    "INDEX",
    "GENERAL INDEX",
    "TO VOLUMES",
    "INDEX TO VOLUMES",
    "KEY",
    "SYMBOLS",
    "KEY TO SYMBOLS",
    "CORRECTION RECORD",
]

# Which boxed region a word announces; the inset detector cares about the first.
CARTOUCHE_KIND: dict[str, str] = {
    "GRAPHIC": "volumes",
    "MAP": "volumes",
    "VOLUMES": "volumes",
    "GRAPHIC MAP": "volumes",
    "MAP OF VOLUMES": "volumes",
    "GRAPHIC MAP OF VOLUMES": "volumes",
    "GENERAL": "volumes",
    "INDEX": "volumes",
    "GENERAL INDEX": "volumes",
    "TO VOLUMES": "volumes",
    "INDEX TO VOLUMES": "volumes",
    "KEY": "legend",
    "SYMBOLS": "legend",
    "KEY TO SYMBOLS": "legend",
    "CORRECTION RECORD": "corrections",
}

# Words that only ever title a volume-index inset. MAP alone reads the sheet's
# own "KEY MAP" title at 1.0 on three of seven sheets, and GENERAL alone fires
# on New Orleans 1951's panel, so on their own they say little; GRAPHIC,
# VOLUMES and INDEX never appear anywhere else on a key-map sheet.
CARTOUCHE_SPECIFIC: frozenset[str] = frozenset(
    {
        "GRAPHIC",
        "VOLUMES",
        "INDEX",
        "GRAPHIC MAP",
        "MAP OF VOLUMES",
        "GRAPHIC MAP OF VOLUMES",
        "GENERAL INDEX",
        "TO VOLUMES",
        "INDEX TO VOLUMES",
    }
)

MIN_CONFIDENCE = 0.3


def is_specific(text: str) -> bool:
    """Whether a cartouche word, on its own, names a volume-index inset."""
    return text in CARTOUCHE_SPECIFIC


def cartouche_sidecar_path(image_path: str | Path) -> Path:
    """``<dir>/<stem>.cartouche.json`` beside the key-map image."""
    image_path = Path(image_path)
    return image_path.parent / (image_stem(str(image_path)) + ".cartouche.json")


def cartouche_reads(
    image_path: str | Path, reader=None, min_confidence: float = MIN_CONFIDENCE
) -> list[dict]:
    """Cartouche-word reads on one sheet, best first.

    Each read keeps detect_text's polygon (in the image's own pixel frame),
    text, confidence and angle, plus ``kind`` (volumes | legend | corrections).
    Requires the sheet's cached CRAFT boxes, like any detect_text call.
    """
    reads = detect_text(str(image_path), CARTOUCHE_VOCAB, min_size=10, reader=reader)
    kept = [
        {
            "text": read["text"],
            "kind": CARTOUCHE_KIND[read["text"]],
            "confidence": round(float(read["confidence"]), 4),
            "angle": read.get("angle", 0),
            "polygon": [[float(x), float(y)] for x, y in read["polygon"]],
        }
        for read in reads
        if read["text"] in CARTOUCHE_KIND and read["confidence"] >= min_confidence
    ]
    return sorted(kept, key=lambda read: -read["confidence"])


def write_cartouche_sidecar(image_path: str | Path, reads: list[dict]) -> Path:
    """Write the reads for ``image_path`` and return the sidecar path."""
    from PIL import Image

    width, height = Image.open(image_path).size
    path = cartouche_sidecar_path(image_path)
    path.write_text(
        json.dumps(
            {
                "image": Path(image_path).name,
                "width": width,
                "height": height,
                "reads": reads,
            },
            indent=1,
        )
    )
    return path


def cartouche_log_lines(reads: list[dict]) -> list[str]:
    """The decision-log lines for a sheet's cartouche reads, specific ones marked."""
    if not reads:
        return ["no cartouche words read"]
    lines = [f"{len(reads)} cartouche read(s):"]
    for read in reads:
        xs = [point[0] for point in read["polygon"]]
        ys = [point[1] for point in read["polygon"]]
        mark = "specific" if is_specific(read["text"]) else "weak"
        lines.append(
            f"  {read['text']} @{read['confidence']:.2f} ({read['kind']}, {mark}) "
            f"at ({sum(xs) / len(xs):.0f}, {sum(ys) / len(ys):.0f})"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read cartouche words (GRAPHIC MAP OF VOLUMES, KEY, ...) on key-map sheets."
    )
    parser.add_argument(
        "images", nargs="+", help="Key-map image(s), e.g. data/<volume>/raw/p0.jpg"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help=f"Drop reads below this confidence (default {MIN_CONFIDENCE}).",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Run the recognizer on the CPU."
    )
    args = parser.parse_args()

    import easyocr

    reader = easyocr.Reader(["en"], gpu=not args.no_gpu, verbose=False)
    for image in args.images:
        reads = cartouche_reads(
            image, reader=reader, min_confidence=args.min_confidence
        )
        path = write_cartouche_sidecar(image, reads)
        append_keymap_log(image, "cartouche", cartouche_log_lines(reads))
        summary = (
            ", ".join(f"{r['text']}@{r['confidence']:.2f}" for r in reads[:6])
            or "nothing"
        )
        print(
            f"{Path(image).name}: {len(reads)} cartouche read(s) -> {path.name}: {summary}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
