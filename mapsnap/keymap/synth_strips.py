"""Synthetic training strips for the key-map page-number recognizer (#316).

The real training set is ~2,400 hand-labelled strips across 25 key maps, and it
demonstrably under-covers some styles — columbia sits at 54% digit accuracy
*in sample*. Page numbers are a tiny, enumerable domain, so synthesis can cover
it: digit strings rendered in fonts matched to the two Sanborn families,
composited over background tiles harvested from real key-map sheets, with the
three letter-suffix conventions the corpus actually uses:

- ``underline``: the letter smaller, with a bar beneath (columbia, richmond);
- ``quotes``: the letter followed by a double-quote-like tick (los_angeles);
- ``plain``: the letter at full size, unadorned (nashville).

Backgrounds come from the labelled sheets themselves via the same safe-crop
logic the trainer's negatives use, so the clutter is the real thing: street
lines through the digits, block numbers, stray small text — not simulated
noise. Fonts are system faces chosen by eye against real strips (see #316):
didone/slab for the ornate family, grotesques for the sans family.

    uv run python -m mapsnap.keymap.synth_strips --preview 64 --out preview.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mapsnap.keymap.crnn_model import CRNN_HEIGHT, CRNN_WIDTH

# Matched by eye against real strips (chicago/hudson/LA digits are a didone;
# columbia a slab-ish serif; nashville a grotesque). Missing faces are skipped
# at load, so the generator degrades rather than fails on another machine.
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Bodoni 72.ttc",
    "/System/Library/Fonts/Supplemental/Rockwell.ttc",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Futura.ttc",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
]

SUFFIX_STYLES = ("none", "underline", "quotes", "plain")

# Digit-count mix: mostly 1-3 digits like the corpus, with enough 4-digit
# strings (los_angeles's 1400s) that long keys are not a novelty at inference.
DIGIT_COUNT_WEIGHTS = {1: 0.3, 2: 0.3, 3: 0.25, 4: 0.15}

# Fraction of strips that carry a letter suffix. Lettered pages are a minority
# of the corpus but the entire point of the charset change, so they are heavily
# over-sampled relative to their natural rate.
SUFFIX_FRACTION = 0.45


def load_fonts() -> list[str]:
    """The available font files, in FONT_PATHS order."""
    return [p for p in FONT_PATHS if Path(p).exists()]


def harvest_background_tiles(
    image: np.ndarray,
    label_centers: list[tuple[float, float]],
    factor: float,
    count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Number-free CRNN-sized tiles from a real key-map sheet.

    Reuses the trainer's safe-crop rule so a tile never contains a real page
    number (which would teach the model to ignore real numbers), but keeps all
    the real clutter -- lines, block numbers, street names.
    """
    from mapsnap.keymap.crnn_model import number_strip
    from mapsnap.keymap.keymap_patches import crop_excludes_numbers

    height, width = image.shape[:2]
    tiles: list[np.ndarray] = []
    for _ in range(count * 30):
        if len(tiles) >= count:
            break
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        if crop_excludes_numbers(
            cx * factor,
            cy * factor,
            [(x * factor, y * factor) for x, y in label_centers],
            crop_half_w=90,
            crop_half_h=34,
        ):
            tiles.append(number_strip(image, cx, cy, factor))
    return tiles


def corpus_background_tiles(
    data_dir: Path, count: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Background tiles pooled across every labelled key map under ``data_dir``."""
    from mapsnap.keymap.keymap_patches import (
        labelled_keymaps,
        load_label_points,
        working_scale,
    )

    keymaps = labelled_keymaps(data_dir)
    if not keymaps:
        return []
    per_sheet = max(1, count // len(keymaps))
    tiles: list[np.ndarray] = []
    for image_path, labels_path in keymaps:
        width, height, points = load_label_points(str(labels_path))
        if not points:
            continue
        factor = working_scale(width, height)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        tiles.extend(
            harvest_background_tiles(
                image, [(x, y) for x, y, _ in points], factor, per_sheet, rng
            )
        )
    rng.shuffle(tiles)  # type: ignore[arg-type]
    return tiles


def random_page_key(rng: np.random.Generator) -> tuple[str, str]:
    """(text, suffix_style) for one synthetic strip."""
    counts = list(DIGIT_COUNT_WEIGHTS)
    weights = np.array(list(DIGIT_COUNT_WEIGHTS.values()))
    n = int(rng.choice(counts, p=weights / weights.sum()))
    digits = str(rng.integers(1, 10)) + "".join(
        str(rng.integers(0, 10)) for _ in range(n - 1)
    )
    if rng.random() >= SUFFIX_FRACTION:
        return digits, "none"
    style = str(rng.choice(["underline", "quotes", "plain"]))
    letter = chr(ord("A") + int(rng.integers(0, 19)))  # A..S, the observed range
    return digits + letter, style


def render_number(
    text: str, style: str, font_path: str, rng: np.random.Generator
) -> Image.Image:
    """The number (and any suffix adornment) as an L-mode image, white background.

    The suffix letter renders per ``style``: smaller with a bar beneath
    (columbia/richmond), followed by a double-quote tick (los_angeles), or at
    full size (nashville). ``none`` means the text is digits only.
    """
    digits = text.rstrip("ABCDEFGHIJKLMNOPQRS") if style != "none" else text
    letter = text[len(digits) :]
    size = int(rng.integers(26, 40))
    font = ImageFont.truetype(font_path, size)
    small = ImageFont.truetype(font_path, max(14, int(size * 0.62)))

    canvas = Image.new("L", (CRNN_WIDTH * 2, CRNN_HEIGHT * 2), 255)
    draw = ImageDraw.Draw(canvas)
    ink = int(rng.integers(10, 90))
    x, y = 20, 30
    draw.text((x, y), digits, font=font, fill=ink)
    dx = draw.textlength(digits, font=font)
    if letter:
        if style == "plain":
            draw.text((x + dx + 2, y), letter, font=font, fill=ink)
        elif style == "underline":
            lx = x + dx + 3
            ly = y + int(size * 0.30)
            draw.text((lx, ly), letter, font=small, fill=ink)
            lw = draw.textlength(letter, font=small)
            uy = ly + int(size * 0.62) + 2
            draw.line([(lx - 1, uy), (lx + lw + 1, uy)], fill=ink, width=2)
        elif style == "quotes":
            # los_angeles renders the double-quote UNDER the raised letter.
            lx = x + dx + 3
            draw.text((lx, y), letter, font=small, fill=ink)
            draw.text((lx + 1, y + int(size * 0.55)), '"', font=small, fill=ink)
    return canvas


def synth_strip(
    fonts: list[str],
    tiles: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """One (CRNN_HEIGHT x CRNN_WIDTH strip, label) pair.

    Composites a rendered number over a real background tile (multiplicative,
    so the sheet's own lines show through the glyph gaps exactly as print
    does), then adds the clutter the corpus actually exhibits: street lines
    THROUGH the number, dots, and the occasional stray small text.
    """
    text, style = random_page_key(rng)
    number = render_number(text, style, fonts[int(rng.integers(len(fonts)))], rng)

    # Rotate slightly and crop off-center, like the localizer's candidates.
    angle = float(rng.uniform(-5, 5))
    number = number.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
    arr = np.asarray(number, dtype=np.float32) / 255.0
    ys, xs = np.where(arr < 0.9)
    if len(xs) == 0:
        return synth_strip(fonts, tiles, rng)
    cx, cy = xs.mean(), ys.mean()
    jx = float(rng.uniform(-12, 12))
    jy = float(rng.uniform(-6, 6))
    x0 = int(np.clip(cx - CRNN_WIDTH / 2 + jx, 0, arr.shape[1] - CRNN_WIDTH))
    y0 = int(np.clip(cy - CRNN_HEIGHT / 2 + jy, 0, arr.shape[0] - CRNN_HEIGHT))
    glyph = arr[y0 : y0 + CRNN_HEIGHT, x0 : x0 + CRNN_WIDTH]

    tile = tiles[int(rng.integers(len(tiles)))].astype(np.float32) / 255.0
    strip = tile * glyph  # print multiplies: ink darkens whatever is beneath

    # Clutter: lines through the number, dots, stray text fragments.
    strip8 = (strip * 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 3))):
        p1 = (int(rng.integers(0, CRNN_WIDTH)), int(rng.integers(0, CRNN_HEIGHT)))
        p2 = (int(rng.integers(0, CRNN_WIDTH)), int(rng.integers(0, CRNN_HEIGHT)))
        cv2.line(strip8, p1, p2, int(rng.integers(30, 120)), 1)
    for _ in range(int(rng.integers(0, 4))):
        center = (int(rng.integers(0, CRNN_WIDTH)), int(rng.integers(0, CRNN_HEIGHT)))
        cv2.circle(
            strip8, center, int(rng.integers(1, 3)), int(rng.integers(20, 90)), -1
        )

    # Scan-quality degradation: mild blur and noise.
    if rng.random() < 0.5:
        strip8 = cv2.GaussianBlur(strip8, (3, 3), 0)
    noise = rng.normal(0, rng.uniform(2, 9), strip8.shape)
    strip8 = np.clip(strip8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return strip8, text


def generate(
    count: int,
    data_dir: Path,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[str]]:
    """``count`` synthetic (strip, label) pairs, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    fonts = load_fonts()
    if not fonts:
        raise RuntimeError("no synthesis fonts available on this system")
    tiles = corpus_background_tiles(data_dir, max(400, count // 25), rng)
    if not tiles:
        raise RuntimeError(f"no background tiles harvested under {data_dir}")
    strips, texts = [], []
    for _ in range(count):
        strip, text = synth_strip(fonts, tiles, rng)
        strips.append(strip)
        texts.append(text)
    return strips, texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview synthetic number strips.")
    parser.add_argument("--preview", type=int, default=48, metavar="N")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("synth_preview.png"))
    args = parser.parse_args()

    strips, texts = generate(args.preview, args.data_dir, args.seed)
    cols = 6
    rows = (len(strips) + cols - 1) // cols
    pad = 22
    sheet = np.full(
        (rows * (CRNN_HEIGHT + pad) + pad, cols * (CRNN_WIDTH + 8) + 8),
        255,
        dtype=np.uint8,
    )
    img = Image.fromarray(sheet)
    draw = ImageDraw.Draw(img)
    for i, (strip, text) in enumerate(zip(strips, texts)):
        r, c = divmod(i, cols)
        y0 = pad + r * (CRNN_HEIGHT + pad)
        x0 = 8 + c * (CRNN_WIDTH + 8)
        img.paste(Image.fromarray(strip), (x0, y0))
        draw.text((x0, y0 - 14), text, fill=0)
    img.save(args.out)
    print(f"wrote {len(strips)} strips to {args.out}")


if __name__ == "__main__":
    main()
