"""Rescale a collection of images by a uniform factor.

Finds the largest image (by short side), computes the scale factor that brings its
short side to --short-side (default 2048), then applies that same factor to every image.
This keeps all images at a consistent pixel-per-metre ratio, which matters for
operations that use pixel measurements (--min-long-side, --min-short-side, scale
filtering). Outputs grayscale JPEGs with a .scaled.jpg suffix.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

from mapsnap.utils import image_stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rescale all images by a uniform factor derived from the largest image, "
            "so pixel measurements are consistent across the collection."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE", help="Input image files.")
    parser.add_argument(
        "--short-side",
        type=int,
        default=2048,
        metavar="PX",
        help="Target short-side pixel length for the largest image (default: 2048).",
    )
    args = parser.parse_args()

    # Pass 1: read dimensions and find the maximum short side.
    # Skip unsplit originals; they exist only as template-matching references.
    args.images = [p for p in args.images if "unsplit" not in Path(p).name]
    sizes: list[tuple[int, int]] = []
    for image_path in args.images:
        with Image.open(image_path) as img:
            sizes.append(img.size)  # (width, height)

    max_short_side = max(min(w, h) for w, h in sizes)

    # Never upscale: cap the scale factor at 1.0.
    scale = min(1.0, args.short_side / max_short_side)
    print(
        f"Scale factor: {scale:.6f}  "
        f"(largest short side {max_short_side}px → {round(max_short_side * scale)}px)",
        file=sys.stderr,
    )

    # Pass 2: convert, resize, and save each image.
    for image_path, (w, h) in zip(args.images, sizes):
        new_w = round(w * scale)
        new_h = round(h * scale)
        stem = image_stem(image_path)
        output_path = Path(image_path).parent / (stem + ".scaled.jpg")
        with Image.open(image_path) as img:
            out = img.convert("L").resize((new_w, new_h), Image.Resampling.LANCZOS)
            out.save(output_path, "JPEG", quality=95)
        print(
            f"{image_path} → {output_path}  ({w}×{h} → {new_w}×{new_h})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
