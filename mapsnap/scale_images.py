"""Rescale a collection of images by a uniform factor.

Finds the largest image (by long side), computes the scale factor that brings its
long side to --long-side (default 2048), then applies that same factor to every image.
This keeps all images at a consistent pixel-per-metre ratio, which matters for
operations that use pixel measurements (--min-long-side, --min-short-side, scale
filtering). Outputs color JPEGs with a .scaled.jpg suffix.
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
        "--percent",
        type=float,
        default=25.0,
        help="Amount by which to scale images (25.0=25%, 100.0=no change)",
    )
    args = parser.parse_args()

    args.images = [p for p in args.images if "unsplit" not in Path(p).name]
    if not args.images:
        print(
            "Error: no images to process (all inputs were filtered as unsplit originals).",
            file=sys.stderr,
        )
        sys.exit(1)

    scale = args.percent / 100.0
    if not (0.0 < scale < 1.0):
        print(f"Error: {args.percent} must be in (0, 100)")
        sys.exit(1)
    print(
        f"Scale factor: {scale:.6f}",
        file=sys.stderr,
    )

    # Pass 2: resize and save each image in color.
    for image_path in args.images:
        with Image.open(image_path) as img:
            w, h = img.size
        new_w = round(w * scale)
        new_h = round(h * scale)
        stem = image_stem(image_path)
        output_path = Path(image_path).parent / (stem + ".scaled.jpg")
        with Image.open(image_path) as img:
            out = img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
            out.save(output_path, "JPEG", quality=95)
        print(
            f"{image_path} → {output_path}  ({w}×{h} → {new_w}×{new_h})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
