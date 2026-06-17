"""Rescale a collection of images by a uniform factor.

Scales every image by --percent (default 25%), the same factor for all of them, so pixel
measurements stay consistent across the collection. Outputs color JPEGs named <stem>.jpg
into --output-dir (default: each image's own directory).
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

from mapsnap.utils import image_stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rescale all images by a uniform --percent factor, so pixel measurements "
            "are consistent across the collection. Writes <stem>.jpg to --output-dir."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE", help="Input image files.")
    parser.add_argument(
        "--percent",
        type=float,
        default=25.0,
        help="Percent to scale images to (25.0 = quarter size; must be in (0, 100)).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write scaled <stem>.jpg files to. Defaults to each input "
        "image's own directory.",
    )
    args = parser.parse_args()

    scale = args.percent / 100.0
    if not (0.0 < scale < 1.0):
        parser.error(f"--percent {args.percent} must be in (0, 100)")
    print(f"Scale factor: {scale:.6f}", file=sys.stderr)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resize and save each image in color.
    for image in args.images:
        image_path = Path(image)
        with Image.open(image_path) as img:
            w, h = img.size
        new_w = round(w * scale)
        new_h = round(h * scale)
        out_dir = args.output_dir if args.output_dir is not None else image_path.parent
        output_path = out_dir / (image_stem(image) + ".jpg")
        with Image.open(image_path) as img:
            out = img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
            out.save(output_path, "JPEG", quality=95)
        print(
            f"{image_path} → {output_path}  ({w}×{h} → {new_w}×{new_h})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
