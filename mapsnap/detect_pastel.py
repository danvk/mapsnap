"""Identify the pastel-colored regions on a Sanborn key map.

A key map page is mostly three things: the light, yellowed color of aged paper;
black ink (text, lines, hatching); and pastel-colored regions (pink, yellow,
green, blue) that show where each detailed sheet sits. This module picks out just
the pastel pixels.

The discriminator is saturation, but with a twist: aged paper is itself yellowish,
so in the yellow hue band the paper and the yellow pastel overlap and we need a
higher saturation threshold to separate them. In the green/blue/pink hue bands the
paper essentially never appears, so a low saturation threshold suffices.

Run as a script to write a sidecar "<stem>.pastel.png" next to each input image: a
copy of the original with every pastel pixel painted bright red.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mapsnap.utils import image_stem

# Hue range (OpenCV 0-179 scale) where aged paper lives, so we demand more
# saturation there to avoid flagging the paper itself.
PAPER_HUE_LO = 12
PAPER_HUE_HI = 40

# Minimum saturation (0-255) to count as pastel, inside vs. outside the paper hue band.
SATURATION_THRESHOLD_YELLOW = 45
SATURATION_THRESHOLD_OTHER = 25

# Minimum value (0-255); excludes black ink and deep shadows.
VALUE_THRESHOLD = 120

# Color used to paint pastel pixels in the sidecar image.
RED = (255, 0, 0)


def pastel_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the pastel-colored pixels in an RGB image.

    Takes an HxWx3 uint8 RGB array and returns an HxW boolean array that is True
    for pixels belonging to a pastel region (and False for paper and ink).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16)
    saturation = hsv[..., 1].astype(np.int16)
    value = hsv[..., 2].astype(np.int16)

    in_paper_hue = (hue >= PAPER_HUE_LO) & (hue < PAPER_HUE_HI)
    threshold = np.where(
        in_paper_hue, SATURATION_THRESHOLD_YELLOW, SATURATION_THRESHOLD_OTHER
    )
    return (saturation >= threshold) & (value >= VALUE_THRESHOLD)


def paint_pastels(rgb: np.ndarray) -> np.ndarray:
    """Copy of an RGB image with every pastel pixel painted bright red."""
    out = rgb.copy()
    out[pastel_mask(rgb)] = RED
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Detect pastel-colored regions on key map images and write a sidecar "
            "<stem>.pastel.png with those pixels painted bright red."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE", help="Input image files.")
    args = parser.parse_args()

    for image in args.images:
        image_path = Path(image)
        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"))
        painted = paint_pastels(rgb)
        output_path = image_path.parent / (image_stem(image) + ".pastel.png")
        Image.fromarray(painted).save(output_path)
        frac = pastel_mask(rgb).mean()
        print(
            f"{image_path} → {output_path}  ({frac:.1%} pastel)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
