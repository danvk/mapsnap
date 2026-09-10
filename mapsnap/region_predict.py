"""Predict every page's content region and write it as a P(region) map.

The content-region model (``mapsnap.region_model``, #226) predicts per pixel
whether a sheet position holds content exclusive to that page, as opposed to
margin, title block, or a strip duplicated from a neighbour. This command runs
it over a volume's 25% page images, parents and split panels alike, and writes
each map as an 8-bit PNG at the page's own resolution:

    data/<vol>/artifacts/region/<stem>.png     0 = margin, 255 = content

The volume viewer draws these in place of the sheets (its Page / Region /
P(road) toggle), and the content-region overlap analysis (#352) reads them.

    mapsnap region data/<vol>            # every p*.jpg that lacks a map
    mapsnap region data/<vol> --force    # redo them all
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from mapsnap.region_model import predict_region

REGION_DIR = Path("artifacts") / "region"


def region_prob_path(volume: Path, stem: str) -> Path:
    """Where a page's P(region) map lives: ``<volume>/artifacts/region/<stem>.png``."""
    return volume / REGION_DIR / f"{stem}.png"


def write_region_maps(
    volume: Path,
    images: list[Path],
    *,
    model: torch.nn.Module,
    device: torch.device,
    force: bool = False,
) -> list[Path]:
    """Predict and write P(region) maps for ``images``; returns the paths written.

    Maps that already exist are kept unless ``force``. Unreadable images are
    reported on stderr and skipped.
    """
    written: list[Path] = []
    for image_path in images:
        out = region_prob_path(volume, image_path.stem)
        if out.exists() and not force:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip (unreadable): {image_path}", file=sys.stderr)
            continue
        prob = predict_region(model, image, device)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.clip(prob * 255, 0, 255).astype(np.uint8))
        written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "volume", type=Path, help="Volume directory holding the p*.jpg pages."
    )
    parser.add_argument(
        "--force", action="store_true", help="Rewrite maps that already exist."
    )
    args = parser.parse_args()

    from mapsnap.region_model import load_region_model

    images = sorted(args.volume.glob("p*.jpg"))
    if not images:
        sys.exit(f"no p*.jpg pages under {args.volume}")
    model, device = load_region_model()
    written = write_region_maps(
        args.volume, images, model=model, device=device, force=args.force
    )
    print(
        f"wrote {len(written)} of {len(images)} P(region) maps"
        f" to {args.volume / REGION_DIR}"
    )


if __name__ == "__main__":
    main()
