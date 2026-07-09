"""Project OIM truth footprints onto a key map: truth data for page-region segmentation.

Each page's human-georeferenced footprint (``main.iiif.json``, clipped by OIM to tile
cleanly) is a world polygon; the key map's own georeferencing maps world coordinates into
key-map pixels. Projecting every footprint gives a ``<stem>.truth.regions.panels.json``
sidecar in the same schema as the detected regions — ground truth for scoring a
segmentation (mapsnap.keymap.score_regions). The footprints won't exactly match the blocks
drawn on the key map (OIM clips to the page's unique area; the key map's blocks are its own
stylization), but they are close enough to rank segmentations.

    uv run python -m mapsnap.keymap.truth_regions data/hudson_co_nj_1950_vol_9
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mapsnap.compare_iiif_georef import truth_polygons_by_page
from mapsnap.keymap.locate import keymap_georef_path, resolve_keymaps
from mapsnap.keymap.page_regions import keymap_image_path, regions_panels_doc


def world_to_pixel_affine(georef: dict) -> np.ndarray:
    """2x3 affine mapping (lon, lat) to key-map pixels, least-squares fit to the corners.

    The georef corner quad ([TL, TR, BR, BL]) is nearly a parallelogram, so an affine fit
    is accurate to a pixel or two — plenty for region truth.
    """
    width, height = georef["width"], georef["height"]
    corners_px = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
    )
    corners_world = np.hstack(
        [np.asarray(georef["corners"], dtype=np.float64), np.ones((4, 1))]
    )
    coefficients, _, _, _ = np.linalg.lstsq(corners_world, corners_px, rcond=None)
    return coefficients.T


def project_truth_regions(
    georef: dict, truth_by_page: dict[int, list[list[list[float]]]]
) -> tuple[dict[int, list[tuple[float, float]]], list[str]]:
    """Truth footprints in key-map pixel space, as (polygons-by-index, labels) for panels.

    Every footprint of every page becomes one panel labeled with its page number (a split
    page contributes several panels sharing a label). Footprints that project entirely
    outside the key map (with a one-page margin) are skipped.
    """
    affine = world_to_pixel_affine(georef)
    width, height = georef["width"], georef["height"]
    polygons: dict[int, list[tuple[float, float]]] = {}
    labels: list[str] = []
    for number in sorted(truth_by_page):
        for footprint in truth_by_page[number]:
            points = np.hstack(
                [np.asarray(footprint, dtype=np.float64), np.ones((len(footprint), 1))]
            )
            pixels = points @ affine.T
            if (
                pixels[:, 0].max() < -width * 0.5
                or pixels[:, 0].min() > width * 1.5
                or pixels[:, 1].max() < -height * 0.5
                or pixels[:, 1].min() > height * 1.5
            ):
                continue
            polygons[len(labels)] = [(float(x), float(y)) for x, y in pixels]
            labels.append(str(number))
    return polygons, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "volume",
        type=Path,
        help="Volume directory holding main.iiif.json and a georeferenced key map.",
    )
    parser.add_argument(
        "--keymap",
        nargs="+",
        metavar="JSON",
        help=(
            "Key-map detections file(s) with georef siblings (default: auto-discovered "
            "next to the pages or under raw/)."
        ),
    )
    args = parser.parse_args()

    truth_path = args.volume / "main.iiif.json"
    if not truth_path.exists():
        sys.exit(f"Not found: {truth_path}")
    truth_by_page = truth_polygons_by_page(truth_path)
    keymaps = resolve_keymaps(args.keymap, False, [str(args.volume / "p0.jpg")])
    if not keymaps:
        sys.exit(f"No georeferenced key map found for {args.volume}.")
    for keymap in keymaps:
        georef = json.load(open(keymap_georef_path(keymap)))
        polygons, labels = project_truth_regions(georef, truth_by_page)
        image_name = keymap_image_path(keymap).name
        doc = regions_panels_doc(
            image_name, (georef["width"], georef["height"]), polygons, labels
        )
        output = keymap.with_name(
            keymap.name.replace(".keymap.json", ".truth.regions.panels.json")
        )
        output.write_text(json.dumps(doc))
        print(
            f"Wrote {output}: {len(labels)} truth footprints "
            f"({len(set(labels))} pages) on {image_name}."
        )


if __name__ == "__main__":
    main()
