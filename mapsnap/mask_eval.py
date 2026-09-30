"""Score a clip masker in isolation: OIM's poses and splits in, our masks out (#548).

    mapsnap mask-eval data/<volume> [--masker region|blocks|voronoi|none] [--allmaps]

Masking is the pipeline's last stage, so scoring a run's masks against OIM mixes in
every upstream error: a misplaced page or a different split changes what the
masker is given. This hands the masker OIM's own poses and splits instead, then
scores its masks with ``mapsnap mask-score``.

Each annotation in ``data/<volume>/main.iiif.json`` becomes a ``georef-final``
sidecar in a work directory: a similarity fitted (in a local metric frame) to all
of its GCPs, written as the page's four corners plus two "initial" intersections
at the two OIM GCPs farthest apart, with ground positions from that similarity.
``make_iiif_georef`` publishes those two points as a Helmert fit, so the masker's
pose and the published one are the same. Split sheets are cut along OIM's own
panel outlines (``data/<volume>/oim/pN.panels.json``, from ``mapsnap oim-panels``);
split sheets without them are skipped.

Maskers: ``region`` is the pipeline's (region_clip_masks); ``blocks`` is the one
before it (clip_masks.compute_all_clip_masks); ``voronoi`` and ``none`` are controls. ``voronoi`` gives each point to the nearest
page centroid among the pages that show it; ``none`` publishes whole sheets and
panels.
"""

import argparse
import json
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import shapely
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union

from mapsnap.annotation_transform import Transform
from mapsnap.clip_masks import _fit_affine, compute_all_clip_masks
from mapsnap.make_iiif_georef import (
    _load_volume_items,
    annotation_page,
    make_annotation,
)
from mapsnap.mask_score import M_PER_DEGREE, score_annotation
from mapsnap.osm_to_centerlines import load_centerlines
from mapsnap.region_clip_masks import compute_region_clip_masks
from mapsnap.split import write_panels
from mapsnap.utils import default_centerlines, source_id_to_page_key

Masker = Callable[..., list[Polygon | None]]


def split_index(label: str) -> int | None:
    """N from a label ending in '[N]', or None for a whole sheet."""
    match = re.search(r"\[(\d+)\]\s*$", label)
    return int(match.group(1)) if match else None


def gcp_pairs(item: dict) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """(canvas px, (lon, lat)) for each of an annotation's GCPs."""
    return [
        (
            tuple(feature["properties"]["resourceCoords"]),
            tuple(feature["geometry"]["coordinates"]),
        )
        for feature in item["body"]["features"]
        if feature.get("properties", {}).get("resourceCoords")
        and feature.get("geometry")
    ]


def fit_similarity(pixels: np.ndarray, lonlat: np.ndarray) -> Transform:
    """A least-squares similarity pixel -> (lon, lat), fitted in a local metric frame."""
    k = math.cos(math.radians(float(lonlat[:, 1].mean())))
    # Pixel y runs down and latitude up; no similarity can mirror, so flip y first.
    z = pixels[:, 0] - 1j * pixels[:, 1]
    w = lonlat[:, 0] * k * M_PER_DEGREE + 1j * lonlat[:, 1] * M_PER_DEGREE
    (a, b), *_ = np.linalg.lstsq(np.c_[z, np.ones_like(z)], w, rcond=None)

    def transform(x: float, y: float) -> tuple[float, float]:
        point = complex(a * complex(x, -y) + b)
        return point.real / (k * M_PER_DEGREE), point.imag / M_PER_DEGREE

    return transform


@dataclass
class ImageFrame:
    """How an image's pixels relate to its canvas: canvas px * scale - offset."""

    scale: float
    offset: tuple[float, float]
    size: tuple[int, int]


def oim_sidecar(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]], frame: ImageFrame
) -> dict | None:
    """A georef-final sidecar for one image from OIM's GCPs; None under two usable GCPs."""
    if len(pairs) < 2:
        return None
    pixels = np.array([pixel for pixel, _ in pairs], float) * frame.scale - frame.offset
    lonlat = np.array([geo for _, geo in pairs], float)
    if np.ptp(pixels, axis=0).max() < 2:
        return None
    transform = fit_similarity(pixels, lonlat)
    width, height = frame.size
    corners = [
        transform(x, y) for x, y in [(0, 0), (width, 0), (width, height), (0, height)]
    ]
    distances = np.linalg.norm(pixels[:, None] - pixels[None], axis=2)
    farthest = np.unravel_index(distances.argmax(), distances.shape)
    intersections = []
    for n in farthest:
        x, y = (float(v) for v in pixels[n])
        lon, lat = transform(x, y)
        intersections.append(
            {
                "label_a": "OIM",
                "label_b": f"GCP {n}",
                "x": x,
                "y": y,
                "lon": lon,
                "lat": lat,
                "inlier": True,
                "initial": True,
            }
        )
    return {
        "width": width,
        "height": height,
        "corners": corners,
        "intersections": intersections,
    }


def write_oim_sidecars(volume: Path, work: Path) -> list[str]:
    """Write a work dir of page links, OIM panel crops and sidecars; returns what was skipped."""
    work.mkdir(parents=True, exist_ok=True)
    truth = json.loads((volume / "main.iiif.json").read_text())
    by_sheet: dict[str, list[dict]] = {}
    for item in truth["items"]:
        source = item["target"]["source"]
        key = source_id_to_page_key(source.get("id"), item.get("label", ""))
        if key:
            by_sheet.setdefault(key.split("__")[0], []).append(item)
    skipped: list[str] = []
    for sheet, items in sorted(by_sheet.items()):
        page = volume / f"{sheet}.jpg"
        if not page.exists():
            skipped.append(f"{sheet} (no image)")
            continue
        link = work / page.name
        if not link.exists():
            link.symlink_to(page.resolve())
        page_width, page_height = Image.open(page).size
        scale = page_width / items[0]["target"]["source"]["width"]
        splits = [item for item in items if split_index(item.get("label", ""))]
        if not splits:
            frame = ImageFrame(scale, (0.0, 0.0), (page_width, page_height))
            doc = oim_sidecar(gcp_pairs(items[0]), frame)
            if doc is None:
                skipped.append(f"{sheet} (under 2 GCPs)")
            else:
                (work / f"{sheet}.georef-final.json").write_text(json.dumps(doc))
            continue
        panels_path = volume / "oim" / f"{sheet}.panels.json"
        if not panels_path.exists():
            skipped.append(f"{sheet} (split, no oim/{sheet}.panels.json)")
            continue
        panels = json.loads(panels_path.read_text())
        rings = [
            np.array(ring, float) * page_width / panels["width"]
            for ring in panels["panels"]
        ]
        # Cut through the work dir's link, so the crops and panels.json land there
        # and never beside the volume's own (the pipeline's) pN__i.jpg.
        write_panels(link, [Polygon(ring).buffer(0) for ring in rings], sheet)
        for item in splits:
            n = split_index(item["label"])
            if n is None or not 1 <= n <= len(rings):
                skipped.append(f"{sheet} [{n}] (no such panel)")
                continue
            crop = work / f"{sheet}__{n}.jpg"
            min_x, min_y = rings[n - 1].min(axis=0)
            frame = ImageFrame(scale, (min_x, min_y), Image.open(crop).size)
            doc = oim_sidecar(gcp_pairs(item), frame)
            if doc is None:
                skipped.append(f"{sheet}__{n} (under 2 GCPs)")
            else:
                (work / f"{sheet}__{n}.georef-final.json").write_text(json.dumps(doc))
    return skipped


def image_footprint(georef: dict, image_path: Path) -> Polygon:
    """What an image can show, in (lon, lat): its scan, or a split panel's outline."""
    a_fwd, _ = _fit_affine(georef)
    width, height = float(georef["width"]), float(georef["height"])
    ring = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    stem = image_path.stem
    if "__" in stem:
        base, index = stem.rsplit("__", 1)
        panels_path = image_path.parent / f"{base}.panels.json"
        if panels_path.exists():
            panel = json.loads(panels_path.read_text())["panels"][int(index) - 1]
            ring = np.array(panel, float)
            ring -= ring.min(axis=0)  # the crop's frame
    lonlat = (a_fwd @ np.c_[ring, np.ones(len(ring))].T).T
    return Polygon(lonlat).buffer(0)


def voronoi_masks(
    georefs: list[dict],
    centerlines_geojson: dict | None = None,
    debug_blocks_out: list[dict] | None = None,
    raw_paths: list[Path] | None = None,
) -> list[Polygon | None]:
    """Control masker: each point goes to the nearest page centroid among pages showing it."""
    if not georefs:
        return []
    k = math.cos(
        math.radians(float(np.mean([c[1] for g in georefs for c in g["corners"]])))
    )

    def scaled(poly: Polygon, sx: float, sy: float) -> Polygon:
        return Polygon([(x * sx, y * sy) for x, y in poly.exterior.coords])

    paths = raw_paths or [Path(f"p{i}.jpg") for i in range(len(georefs))]
    feet = [
        scaled(image_footprint(g, Path(p)), k * M_PER_DEGREE, M_PER_DEGREE)
        for g, p in zip(georefs, paths)
    ]
    centers = [np.array(foot.centroid.coords[0]) for foot in feet]
    reach = max(foot.length for foot in feet) * 4
    masks: list[Polygon | None] = []
    for i, foot in enumerate(feet):
        lost = []
        for j, other in enumerate(feet):
            if i == j or not foot.intersects(other):
                continue
            # The half-plane nearer c_j than c_i, clipped to what page j shows.
            middle = (centers[i] + centers[j]) / 2
            normal = centers[j] - centers[i]
            normal /= np.linalg.norm(normal) or 1.0
            along = np.array([-normal[1], normal[0]])
            half = Polygon(
                [
                    middle + along * reach,
                    middle + along * reach + normal * reach,
                    middle - along * reach + normal * reach,
                    middle - along * reach,
                ]
            )
            lost.append(other.intersection(half))
        mine = foot.difference(unary_union(lost)) if lost else foot
        # Snapping to a millimetre grid drops the slivers float noise leaves where
        # two scans' edges nearly coincide.
        mine = shapely.set_precision(mine, 0.001)
        parts = [p for p in getattr(mine, "geoms", [mine]) if isinstance(p, Polygon)]
        parts = [p for p in parts if not p.is_empty]
        if not parts:
            masks.append(None)
            continue
        largest = max(parts, key=lambda p: p.area)
        masks.append(scaled(largest, 1 / (k * M_PER_DEGREE), 1 / M_PER_DEGREE))
    return masks


def no_masks(
    georefs: list[dict], *args: object, **kwargs: object
) -> list[Polygon | None]:
    """Control masker: no masks, so every image shows its whole sheet or panel."""
    return [None for _ in georefs]


MASKERS: dict[str, Masker] = {
    "region": compute_region_clip_masks,
    "blocks": compute_all_clip_masks,
    "voronoi": voronoi_masks,
    "none": no_masks,
}


def build_annotation_page(
    truth_path: Path, work: Path, masker: Masker, centerlines_path: Path | None
) -> dict:
    """An AnnotationPage for the work dir's sidecars, masked by ``masker``."""
    items, result_id, label = _load_volume_items(
        str(truth_path), str(work / "*.georef-final.json")
    )
    georefs = [georef for _, _, georef, _, _ in items]
    centerlines = load_centerlines(centerlines_path) if centerlines_path else None
    masks = masker(
        georefs,
        centerlines,
        raw_paths=[image_path for _, _, _, image_path, _ in items],
    )
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    annotations = [
        make_annotation(canvas_item, georef, page_key, image_path, now, mask)
        for (page_key, canvas_item, georef, image_path, _), mask in zip(items, masks)
    ]
    return annotation_page(result_id, f"{label} | mask-eval", [], annotations)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mapsnap mask-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("volume", type=Path, help="A volume with main.iiif.json")
    parser.add_argument("--masker", choices=sorted(MASKERS), default="region")
    parser.add_argument(
        "--work",
        type=Path,
        help="Work directory (default: <volume>/artifacts/mask-eval)",
    )
    parser.add_argument(
        "--centerlines",
        type=Path,
        help="Street centerlines (default: the volume's centerlines.geojson)",
    )
    parser.add_argument(
        "--allmaps",
        action="store_true",
        help="Also count the maps Allmaps fails to triangulate (needs Node and app/)",
    )
    args = parser.parse_args()
    volume: Path = args.volume
    work: Path = args.work or volume / "artifacts" / "mask-eval"
    centerlines = args.centerlines or default_centerlines(volume)
    if centerlines is None and args.masker in ("region", "blocks"):
        parser.error(f"the {args.masker} masker needs --centerlines")
    skipped = write_oim_sidecars(volume, work)
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped)}", file=sys.stderr)
    page = build_annotation_page(
        volume / "main.iiif.json", work, MASKERS[args.masker], centerlines
    )
    output = work / f"{args.masker}.iiif.json"
    output.write_text(json.dumps(page, indent=2))
    print(f"Wrote {output}", file=sys.stderr)
    scores = score_annotation(
        output,
        truth_path=volume / "main.iiif.json",
        oim_dir=volume / "oim",
        centerlines_path=centerlines,
        allmaps=args.allmaps,
    )
    json.dump(scores, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
