"""Georeference a Sanborn key map by fitting a transform to its page numbers.

Each georeferenced page in a volume has a footprint (``corners`` in its ``p*.georef.json``)
and hence a centroid in world coordinates. The key-map detector (mapsnap.detect_numbers_crnn)
gives the pixel location of each page number drawn on the index map. Pairing those, a
transform maps key-map pixels to world coordinates.

The fit is robust via plain RANSAC: a minimal point sample defines a candidate transform, and
a page is an inlier when the model maps its page-number pixel inside that page's own frame.
Because a page can be split, its number may appear several times on the key map; every
occurrence is tried as a possible match.

Two models are available. A 4-parameter Helmert (uniform scale, rotation, translation) is the
simplest, but in practice key maps are drawn with anisotropic scale / shear relative to the
ground, so the default is a 6-parameter affine, which fits much more cleanly (on Washington DC
vol 2: affine 51/64 inliers at <120 m vs Helmert ~18). World coordinates are projected to a
local equirectangular metre frame first so distances are meaningful.

    uv run python -m mapsnap.fit_keymap data/washington_dc_1916_vol_2
"""

import argparse
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapsnap.score_keymap_labels import point_in_polygon

# Metres per degree of latitude (and of longitude after the cos(lat) correction).
METERS_PER_DEGREE = 111_320.0

Point = tuple[float, float]
Model = tuple[float, ...]
FitFn = Callable[[list[Point], list[Point]], Model]
ApplyFn = Callable[[Model, Point], Point]


def page_number(stem: str) -> int | None:
    """Page number from a page id like ``p133`` or ``p133s`` (the digits), or None."""
    match = re.search(r"\d+", stem)
    return int(match.group()) if match else None


def project(lon: float, lat: float, lon0: float, lat0: float) -> Point:
    """Equirectangular projection of (lon, lat) to local metres about (lon0, lat0)."""
    x = (lon - lon0) * math.cos(math.radians(lat0)) * METERS_PER_DEGREE
    y = (lat - lat0) * METERS_PER_DEGREE
    return x, y


def unproject(x: float, y: float, lon0: float, lat0: float) -> Point:
    """Inverse of project: local metres back to (lon, lat)."""
    lon = lon0 + x / (math.cos(math.radians(lat0)) * METERS_PER_DEGREE)
    lat = lat0 + y / METERS_PER_DEGREE
    return lon, lat


def helmert_fit(src: list[Point], dst: list[Point]) -> Model:
    """Least-squares 4-parameter Helmert (a, b, tx, ty) mapping ``src`` to ``dst``.

    X = a*x - b*y + tx, Y = b*x + a*y + ty (uniform scale, rotation, translation). Needs at
    least two non-coincident point pairs.
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (big_x, big_y) in zip(src, dst):
        rows.append([x, -y, 1.0, 0.0])
        rhs.append(big_x)
        rows.append([y, x, 0.0, 1.0])
        rhs.append(big_y)
    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return tuple(float(v) for v in solution)


def helmert_apply(model: Model, point: Point) -> Point:
    """Apply a 4-parameter Helmert transform to a point."""
    a, b, tx, ty = model
    x, y = point
    return a * x - b * y + tx, b * x + a * y + ty


def affine_fit(src: list[Point], dst: list[Point]) -> Model:
    """Least-squares 6-parameter affine (a, b, c, d, e, f) mapping ``src`` to ``dst``.

    X = a*x + b*y + c, Y = d*x + e*y + f. Needs at least three non-collinear point pairs.
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (big_x, big_y) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0])
        rhs.append(big_x)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0])
        rhs.append(big_y)
    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return tuple(float(v) for v in solution)


def affine_apply(model: Model, point: Point) -> Point:
    """Apply a 6-parameter affine transform to a point."""
    a, b, c, d, e, f = model
    x, y = point
    return a * x + b * y + c, d * x + e * y + f


def describe_model(model: Model) -> str:
    """Human-readable scale/rotation summary of a Helmert or affine model."""
    if len(model) == 4:
        a, b, _, _ = model
        return f"scale={math.hypot(a, b):.3f} m/px, rotation={math.degrees(math.atan2(b, a)):+.1f}°"
    a, b, _, d, e, _ = model
    scale_x = math.hypot(a, d)
    scale_y = math.hypot(b, e)
    rotation = math.degrees(math.atan2(d, a))
    return (
        f"scale=({scale_x:.3f}, {scale_y:.3f}) m/px, rotation={rotation:+.1f}°, "
        f"anisotropy={max(scale_x, scale_y) / min(scale_x, scale_y):.2f}"
    )


MODELS: dict[str, tuple[FitFn, ApplyFn, int]] = {
    "helmert": (helmert_fit, helmert_apply, 2),
    "affine": (affine_fit, affine_apply, 3),
}


@dataclass
class GeorefPage:
    """A georeferenced page: its id, number, and footprint in local metres."""

    page_id: str
    number: int
    centroid: Point
    frame: list[Point]


@dataclass
class Detection:
    """A key-map page-number detection: its number and pixel centroid."""

    number: int
    pixel: Point


def polygon_centroid(polygon: list[list[float]]) -> Point:
    """Mean of a polygon's vertices."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def load_georef_pages(volume: Path) -> tuple[list[GeorefPage], Point]:
    """Canonical ``p*.georef.json`` pages in a shared local metre frame, plus its (lon0, lat0)."""
    files = sorted(p for p in volume.glob("*.georef.json"))
    raw: list[tuple[str, int, list[list[float]]]] = []
    for path in files:
        page_id = path.name.split(".")[0]
        number = page_number(page_id)
        if number is None:
            continue
        corners = json.load(open(path))["corners"]
        raw.append((page_id, number, corners))

    all_corners = [corner for _, _, corners in raw for corner in corners]
    lon0 = sum(c[0] for c in all_corners) / len(all_corners)
    lat0 = sum(c[1] for c in all_corners) / len(all_corners)

    pages: list[GeorefPage] = []
    for page_id, number, corners in raw:
        frame = [project(lon, lat, lon0, lat0) for lon, lat in corners]
        centroid = polygon_centroid([list(p) for p in frame])
        pages.append(GeorefPage(page_id, number, centroid, frame))
    return pages, (lon0, lat0)


def load_detections(keymap_path: Path) -> list[Detection]:
    """Load key-map page-number detections (numeric text only) with pixel centroids."""
    streets = json.load(open(keymap_path))["streets"]
    detections: list[Detection] = []
    for street in streets:
        text = str(street["text"])
        if not text.isdigit():
            continue
        detections.append(Detection(int(text), polygon_centroid(street["polygon"])))
    return detections


def build_correspondences(
    pages: list[GeorefPage], detections: list[Detection]
) -> list[tuple[int, Point]]:
    """Candidate (page index, key-map pixel) pairs where the page number matches a detection."""
    by_number: dict[int, list[Point]] = {}
    for detection in detections:
        by_number.setdefault(detection.number, []).append(detection.pixel)
    correspondences: list[tuple[int, Point]] = []
    for i, page in enumerate(pages):
        for pixel in by_number.get(page.number, []):
            correspondences.append((i, pixel))
    return correspondences


def inlier_pages(
    model: Model,
    apply_fn: ApplyFn,
    pages: list[GeorefPage],
    correspondences: list[tuple[int, Point]],
) -> set[int]:
    """Page indices whose number maps inside the page's own frame under ``model``."""
    inliers: set[int] = set()
    for page_index, pixel in correspondences:
        if point_in_polygon(apply_fn(model, pixel), pages[page_index].frame):
            inliers.add(page_index)
    return inliers


def ransac(
    pages: list[GeorefPage],
    correspondences: list[tuple[int, Point]],
    *,
    fit_fn: FitFn,
    apply_fn: ApplyFn,
    sample_size: int,
    iterations: int = 5000,
    refit_rounds: int = 5,
    rng: np.random.Generator,
) -> tuple[Model | None, set[int]]:
    """RANSAC a transform (key-map pixels -> world metres); return (model, inlier page indices).

    A minimal sample from distinct pages defines a candidate; the score is the number of pages
    whose number maps inside their frame. The best model is refit by least squares on all inlier
    correspondences and re-scored a few times.
    """
    if len(correspondences) < sample_size:
        return None, set()

    best_model: Model | None = None
    best_inliers: set[int] = set()
    count = len(correspondences)
    for _ in range(iterations):
        idx = [int(v) for v in rng.choice(count, size=sample_size, replace=False)]
        if len({correspondences[k][0] for k in idx}) < sample_size:
            continue  # need distinct pages
        src = [correspondences[k][1] for k in idx]
        dst = [pages[correspondences[k][0]].centroid for k in idx]
        model = fit_fn(src, dst)
        inliers = inlier_pages(model, apply_fn, pages, correspondences)
        if len(inliers) > len(best_inliers):
            best_model, best_inliers = model, inliers

    for _ in range(refit_rounds):
        if best_model is None:
            break
        chosen = [
            (pixel, pages[page_index].centroid)
            for page_index, pixel in correspondences
            if point_in_polygon(apply_fn(best_model, pixel), pages[page_index].frame)
        ]
        if len(chosen) < sample_size:
            break
        best_model = fit_fn([c[0] for c in chosen], [c[1] for c in chosen])
        best_inliers = inlier_pages(best_model, apply_fn, pages, correspondences)
    return best_model, best_inliers


def volume_page_numbers(volume: Path) -> set[int]:
    """All page numbers present in the volume (from its page images)."""
    numbers: set[int] = set()
    for image in volume.glob("p*.jpg"):
        number = page_number(image.name.split(".")[0])
        if number is not None:
            numbers.add(number)
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Georeference a key map by fitting a transform to its page numbers."
    )
    parser.add_argument("volume", type=Path, help="Volume directory.")
    parser.add_argument(
        "--keymap",
        type=Path,
        help="Key-map detections JSON (default <volume>/p1b.keymap.json).",
    )
    parser.add_argument("--model", choices=sorted(MODELS), default="affine")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    keymap_path = args.keymap or (args.volume / "p1b.keymap.json")
    pages, (lon0, lat0) = load_georef_pages(args.volume)
    detections = load_detections(keymap_path)
    correspondences = build_correspondences(pages, detections)

    fit_fn, apply_fn, sample_size = MODELS[args.model]
    rng = np.random.default_rng(args.seed)
    model, inliers = ransac(
        pages,
        correspondences,
        fit_fn=fit_fn,
        apply_fn=apply_fn,
        sample_size=sample_size,
        iterations=args.iterations,
        rng=rng,
    )
    if model is None:
        raise SystemExit("Could not fit a model (too few correspondences).")

    georef_numbers = {page.number for page in pages}
    detection_numbers = {detection.number for detection in detections}
    matched = {i for i, page in enumerate(pages) if page.number in detection_numbers}
    outliers = matched - inliers
    not_detected = [page for i, page in enumerate(pages) if i not in matched]
    all_numbers = volume_page_numbers(args.volume)
    ungeoreferenced = sorted((detection_numbers & all_numbers) - georef_numbers)

    def ids(indices: set[int]) -> str:
        return " ".join(
            pages[i].page_id for i in sorted(indices, key=lambda i: pages[i].number)
        )

    print(f"key map: {keymap_path.name}   detections: {len(detections)}")
    print(
        f"{args.model} fit on {len(pages)} georeferenced pages: {describe_model(model)}"
    )
    print(
        f"  inliers={len(inliers)}  outliers={len(outliers)}  "
        f"not-in-keymap={len(not_detected)}  (of {len(matched)} pages with a match)\n"
    )
    print(f"INLIER pages ({len(inliers)}): number maps inside its own frame")
    print(f"  {ids(inliers)}\n")
    print(
        f"OUTLIER pages ({len(outliers)}): has a key-map number but it maps outside frame"
    )
    print(f"  {ids(outliers)}\n")
    print(f"GEOREFERENCED but NOT in key map ({len(not_detected)}):")
    print(
        "  "
        + " ".join(p.page_id for p in sorted(not_detected, key=lambda p: p.number))
        + "\n"
    )
    print(
        f"NOT georeferenced but matched in key map ({len(ungeoreferenced)}): "
        "key map locates these"
    )
    pixel_by_number = {d.number: d.pixel for d in detections}
    for number in ungeoreferenced:
        lon, lat = unproject(*apply_fn(model, pixel_by_number[number]), lon0, lat0)
        print(f"  p{number}: ~({lat:.5f}, {lon:.5f})")


if __name__ == "__main__":
    main()
