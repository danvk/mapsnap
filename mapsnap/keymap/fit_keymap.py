"""Georeference a Sanborn key map by fitting a transform to its page numbers.

Each georeferenced page in a volume has a footprint (``corners`` in its ``p*.georef.json``)
and hence a centroid in world coordinates. The key-map detector (mapsnap.keymap.detect_numbers_crnn)
gives the pixel location of each page number drawn on the index map. Pairing those, a
transform maps key-map pixels to world coordinates.

The fit is robust via plain RANSAC: a minimal point sample defines a candidate transform, and
a page is an inlier when the model maps its page-number pixel inside that page's own frame.
Because a page can be split, its number may appear several times on the key map; every
occurrence is tried as a possible match.

The model is a 4-parameter reflected similarity (uniform scale, rotation, reflection,
translation) — the same orientation-reversing family the per-page georeferencer uses
(mapsnap.georef_from_labels.solve_similarity_2pts), since a top-down image maps to a north-up
world with a y-flip. On Washington DC vol 2 it fits 49/61 inliers, identical to a full
6-parameter affine (the extra shear/anisotropy DOF buy nothing: the map is near-isotropic).
World coordinates are projected to a local equirectangular metre frame first so distances are
meaningful.

    uv run python -m mapsnap.keymap.fit_keymap data/washington_dc_1916_vol_2
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapsnap.keymap.score_keymap_labels import point_in_polygon

# Metres per degree of latitude (and of longitude after the cos(lat) correction).
METERS_PER_DEGREE = 111_320.0

# Point pairs needed to solve the 4-parameter reflected similarity exactly.
SAMPLE_SIZE = 2

Point = tuple[float, float]
Model = tuple[float, float, float, float]  # a, b, tx, ty


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


def similarity_fit(src: list[Point], dst: list[Point]) -> Model:
    """Least-squares 4-parameter reflected similarity (a, b, tx, ty) mapping ``src`` to ``dst``.

    X = a*x + b*y + tx, Y = b*x - a*y + ty. The 2x2 part [[a, b], [b, -a]] has determinant
    -(a^2 + b^2) < 0, i.e. uniform scale + rotation + a reflection — the orientation-reversing
    family that maps a top-down image to a north-up world (cf. georef_from_labels). Needs at
    least two non-coincident point pairs.
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (big_x, big_y) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0])
        rhs.append(big_x)
        rows.append([-y, x, 0.0, 1.0])
        rhs.append(big_y)
    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    a, b, tx, ty = (float(v) for v in solution)
    return a, b, tx, ty


def similarity_apply(model: Model, point: Point) -> Point:
    """Apply a 4-parameter reflected similarity to a point."""
    a, b, tx, ty = model
    x, y = point
    return a * x + b * y + tx, b * x - a * y + ty


def describe_model(model: Model) -> str:
    """Human-readable scale/rotation summary of a reflected-similarity model."""
    a, b, _, _ = model
    scale = math.hypot(a, b)
    rotation = math.degrees(math.atan2(b, a))
    return f"scale={scale:.3f} m/px, rotation={rotation:+.1f}° (reflected)"


@dataclass
class GeorefPage:
    """A georeferenced page number and its footprint(s) in local metres.

    A page number can be mapped by several georeferenced pieces — a full-colour scan and a
    lower-fidelity ``s`` skeleton of the same area, and each may be split into ``__N`` parts —
    so ``frames`` holds one polygon per piece and the number maps inside the page when it
    lands in ANY of them.
    """

    number: int
    centroid: Point
    frames: list[list[Point]]
    piece_ids: list[str]


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


def superseded_stems(volume: Path) -> set[str]:
    """Page stems made obsolete by the splitter: if ``p239__1.jpg`` exists, ``p239`` is dead."""
    stems = {f.name[: -len(".jpg")] for f in volume.glob("p*.jpg")}
    return {s.split("__")[0] for s in stems if "__" in s}


def georef_variant(filename: str) -> tuple[str, str] | None:
    """(stem, variant) of a georef file, e.g. ('p239__2', '1gcp'); 'canonical' if no suffix."""
    match = re.match(r"^(p\w+)\.georef(?:-(\w+))?\.json$", filename)
    if not match:
        return None
    return match.group(1), match.group(2) or "canonical"


def load_georef_pages(volume: Path) -> tuple[list[GeorefPage], Point]:
    """Georeferenced page footprints grouped by page number, in a shared metre frame.

    Uses only canonical (untossed) georef pieces whose stem was not superseded by the
    splitter, merging a page's full/skeleton/split pieces into one entry. Also returns the
    projection origin (lon0, lat0) so results can be reported as lon/lat.
    """
    dead = superseded_stems(volume)
    by_number: dict[int, list[tuple[str, list[list[float]]]]] = {}
    for path in sorted(volume.glob("*.georef.json")):
        parsed = georef_variant(path.name)
        if parsed is None:
            continue
        stem, _ = parsed
        number = page_number(stem)
        if number is None or stem in dead:
            continue
        corners = json.load(open(path))["corners"]
        by_number.setdefault(number, []).append((stem, corners))

    all_corners = [c for pieces in by_number.values() for _, cs in pieces for c in cs]
    lon0 = sum(c[0] for c in all_corners) / len(all_corners)
    lat0 = sum(c[1] for c in all_corners) / len(all_corners)

    pages: list[GeorefPage] = []
    for number, pieces in sorted(by_number.items()):
        frames = [
            [project(lon, lat, lon0, lat0) for lon, lat in cs] for _, cs in pieces
        ]
        points = [pt for frame in frames for pt in frame]
        centroid = (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
        pages.append(GeorefPage(number, centroid, frames, [stem for stem, _ in pieces]))
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


def maps_inside(model: Model, page: GeorefPage, pixel: Point) -> bool:
    """Whether ``pixel`` maps inside any of the page's piece frames under ``model``."""
    world = similarity_apply(model, pixel)
    return any(point_in_polygon(world, frame) for frame in page.frames)


def inlier_pages(
    model: Model,
    pages: list[GeorefPage],
    correspondences: list[tuple[int, Point]],
) -> set[int]:
    """Page indices whose number maps inside one of the page's own frames under ``model``."""
    inliers: set[int] = set()
    for page_index, pixel in correspondences:
        if maps_inside(model, pages[page_index], pixel):
            inliers.add(page_index)
    return inliers


def ransac(
    pages: list[GeorefPage],
    correspondences: list[tuple[int, Point]],
    *,
    iterations: int = 5000,
    refit_rounds: int = 5,
    rng: np.random.Generator,
) -> tuple[Model | None, set[int]]:
    """RANSAC the reflected similarity (key-map pixels -> world metres).

    A 2-point sample from distinct pages defines a candidate; the score is the number of pages
    whose number maps inside their frame. The best model is refit by least squares on all inlier
    correspondences and re-scored a few times. Returns (model, inlier page indices).
    """
    if len(correspondences) < SAMPLE_SIZE:
        return None, set()

    best_model: Model | None = None
    best_inliers: set[int] = set()
    count = len(correspondences)
    for _ in range(iterations):
        idx = [int(v) for v in rng.choice(count, size=SAMPLE_SIZE, replace=False)]
        if len({correspondences[k][0] for k in idx}) < SAMPLE_SIZE:
            continue  # need distinct pages
        src = [correspondences[k][1] for k in idx]
        dst = [pages[correspondences[k][0]].centroid for k in idx]
        model = similarity_fit(src, dst)
        inliers = inlier_pages(model, pages, correspondences)
        if len(inliers) > len(best_inliers):
            best_model, best_inliers = model, inliers

    for _ in range(refit_rounds):
        if best_model is None:
            break
        chosen = [
            (pixel, pages[page_index].centroid)
            for page_index, pixel in correspondences
            if maps_inside(best_model, pages[page_index], pixel)
        ]
        if len(chosen) < SAMPLE_SIZE:
            break
        best_model = similarity_fit([c[0] for c in chosen], [c[1] for c in chosen])
        best_inliers = inlier_pages(best_model, pages, correspondences)
    return best_model, best_inliers


def volume_page_numbers(volume: Path) -> set[int]:
    """All page numbers present in the volume (from its page images).

    Split panels contribute their base page's number: parsing ``pa__1`` whole
    would read the panel index as a page number (a letter page has none).
    """
    numbers: set[int] = set()
    for image in volume.glob("p*.jpg"):
        number = page_number(image.name.split(".")[0].split("__")[0])
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
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    keymap_path = args.keymap or (args.volume / "p1b.keymap.json")
    pages, (lon0, lat0) = load_georef_pages(args.volume)
    detections = load_detections(keymap_path)
    correspondences = build_correspondences(pages, detections)

    rng = np.random.default_rng(args.seed)
    model, inliers = ransac(pages, correspondences, iterations=args.iterations, rng=rng)
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
            f"p{pages[i].number}"
            for i in sorted(indices, key=lambda i: pages[i].number)
        )

    print(f"key map: {keymap_path.name}   detections: {len(detections)}")
    print(f"fit on {len(pages)} georeferenced pages: {describe_model(model)}")
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
        + " ".join(f"p{p.number}" for p in sorted(not_detected, key=lambda p: p.number))
        + "\n"
    )
    print(
        f"NOT georeferenced but matched in key map ({len(ungeoreferenced)}): "
        "key map locates these"
    )
    pixel_by_number = {d.number: d.pixel for d in detections}
    for number in ungeoreferenced:
        lon, lat = unproject(
            *similarity_apply(model, pixel_by_number[number]), lon0, lat0
        )
        print(f"  p{number}: ~({lat:.5f}, {lon:.5f})")


if __name__ == "__main__":
    main()
