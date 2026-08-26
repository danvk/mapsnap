"""Place a page by matching its content region to the key map's region polygon (#226 phase 2).

The region model (mapsnap.region_model) says which part of a page holds its own
content; the key map says where that page's ground is. Matching the two shapes
determines a 4-DOF pose with no OCR at all — the channel for pages whose street
labels are unreadable (asheville's coarse sheets) or whose grid no longer exists
(hudson's fill, detroit's east side).

Shape matching aliases by construction: a rectangular page matches a rectangular
region at four rotations, and a soft region boundary (hudson's waterfront) admits
a range of poses. So this emits CANDIDATES, ranked by shape IoU, for the
reconciler (#270/#340) to judge against evidence this channel cannot see —
neighbour stamps, P(road) chamfer, volume context. It never publishes a fit.

    uv run python -m mapsnap.region_match data/detroit_mich_1929_vol_11 --grade
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry.base import BaseGeometry

from mapsnap.edge_join_experiment import volume_median_scale
from mapsnap.keymap.align_page_region import (
    Model,
    icp_refine,
    polygon_exterior,
    polygon_iou,
    region_polygon_metres,
    resample_ring,
    ring_centroid,
    similarity_from_pose,
    transformed_page_polygon,
)
from mapsnap.keymap.fit_keymap import similarity_apply, unproject
from mapsnap.keymap.locate import KeymapLocator, discover_keymaps
from mapsnap.osm_snap_experiment import grid_rmse_ft_between, load_page_units
from mapsnap.region_model import REGION_MODEL_PATH, predict_region
from mapsnap.road_model import UNet
from mapsnap.street_solve_experiment import corners_to_affine
from mapsnap.utils import pose_is_upside_down

Point = tuple[float, float]

# Rotation sweep: the shape score is the only orientation evidence this channel
# has, so the sweep is dense enough to find the true pose and the aliases are
# kept as candidates rather than resolved here.
ROTATION_STEP_DEG = 5.0
ICP_ITERATIONS = 3
RING_POINTS = 96
MAX_CANDIDATES = 4
# Two candidates are the same placement when their centres and rotations agree.
DISTINCT_CENTRE_M = 60.0
DISTINCT_ROTATION_DEG = 12.0
# Below this the shape match says nothing (a blob against a blob).
MIN_IOU = 0.35


@dataclass
class RegionCandidate:
    """One shape-matched placement of a page, with the evidence behind it."""

    iou: float
    rotation_deg: float
    scale_m_per_px: float
    corners: list[list[float]]
    rmse_ft: float | None = None


def page_outline(prob: np.ndarray, threshold: float = 0.5) -> list[Point] | None:
    """The page's content outline in page pixels: largest blob above ``threshold``.

    Largest-component rather than every component, and a simplified contour
    rather than the raw mask: the model's boundaries are soft on exactly the
    pages this channel serves (hudson's waterfront median IoU 0.78), so the
    usable signal is the coarse shape, not the pixels.
    """
    mask = (prob >= threshold).astype(np.uint8)
    if mask.sum() < 0.02 * mask.size:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    if len(simplified) < 3:
        return None
    return [(float(p[0][0]), float(p[0][1])) for p in simplified]


def outline_area_px(outline: list[Point]) -> float:
    """Polygon area of an outline in square pixels."""
    return float(abs(cv2.contourArea(np.array(outline, np.float32))))


def pose_corners(
    model: Model, size: tuple[int, int], origin: Point
) -> list[list[float]]:
    """The page's four corners in (lon, lat) under a similarity model."""
    width, height = size
    return [
        list(unproject(*similarity_apply(model, corner), origin[0], origin[1]))
        for corner in [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    ]


def model_rotation_deg(model: Model) -> float:
    """The model's rotation in degrees (its reflected part removed)."""
    return math.degrees(math.atan2(model[1], model[0]))


def fixed_scale_fit(source: list[Point], target: list[Point], scale: float) -> Model:
    """Best reflected similarity from ``source`` to ``target`` at a KNOWN scale.

    The mapped point is s*R(theta)*p with R the reflection [[c, s], [s, -c]],
    so the optimum is theta = atan2(B, A) over the centred correspondences
    (A = sum(qx*px - qy*py), B = sum(qx*py + qy*px)) and the translation
    follows. Fitting scale as well -- what similarity_fit does -- is what let
    ICP overwrite the volume's scale prior with the shape correspondence's own
    ~20% bias.
    """
    source_array = np.array(source, dtype=float)
    target_array = np.array(target, dtype=float)
    source_centre = source_array.mean(axis=0)
    target_centre = target_array.mean(axis=0)
    p = source_array - source_centre
    q = target_array - target_centre
    a_term = float((q[:, 0] * p[:, 0] - q[:, 1] * p[:, 1]).sum())
    b_term = float((q[:, 0] * p[:, 1] + q[:, 1] * p[:, 0]).sum())
    angle = math.atan2(b_term, a_term)
    a = scale * math.cos(angle)
    b = scale * math.sin(angle)
    tx = target_centre[0] - (a * source_centre[0] + b * source_centre[1])
    ty = target_centre[1] - (b * source_centre[0] - a * source_centre[1])
    return (a, b, float(tx), float(ty))


def fixed_scale_icp(
    source: list[Point],
    target: list[Point],
    model: Model,
    iterations: int,
    scale: float,
) -> Model:
    """icp_refine with the scale held at ``scale`` (see fixed_scale_fit)."""
    target_array = np.array(target)
    for _ in range(iterations):
        matched: list[Point] = []
        for point in source:
            transformed = similarity_apply(model, point)
            distances = np.hypot(
                target_array[:, 0] - transformed[0], target_array[:, 1] - transformed[1]
            )
            nearest = target_array[int(distances.argmin())]
            matched.append((float(nearest[0]), float(nearest[1])))
        model = fixed_scale_fit(source, matched, scale)
    return model


def match_page(
    outline: list[Point],
    region: BaseGeometry,
    size: tuple[int, int],
    origin: Point,
    volume_scale_m_per_px: float | None = None,
) -> list[RegionCandidate]:
    """Shape-matched placements of one page, best IoU first.

    Translation comes from centroid correspondence and rotation from a sweep,
    then ICP polishes each and shape IoU ranks them.

    Scale prefers the VOLUME's median fitted scale over the shape's area
    ratio. Measured on detroit/hudson/NO-1896, the area ratio is biased ~20-25%
    small (median ratio 0.75-0.83 against truth, essentially none within 5%):
    the model's content region and the key map's drawn region are not the same
    fraction of their respective frames, and that mismatch enters as scale
    error. Sheets in a volume share a scale family, so the volume's own fits
    are the better estimate; the area ratio remains the fallback for a volume
    with no fitted page.
    """
    target_ring = polygon_exterior(region)
    if len(target_ring) < 3 or region.area <= 0:
        return []
    source_area = outline_area_px(outline)
    if source_area <= 0:
        return []
    scale = volume_scale_m_per_px or math.sqrt(region.area / source_area)
    source_ring = resample_ring(outline, RING_POINTS)
    target_resampled = resample_ring(target_ring, RING_POINTS)
    source_centroid = ring_centroid(source_ring)
    target_centroid = ring_centroid(target_resampled)

    scored: list[RegionCandidate] = []
    steps = round(360.0 / ROTATION_STEP_DEG)
    for step in range(steps):
        angle = math.radians(step * ROTATION_STEP_DEG)
        unit = (math.cos(angle), math.sin(angle))
        model = similarity_from_pose(unit, scale, source_centroid, target_centroid)
        if volume_scale_m_per_px:
            model = fixed_scale_icp(
                source_ring, target_resampled, model, ICP_ITERATIONS, scale
            )
        else:
            model = icp_refine(source_ring, target_resampled, model, ICP_ITERATIONS)
        placed = transformed_page_polygon(source_ring, model)
        iou = polygon_iou(placed, region)
        if iou < MIN_IOU:
            continue
        corners = pose_corners(model, size, origin)
        affine = corners_to_affine(corners, size)
        if pose_is_upside_down(affine):
            continue  # #324: corpus-impossible, 0 of 1,332 truth pages
        scored.append(
            RegionCandidate(
                iou=iou,
                rotation_deg=model_rotation_deg(model),
                scale_m_per_px=math.hypot(model[0], model[1]),
                corners=corners,
            )
        )

    scored.sort(key=lambda c: -c.iou)
    kept: list[RegionCandidate] = []
    for candidate in scored:
        centre = np.mean(np.array(candidate.corners), axis=0)
        if any(_same_placement(candidate, centre, other, origin) for other in kept):
            continue
        kept.append(candidate)
        if len(kept) >= MAX_CANDIDATES:
            break
    return kept


def _same_placement(
    candidate: RegionCandidate,
    centre: np.ndarray,
    other: RegionCandidate,
    origin: Point,
) -> bool:
    """Whether two candidates describe the same placement (centre and rotation)."""
    other_centre = np.mean(np.array(other.corners), axis=0)
    kx = 111_320.0 * math.cos(math.radians(origin[1]))
    metres = math.hypot(
        (centre[0] - other_centre[0]) * kx, (centre[1] - other_centre[1]) * 110_540.0
    )
    rotation = abs(candidate.rotation_deg - other.rotation_deg) % 360.0
    rotation = min(rotation, 360.0 - rotation)
    return metres < DISTINCT_CENTRE_M and rotation < DISTINCT_ROTATION_DEG


def load_region_model(path: Path, device) -> UNet:
    """The trained region UNet, its channel width read from the checkpoint."""
    state = torch.load(path, map_location=device)
    base = state["enc1.block.0.weight"].shape[0]
    model = UNet(base=base, in_channels=3, norm="group").to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_volume(volume: Path, args: argparse.Namespace) -> list[dict]:
    """Match every eligible page of a volume; returns one record per page."""
    from mapsnap.keymap.number_model import select_device

    keymaps = discover_keymaps([str(volume / "p1.jpg")])
    if not keymaps:
        sys.exit(f"{volume} has no key map")
    locator = KeymapLocator.from_keymaps(keymaps)
    device = select_device()
    model = load_region_model(args.model, device)

    units = load_page_units(volume)
    volume_scale = volume_median_scale(units) or None
    print(f"volume scale: {volume_scale} m/px (median of fitted pages)")
    records = []
    for unit in units:
        if unit.fit_state == "fitted" and not args.all_pages:
            continue
        key = unit.stem[1:]
        rings = locator.regions_for(key) or locator.regions_for(key.upper())
        record: dict = {"stem": unit.stem, "fit_state": unit.fit_state}
        if not rings:
            record["status"] = "no-region"
            records.append(record)
            continue
        image = cv2.imread(str(volume / f"{unit.stem}.jpg"))
        if image is None:
            record["status"] = "no-image"
            records.append(record)
            continue
        prob = predict_region(model, image, device)
        outline = page_outline(prob)
        if outline is None:
            record["status"] = "no-outline"
            records.append(record)
            continue
        origin = (
            float(np.mean([p[0] for ring in rings for p in ring])),
            float(np.mean([p[1] for ring in rings for p in ring])),
        )
        region = region_polygon_metres(rings, origin)
        candidates = match_page(
            outline,
            region,
            (image.shape[1], image.shape[0]),
            origin,
            volume_scale_m_per_px=volume_scale,
        )
        if not candidates:
            record["status"] = "no-match"
            records.append(record)
            continue
        if unit.truth is not None:
            for candidate in candidates:
                affine = corners_to_affine(candidate.corners, (unit.width, unit.height))
                candidate.rmse_ft = round(
                    grid_rmse_ft_between(
                        affine, unit.truth.affine_local, unit.width, unit.height
                    ),
                    1,
                )
        record["status"] = "ok"
        record["candidates"] = [
            {
                "iou": round(c.iou, 4),
                "rotation_deg": round(c.rotation_deg, 1),
                "scale_m_per_px": round(c.scale_m_per_px, 4),
                "corners": [[round(v, 7) for v in corner] for corner in c.corners],
                **({"rmse_ft": c.rmse_ft} if c.rmse_ft is not None else {}),
            }
            for c in candidates
        ]
        records.append(record)
        top = candidates[0]
        print(
            f"{unit.stem:10s} {unit.fit_state:10s} iou={top.iou:.3f} "
            f"cands={len(candidates)} rmse={top.rmse_ft}",
            flush=True,
        )
    return records


def report(records: list[dict]) -> None:
    """Print how well the channel's candidates would place graded pages."""
    graded = [
        r
        for r in records
        if r.get("status") == "ok" and "rmse_ft" in r["candidates"][0]
    ]
    if not graded:
        print("\nno truth-graded pages")
        return
    print(f"\n{len(graded)} graded pages")
    for label, pick in (
        ("top-1 (best IoU)", lambda cs: cs[0]["rmse_ft"]),
        ("best of candidates", lambda cs: min(c["rmse_ft"] for c in cs)),
    ):
        errors = sorted(pick(r["candidates"]) for r in graded)
        median = errors[len(errors) // 2]
        buckets = " ".join(
            f"<={t}ft {sum(1 for e in errors if e <= t):2d}" for t in (25, 50, 100, 250)
        )
        print(f"  {label:20s} median {median:8.1f} ft   {buckets}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volumes", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=REGION_MODEL_PATH)
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Also match pages the pipeline already fitted (for measurement).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    for volume in args.volumes:
        print(f"=== {volume.name}")
        records = run_volume(volume, args)
        report(records)
        out_dir = args.out_dir or (volume / "artifacts" / "region_match")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "candidates.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        print(f"wrote {path} ({len(records)} pages)")


if __name__ == "__main__":
    main()
