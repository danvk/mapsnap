"""Truth-aware harness for the streets-only georeferencer (issue #168).

Runs :mod:`mapsnap.street_solve` over a volume's pages and scores the result against
the human georeferencing, beside the RANSAC fit the pipeline produced for the same
page. The question it answers is narrow and deliberately isolated from the rest of the
pipeline: *given a coarse location, do street constraints alone place a page better
than intersection-GCP RANSAC does?* Nothing here writes production sidecars.

Each page gets a location prior from the first rung that applies, recorded per page so
a reader can tell what the fit was given:

  keymap-exact    the key map places this page key on its own (the normal case)
  keymap-family   only the page's stem family is placed; usable when those centers
                  are tight, useless when they are a whole lettered family apart
  fit-center      the existing RANSAC fit's own centre — truth-free, and the honest
                  fallback for pages whose key-map prior is degenerate
  truth-centroid  the truth footprint's centre; an experiment-only ceiling, never
                  mixed into the headline comparison

    uv run python -m mapsnap.street_solve_experiment candidates data/los_angeles_ca_1949_vol_14
    uv run python -m mapsnap.street_solve_experiment report data/*/
"""

import argparse
import contextlib
import io
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    extract_gcps,
    fit_transform,
    truth_polygons_by_page,
)
from mapsnap.edge_join_experiment import (
    PageUnit,
    TruthFit,
    grid_rmse_ft_between,
    load_page_units,
    load_truth_units,
    scale_affine_to_local,
)
from mapsnap.georef_from_labels import LabelFeature, prepare_label_features
from mapsnap.keymap.align_page_region import (
    angle_difference_mod180,
    pose_corners_world,
    pose_world_of,
    volume_filter_params,
    volume_median_scale_px_per_m,
)
from mapsnap.keymap.fit_keymap import project, unproject
from mapsnap.keymap.locate import KeymapLocator, discover_keymaps
from mapsnap.osm_snap import dedupe_thetas, label_osm_rotations
from mapsnap.road_model import page_world_affine
from mapsnap.street_solve import (
    PriorLocation,
    StreetGates,
    StreetSolveResult,
    assemble_constraints,
    psi_from_theta,
    psi_votes,
    residuals_at,
    solve_streets_pose,
)
from mapsnap.streets import build_block_index

ARTIFACT_DIR = "artifacts/street_solve"
# A stem family whose centers span more than this is not a location (LA's 1499 family
# covers twenty blocks); such a page falls through to the next prior rung.
MAX_FAMILY_SPREAD_M = 400.0


def family_spread_m(centers: list[tuple[float, float]]) -> float:
    """Largest distance between any two key-map centers, in metres."""
    if len(centers) < 2:
        return 0.0
    kx = 111_320.0 * math.cos(math.radians(centers[0][1]))
    points = [((lon * kx), lat * 110_540.0) for lon, lat in centers]
    return max(math.dist(a, b) for i, a in enumerate(points) for b in points[i + 1 :])


def page_prior(
    unit: PageUnit,
    locator: KeymapLocator | None,
    *,
    allow_truth: bool = False,
) -> PriorLocation | None:
    """The best available coarse location for a page, by the prior ladder."""
    if locator is not None:
        key = unit.stem[1:].upper()
        exact = locator.locations.get(key)
        if exact:
            center = (
                sum(c[0] for c in exact) / len(exact),
                sum(c[1] for c in exact) / len(exact),
            )
            return PriorLocation(center, locator.radius_m, "keymap-exact", list(exact))
        family = locator.centers_for(key)
        if family and family_spread_m(family) <= MAX_FAMILY_SPREAD_M:
            center = (
                sum(c[0] for c in family) / len(family),
                sum(c[1] for c in family) / len(family),
            )
            return PriorLocation(
                center, locator.radius_m, "keymap-family", list(family)
            )
    radius = locator.radius_m if locator is not None else 500.0
    if unit.gen_affine is not None:
        lon, lat = unit.gen_affine @ np.array(
            [unit.width / 2.0, unit.height / 2.0, 1.0]
        )
        return PriorLocation((float(lon), float(lat)), radius, "fit-center")
    if allow_truth and unit.truth is not None:
        lon, lat = unit.truth.affine_local @ np.array(
            [unit.width / 2.0, unit.height / 2.0, 1.0]
        )
        return PriorLocation((float(lon), float(lat)), radius, "truth-centroid")
    return None


def page_features(
    volume: Path,
    unit: PageUnit,
    prior: PriorLocation,
    centerlines: list[dict],
    filter_params: dict,
) -> tuple[list[LabelFeature], dict, tuple[int, int]] | None:
    """(features, block index, label frame) for a page, restricted to its prior.

    The vocabulary is the streets near the prior only — no rectangle or volume-wide
    fallback, which is the whole point of running this channel on a located page.
    """
    streets_path = volume / f"{unit.stem}.streets.json"
    if not streets_path.exists():
        return None
    # A one-page locator over the prior's centers: reuses the same radius search the
    # pipeline uses, without needing the page to be in the real key map.
    near = KeymapLocator(
        {"1": list(prior.centers)}, prior.radius_m
    ).restricted_features("1", centerlines)
    if not near:
        return None
    block_index = build_block_index({"type": "FeatureCollection", "features": near})
    doc = json.loads(streets_path.read_text())
    label_size = (int(doc["width"]), int(doc["height"]))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        features = prepare_label_features(
            str(streets_path), block_index, label_size, **filter_params
        )
    return features, block_index, label_size


def psi_priors_for(
    features: list[LabelFeature], block_index: dict, prior: PriorLocation
) -> list[tuple[float, str]]:
    """Evidence-seeded page bearings, best rung first (never a blind sweep, see G6)."""
    rotations = dedupe_thetas(label_osm_rotations(features, block_index, prior.center))
    return [(psi_from_theta(r.theta_deg), r.source) for r in rotations]


def solve_page(
    volume: Path,
    unit: PageUnit,
    locator: KeymapLocator | None,
    centerlines: list[dict],
    filter_params: dict,
    scale_px_per_m: float | None,
    gates: StreetGates,
    *,
    allow_truth_prior: bool = False,
) -> dict:
    """One page's streets-only fit, scored against truth and against RANSAC."""
    record: dict = {
        "stem": unit.stem,
        "fit_state": unit.fit_state,
        "ransac_rmse_ft": unit.rmse_ft,
        "ransac_inlier_intersections": unit.inlier_intersections,
        "ransac_inlier_streets": unit.inlier_streets,
        "has_truth": unit.truth is not None,
    }
    prior = page_prior(unit, locator, allow_truth=allow_truth_prior)
    if prior is None:
        record["status"] = "no-prior"
        return record
    record["prior_source"] = prior.source
    record["prior_radius_m"] = round(prior.radius_m, 1)

    prepared = page_features(volume, unit, prior, centerlines, filter_params)
    if prepared is None:
        record["status"] = "no-vocabulary"
        return record
    features, block_index, label_size = prepared
    size = (unit.width, unit.height)
    constraints = assemble_constraints(
        features,
        block_index,
        prior=prior,
        label_size=label_size,
        working_size=size,
        scale_px_per_m=scale_px_per_m or 1.0,
        gates=gates,
    )
    record["n_constraints"] = len(constraints)
    record["constraint_names"] = sorted({c[2] for c in constraints})

    # Constraint votes first (they read the tangent under each label), then the
    # osm_snap rungs, which see streets the constraint set may have dropped.
    psi_priors = psi_votes(constraints, gates) + psi_priors_for(
        features, block_index, prior
    )
    record["n_psi_priors"] = len(psi_priors)
    prior_log_scale = math.log(scale_px_per_m) if scale_px_per_m else 0.0
    result: StreetSolveResult = solve_streets_pose(
        constraints,
        size=size,
        prior_log_scale=prior_log_scale,
        psi_priors=psi_priors,
        gates=gates,
        prior_radius_m=prior.radius_m,
    )
    record["psi_source"] = result.psi_source
    record["scale_source"] = result.scale_source
    record["n_inliers"] = result.n_inliers
    record["bearing_spread_deg"] = round(result.bearing_spread_deg, 1)
    record["diagnostics"] = [
        {
            "name": d.name,
            "position_m": None if d.position_m is None else round(d.position_m, 1),
            "angle_deg": None if d.angle_deg is None else round(d.angle_deg, 2),
            "inlier": d.inlier,
        }
        for d in result.diagnostics
    ]
    if result.pose is None:
        record["status"] = f"abstain-{result.abstain}"
        return record

    record["status"] = "posed"
    record["pose"] = [round(v, 6) for v in result.pose]
    corners = pose_corners_world(result.pose, size, prior.center)
    record["corners"] = corners
    affine = corners_to_affine(corners, size)
    record["street_scale_px_per_m"] = round(math.exp(result.pose[3]), 4)
    if unit.truth is not None:
        record["street_rmse_ft"] = round(
            grid_rmse_ft_between(
                affine, unit.truth.affine_local, unit.width, unit.height
            ),
            1,
        )
    return record


def corners_to_affine(corners: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    """Local-pixel affine from TL/TR/BR/BL world corners (the quad is a parallelogram)."""
    top_left = np.array(corners[0], dtype=float)
    top_right = np.array(corners[1], dtype=float)
    bottom_left = np.array(corners[3], dtype=float)
    width, height = size
    column_x = (top_right - top_left) / width
    column_y = (bottom_left - top_left) / height
    return np.array(
        [
            [column_x[0], column_y[0], top_left[0]],
            [column_x[1], column_y[1], top_left[1]],
        ]
    )


def attach_case_folded_truth(volume: Path, units: list[PageUnit]) -> int:
    """Attach truth to pages whose truth key differs from the jpg stem only in case.

    Lettered pages are written ``p1499J`` in the truth annotations and ``p1499j`` on
    disk, so an exact-key lookup silently leaves them unscored — and unscored pages
    would quietly vanish from a head-to-head comparison. Returns how many were fixed.
    """
    truth_by_key, _ = load_truth_units(volume)
    folded = {key.lower(): item for key, item in truth_by_key.items()}
    fixed = 0
    for unit in units:
        if unit.truth is not None:
            continue
        item = folded.get(unit.stem.lower())
        if item is None:
            continue
        source = item["target"]["source"]
        affine_full = fit_transform(extract_gcps(item), annotation_transform_type(item))
        unit.truth = TruthFit(
            affine_local=scale_affine_to_local(
                affine_full, source["width"], unit.width
            ),
            gcp_count=len(extract_gcps(item)),
            transform_type=annotation_transform_type(item),
        )
        if unit.gen_affine is not None:
            unit.rmse_ft = grid_rmse_ft_between(
                unit.truth.affine_local, unit.gen_affine, unit.width, unit.height
            )
        fixed += 1
    return fixed


def street_records(
    constraints: list,
    pose,
    size: tuple[int, int],
    origin: tuple[float, float],
    gates: StreetGates,
    label_scale: tuple[float, float],
    raw_soups: dict | None = None,
) -> list[dict]:
    """Every considered detection under a pose, in the .georef.json street schema.

    Adds the diagnostics this channel turns on — the distance and angle to the
    matched centerline, and the point it snapped to — so the debugger can draw the
    residual that decided inlier or outlier.
    """
    records = []
    for constraint in constraints:
        center_px, dir_pix, name, starts, ends = constraint
        position_m, angle_deg = residuals_at(pose, constraint, size)
        world = pose_world_of(pose, center_px, size)
        lon, lat = unproject(float(world[0]), float(world[1]), origin[0], origin[1])
        step = 20.0
        tip_px = (
            center_px[0] + step * math.cos(dir_pix),
            center_px[1] + step * math.sin(dir_pix),
        )
        tip = pose_world_of(pose, tip_px, size)
        tip_lon, tip_lat = unproject(float(tip[0]), float(tip[1]), origin[0], origin[1])
        norm = math.hypot(tip_lon - lon, tip_lat - lat) or 1.0
        snapped = nearest_point_on(world, starts, ends)
        snap_lon, snap_lat = unproject(
            float(snapped[0]), float(snapped[1]), origin[0], origin[1]
        )
        record = {
            "street": name,
            "x": round(center_px[0] / label_scale[0]),
            "y": round(center_px[1] / label_scale[1]),
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "dir_x": round(math.cos(dir_pix), 6),
            "dir_y": round(math.sin(dir_pix), 6),
            "dir_lon": round((tip_lon - lon) / norm, 6),
            "dir_lat": round((tip_lat - lat) / norm, 6),
            "inlier": position_m <= gates.position_gate_m
            and abs(angle_deg) <= gates.angle_gate_deg,
            "position_m": round(position_m, 1),
            "angle_deg": round(angle_deg, 2),
            "snap_lon": round(snap_lon, 7),
            "snap_lat": round(snap_lat, 7),
        }
        if raw_soups is not None:
            record.update(extension_diagnostics(raw_soups.get(name), world, position_m))
        records.append(record)
    return records


def extension_diagnostics(
    raw: tuple[np.ndarray, np.ndarray] | None, world: np.ndarray, used_m: float
) -> dict:
    """How much this constraint leaned on the terminal extension, and whether it should.

    ``distance_to_drawn_m`` is the distance to the street as OSM actually draws it, so
    the gap against the reported distance is what extending the open end bought.
    ``local_bend_deg`` is the widest bearing disagreement among the drawn segments
    within 100 m of that nearest point: a street that is turning there is one whose
    straight continuation is a guess, which is the case a fixed allowance cannot see.
    """
    if raw is None or not len(raw[0]):
        return {}
    starts, ends = raw
    delta = ends - starts
    length_sq = (delta * delta).sum(axis=1)
    t = np.clip(
        ((world - starts) * delta).sum(axis=1) / np.maximum(length_sq, 1e-9), 0, 1
    )
    projected = starts + t[:, None] * delta
    distances = np.linalg.norm(world - projected, axis=1)
    nearest = int(distances.argmin())
    near = np.linalg.norm(projected - projected[nearest], axis=1) <= 100.0
    bearings = np.degrees(np.arctan2(delta[near, 0], delta[near, 1])) % 180.0
    bend = 0.0
    for i, first in enumerate(bearings):
        for second in bearings[i + 1 :]:
            bend = max(bend, abs(angle_difference_mod180(float(first), float(second))))
    return {
        "distance_to_drawn_m": round(float(distances[nearest]), 1),
        "extension_gain_m": round(float(distances[nearest]) - round(used_m, 1), 1),
        "local_bend_deg": round(bend, 1),
    }


def nearest_point_on(point, starts, ends):
    """The closest point to ``point`` on a segment soup (metre frame)."""
    delta = ends - starts
    length_sq = (delta * delta).sum(axis=1)
    t = np.clip(
        ((point - starts) * delta).sum(axis=1) / np.maximum(length_sq, 1e-9), 0, 1
    )
    projected = starts + t[:, None] * delta
    return projected[int(np.linalg.norm(point - projected, axis=1).argmin())]


def truth_pose_for(unit: PageUnit, origin: tuple[float, float]):
    """The truth transform expressed as a StreetPose in the prior's metre frame."""
    if unit.truth is None:
        return None
    affine = unit.truth.affine_local

    def world(px: float, py: float):
        lon, lat = affine @ np.array([px, py, 1.0])
        return np.array(project(float(lon), float(lat), origin[0], origin[1]))

    center = world(unit.width / 2, unit.height / 2)
    up = world(unit.width / 2, unit.height / 2 - 1.0) - center
    psi = math.degrees(math.atan2(up[0], up[1]))
    return (
        float(center[0]),
        float(center[1]),
        psi,
        math.log(1.0 / float(np.linalg.norm(up))),
    )


def truth_rings(volume: Path, unit: PageUnit) -> list | None:
    """The page's human footprint ring(s) as [lon, lat] lists, if truth exists."""
    truth_path = volume / "main.iiif.json"
    if not truth_path.exists():
        return None
    polygons = truth_polygons_by_page(truth_path).get(unit.number)
    if not polygons:
        return None
    return [[[float(x), float(y)] for x, y in ring] for ring in polygons]


def write_georef_streets(
    volume: Path,
    unit: PageUnit,
    prior: PriorLocation,
    constraints: list,
    pose,
    gates: StreetGates,
    label_size: tuple[int, int],
    suffix: str,
    extra: dict,
    raw_soups: dict | None = None,
) -> Path:
    """Write one <stem>.georef-streets*.json in the pipeline's sidecar schema."""
    size = (unit.width, unit.height)
    label_scale = (size[0] / label_size[0], size[1] / label_size[1])
    doc = {
        "width": label_size[0],
        "height": label_size[1],
        "corners": pose_corners_world(pose, size, prior.center),
        "streets": street_records(
            constraints, pose, size, prior.center, gates, label_scale, raw_soups
        ),
        "intersections": [],
        "keymap": {
            "lat": round(prior.center[1], 7),
            "lon": round(prior.center[0], 7),
            "radius_m": round(prior.radius_m, 1),
            "centers": [[round(c[0], 7), round(c[1], 7)] for c in prior.centers],
            "source": prior.source,
        },
        "street_solve": extra,
    }
    rings = truth_rings(volume, unit)
    if rings:
        # The human footprint, so one file shows where the page belongs beside where
        # this fit put it.
        doc["truth"] = rings
    path = volume / f"{unit.stem}.georef-{suffix}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def volume_context(volume: Path):
    """(locator, centerlines, filter params, volume scale) for a volume."""
    keymaps = discover_keymaps([str(volume / "p1.jpg")])
    locator = KeymapLocator.from_keymaps(keymaps) if keymaps else None
    centerlines_path = volume / "centerlines.geojson"
    centerlines = (
        json.loads(centerlines_path.read_text())["features"]
        if centerlines_path.exists()
        else []
    )
    return (
        locator,
        centerlines,
        volume_filter_params(volume),
        volume_median_scale_px_per_m(volume),
    )


def cmd_candidates(args: argparse.Namespace) -> None:
    """Solve every eligible page of a volume and write candidates.jsonl."""
    volume = Path(args.volume)
    locator, centerlines, filter_params, scale = volume_context(volume)
    if not centerlines:
        sys.exit(f"{volume} has no centerlines.geojson")
    wanted = set(args.pages.split(",")) if args.pages else None
    units = load_page_units(volume)
    fixed = attach_case_folded_truth(volume, units)
    if fixed:
        print(f"attached truth to {fixed} case-mismatched page(s)", file=sys.stderr)
    gates = StreetGates(**parse_gate_overrides(args.gates))
    records = []
    for unit in units:
        if wanted is not None and unit.stem not in wanted:
            continue
        record = solve_page(
            volume,
            unit,
            locator,
            centerlines,
            filter_params,
            scale,
            gates,
            allow_truth_prior=args.truth_prior,
        )
        records.append(record)
        print(
            f"{unit.stem:<10} {record.get('status', '?'):<26} "
            f"prior={record.get('prior_source', '-'):<14} "
            f"constraints={record.get('n_constraints', 0):<3} "
            f"inliers={record.get('n_inliers', 0):<3} "
            f"streets={format_ft(record.get('street_rmse_ft'))} "
            f"ransac={format_ft(record.get('ransac_rmse_ft'))}",
            flush=True,
        )
    out_dir = volume / ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.jsonl"
    out_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(f"\nwrote {out_path} ({len(records)} pages)")


def format_ft(value: float | None) -> str:
    """A right-aligned RMSE cell, or a dash when the page has no such fit."""
    return "     -" if value is None else f"{value:6.0f}"


def parse_gate_overrides(text: str | None) -> dict:
    """``angle_gate_deg=6,position_gate_m=60`` -> kwargs for StreetGates."""
    if not text:
        return {}
    overrides = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        overrides[key.strip()] = float(value)
    return overrides


# Two poses closer than this are not a disagreement worth refereeing.
DISAGREE_FT = 25.0
# The streets pose is adopted only when the referee prefers it by more than this.
# Swept over all 123 production-incumbent disagreements: every threshold loses
# zero pages and creates zero disasters, so this only trades how much upside is
# taken; 0.2 sits in the middle of a flat plateau at ~95% precision.
ADOPT_GAP = 0.2


def incumbent_pose(volume: Path, unit: PageUnit) -> tuple[np.ndarray | None, str]:
    """(affine, source) of the pose the pipeline publishes for a page.

    Snap's sidecar takes priority in the production glob, so on a page it acted on
    that -- not the RANSAC fit -- is what a challenger has to beat.
    """
    osm_path = volume / f"{unit.stem}.georef-osm.json"
    if osm_path.exists():
        try:
            return page_world_affine(json.loads(osm_path.read_text())), "snap"
        except (KeyError, TypeError, ValueError):
            pass
    return unit.gen_affine, "ransac"


def cmd_select(args: argparse.Namespace) -> None:
    """Adopt the streets pose where an independent referee prefers it.

    Neither channel can say which of the two is right -- their agreement is a coin
    flip and neither ranks its own fits. `osm_snap.evaluate_pose` can: it scores any
    pose by road-skeleton chamfer against OSM plus name alignment, evidence derived
    from neither channel's fitting criterion. Both poses are scored by identical
    features, and the streets pose is written only when the referee prefers it by
    ADOPT_GAP. Measured over every disagreement in the twelve truth volumes, that
    picks the closer pose 86% of the time and never costs a page its <=25 ft
    placement.
    """
    from mapsnap.osm_snap import evaluate_pose
    from mapsnap.osm_snap_experiment import build_page_context, load_volume_context

    volume = Path(args.volume)
    locator, centerlines, filter_params, scale = volume_context(volume)
    gates = StreetGates(**parse_gate_overrides(args.gates))
    units = load_page_units(volume)
    attach_case_folded_truth(volume, units)
    posed = {}
    path = volume / ARTIFACT_DIR / "candidates.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") == "posed" and record.get("corners"):
                posed[record["stem"]] = record
    if not posed:
        sys.exit(f"no posed candidates in {path}; run `candidates` first")

    for stale in volume.glob("p*.georef-streets.json"):
        stale.unlink()  # this command owns them; never leave a previous run's pick

    vctx = load_volume_context(volume, units)
    adopted = 0
    for unit in units:
        record = posed.get(unit.stem)
        if record is None:
            continue
        incumbent_affine, source = incumbent_pose(volume, unit)
        if incumbent_affine is None:
            continue
        size = (unit.width, unit.height)
        streets_affine = corners_to_affine(record["corners"], size)
        apart = grid_rmse_ft_between(
            streets_affine, incumbent_affine, unit.width, unit.height
        )
        if apart < DISAGREE_FT:
            continue
        ctx, _status = build_page_context(vctx, unit)
        if ctx is None:
            continue
        incumbent = evaluate_pose(ctx, vctx.feature_index, incumbent_affine)
        challenger = evaluate_pose(ctx, vctx.feature_index, streets_affine)
        if incumbent is None or challenger is None:
            continue
        gap = challenger["verification"] - incumbent["verification"]
        if gap <= args.adopt_gap:
            continue
        prior = page_prior(unit, locator)
        if prior is None:
            continue
        prepared = page_features(volume, unit, prior, centerlines, filter_params)
        if prepared is None:
            continue
        features, block_index, label_size = prepared
        constraints = assemble_constraints(
            features,
            block_index,
            prior=prior,
            label_size=label_size,
            working_size=size,
            scale_px_per_m=scale or 1.0,
            gates=gates,
        )
        write_georef_streets(
            volume,
            unit,
            prior,
            constraints,
            tuple(record["pose"]),
            gates,
            label_size,
            "streets",
            {
                "pose": record["pose"],
                "adopted_over": source,
                "verification": {
                    "streets": round(challenger["verification"], 4),
                    "incumbent": round(incumbent["verification"], 4),
                    "gap": round(gap, 4),
                },
                "disagreement_ft": round(apart, 1),
                "rmse_ft": record.get("street_rmse_ft"),
            },
        )
        adopted += 1
        print(
            f"{unit.stem:<10} adopted over {source:<6} gap {gap:+.3f} "
            f"(apart {apart:.0f} ft)",
            flush=True,
        )
    print(f"\n{adopted} page(s) adopted in {volume.name}")


def cmd_report(args: argparse.Namespace) -> None:
    """Head-to-head table: streets-only vs RANSAC, per volume and aggregate."""
    rows: list[dict] = []
    for volume_arg in args.volumes:
        path = Path(volume_arg) / ARTIFACT_DIR / "candidates.jsonl"
        if not path.exists():
            print(f"skip {volume_arg}: no candidates.jsonl", file=sys.stderr)
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            record["volume"] = Path(volume_arg).name
            rows.append(record)
    if not rows:
        sys.exit("no candidate records; run `candidates` first")

    posed = [r for r in rows if r.get("status") == "posed"]
    comparable = [
        r
        for r in posed
        if r.get("street_rmse_ft") is not None and r.get("ransac_rmse_ft") is not None
    ]
    print(f"{len(rows)} pages, {len(posed)} posed, {len(comparable)} comparable\n")
    print(f"{'page':<22} {'prior':<14} {'streets':>8} {'ransac':>8} {'delta':>8}")
    for record in sorted(
        comparable, key=lambda r: r["ransac_rmse_ft"] - r["street_rmse_ft"]
    ):
        delta = record["ransac_rmse_ft"] - record["street_rmse_ft"]
        print(
            f"{record['volume'][:12]}/{record['stem']:<9} "
            f"{record.get('prior_source', '-'):<14} "
            f"{record['street_rmse_ft']:8.0f} {record['ransac_rmse_ft']:8.0f} "
            f"{delta:+8.0f}"
        )
    wins = sum(1 for r in comparable if r["street_rmse_ft"] < r["ransac_rmse_ft"] - 5)
    losses = sum(1 for r in comparable if r["street_rmse_ft"] > r["ransac_rmse_ft"] + 5)
    print(
        f"\nstreets better on {wins}, worse on {losses}, "
        f"within 5 ft on {len(comparable) - wins - losses}"
    )
    print("\nabstentions:")
    reasons: dict[str, int] = {}
    for record in rows:
        status = record.get("status", "?")
        if status != "posed":
            reasons[status] = reasons.get(status, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<32} {count}")


def cmd_materialize(args: argparse.Namespace) -> None:
    """Write .georef-streets.json (and the truth-pose twin) for chosen pages."""
    volume = Path(args.volume)
    locator, centerlines, filter_params, scale = volume_context(volume)
    gates = StreetGates(**parse_gate_overrides(args.gates))
    units = load_page_units(volume)
    attach_case_folded_truth(volume, units)
    wanted = set(args.pages.split(","))
    for unit in units:
        if unit.stem not in wanted:
            continue
        prior = page_prior(unit, locator, allow_truth=args.truth_prior)
        if prior is None:
            print(f"{unit.stem}: no prior", file=sys.stderr)
            continue
        prepared = page_features(volume, unit, prior, centerlines, filter_params)
        if prepared is None:
            print(f"{unit.stem}: no vocabulary", file=sys.stderr)
            continue
        features, block_index, label_size = prepared
        size = (unit.width, unit.height)
        constraints = assemble_constraints(
            features,
            block_index,
            prior=prior,
            label_size=label_size,
            working_size=size,
            scale_px_per_m=scale or 1.0,
            gates=gates,
        )
        # The same constraints without the terminal extension, so each record can say
        # what the extension bought and whether the street bends where it was applied.
        drawn = assemble_constraints(
            features,
            block_index,
            prior=prior,
            label_size=label_size,
            working_size=size,
            scale_px_per_m=scale or 1.0,
            gates=StreetGates(**{**asdict(gates), "terminal_extrapolation_m": 0.0}),
        )
        raw_soups = {c[2]: (c[3], c[4]) for c in drawn}
        psi_priors = psi_votes(constraints, gates) + psi_priors_for(
            features, block_index, prior
        )
        result = solve_streets_pose(
            constraints,
            size=size,
            prior_log_scale=math.log(scale) if scale else 0.0,
            psi_priors=psi_priors,
            gates=gates,
            prior_radius_m=prior.radius_m,
        )
        written = []
        if result.pose is not None:
            rmse = (
                round(
                    grid_rmse_ft_between(
                        corners_to_affine(
                            pose_corners_world(result.pose, size, prior.center), size
                        ),
                        unit.truth.affine_local,
                        unit.width,
                        unit.height,
                    ),
                    1,
                )
                if unit.truth is not None
                else None
            )
            written.append(
                write_georef_streets(
                    volume,
                    unit,
                    prior,
                    constraints,
                    result.pose,
                    gates,
                    label_size,
                    args.suffix,
                    {
                        "pose": [round(v, 6) for v in result.pose],
                        "psi_source": result.psi_source,
                        "scale_source": result.scale_source,
                        "n_inliers": result.n_inliers,
                        "n_constraints": len(constraints),
                        "rmse_ft": rmse,
                        "ransac_rmse_ft": (
                            None if unit.rmse_ft is None else round(unit.rmse_ft, 1)
                        ),
                    },
                    raw_soups,
                )
            )
        else:
            print(f"{unit.stem}: abstained ({result.abstain})", file=sys.stderr)
        truth_pose = truth_pose_for(unit, prior.center)
        if truth_pose is not None:
            written.append(
                write_georef_streets(
                    volume,
                    unit,
                    prior,
                    constraints,
                    truth_pose,
                    gates,
                    label_size,
                    f"{args.suffix}-truth",
                    {
                        "pose": [round(v, 6) for v in truth_pose],
                        "source": "truth",
                        "n_constraints": len(constraints),
                        "note": "detections scored against the human georeference",
                    },
                    raw_soups,
                )
            )
        for path in written:
            print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    candidates = sub.add_parser("candidates", help="Solve a volume's pages.")
    candidates.add_argument("volume")
    candidates.add_argument("--pages", help="Comma-separated stems (default: all).")
    candidates.add_argument("--gates", help="Overrides, e.g. 'angle_gate_deg=6'.")
    candidates.add_argument(
        "--truth-prior",
        action="store_true",
        help="Allow the truth-centroid prior rung (experiment ceiling only).",
    )
    candidates.set_defaults(func=cmd_candidates)

    materialize = sub.add_parser(
        "materialize", help="Write .georef-streets.json sidecars for chosen pages."
    )
    materialize.add_argument("volume")
    materialize.add_argument("--pages", required=True)
    materialize.add_argument("--gates")
    materialize.add_argument("--truth-prior", action="store_true")
    materialize.add_argument(
        "--suffix",
        default="streets",
        help="Sidecar suffix: <stem>.georef-<suffix>.json (default: %(default)s).",
    )
    materialize.set_defaults(func=cmd_materialize)

    select = sub.add_parser(
        "select", help="Adopt the streets pose where the referee prefers it."
    )
    select.add_argument("volume")
    select.add_argument("--gates")
    select.add_argument(
        "--adopt-gap",
        type=float,
        default=ADOPT_GAP,
        help="Verification margin the streets pose must win by (default: %(default)s).",
    )
    select.set_defaults(func=cmd_select)

    report = sub.add_parser("report", help="Head-to-head vs RANSAC.")
    report.add_argument("volumes", nargs="+")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
