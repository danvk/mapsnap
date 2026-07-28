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
from pathlib import Path

import numpy as np

from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    extract_gcps,
    fit_transform,
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
    pose_corners_world,
    volume_filter_params,
    volume_median_scale_px_per_m,
)
from mapsnap.keymap.locate import KeymapLocator, discover_keymaps
from mapsnap.osm_snap import dedupe_thetas, label_osm_rotations
from mapsnap.street_solve import (
    PriorLocation,
    StreetGates,
    StreetSolveResult,
    assemble_constraints,
    psi_from_theta,
    psi_votes,
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

    report = sub.add_parser("report", help="Head-to-head vs RANSAC.")
    report.add_argument("volumes", nargs="+")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
