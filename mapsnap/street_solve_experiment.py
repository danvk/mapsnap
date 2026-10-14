"""Truth-aware harness for the streets-only georeferencer (issue #168).

Scores street_solve_volume's candidates against the human georeferencing, beside
the RANSAC fit the pipeline produced for the same page, and keeps the truth-centroid
prior that is an experiment-only ceiling. The production commands (candidates,
select) live in street_solve_volume.py and run as `mapsnap street-solve`.

    uv run python -m mapsnap.street_solve_experiment report data/*/
"""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mapsnap.keymap.align_page_region import (
    pose_corners_world,
)
from mapsnap.keymap.fit_keymap import project
from mapsnap.page_units import PageUnit, grid_rmse_ft_between, load_page_units
from mapsnap.street_solve import (
    StreetGates,
    assemble_constraints,
    psi_votes,
    solve_streets_pose,
)
from mapsnap.street_solve_volume import (
    ADOPT_GAP,
    ARTIFACT_DIR,
    STREET_SUFFIX,
    attach_case_folded_truth,
    cmd_candidates,
    cmd_select,
    corners_to_affine,
    page_features,
    page_prior,
    parse_gate_overrides,
    psi_priors_for,
    volume_context,
    write_georef_streets,
)


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


"""This channel's sidecar name: p<stem>.georef-street.json.

A constant, not a literal at each call site: the name has to agree with
reconcile's CHANNEL_ORDER and fit's glob, and when it was spelled out three
times a rename reached two of them. The one it missed still wrote a valid
sidecar, so nothing failed -- the pose just stopped counting as the incumbent.
"""


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
    """Write .georef-street.json (and the truth-pose twin) for chosen pages."""
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
                    suffix=args.suffix,
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
                    {
                        "pose": [round(v, 6) for v in truth_pose],
                        "source": "truth",
                        "n_constraints": len(constraints),
                        "note": "detections scored against the human georeference",
                    },
                    raw_soups,
                    suffix=f"{args.suffix}-truth",
                )
            )
        for path in written:
            print(path)


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser, separated from main() so its defaults are testable.

    The sidecar suffix in particular is a default, not a literal, and a channel
    rename that misses it fails silently (see fit_test's channel-name test)."""
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
        "materialize", help="Write .georef-street.json sidecars for chosen pages."
    )
    materialize.add_argument("volume")
    materialize.add_argument("--pages", required=True)
    materialize.add_argument("--gates")
    materialize.add_argument("--truth-prior", action="store_true")
    materialize.add_argument(
        "--suffix",
        default=STREET_SUFFIX,
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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
