"""Truth-aware harness for the geometry-first OSM matcher (osm_snap.py).

Commands:
  candidates DIR   generate snap candidates for the volume's unplaced pages
                   -> artifacts/osm_snap/candidates.jsonl (+ vis/ contact sheets)
  report DIR       recall / rank-1 / near-miss diagnostics against truth

The matcher itself lives in osm_snap.py and is truth-free; truth
(main.iiif.json) is used here only to annotate each candidate with its
rmse_ft so ranking quality can be measured. Production selection and
materialization run from candidates.jsonl without touching truth.
"""

import argparse
import contextlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mapsnap.edge_join_experiment import (
    PageUnit,
    detected_pairs,
    grid_rmse_ft_between,
    keymap_region_adjacency,
    load_page_units,
    load_prob,
    volume_median_scale,
)
from mapsnap.georef_from_labels import LabelFeature, prepare_label_features
from mapsnap.keymap.align_page_region import (
    image_neighbor_directions,
    load_adjacency,
    volume_filter_params,
)
from mapsnap.keymap.locate import KeymapLocator
from mapsnap.osm_snap import (
    PageContext,
    RotationPrior,
    SnapCandidate,
    affine_theta_deg,
    calibrated_radius_m,
    frame_around,
    label_osm_rotations,
    adjacency_keymap_rotations,
    osm_rasters,
    page_scale_priors,
    snap_page,
)
from mapsnap.streets import Block, build_block_index
from mapsnap.utils import default_centerlines, haversine_m

# Pages the rescue channel may place: everything the iiif glob does not see.
RESCUE_STATES = {"nofit", "misscale", "1gcp", "outlier", "none"}


def artifacts_dir(volume: Path) -> Path:
    return volume / "artifacts" / "osm_snap"


@dataclass
class VolumeContext:
    """Once-per-volume inputs shared by every page's candidate generation."""

    volume: Path
    units: list[PageUnit]
    features: list[dict]
    locator: KeymapLocator | None
    volume_m_per_px: float
    adjacency: dict
    region_centroids: dict[int, tuple[float, float]]
    filter_params: dict
    radius_m: float
    radius_source: str
    median_theta_deg: float | None


def ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """Vertex-mean centroid of a [lon, lat] ring."""
    return (
        sum(p[0] for p in ring) / len(ring),
        sum(p[1] for p in ring) / len(ring),
    )


def unit_theta_deg(unit: PageUnit) -> float | None:
    """The cv2 rotation of a fitted unit's affine, or None."""
    if unit.gen_affine is None:
        return None
    lon = unit.gen_affine[0, 2]
    lat = unit.gen_affine[1, 2]
    frame = frame_around((lon, lat), half_m=100.0)
    return affine_theta_deg(unit.gen_affine, frame)


def volume_median_theta(units: list[PageUnit]) -> float | None:
    """Circular-mean rotation of the volume's fitted pages, in cv2 degrees."""
    sines = cosines = 0.0
    n = 0
    for unit in units:
        theta = unit_theta_deg(unit) if unit.fit_state == "fitted" else None
        if theta is None:
            continue
        sines += math.sin(math.radians(theta))
        cosines += math.cos(math.radians(theta))
        n += 1
    if n == 0:
        return None
    return math.degrees(math.atan2(sines, cosines))


def keymap_fit_residuals(units: list[PageUnit]) -> list[float]:
    """Fitted pages' distances (m) from their keymap location to the fit center."""
    residuals = []
    for unit in units:
        if unit.fit_state != "fitted" or unit.gen_affine is None:
            continue
        anchors = list(unit.keymap_centers)
        for ring in unit.keymap_regions or []:
            anchors.append(ring_centroid(ring))
        if not anchors:
            continue
        lon_c = (
            unit.gen_affine[0, 0] * unit.width / 2
            + unit.gen_affine[0, 1] * unit.height / 2
            + unit.gen_affine[0, 2]
        )
        lat_c = (
            unit.gen_affine[1, 0] * unit.width / 2
            + unit.gen_affine[1, 1] * unit.height / 2
            + unit.gen_affine[1, 2]
        )
        residuals.append(
            min(haversine_m(lat_c, lon_c, lat, lon) for lon, lat in anchors)
        )
    return residuals


def load_volume_context(volume: Path) -> VolumeContext:
    units = load_page_units(volume)
    centerlines_path = default_centerlines(volume)
    if centerlines_path is None:
        sys.exit(f"no centerlines.geojson under {volume}")
    features = json.loads(centerlines_path.read_text())["features"]
    keymaps = sorted((volume / "raw").glob("*.keymap.json"))
    locator = KeymapLocator.from_keymaps(keymaps) if keymaps else None
    _, region_centroids = keymap_region_adjacency(volume)
    residuals = keymap_fit_residuals(units)
    locator_radius = locator.radius_m if locator else 600.0
    radius, radius_source = calibrated_radius_m(residuals, locator_radius)
    return VolumeContext(
        volume=volume,
        units=units,
        features=features,
        locator=locator,
        volume_m_per_px=volume_median_scale(units),
        adjacency=load_adjacency(volume),
        region_centroids=region_centroids,
        filter_params=volume_filter_params(volume),
        radius_m=radius,
        radius_source=radius_source,
        median_theta_deg=volume_median_theta(units),
    )


def page_keymap_data(
    vctx: VolumeContext, unit: PageUnit
) -> tuple[list[tuple[float, float]], list[list[list[float]]] | None]:
    """(search centers, region rings) for a page, from its sidecar or the locator.

    Search centers are every keymap detection of the page number plus each
    region ring's centroid (split blocks can sit far apart; the matcher tries
    each). Deduped within 50 m.
    """
    centers = list(unit.keymap_centers)
    regions = unit.keymap_regions
    if not centers and vctx.locator is not None:
        entry = vctx.locator.page_keymap(unit.number)
        if entry:
            centers = [tuple(c) for c in entry["centers"]]
            regions = entry.get("regions")
    for ring in regions or []:
        centers.append(ring_centroid(ring))
    deduped: list[tuple[float, float]] = []
    for lon, lat in centers:
        if all(haversine_m(lat, lon, b, a) > 50.0 for a, b in deduped):
            deduped.append((lon, lat))
    return deduped, regions


def page_label_features(
    vctx: VolumeContext, unit: PageUnit
) -> tuple[list[LabelFeature], dict[str, list[Block]]] | None:
    """(label features, restricted block index) for a page, or None.

    The vocabulary is restricted to streets near the page's keymap location
    (falling back to the key-map rectangles), exactly as the main pipeline's
    --keymap path does, so label matching behaves the same way here.
    """
    streets_path = vctx.volume / f"{unit.stem}.streets.json"
    if not streets_path.exists() or vctx.locator is None:
        return None
    near = vctx.locator.restricted_features(unit.number, vctx.features)
    if near is None:
        near = vctx.locator.rectangle_features(vctx.features)
    if not near:
        return None
    block_index = build_block_index({"type": "FeatureCollection", "features": near})
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        features = prepare_label_features(
            str(streets_path),
            block_index,
            (unit.width, unit.height),
            **vctx.filter_params,
        )
    return features, block_index


def rotation_priors_for(
    vctx: VolumeContext,
    unit: PageUnit,
    search_centers: list[tuple[float, float]],
    labels: tuple[list[LabelFeature], dict[str, list[Block]]] | None,
) -> list[RotationPrior]:
    """The rung-ordered rotation-prior ladder for one page (mask rung excluded)."""
    priors: list[RotationPrior] = []
    if labels is not None and search_centers:
        features, block_index = labels
        priors.extend(label_osm_rotations(features, block_index, search_centers[0]))
    own_centroid = vctx.region_centroids.get(unit.number)
    if own_centroid is None and search_centers:
        own_centroid = search_centers[0]
    if own_centroid is not None:
        image_directions = image_neighbor_directions(vctx.adjacency, unit.stem)
        priors.extend(
            adjacency_keymap_rotations(
                image_directions, vctx.region_centroids, own_centroid
            )
        )
    by_number = {u.number: u for u in vctx.units}
    neighbor_thetas: list[float] = []
    for pair in detected_pairs(vctx.volume):
        if unit.number not in pair:
            continue
        (other,) = pair - {unit.number}
        neighbor = by_number.get(other)
        if neighbor is not None and neighbor.fit_state == "fitted":
            theta = unit_theta_deg(neighbor)
            if theta is not None:
                neighbor_thetas.append(theta)
    if neighbor_thetas:
        sines = sum(math.sin(math.radians(t)) for t in neighbor_thetas)
        cosines = sum(math.cos(math.radians(t)) for t in neighbor_thetas)
        priors.append(
            RotationPrior(
                math.degrees(math.atan2(sines, cosines)), 8.0, "ransac-neighbor"
            )
        )
    elif vctx.median_theta_deg is not None:
        priors.append(RotationPrior(vctx.median_theta_deg, 15.0, "volume-median-theta"))
    return priors


def build_page_context(
    vctx: VolumeContext, unit: PageUnit
) -> tuple[PageContext | None, str]:
    """(PageContext, status) for one page; context is None unless status 'ok'."""
    prob = load_prob(vctx.volume, unit.stem)
    if prob is None:
        return None, "no_prob"
    centers, regions = page_keymap_data(vctx, unit)
    if not centers:
        return None, "no_keymap"
    labels = page_label_features(vctx, unit)
    priors = rotation_priors_for(vctx, unit, centers, labels)
    scales = page_scale_priors(vctx.volume_m_per_px, regions, unit.width, unit.height)
    ctx = PageContext(
        stem=unit.stem,
        number=unit.number,
        width=unit.width,
        height=unit.height,
        prob=prob,
        search_centers=centers,
        radius_m=vctx.radius_m,
        rotation_priors=priors,
        scale_priors=scales,
        keymap_regions=regions,
        label_features=labels[0] if labels else None,
        block_index=labels[1] if labels else None,
    )
    return ctx, "ok"


def candidate_record(candidate: SnapCandidate, unit: PageUnit) -> dict:
    """JSON-serializable record of one candidate, with truth rmse if known."""
    record = {
        "world_affine": [[float(v) for v in row] for row in candidate.world_affine],
        "center": [round(candidate.center[0], 7), round(candidate.center[1], 7)],
        "theta_deg": round(candidate.theta_deg, 2),
        "theta_source": candidate.theta_source,
        "scale_m_per_px": round(candidate.scale_m_per_px, 4),
        "scale_source": candidate.scale_source,
        "scale_adjust": round(candidate.scale_adjust, 4),
        "ncc": round(candidate.ncc, 4),
        "ncc_fine": round(candidate.ncc_fine, 4),
        "chamfer_mean_m": round(candidate.chamfer_mean_m, 2),
        "inlier_frac": round(candidate.inlier_frac, 4),
        "n_points": candidate.n_points,
        "jtj_eig_ratio": round(candidate.jtj_eig_ratio, 6),
        "overlap_frac": round(candidate.overlap_frac, 4),
        "refine_shift_m": round(candidate.refine_shift_m, 1),
        "center_dist_m": round(candidate.center_dist_m, 1),
        "verification": round(candidate.verification, 4)
        if math.isfinite(candidate.verification)
        else None,
        "select_score": round(candidate.select_score(), 4)
        if math.isfinite(candidate.select_score())
        else None,
        "plausible": candidate.plausible,
        "gate_reasons": candidate.gate_reasons,
    }
    if candidate.region_containment is not None:
        record["region_containment"] = round(candidate.region_containment, 3)
    if candidate.prior_theta_residual_sigma is not None:
        record["prior_theta_residual_sigma"] = round(
            candidate.prior_theta_residual_sigma, 2
        )
    if candidate.name is not None:
        record["name"] = {
            "score": round(candidate.name.score, 4),
            "n_labels": candidate.name.n_labels,
            "n_hits": candidate.name.n_hits,
            "hits": candidate.name.hits,
        }
    if unit.truth is not None:
        record["rmse_ft"] = round(
            grid_rmse_ft_between(
                unit.truth.affine_local,
                candidate.world_affine,
                unit.width,
                unit.height,
            ),
            1,
        )
    return record


def page_record(
    vctx: VolumeContext, unit: PageUnit, limit_note: str | None = None
) -> dict:
    """Generate the full candidates.jsonl record for one page."""
    ctx, status = build_page_context(vctx, unit)
    record: dict = {
        "target": unit.stem,
        "status": status,
        "fit_state": unit.fit_state,
        "width": unit.width,
        "height": unit.height,
        "has_truth": unit.truth is not None,
    }
    if ctx is None:
        return record
    record["search"] = {
        "centers": [[round(c[0], 7), round(c[1], 7)] for c in ctx.search_centers],
        "radius_m": round(vctx.radius_m, 1),
        "radius_source": vctx.radius_source,
    }
    record["priors"] = {
        "rotation": [
            {
                "theta_deg": round(p.theta_deg, 2),
                "sigma_deg": p.sigma_deg,
                "source": p.source,
            }
            for p in ctx.rotation_priors
        ],
        "scale": [
            {
                "m_per_px": round(p.m_per_px, 4),
                "sigma_log": p.sigma_log,
                "source": p.source,
            }
            for p in ctx.scale_priors
        ],
    }
    candidates = snap_page(ctx, vctx.features)
    if not candidates:
        record["status"] = "no_candidates"
        return record
    record["candidates"] = [candidate_record(c, unit) for c in candidates]
    scores = [
        c["select_score"] for c in record["candidates"] if c["select_score"] is not None
    ]
    record["margin"] = (
        round(scores[0] - scores[1], 4)
        if len(scores) >= 2
        else (round(scores[0], 4) if scores else None)
    )
    return record


def write_contact_sheet(
    vctx: VolumeContext, unit: PageUnit, record: dict, out_dir: Path
) -> None:
    """Red/green overlay PNGs of the top-2 candidates for eyeballing."""
    prob = load_prob(vctx.volume, unit.stem)
    if prob is None:
        return
    for rank, candidate in enumerate(record.get("candidates", [])[:2]):
        world = np.array(candidate["world_affine"])
        center = (candidate["center"][0], candidate["center"][1])
        diag_m = math.hypot(unit.width, unit.height) * candidate["scale_m_per_px"] / 2
        frame = frame_around(center, half_m=diag_m + 100.0)
        osm_prob, _, _ = osm_rasters(frame, vctx.features)
        pose = frame.page_to_raster_affine(world)
        warped = cv2.warpAffine(prob, pose, (frame.shape[1], frame.shape[0]))
        rgb = np.zeros((*frame.shape, 3), np.uint8)
        rgb[:, :, 1] = (osm_prob * 200).astype(np.uint8)
        rgb[:, :, 2] = (warped * 255).astype(np.uint8)
        scale = min(1.0, 1000.0 / max(rgb.shape[:2]))
        if scale < 1.0:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale)
        rmse = candidate.get("rmse_ft")
        suffix = f"_{rmse:.0f}ft" if rmse is not None else ""
        cv2.imwrite(str(out_dir / f"{unit.stem}_{rank + 1}{suffix}.png"), rgb)


def cmd_candidates(
    volume: Path,
    pages: list[str] | None,
    all_pages: bool,
    limit: int | None,
    recompute: bool,
    vis: bool,
) -> None:
    """Generate candidates.jsonl for the volume's rescue targets."""
    vctx = load_volume_context(volume)
    out_dir = artifacts_dir(volume)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.jsonl"
    existing: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["target"]] = record

    targets = [
        u
        for u in vctx.units
        if (all_pages and u.fit_state != "split") or u.fit_state in RESCUE_STATES
    ]
    if pages:
        wanted = set(pages)
        targets = [u for u in targets if u.stem in wanted]
    if limit is not None:
        targets = targets[:limit]

    print(
        f"{volume.name}: {len(targets)} target pages, radius "
        f"{vctx.radius_m:.0f}m ({vctx.radius_source}), scale "
        f"{vctx.volume_m_per_px:.3f} m/px"
    )
    vis_dir = out_dir / "vis"
    if vis:
        vis_dir.mkdir(exist_ok=True)
    done = 0
    for unit in targets:
        if unit.stem in existing and not recompute and not pages:
            continue
        record = page_record(vctx, unit)
        existing[unit.stem] = record
        done += 1
        best = (record.get("candidates") or [{}])[0]
        rmse = best.get("rmse_ft")
        print(
            f"  {unit.stem:<8} {record['status']:<14}"
            f" cands={len(record.get('candidates', []))}"
            + (f" best_rmse={rmse:.0f}ft" if rmse is not None else "")
            + (
                f" score={best['select_score']}"
                if best.get("select_score") is not None
                else ""
            )
        )
        if vis and record.get("candidates"):
            write_contact_sheet(vctx, unit, record, vis_dir)
        # Rewrite after every page so an interrupted run keeps its progress.
        with out_path.open("w") as handle:
            for stem in sorted(existing):
                handle.write(json.dumps(existing[stem]) + "\n")
    print(f"{done} pages computed; {len(existing)} total in {out_path}")


def load_candidates(volume: Path) -> list[dict]:
    path = artifacts_dir(volume) / "candidates.jsonl"
    if not path.exists():
        sys.exit(f"{path} missing; run `candidates` first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_report(volume: Path) -> None:
    """Recall / ranking diagnostics for the cached candidates, against truth."""
    records = load_candidates(volume)
    by_status: dict[str, int] = {}
    for record in records:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
    print(f"== {volume.name}: {len(records)} pages ==")
    print("  status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))

    scored = [
        r
        for r in records
        if r["status"] == "ok" and r.get("has_truth") and r.get("candidates")
    ]
    if not scored:
        print("  no truth-scored pages")
        return
    recall50 = rank1_50 = rank1_25 = 0
    best_rmses: list[float] = []
    print(
        f"  {'page':<9}{'state':<9}{'cands':>6}{'best set':>10}{'rank-1':>9}"
        f"{'score':>8}{'margin':>8}  theta_src"
    )
    for record in scored:
        candidates = record["candidates"]
        rmses = [c["rmse_ft"] for c in candidates if "rmse_ft" in c]
        if not rmses:
            continue
        best_in_set = min(rmses)
        top = candidates[0]
        top_rmse = top.get("rmse_ft")
        if best_in_set <= 50.0:
            recall50 += 1
        if top_rmse is not None and top_rmse <= 50.0:
            rank1_50 += 1
        if top_rmse is not None and top_rmse <= 25.0:
            rank1_25 += 1
        if top_rmse is not None:
            best_rmses.append(top_rmse)
        print(
            f"  {record['target']:<9}{record['fit_state']:<9}{len(candidates):>6}"
            f"{best_in_set:>9.0f}f{top_rmse:>8.0f}f"
            f"{top.get('select_score') if top.get('select_score') is not None else float('nan'):>8.2f}"
            f"{record.get('margin') if record.get('margin') is not None else float('nan'):>8.2f}"
            f"  {top['theta_source']}"
        )
    n = len(scored)
    print(
        f"  truth-in-top-K (<=50ft): {recall50}/{n} ({recall50 / n:.0%})   "
        f"rank-1 <=50ft: {rank1_50}/{n} ({rank1_50 / n:.0%})   "
        f"rank-1 <=25ft: {rank1_25}/{n} ({rank1_25 / n:.0%})"
    )
    if best_rmses:
        best_rmses.sort()
        median = best_rmses[len(best_rmses) // 2]
        print(f"  rank-1 rmse: median {median:.0f}ft, max {best_rmses[-1]:.0f}ft")


def truth_land_weights(volume: Path) -> tuple[dict[str, float], float]:
    """(land m² per unsplit truth page key, total land over ALL truth items).

    Approximates the `mapsnap score` land weighting closely enough to tune
    gates on cached candidates without re-running iiif+score per setting; the
    final numbers always come from the real pipeline.
    """
    from shapely.geometry import Polygon

    from mapsnap.score import (
        LocalFrame,
        land_fraction,
        street_tree,
        truth_footprint_ring,
    )
    from mapsnap.utils import source_id_to_page_key

    items = json.loads((volume / "main.iiif.json").read_text()).get("items", [])
    centerlines = default_centerlines(volume)
    weights: dict[str, float] = {}
    total = 0.0
    frame: LocalFrame | None = None
    tree = None
    for item in items:
        ring = truth_footprint_ring(item)
        if not ring:
            continue
        if frame is None:
            frame = LocalFrame(ring[0][0], ring[0][1])
            assert centerlines is not None
            tree = street_tree(centerlines, frame)
        polygon = Polygon([frame.to_xy(lon, lat) for lon, lat in ring]).buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            continue
        assert tree is not None
        land = polygon.area * land_fraction(polygon, tree)
        total += land
        key = source_id_to_page_key(
            item.get("target", {}).get("source", {}).get("id"), item.get("label", "")
        )
        if "__" not in key:
            weights[key] = weights.get(key, 0.0) + land
    return weights, total


def simulate_delta_net(
    records: list[dict],
    weights: dict[str, float],
    total_land: float,
    gate_score: float,
    gate_margin: float,
) -> tuple[float, int, int, int]:
    """(simulated Δnet, accepted, good adds, disaster adds) at one gate setting."""
    accepted = good = disaster = 0
    delta = 0.0
    for record in records:
        if record.get("status") != "ok" or not record.get("candidates"):
            continue
        top = record["candidates"][0]
        score = top.get("select_score")
        margin = record.get("margin")
        if score is None or score < gate_score:
            continue
        if margin is None or margin < gate_margin:
            continue
        accepted += 1
        rmse = top.get("rmse_ft")
        weight = weights.get(record["target"])
        if rmse is None or weight is None:
            continue
        if rmse <= 25.0:
            good += 1
            delta += weight
        elif rmse >= 200.0:
            disaster += 1
            delta -= weight
    return (delta / total_land if total_land else 0.0), accepted, good, disaster


def cmd_sweep(volume: Path) -> None:
    """Grid the abstention gates and print the simulated Δnet for each."""
    records = load_candidates(volume)
    weights, total_land = truth_land_weights(volume)
    print(f"== {volume.name}: simulated Δnet over gate grid ==")
    print(f"  {'gate':>6} " + "".join(f"m>={m:<4.1f}" + " " * 14 for m in GATE_MARGINS))
    for gate in GATE_SCORES:
        cells = []
        for margin in GATE_MARGINS:
            delta, accepted, good, disaster = simulate_delta_net(
                records, weights, total_land, gate, margin
            )
            cells.append(
                f"{delta * 100:+5.1f}% ({accepted:>2}a {good:>2}g {disaster}d)"
            )
        print(f"  {gate:>6.2f} " + "  ".join(cells))
    print("  (a=accepted, g=good <=25ft, d=disaster >=200ft; Δnet is land-weighted)")


GATE_SCORES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
GATE_MARGINS = [0.0, 0.1, 0.25, 0.5]


def cmd_select(volume: Path, mode: str, gate_score: float, gate_margin: float) -> None:
    """Pick one candidate (or abstain) per page; write selection_<mode>.jsonl."""
    records = load_candidates(volume)
    out_path = artifacts_dir(volume) / f"selection_{mode}.jsonl"
    selections = []
    accepted = 0
    for record in records:
        stem = record["target"]
        choice: dict = {"target": stem, "chosen": None, "reason": record["status"]}
        if record.get("status") == "ok" and record.get("candidates"):
            top = record["candidates"][0]
            score = top.get("select_score")
            margin = record.get("margin")
            if score is None:
                choice["reason"] = "implausible"
            elif score < gate_score:
                choice["reason"] = f"score {score:.2f} < {gate_score}"
            elif margin is None or margin < gate_margin:
                choice["reason"] = f"margin {margin} < {gate_margin}"
            else:
                choice = {
                    "target": stem,
                    "chosen": 0,
                    "reason": "accepted",
                    "select_score": score,
                    "margin": margin,
                }
                accepted += 1
        selections.append(choice)
    with out_path.open("w") as handle:
        for choice in selections:
            handle.write(json.dumps(choice) + "\n")
    print(f"{accepted}/{len(selections)} pages accepted -> {out_path}")


def osm_variant_path(volume: Path, stem: str) -> Path:
    return volume / f"{stem}.georef-osm.json"


def cmd_materialize(volume: Path, mode: str) -> None:
    """Write pN.georef-osm.json sidecars for the selection's accepted pages."""
    from mapsnap.edge_join_experiment import page_fit_state

    records = {r["target"]: r for r in load_candidates(volume)}
    selection_path = artifacts_dir(volume) / f"selection_{mode}.jsonl"
    if not selection_path.exists():
        sys.exit(f"{selection_path} missing; run `select --mode {mode}` first")
    # Remove every sidecar this channel owns before writing: a stale one from
    # a looser gate would silently keep scoring.
    for stale in volume.glob("p*.georef-osm.json"):
        stale.unlink()
    written = 0
    for line in selection_path.read_text().splitlines():
        choice = json.loads(line)
        if choice.get("chosen") is None:
            continue
        record = records[choice["target"]]
        candidate = record["candidates"][choice["chosen"]]
        affine = np.array(candidate["world_affine"])
        width, height = record["width"], record["height"]
        corners = []
        for x, y in [(0, 0), (width, 0), (width, height), (0, height)]:
            corners.append(
                [
                    affine[0, 0] * x + affine[0, 1] * y + affine[0, 2],
                    affine[1, 0] * x + affine[1, 1] * y + affine[1, 2],
                ]
            )
        _, georef = page_fit_state(volume, choice["target"])
        doc: dict = {
            "width": width,
            "height": height,
            "corners": corners,
            "streets": [],
            "intersections": [],
            "osm_snap": {
                "previous_state": record["fit_state"],
                "mode": mode,
                "select_score": candidate.get("select_score"),
                "margin": record.get("margin"),
                "verification": candidate.get("verification"),
                "ncc_fine": candidate.get("ncc_fine"),
                "inlier_frac": candidate.get("inlier_frac"),
                "name_score": (candidate.get("name") or {}).get("score"),
                "theta_deg": candidate.get("theta_deg"),
                "theta_source": candidate.get("theta_source"),
                "scale_source": candidate.get("scale_source"),
                "rmse_ft": candidate.get("rmse_ft"),
            },
        }
        if georef and georef.get("keymap"):
            doc["keymap"] = georef["keymap"]
        osm_variant_path(volume, choice["target"]).write_text(json.dumps(doc, indent=2))
        written += 1
    print(f"{written} pN.georef-osm.json sidecars written in {volume}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_cand = sub.add_parser("candidates", help="generate snap candidates")
    p_cand.add_argument("volume", type=Path)
    p_cand.add_argument("--pages", type=str, default=None, help="comma-separated stems")
    p_cand.add_argument(
        "--all-pages",
        action="store_true",
        help="include fitted pages (arbitration study), not just rescue targets",
    )
    p_cand.add_argument("--limit", type=int, default=None)
    p_cand.add_argument("--recompute", action="store_true")
    p_cand.add_argument("--no-vis", action="store_true", help="skip contact sheets")

    p_rep = sub.add_parser("report", help="ranking diagnostics vs truth")
    p_rep.add_argument("volume", type=Path)
    p_rep.add_argument(
        "--sweep", action="store_true", help="grid the gates, print simulated Δnet"
    )

    p_sel = sub.add_parser("select", help="pick candidates / abstain per page")
    p_sel.add_argument("volume", type=Path)
    p_sel.add_argument("--mode", choices=["argmax", "volume"], default="argmax")
    p_sel.add_argument("--gate-score", type=float, default=1.0)
    p_sel.add_argument("--gate-margin", type=float, default=0.25)

    p_mat = sub.add_parser("materialize", help="write pN.georef-osm.json sidecars")
    p_mat.add_argument("volume", type=Path)
    p_mat.add_argument("--mode", choices=["argmax", "volume"], default="argmax")

    args = parser.parse_args()
    if args.command == "candidates":
        cmd_candidates(
            args.volume,
            args.pages.split(",") if args.pages else None,
            args.all_pages,
            args.limit,
            args.recompute,
            vis=not args.no_vis,
        )
    elif args.command == "report":
        if args.sweep:
            cmd_sweep(args.volume)
        else:
            cmd_report(args.volume)
    elif args.command == "select":
        cmd_select(args.volume, args.mode, args.gate_score, args.gate_margin)
    elif args.command == "materialize":
        cmd_materialize(args.volume, args.mode)


if __name__ == "__main__":
    main()
