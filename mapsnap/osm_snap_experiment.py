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
    adjacency_keymap_rotations,
    affine_theta_deg,
    calibrated_radius_m,
    evaluate_pose,
    frame_around,
    label_osm_rotations,
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
    panel_units: list[PageUnit]
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


def attach_missing_truth(volume: Path, units: list[PageUnit]) -> int:
    """Attach truth to pages load_page_units missed; returns pages annotated.

    Two classes were invisible to the harness (their candidates carried no
    rmse, so the gate sweep was blind to them) even though `mapsnap score`
    scores their placements: (1) unsplit truth items whose page key differs
    from the jpg stem only in letter case (Chicago 'p51N' vs 'p51n'); (2)
    pages whose truth exists only as split items — a whole-page placement of
    such a sheet is scored against its LARGEST truth split (the whole-canvas
    stand-in rule in match_split_pairs), so that split's transform is the
    right truth here, with rmse sampled over the full canvas exactly like
    analyze_pair's whole-page grid.
    """
    from mapsnap.compare_iiif_georef import (
        annotation_transform_type,
        extract_gcps,
        fit_transform,
        label_split_index,
        load_split_polygons,
    )
    from mapsnap.edge_join_experiment import TruthFit, scale_affine_to_local
    from mapsnap.utils import source_id_to_page_key

    truth_path = volume / "main.iiif.json"
    if not truth_path.exists():
        return 0
    unsplit_by_lower: dict[str, dict] = {}
    splits_by_parent: dict[str, list[dict]] = {}
    for item in json.loads(truth_path.read_text()).get("items", []):
        key = source_id_to_page_key(
            item.get("target", {}).get("source", {}).get("id"), item.get("label", "")
        )
        if "__" in key:
            splits_by_parent.setdefault(key.split("__")[0].lower(), []).append(item)
        else:
            unsplit_by_lower.setdefault(key.lower(), item)

    def truth_fit(item: dict, local_width: int) -> TruthFit | None:
        gcps = extract_gcps(item)
        if len(gcps) < 2:
            return None
        affine_full = fit_transform(gcps, annotation_transform_type(item))
        return TruthFit(
            affine_local=scale_affine_to_local(
                affine_full, item["target"]["source"]["width"], local_width
            ),
            gcp_count=len(gcps),
            transform_type=annotation_transform_type(item),
        )

    annotated = 0
    for unit in units:
        if unit.truth is not None:
            continue
        stem_lower = unit.stem.lower()
        item = unsplit_by_lower.get(stem_lower)
        if item is None:
            items = splits_by_parent.get(stem_lower)
            if not items:
                continue
            panels_path = next(
                (
                    p
                    for p in (volume / "oim").glob("*.panels.json")
                    if p.name.lower() == f"{stem_lower}.panels.json"
                ),
                None,
            )
            panels = load_split_polygons(panels_path) if panels_path else {}

            def panel_area(split_item: dict) -> float:
                index = label_split_index(split_item)
                polygon = panels.get(index) if index is not None else None
                return polygon.area if polygon is not None else 0.0

            item = max(items, key=lambda i: (panel_area(i), len(extract_gcps(i))))
        fit = truth_fit(item, unit.width)
        if fit is not None:
            unit.truth = fit
            annotated += 1
    return annotated


def panel_base(stem: str) -> str | None:
    """The base page stem of a panel stem ('p474__1' -> 'p474'), or None."""
    if "__" not in stem:
        return None
    return stem.rpartition("__")[0]


def load_panel_units(volume: Path) -> list[PageUnit]:
    """One PageUnit per split-panel jpg (pN__k.jpg), with truth attached.

    A panel jpg is the base jpg cropped to its panel polygon's bounding box
    (split.write_panels), so panel px -> base px is a pure translation by the
    bbox origin. Truth comes from the truth split item whose OIM panel polygon
    best overlaps ours in canvas coordinates — the same IoU rule compare_pages
    uses, so OIM's split numbering need not match ours — translated into the
    panel's own pixel frame.
    """
    from mapsnap.compare_iiif_georef import (
        MIN_SPLIT_IOU,
        annotation_transform_type,
        extract_gcps,
        fit_transform,
        label_split_index,
        load_split_polygons,
        polygon_iou,
        ring_to_polygon,
    )
    from mapsnap.edge_join_experiment import (
        TruthFit,
        page_fit_state,
        scale_affine_to_local,
    )
    from mapsnap.keymap.fit_keymap import page_number
    from mapsnap.road_model import effective_gcp_count, page_world_affine
    from mapsnap.utils import jpeg_dimensions, source_id_to_page_key

    truth_path = volume / "main.iiif.json"
    splits_by_parent: dict[str, list[dict]] = {}
    if truth_path.exists():
        for item in json.loads(truth_path.read_text()).get("items", []):
            key = source_id_to_page_key(
                item.get("target", {}).get("source", {}).get("id"),
                item.get("label", ""),
            )
            if "__" in key:
                splits_by_parent.setdefault(key.split("__")[0].lower(), []).append(item)

    units: list[PageUnit] = []
    for jpg in sorted(volume.glob("p*__*.jpg")):
        stem = jpg.stem
        base = panel_base(stem)
        index_str = stem.rpartition("__")[2]
        if base is None or not index_str.isdigit():
            continue
        number = page_number(base)
        panels_path = volume / f"{base}.panels.json"
        if number is None or not panels_path.exists():
            continue
        panels_doc = json.loads(panels_path.read_text())
        rings = panels_doc.get("panels", [])
        index = int(index_str)
        if not (1 <= index <= len(rings)):
            continue
        ring = rings[index - 1]
        width, height = jpeg_dimensions(jpg)
        state, georef = page_fit_state(volume, stem)

        truth: TruthFit | None = None
        items = splits_by_parent.get(base.lower())
        if items:
            source = items[0]["target"]["source"]
            canvas_scale = float(source["width"]) / panels_doc["width"]
            our_polygon = ring_to_polygon(
                [[x * canvas_scale, y * canvas_scale] for x, y in ring]
            )
            oim_path = next(
                (
                    p
                    for p in (volume / "oim").glob("*.panels.json")
                    if p.name.lower() == f"{base.lower()}.panels.json"
                ),
                None,
            )
            oim_polygons = load_split_polygons(oim_path) if oim_path else {}
            best_iou, best_item = 0.0, None
            for item in items:
                item_index = label_split_index(item)
                polygon = oim_polygons.get(item_index) if item_index else None
                if polygon is None:
                    continue
                iou = polygon_iou(our_polygon, polygon)
                if iou > best_iou:
                    best_iou, best_item = iou, item
            if best_item is not None and best_iou >= MIN_SPLIT_IOU:
                gcps = extract_gcps(best_item)
                if len(gcps) >= 2:
                    base_local = scale_affine_to_local(
                        fit_transform(gcps, annotation_transform_type(best_item)),
                        best_item["target"]["source"]["width"],
                        panels_doc["width"],
                    )
                    x0 = max(0, int(min(x for x, _ in ring)))
                    y0 = max(0, int(min(y for _, y in ring)))
                    panel_affine = base_local.copy()
                    panel_affine[:, 2] = base_local @ np.array([x0, y0, 1.0])
                    truth = TruthFit(
                        affine_local=panel_affine,
                        gcp_count=len(gcps),
                        transform_type=annotation_transform_type(best_item),
                    )

        gen_affine = None
        effective_gcps = 0
        keymap_centers: list[tuple[float, float]] = []
        keymap_radius = 0.0
        keymap_regions = None
        if georef is not None:
            if state == "fitted":
                gen_affine = page_world_affine(georef)
                effective_gcps = effective_gcp_count(georef)
            keymap = georef.get("keymap") or {}
            keymap_centers = [tuple(c) for c in keymap.get("centers", [])]
            keymap_radius = float(keymap.get("radius_m") or 0.0)
            keymap_regions = keymap.get("regions") or None

        units.append(
            PageUnit(
                stem=stem,
                number=number,
                width=width,
                height=height,
                fit_state=state,
                truth=truth,
                split_truth=False,
                gen_affine=gen_affine,
                inlier_intersections=effective_gcps,
                inlier_streets=0,
                keymap_centers=keymap_centers,
                keymap_radius_m=keymap_radius,
                keymap_regions=keymap_regions,
            )
        )
    return units


def load_volume_context(volume: Path) -> VolumeContext:
    units = load_page_units(volume)
    attach_missing_truth(volume, units)
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
        panel_units=load_panel_units(volume),
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
    base = panel_base(unit.stem)
    if base is not None:
        # A fitted sibling panel is the same physical sheet: its rotation is
        # the most reliable directed prior a panel can get.
        for sibling in vctx.panel_units:
            if (
                sibling.stem != unit.stem
                and panel_base(sibling.stem) == base
                and sibling.fit_state == "fitted"
            ):
                theta = unit_theta_deg(sibling)
                if theta is not None:
                    priors.append(RotationPrior(theta, 6.0, "ransac-neighbor"))
    own_centroid = vctx.region_centroids.get(unit.number)
    if own_centroid is None and search_centers:
        own_centroid = search_centers[0]
    if own_centroid is not None:
        # Printed neighbor claims live on the base sheet's margins; the crop
        # preserves orientation, so the base stem's directions apply to a panel.
        image_directions = image_neighbor_directions(vctx.adjacency, base or unit.stem)
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
    if unit.fit_state == "fitted" and unit.gen_affine is not None:
        # For arbitration the incumbent pose itself is the natural search
        # init — it also reaches fitted pages the keymap never placed.
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
        if all(haversine_m(lat_c, lon_c, b, a) > 50.0 for a, b in centers):
            centers = centers + [(lon_c, lat_c)]
    if not centers:
        return None, "no_keymap"
    labels = page_label_features(vctx, unit)
    priors = rotation_priors_for(vctx, unit, centers, labels)
    base = panel_base(unit.stem)
    radius = vctx.radius_m
    if base is not None:
        # The keymap places the SHEET; a panel's center can sit up to half the
        # base diagonal away from it, so widen the center-search accordingly.
        # The region's area implies the sheet's scale (which the panel shares),
        # so the family-rung test runs against the BASE dims, not the panel's.
        base_unit = next((u for u in vctx.units if u.stem == base), None)
        if base_unit is not None:
            scales = page_scale_priors(
                vctx.volume_m_per_px, regions, base_unit.width, base_unit.height
            )
            base_diag = (
                math.hypot(base_unit.width, base_unit.height) * vctx.volume_m_per_px
            )
            panel_diag = math.hypot(unit.width, unit.height) * vctx.volume_m_per_px
            radius = vctx.radius_m + max(0.0, (base_diag - panel_diag) / 2)
        else:
            scales = page_scale_priors(
                vctx.volume_m_per_px, None, unit.width, unit.height
            )
    else:
        scales = page_scale_priors(
            vctx.volume_m_per_px, regions, unit.width, unit.height
        )
    ctx = PageContext(
        stem=unit.stem,
        number=unit.number,
        width=unit.width,
        height=unit.height,
        prob=prob,
        search_centers=centers,
        radius_m=radius,
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
    if unit.fit_state == "fitted" and unit.gen_affine is not None:
        # Arbitration head-to-head: score the incumbent RANSAC pose with the
        # same evidence the challenger candidates carry.
        incumbent = evaluate_pose(ctx, vctx.features, unit.gen_affine)
        if incumbent is not None:
            incumbent["world_affine"] = [
                [float(v) for v in row] for row in unit.gen_affine
            ]
            incumbent["effective_gcps"] = unit.inlier_intersections
            if unit.rmse_ft is not None:
                incumbent["rmse_ft"] = round(unit.rmse_ft, 1)
            record["incumbent"] = incumbent
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


def ensure_probs(volume: Path, stems: list[str]) -> None:
    """Run road-UNet inference for stems missing a cached P(road) map."""
    missing = [
        s
        for s in stems
        if load_prob(volume, s) is None and (volume / f"{s}.jpg").exists()
    ]
    if not missing:
        return
    import torch  # noqa: F401  (import check before loading model)

    from mapsnap.keymap.number_model import select_device
    from mapsnap.road_model import load_model, predict_page

    out_dir = volume / "artifacts" / "edge_join" / "roadprob"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    model = load_model(Path("models/road_unet.pt"), device)
    for stem in missing:
        gray = cv2.imread(str(volume / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        prob = predict_page(model, gray, device)
        cv2.imwrite(str(out_dir / f"{stem}.png"), (prob * 255).round().astype(np.uint8))
    print(f"  inferred {len(missing)} P(road) maps")


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
    # Panels are rescue-only even under --all-pages: arbitration challenges
    # base fitted pages, and fitted panels are the least reliable fits in the
    # volume — not worth the compute to challenge.
    targets += [u for u in vctx.panel_units if u.fit_state in RESCUE_STATES]
    if pages:
        wanted = set(pages)
        targets = [u for u in targets if u.stem in wanted]
    if limit is not None:
        targets = targets[:limit]
    ensure_probs(
        volume,
        [
            u.stem
            for u in targets
            if "__" in u.stem and (u.stem not in existing or recompute or pages)
        ],
    )

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
    records = [
        r for r in load_candidates(volume) if r.get("fit_state") in RESCUE_STATES
    ]
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


DISTINCT_SEPARATION_M = 100.0
DISTINCT_THETA_DEG = 10.0


def distinct_margin(record: dict) -> float | None:
    """Rank-1's select_score lead over the best *distinct* alternative lock.

    Near-identical twins (the same lock found from two search centers, within
    100 m and 10 degrees) are not ambiguity — a margin computed against them
    wrongly abstains on confident pages. If no distinct alternative exists the
    margin is infinite. None when the top candidate is implausible.
    """
    candidates = record.get("candidates") or []
    if not candidates or candidates[0].get("select_score") is None:
        return None
    top = candidates[0]
    for candidate in candidates[1:]:
        if candidate.get("select_score") is None:
            continue
        separation = haversine_m(
            top["center"][1],
            top["center"][0],
            candidate["center"][1],
            candidate["center"][0],
        )
        theta_gap = abs(
            (candidate["theta_deg"] - top["theta_deg"] + 180.0) % 360.0 - 180.0
        )
        if separation > DISTINCT_SEPARATION_M or theta_gap > DISTINCT_THETA_DEG:
            # Candidates are sorted by select_score, so the first distinct
            # alternative is the strongest one.
            return top["select_score"] - candidate["select_score"]
    return math.inf


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
        margin = distinct_margin(record)
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
    records = [
        r for r in load_candidates(volume) if r.get("fit_state") in RESCUE_STATES
    ]
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
ARBITRATE_GATES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75]


def cmd_sweep_arbitrate(volume: Path) -> None:
    """Grid the arbitration gate; print the simulated Δnet from challenges."""
    records = load_candidates(volume)
    weights, total_land = truth_land_weights(volume)

    def bucket_value(rmse: float | None) -> int:
        if rmse is None:
            return 0
        if rmse <= 25.0:
            return 1
        if rmse >= 200.0:
            return -1
        return 0

    print(f"== {volume.name}: simulated arbitration Δnet ==")
    for gate in ARBITRATE_GATES:
        delta = 0.0
        challenged = improved = worsened = unweighted = 0
        details = []
        for record in records:
            if record.get("fit_state") != "fitted":
                continue
            challenge = arbitrate_challenge(record, gate)
            if challenge is None:
                continue
            challenged += 1
            old_rmse = (record.get("incumbent") or {}).get("rmse_ft")
            new_rmse = record["candidates"][0].get("rmse_ft")
            if new_rmse is not None and old_rmse is not None:
                if new_rmse < old_rmse:
                    improved += 1
                elif new_rmse > old_rmse:
                    worsened += 1
                details.append(f"{record['target']}:{old_rmse:.0f}->{new_rmse:.0f}")
            weight = weights.get(record["target"])
            if weight is None or old_rmse is None or new_rmse is None:
                unweighted += 1
                continue
            delta += weight * (bucket_value(new_rmse) - bucket_value(old_rmse))
        net = delta / total_land if total_land else 0.0
        print(
            f"  gate {gate:>5.2f}: {net * 100:+5.1f}%  {challenged} challenged"
            f" ({improved} better, {worsened} worse, {unweighted} unweighted)"
        )
        if details:
            print("      " + "  ".join(details[:10]))


VOLUME_MODE_GATE = 1.5  # the dev-chosen conservative elbow for the energy mode

# Arbitration: a snap candidate may replace a placed RANSAC fit only when it
# is a confident, unambiguous lock that clearly disagrees with the incumbent
# AND beats it on the shared evidence (geometry verification + name score).
ARBITRATE_MIN_DISAGREE_FT = 100.0


def arbitrate_challenge(record: dict, arbitrate_gate: float) -> dict | None:
    """The challenge decision for one fitted page, or None to keep RANSAC.

    Truth-free head-to-head: the challenger must clear the (high) arbitration
    score gate with an unambiguous margin, disagree with the incumbent by more
    than the mid-tier threshold, and win on BOTH shared-evidence axes — the
    matcher's geometric verification and the street-name alignment. Ties keep
    the incumbent: replacing a placed page is the risky direction.
    """
    incumbent = record.get("incumbent")
    candidates = record.get("candidates") or []
    if not incumbent or not candidates:
        return None
    top = candidates[0]
    score = top.get("select_score")
    if score is None or score < arbitrate_gate:
        return None
    margin = distinct_margin(record)
    if margin is None or margin < 0.25:
        return None
    disagreement = grid_rmse_ft_between(
        np.array(incumbent["world_affine"]),
        np.array(top["world_affine"]),
        record["width"],
        record["height"],
    )
    if disagreement < ARBITRATE_MIN_DISAGREE_FT:
        return None
    incumbent_verification = incumbent.get("verification")
    top_verification = top.get("verification")
    if incumbent_verification is None or top_verification is None:
        return None
    if top_verification <= incumbent_verification:
        return None
    incumbent_name = (incumbent.get("name") or {}).get("score") or 0.0
    top_name = (top.get("name") or {}).get("score") or 0.0
    if top_name < incumbent_name:
        return None
    return {
        "target": record["target"],
        "chosen": 0,
        "reason": "challenge",
        "challenge": True,
        "select_score": score,
        "disagreement_ft": round(disagreement, 1),
        "incumbent_verification": incumbent_verification,
        "challenger_verification": top_verification,
    }


# Sheet-integrity gate for split panels: every panel is a rigid crop of one
# sheet, so a panel placement determines the WHOLE sheet's placement exactly
# (the crop is a pure translation). Two placements of the same sheet — from a
# candidate and a fitted sibling, or from two co-accepted candidates — must
# agree to within this corner tolerance. Correct fits agree to ~10-30 m;
# LA p1408__2's along-the-cut-line alias disagreed by 258 m while still
# touching its sibling, which is why mere contiguity was not enough.
SHEET_AGREE_TOL_M = 60.0
# A fitted sibling anchors the sheet-agreement gate only when its own fit has
# real evidence: eff>=3 panels measure <=57ft everywhere truth exists, while
# eff<=2 panels range to 6640ft — agreeing with those rejects true candidates.
SIBLING_ANCHOR_MIN_GCPS = 3
# A panel with no reliable anchor and no co-accepted sibling may still be
# accepted on its own score, at a much higher bar than base pages: small
# pages alias more readily, and this is where their confident failures live.
PANEL_SOLO_GATE = 2.2


def panel_ring_origin(ring: list[list[float]]) -> tuple[int, int]:
    """The bbox origin write_panels cropped this panel at (base-jpg px)."""
    return (
        max(0, int(min(x for x, _ in ring))),
        max(0, int(min(y for _, y in ring))),
    )


def implied_sheet_corners(
    affine: np.ndarray,
    origin: tuple[int, int],
    sheet_size: tuple[float, float],
) -> list[tuple[float, float]]:
    """The full sheet's corner lon/lats implied by one panel's affine."""
    x0, y0 = origin
    sheet = affine.copy()
    sheet[:, 2] = affine @ np.array([-x0, -y0, 1.0])
    width, height = sheet_size
    corners = []
    for x, y in [(0, 0), (width, 0), (width, height), (0, height)]:
        corners.append(
            (
                sheet[0, 0] * x + sheet[0, 1] * y + sheet[0, 2],
                sheet[1, 0] * x + sheet[1, 1] * y + sheet[1, 2],
            )
        )
    return corners


def sheets_agree(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """Whether two implied-sheet placements agree within SHEET_AGREE_TOL_M."""
    worst = max(haversine_m(pa[1], pa[0], pb[1], pb[0]) for pa, pb in zip(a, b))
    return worst <= SHEET_AGREE_TOL_M


def panel_allowed_candidates(
    volume: Path, records: list[dict]
) -> dict[str, set[int] | None]:
    """Per panel stem: candidate indices whose implied sheet placement agrees
    with every fitted sibling's. None = no fitted sibling to check against
    (the volume-energy mutual-agreement terms are then the only constraint).
    """
    panel_units = load_panel_units(volume)
    by_base: dict[str, list[PageUnit]] = {}
    for unit in panel_units:
        base = panel_base(unit.stem)
        if base is not None:
            by_base.setdefault(base, []).append(unit)

    result: dict[str, set[int] | None] = {}
    panels_cache: dict[str, dict] = {}

    def doc_of(base: str) -> dict:
        if base not in panels_cache:
            panels_cache[base] = json.loads(
                (volume / f"{base}.panels.json").read_text()
            )
        return panels_cache[base]

    for record in records:
        stem = record["target"]
        base = panel_base(stem)
        if base is None or record.get("status") != "ok":
            continue
        siblings = [
            u
            for u in by_base.get(base, [])
            if u.stem != stem
            and u.fit_state == "fitted"
            and u.gen_affine is not None
            and u.inlier_intersections >= SIBLING_ANCHOR_MIN_GCPS
        ]
        if not siblings:
            result[stem] = None
            continue
        panels_doc = doc_of(base)
        rings = panels_doc["panels"]
        sheet_size = (panels_doc["width"], panels_doc["height"])
        origin = panel_ring_origin(rings[int(stem.rpartition("__")[2]) - 1])
        sibling_sheets = []
        for sibling in siblings:
            assert sibling.gen_affine is not None
            sibling_origin = panel_ring_origin(
                rings[int(sibling.stem.rpartition("__")[2]) - 1]
            )
            sibling_sheets.append(
                implied_sheet_corners(sibling.gen_affine, sibling_origin, sheet_size)
            )
        allowed: set[int] = set()
        for k, candidate in enumerate(record.get("candidates") or []):
            candidate_sheet = implied_sheet_corners(
                np.array(candidate["world_affine"]), origin, sheet_size
            )
            if all(sheets_agree(candidate_sheet, s) for s in sibling_sheets):
                allowed.add(k)
        result[stem] = allowed
    return result


def select_union(
    volume: Path,
    records: list[dict],
    gate_score: float,
    gate_margin: float,
    allowed: dict[str, set[int] | None],
) -> list[dict]:
    """The two dev-calibrated committees, combined: the energy mode (at its
    conservative gate) resolves joint/ambiguous pages, and the per-page argmax
    gate is the floor for pages the energy abstains on."""
    by_target = {
        s["target"]: s
        for s in select_volume(volume, records, VOLUME_MODE_GATE, allowed)
    }
    selections = []
    for choice in select_argmax(records, gate_score, gate_margin, allowed):
        volume_choice = by_target.get(choice["target"])
        if volume_choice is not None and volume_choice.get("chosen") is not None:
            selections.append(volume_choice)
        else:
            selections.append(choice)
    return selections


def cmd_select(
    volume: Path,
    mode: str,
    gate_score: float,
    gate_margin: float,
    arbitrate_gate: float = 2.0,
) -> None:
    """Pick one candidate (or abstain) per page; write selection_<mode>.jsonl."""
    records = load_candidates(volume)
    # Fitted pages' records (from `candidates --all-pages`) exist solely for
    # arbitration; the rescue committees must never treat them as targets.
    rescue = [r for r in records if r.get("fit_state") in RESCUE_STATES]
    allowed = panel_allowed_candidates(volume, rescue)
    out_path = artifacts_dir(volume) / f"selection_{mode}.jsonl"
    if mode == "volume":
        selections = select_volume(volume, rescue, gate_score, allowed)
    elif mode == "union":
        selections = select_union(volume, rescue, gate_score, gate_margin, allowed)
    elif mode == "arbitrate":
        # The union rescue selection, plus challenges to placed RANSAC fits.
        selections = select_union(volume, rescue, gate_score, gate_margin, allowed)
        challenged = 0
        for record in records:
            if record.get("fit_state") != "fitted":
                continue
            challenge = arbitrate_challenge(record, arbitrate_gate)
            if challenge is not None:
                selections.append(challenge)
                challenged += 1
        print(f"{challenged} challenges accepted")
    else:
        selections = select_argmax(rescue, gate_score, gate_margin, allowed)
    accepted = sum(1 for s in selections if s.get("chosen") is not None)
    with out_path.open("w") as handle:
        for choice in selections:
            handle.write(json.dumps(choice) + "\n")
    print(f"{accepted}/{len(selections)} pages accepted -> {out_path}")


def select_argmax(
    records: list[dict],
    gate_score: float,
    gate_margin: float,
    panel_allowed: dict[str, set[int] | None] | None = None,
) -> list[dict]:
    """v0: per-page rank-1 with abstention gates.

    Panels face the sheet-integrity gate on top of the usual ones: rank-1 must
    agree with any reliable fitted sibling's implied sheet; a panel with no
    reliable anchor is held to the much higher PANEL_SOLO_GATE instead.
    """
    panel_allowed = panel_allowed or {}
    selections = []
    for record in records:
        stem = record["target"]
        choice: dict = {"target": stem, "chosen": None, "reason": record["status"]}
        if record.get("status") == "ok" and record.get("candidates"):
            if panel_base(stem) is not None:
                allowed = panel_allowed.get(stem)
                if allowed is not None and 0 not in allowed:
                    choice["reason"] = "sheet-agreement"
                    selections.append(choice)
                    continue
                if allowed is None:
                    top_score = record["candidates"][0].get("select_score")
                    if top_score is None or top_score < PANEL_SOLO_GATE:
                        choice["reason"] = (
                            f"panel-solo score {top_score} < {PANEL_SOLO_GATE}"
                        )
                        selections.append(choice)
                        continue
            top = record["candidates"][0]
            score = top.get("select_score")
            margin = distinct_margin(record)
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
                    "margin": None if math.isinf(margin) else round(margin, 4),
                }
        selections.append(choice)
    return selections


# --- v1: volume-wide discrete selection ------------------------------------

W_OVERLAP = 3.0
W_ADJACENT = 0.5
W_SIDE = 0.3
OVERLAP_SOFT = 0.15  # IoU-over-min where the penalty starts
OVERLAP_HARD = 0.5  # and where it becomes prohibitive
ADJACENT_SIGMA_M = 150.0  # schematic keymap-centroid geometry
EXHAUSTIVE_LIMIT = 20_000


def overlap_penalty(iou_over_min: float, soft: float, hard: float) -> float:
    """0 below soft, quadratic to hard, prohibitive above."""
    if iou_over_min <= soft:
        return 0.0
    if iou_over_min >= hard:
        return 1e6
    frac = (iou_over_min - soft) / (hard - soft)
    return frac * frac


def select_volume(
    volume: Path,
    records: list[dict],
    gate_score: float,
    panel_allowed: dict[str, set[int] | None] | None = None,
) -> list[dict]:
    """v1: joint selection over per-page candidate sets.

    Energy = per-page unary (a pick must beat abstention by its select_score
    over the gate) + pairwise terms on keymap/printed adjacency edges:
    footprint-overlap penalty (including against FITTED pages — this is what
    kills a one-block slide, which collides with a placed neighbor where the
    truth does not), keymap-centroid distance consistency, and a side-agreement
    reward. Connected components solve exhaustively when small, else ICM from
    several greedy orderings.

    Panels additionally face sheet-integrity constraints: options inconsistent
    with a fitted sibling are dropped up front, page-space-adjacent sibling
    picks must land next to each other, and an accepted panel with no fitted
    sibling must have a co-accepted contiguous sibling backing it up.
    """
    from shapely.geometry import Polygon

    from mapsnap.score import LocalFrame

    panel_allowed = panel_allowed or {}
    units = load_page_units(volume) + load_panel_units(volume)
    stem_to_number = {u.stem: u.number for u in units}
    region_pairs, centroids = keymap_region_adjacency(volume)
    pairs = detected_pairs(volume) | region_pairs

    eligible: dict[str, dict] = {}
    for record in records:
        if record.get("status") != "ok" or not record.get("candidates"):
            continue
        options = [
            (k, c)
            for k, c in enumerate(record["candidates"])
            if c.get("select_score") is not None
        ]
        allowed = panel_allowed.get(record["target"])
        if allowed is not None:
            options = [(k, c) for k, c in options if k in allowed]
        if options:
            eligible[record["target"]] = {"record": record, "options": options}
    if not eligible:
        return select_argmax(records, gate_score, gate_margin=0.0)

    # Implied-sheet corners per eligible panel candidate, for the mutual
    # sheet-agreement constraint between co-accepted siblings.
    panels_cache: dict[str, dict] = {}

    def sheet_corners_of(stem: str, candidate: dict) -> list[tuple[float, float]]:
        base = panel_base(stem)
        assert base is not None
        if base not in panels_cache:
            panels_cache[base] = json.loads(
                (volume / f"{base}.panels.json").read_text()
            )
        doc = panels_cache[base]
        origin = panel_ring_origin(doc["panels"][int(stem.rpartition("__")[2]) - 1])
        return implied_sheet_corners(
            np.array(candidate["world_affine"]), origin, (doc["width"], doc["height"])
        )

    sheet_corners: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for stem, entry in eligible.items():
        if panel_base(stem) is not None:
            for k, candidate in entry["options"]:
                sheet_corners[(stem, k)] = sheet_corners_of(stem, candidate)

    first = next(iter(eligible.values()))["options"][0][1]
    frame = LocalFrame(first["center"][0], first["center"][1])

    def polygon_of(affine: list[list[float]], width: int, height: int) -> Polygon:
        ring = []
        for x, y in [(0, 0), (width, 0), (width, height), (0, height)]:
            lon = affine[0][0] * x + affine[0][1] * y + affine[0][2]
            lat = affine[1][0] * x + affine[1][1] * y + affine[1][2]
            ring.append(frame.to_xy(lon, lat))
        return Polygon(ring).buffer(0)

    polygons: dict[tuple[str, int], Polygon] = {}
    centers_xy: dict[tuple[str, int], tuple[float, float]] = {}
    for stem, entry in eligible.items():
        record = entry["record"]
        for k, candidate in entry["options"]:
            polygons[(stem, k)] = polygon_of(
                candidate["world_affine"], record["width"], record["height"]
            )
            centers_xy[(stem, k)] = frame.to_xy(*candidate["center"])

    # Fitted context, with skeleton twins deduped: a fitted pNs maps the same
    # ground as fitted pN, and its ~100% overlap would poison both the
    # calibration below and every candidate that legitimately touches pN.
    # Fitted PANELS are excluded outright — their own placements are the least
    # reliable in the volume (LA's fitted p1499n__3 sits 392ft off and lies on
    # p1484's true ground), so they cannot serve as overlap evidence; panel
    # consistency is enforced by the sheet-agreement machinery instead.
    fitted_stems = {u.stem for u in units if u.fit_state == "fitted"}
    fitted_units = [
        u
        for u in units
        if u.fit_state == "fitted"
        and u.gen_affine is not None
        and panel_base(u.stem) is None
        and not (u.stem.endswith("s") and u.stem[:-1] in fitted_stems)
    ]
    fitted_polys: dict[str, Polygon] = {}
    number_to_fitted: dict[int, list[str]] = {}
    for unit in fitted_units:
        assert unit.gen_affine is not None
        fitted_polys[unit.stem] = polygon_of(
            [[float(v) for v in row] for row in unit.gen_affine],
            unit.width,
            unit.height,
        )
        number_to_fitted.setdefault(unit.number, []).append(unit.stem)
    fitted_polygons = list(fitted_polys.values())

    def iou_over_min(a: Polygon, b: Polygon) -> float:
        smaller = min(a.area, b.area)
        if smaller <= 0:
            return 0.0
        return a.intersection(b).area / smaller

    # Calibrate the overlap thresholds from the volume's own adjacent fitted
    # BASE pairs: Sanborn sheets legitimately share strips (Hudson true locks
    # sit at 0.32-0.43 IoU-over-min against their fitted neighbors), so a
    # fixed threshold either misses slides or punishes correct seams. One
    # value per NUMBER pair — the max over member sheets — because a page
    # number can name a whole lettered family (LA p1499a..q) of which only
    # one member actually neighbors the partner; flooding the distribution
    # with the other members' near-zero overlaps drags the percentile down
    # and tightens the gate onto true seams.
    observed = []
    for a, b in (tuple(pair) for pair in pairs):
        values = [
            iou_over_min(fitted_polys[sa], fitted_polys[sb])
            for sa in number_to_fitted.get(a, [])
            for sb in number_to_fitted.get(b, [])
        ]
        if values:
            observed.append(max(values))
    if len(observed) >= 5:
        p90 = float(np.percentile(observed, 90))
        overlap_soft = min(0.5, max(OVERLAP_SOFT, p90 + 0.05))
    else:
        overlap_soft = OVERLAP_SOFT
    overlap_hard = min(0.85, overlap_soft + 0.3)

    def unary(stem: str, k: int | None) -> float:
        if k is None:
            return 0.0
        candidate = dict(eligible[stem]["options"])[k]
        energy = -(candidate["select_score"] - gate_score)
        polygon = polygons[(stem, k)]
        for fitted in fitted_polygons:
            energy += W_OVERLAP * overlap_penalty(
                iou_over_min(polygon, fitted), overlap_soft, overlap_hard
            )
        return energy

    def centroid_xy(number: int | None) -> tuple[float, float] | None:
        if number is None:
            return None
        centroid = centroids.get(number)
        return frame.to_xy(*centroid) if centroid else None

    def coupled(stem_a: str, stem_b: str) -> str | None:
        """'sibling' | 'adjacent' | None: whether two stems interact at all.

        Panels couple ONLY with their siblings: coupling them into the wider
        adjacency graph balloons component sizes (degrading the solver) for a
        constraint the unary fitted-overlap term already carries.
        """
        base_a, base_b = panel_base(stem_a), panel_base(stem_b)
        if base_a is not None and base_a == base_b:
            return "sibling"
        if base_a is not None or base_b is not None:
            return None
        na, nb = stem_to_number.get(stem_a), stem_to_number.get(stem_b)
        if na is not None and nb is not None and frozenset((na, nb)) in pairs:
            return "adjacent"
        return None

    def pairwise(stem_a: str, ka: int | None, stem_b: str, kb: int | None) -> float:
        if ka is None or kb is None:
            return 0.0
        coupling = coupled(stem_a, stem_b)
        if coupling is None:
            return 0.0
        if coupling == "sibling":
            # Rigid-sheet constraint: two co-accepted panels of one sheet must
            # imply (nearly) the same full-sheet placement.
            if sheets_agree(sheet_corners[(stem_a, ka)], sheet_corners[(stem_b, kb)]):
                return 0.0
            return 1e6
        energy = W_OVERLAP * overlap_penalty(
            iou_over_min(polygons[(stem_a, ka)], polygons[(stem_b, kb)]),
            overlap_soft,
            overlap_hard,
        )
        na, nb = stem_to_number.get(stem_a), stem_to_number.get(stem_b)
        ca, cb = centroid_xy(na), centroid_xy(nb)
        if ca and cb:
            xa, ya = centers_xy[(stem_a, ka)]
            xb, yb = centers_xy[(stem_b, kb)]
            realized = (xb - xa, yb - ya)
            expected = (cb[0] - ca[0], cb[1] - ca[1])
            gap = math.hypot(realized[0] - expected[0], realized[1] - expected[1])
            energy += W_ADJACENT * min(3.0, (gap / ADJACENT_SIGMA_M) ** 2)
            norm_r = math.hypot(*realized)
            norm_e = math.hypot(*expected)
            if norm_r > 1e-6 and norm_e > 1e-6:
                cosine = (realized[0] * expected[0] + realized[1] * expected[1]) / (
                    norm_r * norm_e
                )
                energy -= W_SIDE * max(0.0, cosine)
        return energy

    # Connected components over the coupling graph among eligible pages.
    stems = sorted(eligible)
    neighbors: dict[str, set[str]] = {s: set() for s in stems}
    for sa in stems:
        for sb in stems:
            if sa >= sb:
                continue
            if coupled(sa, sb) is not None:
                neighbors[sa].add(sb)
                neighbors[sb].add(sa)
    components: list[list[str]] = []
    seen: set[str] = set()
    for stem in stems:
        if stem in seen:
            continue
        component = []
        queue = [stem]
        seen.add(stem)
        while queue:
            current = queue.pop()
            component.append(current)
            for other in neighbors[current]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        components.append(sorted(component))

    def stem_options(stem: str) -> list[int | None]:
        choices: list[int | None] = [None]
        choices.extend(k for k, _ in eligible[stem]["options"])
        return choices

    assignment: dict[str, int | None] = {}
    for component in components:
        choices_per_stem = [stem_options(stem) for stem in component]
        size = 1
        for choices in choices_per_stem:
            size *= len(choices)

        def total_energy(assign: dict[str, int | None]) -> float:
            energy = sum(unary(s, assign[s]) for s in component)
            for i, sa in enumerate(component):
                for sb in component[i + 1 :]:
                    energy += pairwise(sa, assign[sa], sb, assign[sb])
            return energy

        if size <= EXHAUSTIVE_LIMIT:
            import itertools

            best_assign_c: dict[str, int | None] | None = None
            best_energy = math.inf
            for combo in itertools.product(*choices_per_stem):
                assign: dict[str, int | None] = dict(zip(component, combo))
                energy = total_energy(assign)
                if energy < best_energy:
                    best_energy = energy
                    best_assign_c = assign
            assert best_assign_c is not None
            assignment.update(best_assign_c)
        else:
            # ICM from two deterministic starts: best-unary and abstain-all.
            # Deliberately NOT more: the best-unary start biases convergence
            # toward acceptance, and on the dev volumes that bias scores
            # better than the energy model's own global optimum — the ADJ
            # term over-penalizes true placements where keymap centroids are
            # unreliable (LA's lettered sheets), so a stronger optimizer
            # abstains on pages the weak one correctly keeps.
            best_assign = None
            best_energy = math.inf
            starts: list[dict[str, int | None]] = [
                {s: eligible[s]["options"][0][0] for s in component},
                {s: None for s in component},
            ]
            for start in starts:
                assign = dict(start)
                for _ in range(20):
                    changed = False
                    for stem in component:
                        stem_choices = stem_options(stem)
                        best_k = assign[stem]
                        best_local = math.inf
                        for k in stem_choices:
                            trial = dict(assign)
                            trial[stem] = k
                            energy = total_energy(trial)
                            if energy < best_local:
                                best_local = energy
                                best_k = k
                        if best_k != assign[stem]:
                            assign[stem] = best_k
                            changed = True
                    if not changed:
                        break
                energy = total_energy(assign)
                if energy < best_energy:
                    best_energy = energy
                    best_assign = assign
            assert best_assign is not None
            assignment.update(best_assign)

    # Post-pass: an accepted panel with no reliable fitted anchor needs either
    # a co-accepted sibling backing it up (their mutual sheet agreement is
    # enforced by the pairwise term) or a solo score above PANEL_SOLO_GATE —
    # an ordinary score on a lone small panel is not trustworthy evidence.
    # Dropping one panel can orphan another, so iterate.
    changed_post = True
    while changed_post:
        changed_post = False
        for stem, chosen in list(assignment.items()):
            if chosen is None:
                continue
            if stem not in panel_allowed:
                continue  # not a panel
            if panel_allowed[stem] is not None:
                continue  # has reliable anchors: gated in the options filter
            base = panel_base(stem)
            supported = any(
                other != stem and other_chosen is not None and panel_base(other) == base
                for other, other_chosen in assignment.items()
            )
            if not supported:
                score = dict(eligible[stem]["options"])[chosen].get("select_score")
                if score is None or score < PANEL_SOLO_GATE:
                    assignment[stem] = None
                    changed_post = True

    selections = []
    for record in records:
        stem = record["target"]
        if stem not in eligible:
            selections.append(
                {"target": stem, "chosen": None, "reason": record["status"]}
            )
            continue
        chosen = assignment.get(stem)
        if chosen is None:
            selections.append(
                {"target": stem, "chosen": None, "reason": "energy-abstain"}
            )
        else:
            candidate = dict(eligible[stem]["options"])[chosen]
            selections.append(
                {
                    "target": stem,
                    "chosen": chosen,
                    "reason": "energy",
                    "select_score": candidate["select_score"],
                    "rank": chosen,
                }
            )
    return selections


def cmd_reannotate(volume: Path) -> None:
    """Refresh the rmse_ft annotations on cached candidates from current truth.

    Cheap (no matching): recomputes every candidate's grid rmse against the
    unit's truth affine, including pages that attach_missing_truth
    now covers (case-mismatched keys and split-only truth). Rewrites candidates.jsonl in place.
    """
    unit_list = load_page_units(volume) + load_panel_units(volume)
    units = {u.stem: u for u in unit_list}
    newly = attach_missing_truth(volume, unit_list)
    records = load_candidates(volume)
    changed = 0
    for record in records:
        unit = units.get(record["target"])
        if unit is None:
            continue
        has_truth = unit.truth is not None
        if record.get("has_truth") != has_truth:
            record["has_truth"] = has_truth
            changed += 1
        for candidate in record.get("candidates") or []:
            if unit.truth is None:
                candidate.pop("rmse_ft", None)
                continue
            candidate["rmse_ft"] = round(
                grid_rmse_ft_between(
                    unit.truth.affine_local,
                    np.array(candidate["world_affine"]),
                    unit.width,
                    unit.height,
                ),
                1,
            )
    out_path = artifacts_dir(volume) / "candidates.jsonl"
    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(
        f"{volume.name}: {newly} split-truth pages attached, "
        f"{changed} records flipped has_truth"
    )


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
        if not georef or not georef.get("keymap"):
            # A panel with no own sidecar borrows the keymap block from any
            # sibling variant of the same sheet (the sheet is what the keymap
            # places). The stale-osm cleanup above already ran, so a sibling's
            # georef-osm sidecar can only be one written earlier this loop.
            base = panel_base(choice["target"])
            if base is not None:
                for sibling in sorted(volume.glob(f"{base}__*.georef*.json")):
                    sibling_doc = json.loads(sibling.read_text())
                    if sibling_doc.get("keymap"):
                        georef = sibling_doc
                        break
        doc: dict = {
            "width": width,
            "height": height,
            "corners": corners,
            "streets": [],
            "intersections": [],
            "osm_snap": {
                "previous_state": record["fit_state"],
                "mode": mode,
                "challenge": bool(choice.get("challenge")),
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
    p_rep.add_argument(
        "--sweep-arbitrate",
        action="store_true",
        help="grid the arbitration gate over fitted-page challenges",
    )

    p_sel = sub.add_parser("select", help="pick candidates / abstain per page")
    p_sel.add_argument("volume", type=Path)
    p_sel.add_argument(
        "--mode", choices=["argmax", "volume", "union", "arbitrate"], default="argmax"
    )
    p_sel.add_argument("--gate-score", type=float, default=1.0)
    p_sel.add_argument("--gate-margin", type=float, default=0.25)
    p_sel.add_argument("--arbitrate-gate", type=float, default=2.0)

    p_mat = sub.add_parser("materialize", help="write pN.georef-osm.json sidecars")
    p_mat.add_argument("volume", type=Path)
    p_mat.add_argument(
        "--mode",
        choices=["argmax", "volume", "union", "arbitrate"],
        default="argmax",
    )

    p_re = sub.add_parser("reannotate", help="refresh cached rmse annotations")
    p_re.add_argument("volume", type=Path)

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
        if args.sweep_arbitrate:
            cmd_sweep_arbitrate(args.volume)
        elif args.sweep:
            cmd_sweep(args.volume)
        else:
            cmd_report(args.volume)
    elif args.command == "select":
        cmd_select(
            args.volume,
            args.mode,
            args.gate_score,
            args.gate_margin,
            args.arbitrate_gate,
        )
    elif args.command == "materialize":
        cmd_materialize(args.volume, args.mode)
    elif args.command == "reannotate":
        cmd_reannotate(args.volume)


if __name__ == "__main__":
    main()
