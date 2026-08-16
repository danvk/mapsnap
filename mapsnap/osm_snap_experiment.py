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
import multiprocessing
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from mapsnap.feature_index import FeatureIndex
from mapsnap.georef_from_labels import LabelFeature, prepare_label_features
from mapsnap.keymap.align_page_region import (
    image_neighbor_directions,
    load_adjacency,
    volume_filter_params,
)
from mapsnap.keymap.locate import KeymapLocator, usable_keymaps
from mapsnap.osm_snap import (
    PageContext,
    RotationPrior,
    SnapCandidate,
    adjacency_keymap_rotations,
    affine_theta_deg,
    calibrated_radius_m,
    cluster_search_centers,
    evaluate_pose,
    frame_around,
    label_osm_rotations,
    osm_rasters,
    page_scale_priors,
    snap_page,
)
from mapsnap.streets import Block, build_block_index
from mapsnap.utils import default_centerlines, haversine_m, pose_is_upside_down

# Pages the rescue channel may place: everything the iiif glob does not see.
LOCAL_CHALLENGE_RADIUS_M = 150.0
"""Search radius when challenging a defensible incumbent: covers refinement
(<100 ft agreement) and rung flips (co-located) with margin, nothing more."""

RESCUE_STATES = {"nofit", "misscale", "1gcp", "outlier", "none"}

# Rotation-prior sigma for a demoted pose used as a search seed (#315). Wide
# enough that the ladder still explores, tight enough that the demoted fit's
# orientation -- its most trustworthy component -- ranks first.
DEMOTED_SEED_SIGMA_DEG = 10.0


def artifacts_dir(volume: Path) -> Path:
    return volume / "artifacts" / "osm_snap"


@dataclass
class VolumeContext:
    """Once-per-volume inputs shared by every page's candidate generation."""

    volume: Path
    units: list[PageUnit]
    panel_units: list[PageUnit]
    features: list[dict]
    feature_index: FeatureIndex  # the same features, spatially indexed
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
    pages whose truth exists only as split items. The scorer now grades each
    truth panel over its own region; for this harness's page-level
    diagnostics we approximate with the LARGEST truth split's transform
    (rmse over the full canvas), which matches the scorer for the panel a
    whole-page placement actually fits.
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

            # `panels` is bound as a default rather than captured, so this
            # page's polygons cannot be re-read on a later loop iteration.
            def panel_area(split_item: dict, panels: dict = panels) -> float:
                index = label_split_index(split_item)
                polygon = panels.get(index) if index is not None else None
                return polygon.area if polygon is not None else 0.0

            item = max(items, key=lambda i: (panel_area(i), len(extract_gcps(i))))
        fit = truth_fit(item, unit.width)
        if fit is not None:
            unit.truth = fit
            if unit.gen_affine is not None:
                unit.rmse_ft = grid_rmse_ft_between(
                    fit.affine_local, unit.gen_affine, unit.width, unit.height
                )
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
        demoted_affine = None
        gcp_hints: list[tuple[float, float]] = []
        if georef is not None:
            if state == "fitted":
                gen_affine = page_world_affine(georef)
                effective_gcps = effective_gcp_count(georef)
            else:
                try:
                    demoted_affine = page_world_affine(georef)
                except (KeyError, TypeError, ValueError):
                    demoted_affine = None  # nofit-style doc: no corners
            keymap = georef.get("keymap") or {}
            keymap_centers = [tuple(c) for c in keymap.get("centers", [])]
            keymap_radius = float(keymap.get("radius_m") or 0.0)
            gcp_hints = [tuple(h) for h in georef.get("gcp_hints") or []]
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
                demoted_affine=demoted_affine,
                gcp_hints=gcp_hints,
                inlier_intersections=effective_gcps,
                inlier_streets=0,
                keymap_centers=keymap_centers,
                keymap_radius_m=keymap_radius,
                keymap_regions=keymap_regions,
            )
        )
    return units


def georef_variant_mtime(volume: Path, stem: str) -> int | None:
    """mtime of the stem's current georef-variant sidecar, or None for none.

    The staleness key for cached candidates: `mapsnap fit` re-runs georef
    before snap, and a page whose fit changed (or whose fit STATE changed)
    must be re-matched — its cached record carries the old incumbent pose.
    """
    from mapsnap.edge_join_experiment import GEOREF_VARIANTS

    for variant in GEOREF_VARIANTS:
        path = volume / f"{stem}.{variant}.json"
        if path.exists():
            return int(path.stat().st_mtime)
    return None


def candidates_record_fresh(
    record: dict,
    unit: PageUnit,
    mtime: int | None,
    hint_mtime: int | None = None,
    keymap_mtime: int | None = None,
) -> bool:
    """Whether a cached candidates record still matches the page's fit state.

    Only a successful record is ever fresh. A failure ('no_prob', 'no_keymap')
    turns on inputs this check does not track — the P(road) cache and the key
    map's own sidecars — so caching one would pin the page behind a stale
    failure even after the input is fixed. Recomputing a failure is cheap: it
    fails at the same gate before any matching work.

    ``hint_mtime`` is the contradiction-hint sidecar's mtime: a page the
    adjacency gate demoted twice has georef_mtime None both times, and without
    this key the second snap pass reuses the first pass's candidates —
    generated before the hint (or the stamp-consistency gate) existed — and
    re-adopts the very alias the gate demoted (KC p551).

    ``keymap_mtime`` covers the key map's own sidecars, which supply every
    page's search centers, radius and region rings. The adjacency assignment
    repair (#213) rewrites them without touching any page's georef, so a
    page that just gained a key-map location — or had a wrong one corrected —
    would otherwise keep candidates searched around the old place, or none at
    all.
    """
    if record.get("status") != "ok":
        return False
    return (
        record.get("fit_state") == unit.fit_state
        and record.get("georef_mtime") == mtime
        and record.get("contradiction_mtime") == hint_mtime
        and record.get("keymap_mtime") == keymap_mtime
    )


def keymap_sidecar_mtime(volume: Path) -> int | None:
    """Newest mtime across the volume's key-map sidecars, or None when absent.

    One number for the whole volume, since every page's coarse location comes
    from the same key maps: any change to the numbers or the regions can move
    any page's search centers.
    """
    raw = volume / "raw"
    if not raw.is_dir():
        return None
    mtimes = [
        int(path.stat().st_mtime)
        for pattern in ("*.keymap.json", "*.regions.panels.json")
        for path in raw.glob(pattern)
        if ".truth." not in path.name
    ]
    return max(mtimes) if mtimes else None


def contradiction_hint_mtime(volume: Path, stem: str) -> int | None:
    """mtime of the page's contradiction-hint sidecar, or None for none."""
    path = volume / f"{stem}.contradiction.json"
    return int(path.stat().st_mtime) if path.exists() else None


def load_volume_context(
    volume: Path, units: list[PageUnit] | None = None
) -> VolumeContext:
    units = units if units is not None else load_page_units(volume)
    attach_missing_truth(volume, units)
    centerlines_path = default_centerlines(volume)
    if centerlines_path is None:
        sys.exit(f"no centerlines.geojson under {volume}")
    features = json.loads(centerlines_path.read_text())["features"]
    keymaps = usable_keymaps(volume / "raw")
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
        feature_index=FeatureIndex(features),
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
        # A fitted sibling panel's rotation, as one rung of the ladder. Split
        # sheets are usually inset composites whose insets can be rotated
        # differently on the paper, so this is a useful hint, not a certainty
        # — the mask sweep below covers the disagreeing cases.
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


_note_calibration_cache: dict[str, tuple[float, str]] = {}


def volume_note_calibration(vctx: VolumeContext) -> tuple[float, str]:
    """Per-volume px-per-paper-inch for printed-scale notes (memoized).

    Self-calibrates from fitted pages whose notes read; volumes that print
    notes only on their unfittable odd sheets (Columbus) get the median-rung
    estimate from their own fitted scales instead -- Columbus's scan runs 22%
    denser than the corpus default assumed, and the median rung absorbs that.
    Working-scale px per paper inch, matching the 25% images.
    """
    from mapsnap.printed_scale import printed_scale_ft, resolve_px_per_paper_inch

    key = str(vctx.volume)
    if key not in _note_calibration_cache:
        pairs = []
        for unit in vctx.units:
            if unit.fit_state != "fitted" or unit.gen_affine is None:
                continue
            note = printed_scale_ft(vctx.volume / f"{unit.stem}.streets.json")
            if note is None:
                continue
            m_per_px = (
                math.hypot(unit.gen_affine[1, 0], unit.gen_affine[1, 1]) * 110_540.0
            )
            # px per paper inch = px_per_ft * printed_ft = ft*0.3048/m_per_px...
            pairs.append((0.3048 / m_per_px, note[0]))
        _note_calibration_cache[key] = resolve_px_per_paper_inch(
            pairs, median_px_per_ft=0.3048 / vctx.volume_m_per_px
        )
    return _note_calibration_cache[key]


def build_page_context(
    vctx: VolumeContext, unit: PageUnit
) -> tuple[PageContext | None, str]:
    """(PageContext, status) for one page; context is None unless status 'ok'."""
    prob = load_prob(vctx.volume, unit.stem)
    if prob is None:
        return None, "no_prob"
    centers, regions = page_keymap_data(vctx, unit)
    # A page the adjacency gate demoted carries its neighbors' printed-claim
    # positions — world points on the shared seam the page must sit against.
    # They join the search centers, and stand alone for pages the keymap
    # never placed.
    from mapsnap.adjacency_gate import contradiction_centers

    for lon, lat in contradiction_centers(
        vctx.volume, panel_base(unit.stem) or unit.stem
    ):
        if all(haversine_m(lat, lon, b, a) > 50.0 for a, b in centers):
            centers = centers + [(lon, lat)]
    # Candidate-GCP hints (#335): a failed fit's matched crossings, weighed
    # like contradiction hints -- for a keymap-less page they are the only
    # centers, which is what makes rescue reachable at all there.
    for lon, lat in unit.gcp_hints:
        if all(haversine_m(lat, lon, b, a) > 50.0 for a, b in centers):
            centers = centers + [(lon, lat)]
    seed_affine = None
    if unit.fit_state == "fitted" and unit.gen_affine is not None:
        # For arbitration the incumbent pose itself is the natural search
        # init — it also reaches fitted pages the keymap never placed.
        seed_affine = unit.gen_affine
    elif unit.demoted_affine is not None:
        # #315: a demoted pose is denied publication, not usefulness. It
        # anchors the rescue search exactly as an incumbent would — which
        # also reaches demoted pages the keymap never placed (richmond p353:
        # misscale, no keymap location, previously status no_keymap with no
        # search at all; its own 3.06x-mis-scaled pose is a good init).
        seed_affine = unit.demoted_affine
    if seed_affine is not None:
        lon_c = (
            seed_affine[0, 0] * unit.width / 2
            + seed_affine[0, 1] * unit.height / 2
            + seed_affine[0, 2]
        )
        lat_c = (
            seed_affine[1, 0] * unit.width / 2
            + seed_affine[1, 1] * unit.height / 2
            + seed_affine[1, 2]
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
    # The page's own printed scale note, where read, is the most authoritative
    # scale rung of all -- Columbus p297 is a 200 ft district sheet in a 50 ft
    # volume, ~5x the median, a rung no other prior source proposes.
    from mapsnap.osm_snap import ScalePrior as _ScalePrior
    from mapsnap.printed_scale import note_m_per_px, printed_scale_ft

    note = printed_scale_ft(vctx.volume / f"{unit.stem}.streets.json")
    if note is not None:
        calibration, _source = volume_note_calibration(vctx)
        implied = note_m_per_px(note[0], calibration)
        if all(abs(math.log(implied / prior.m_per_px)) > 0.15 for prior in scales):
            scales = [_ScalePrior(implied, 0.05, "printed-note"), *scales]
    # Overlapping discs are one search; see cluster_search_centers.
    centers = cluster_search_centers(centers, 0.75 * radius)
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
    if candidate.stamp_separation_m is not None:
        record["stamp_separation_m"] = round(candidate.stamp_separation_m, 1)
    if candidate.stamp_median_m is not None:
        record["stamp_median_m"] = round(candidate.stamp_median_m, 1)
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


def page_record(vctx: VolumeContext, unit: PageUnit) -> dict:
    """Generate the full candidates.jsonl record for one page."""
    ctx, status = build_page_context(vctx, unit)
    record: dict = {
        "target": unit.stem,
        "status": status,
        "fit_state": unit.fit_state,
        "georef_mtime": georef_variant_mtime(vctx.volume, unit.stem),
        "keymap_mtime": keymap_sidecar_mtime(vctx.volume),
        "contradiction_mtime": contradiction_hint_mtime(vctx.volume, unit.stem),
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
        incumbent = evaluate_pose(ctx, vctx.feature_index, unit.gen_affine)
        if incumbent is not None:
            incumbent["world_affine"] = [
                [float(v) for v in row] for row in unit.gen_affine
            ]
            incumbent["effective_gcps"] = unit.inlier_intersections
            if unit.rmse_ft is not None:
                incumbent["rmse_ft"] = round(unit.rmse_ft, 1)
            record["incumbent"] = incumbent
            # A defensible incumbent can only be improved LOCALLY: refinement
            # requires <100 ft agreement and a rung flip is co-located by
            # construction, while arbitration (the only rule that moves a page
            # far) demands an indefensible incumbent. So the challenge search
            # collapses to one center at the incumbent with a small radius --
            # the 29-center hunts on LA's lettered sheets were 87% of corpus
            # snap time spent challenging healthy fits (issue #155).
            if (
                incumbent.get("verification") is not None
                and incumbent["verification"] >= INCUMBENT_DEFENSIBLE_VERIFICATION
            ):
                lon_c = float(
                    unit.gen_affine[0, 0] * unit.width / 2
                    + unit.gen_affine[0, 1] * unit.height / 2
                    + unit.gen_affine[0, 2]
                )
                lat_c = float(
                    unit.gen_affine[1, 0] * unit.width / 2
                    + unit.gen_affine[1, 1] * unit.height / 2
                    + unit.gen_affine[1, 2]
                )
                ctx.search_centers = [(lon_c, lat_c)]
                ctx.radius_m = min(ctx.radius_m, LOCAL_CHALLENGE_RADIUS_M)
                record["search"] = {
                    "centers": [[round(lon_c, 7), round(lat_c, 7)]],
                    "radius_m": round(ctx.radius_m, 1),
                    "radius_source": "local-challenge",
                }
    if unit.fit_state in RESCUE_STATES and unit.demoted_affine is not None:
        # #315: the demoted pose's center joined the search in
        # build_page_context; here its rotation — the pose's most trustworthy
        # component (its scale is often the very reason for the demotion) —
        # enters as a prior, and the record is marked so the debugger can show
        # the seed's provenance.
        a = unit.demoted_affine
        theta = math.degrees(math.atan2(-a[1, 0], a[0, 0]))
        ctx.rotation_priors = [
            *ctx.rotation_priors,
            RotationPrior(
                theta_deg=theta, sigma_deg=DEMOTED_SEED_SIGMA_DEG, source="demoted-pose"
            ),
        ]
        record["search"]["demoted_seed"] = True
        record["priors"]["rotation"].append(
            {
                "theta_deg": round(theta, 2),
                "sigma_deg": DEMOTED_SEED_SIGMA_DEG,
                "source": "demoted-pose",
            }
        )
    candidates = snap_page(ctx, vctx.feature_index)
    # #324: a candidate whose pose reads the page upside-down is
    # corpus-impossible (0/1,332 truth pages in the zone); snap's rotation
    # ladder has admitted 180-off aliases before (detroit p77 at 116 deg,
    # 687 ft). Marked implausible the same way the stamp gate does, so it can
    # never win rescue, challenge, or refinement.
    for candidate in candidates:
        if pose_is_upside_down(candidate.world_affine):
            candidate.plausible = False
            candidate.gate_reasons.append("upside-down")
            candidate.verification = -math.inf

    if unit.fit_state in RESCUE_STATES:
        # A contradiction-demoted page may only be re-adopted at a pose that
        # satisfies the invariant that demoted it: its printed claim of a
        # hinting neighbor must land back on that neighbor's stamp. Without
        # this, a verification-confident alias simply re-verifies from the
        # very hints meant to displace it (KC p551, re-adopted 686 -> 692 ft).
        from mapsnap.adjacency_gate import STAMP_REHOME_M, load_stamp_gate

        stamp_gate = load_stamp_gate(vctx.volume, unit.stem, unit.width, unit.height)
        if stamp_gate is not None:
            for candidate in candidates:
                separations = stamp_gate.partner_separations_m(candidate.world_affine)
                separation = min(separations) if separations else None
                candidate.stamp_separation_m = separation
                if separations:
                    candidate.stamp_median_m = float(statistics.median(separations))
                if separation is not None and separation > STAMP_REHOME_M:
                    candidate.plausible = False
                    candidate.gate_reasons.append(
                        f"stamp-inconsistent({separation:.0f}m)"
                    )
                    candidate.verification = -math.inf
            candidates.sort(key=lambda c: -c.select_score())
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
        osm_prob, _, _ = osm_rasters(frame, vctx.feature_index)
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
    from mapsnap.road_model import ROAD_MODEL_PATH, load_model, predict_page

    out_dir = volume / "artifacts" / "edge_join" / "roadprob"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    model = load_model(ROAD_MODEL_PATH, device)
    for stem in missing:
        gray = cv2.imread(str(volume / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        prob = predict_page(model, gray, device)
        cv2.imwrite(str(out_dir / f"{stem}.png"), (prob * 255).round().astype(np.uint8))
    print(f"  inferred {len(missing)} P(road) maps")


# Module-level state populated by init_worker and read by snap_one_page. Each
# multiprocessing worker gets its own copy; at --num-workers=1 the main process
# fills it directly instead of spawning a pool.
worker_state: dict[str, Any] = {}


def init_worker(volume: Path, vctx: VolumeContext | None = None) -> None:
    """Give this process the volume context snap_one_page reads.

    A pool worker receives only the volume path and rebuilds the context
    itself: the centerlines run to hundreds of thousands of features and
    pickling them to every worker costs far more than re-reading the file
    (~1 s), and the rebuild is deterministic — it reads the same sidecars. The
    sequential path passes the context the caller already built.
    """
    context = load_volume_context(volume) if vctx is None else vctx
    worker_state["vctx"] = context
    worker_state["units"] = {
        unit.stem: unit for unit in list(context.units) + list(context.panel_units)
    }


def snap_one_page(stem: str) -> tuple[str, dict]:
    """Generate one page's candidates record from worker_state.

    Runs the same way whether dispatched to a pool worker or called directly,
    so a page's record is identical either way. The record carries the page's
    own wall-clock cost as ``elapsed_s``: matching dominates a fit run, and the
    per-page number is what distinguishes a uniformly slow volume from a few
    pathological pages. It is wall time inside one worker, so with
    ``--num-workers N`` the sum over pages exceeds the run's elapsed time.
    """
    started = time.perf_counter()
    record = page_record(worker_state["vctx"], worker_state["units"][stem])
    record["elapsed_s"] = round(time.perf_counter() - started, 2)
    return stem, record


def cmd_candidates(
    volume: Path,
    pages: list[str] | None,
    all_pages: bool,
    limit: int | None,
    recompute: bool,
    vis: bool,
    *,
    num_workers: int = 1,
) -> None:
    """Generate candidates.jsonl for the volume's rescue targets."""
    out_dir = artifacts_dir(volume)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.jsonl"
    units = load_page_units(volume)
    if not any(u.fit_state == "fitted" and u.gen_affine is not None for u in units):
        # Nothing to calibrate scale/radius/rotation against — and nothing for
        # the channel to do that could be trusted. Leave an empty candidates
        # file so select/materialize no-op cleanly instead of erroring.
        print(
            f"{volume.name}: no fitted pages to calibrate against; "
            "skipping snap candidates."
        )
        if not out_path.exists():
            out_path.write_text("")
        return
    vctx = load_volume_context(volume, units)
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
    # Every target needs a P(road) map, whole pages included. (This once
    # inferred only "__" panels, on the assumption that whole-page maps already
    # existed from an edge-join `infer` run — true of the dev volumes, false of
    # any new volume, which then had every whole page rejected as 'no_prob'.)
    ensure_probs(
        volume,
        [
            u.stem
            for u in targets
            if load_prob(volume, u.stem) is None or recompute or pages
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
    keymap_mtime = keymap_sidecar_mtime(volume)
    stale = [
        unit
        for unit in targets
        if recompute
        or pages
        or (cached := existing.get(unit.stem)) is None
        or not candidates_record_fresh(
            cached,
            unit,
            georef_variant_mtime(volume, unit.stem),
            contradiction_hint_mtime(volume, unit.stem),
            keymap_mtime,
        )
    ]
    by_stem = {unit.stem: unit for unit in targets}

    def record_done(stem: str, record: dict) -> None:
        """Log one finished page, draw its contact sheet, and checkpoint the file."""
        existing[stem] = record
        best = (record.get("candidates") or [{}])[0]
        rmse = best.get("rmse_ft")
        print(
            f"  {stem:<8} {record['status']:<14}"
            f" cands={len(record.get('candidates', []))}"
            + (f" best_rmse={rmse:.0f}ft" if rmse is not None else "")
            + (
                f" score={best['select_score']}"
                if best.get("select_score") is not None
                else ""
            )
            + (
                f" [{record['elapsed_s']:.0f}s]"
                if record.get("elapsed_s") is not None
                else ""
            )
        )
        if vis and record.get("candidates"):
            write_contact_sheet(vctx, by_stem[stem], record, vis_dir)
        # Rewrite after every page so an interrupted run keeps its progress.
        with out_path.open("w") as handle:
            for target in sorted(existing):
                handle.write(json.dumps(existing[target]) + "\n")

    if num_workers > 1 and len(stale) > 1:
        # Matching is independent per page and CPU-bound. Workers rebuild the
        # volume context from disk rather than receive it pickled, so start-up
        # costs about a second each; results are consumed as they land so the
        # checkpoint file keeps pace with an interrupted run.
        with multiprocessing.Pool(
            num_workers, initializer=init_worker, initargs=(volume,)
        ) as pool:
            for stem, record in pool.imap_unordered(
                snap_one_page, [unit.stem for unit in stale]
            ):
                record_done(stem, record)
    else:
        init_worker(volume, vctx)
        for unit in stale:
            # Which page we are ON, to stderr, before the work starts. The
            # finished-page line below only prints on success, so a native crash
            # (#296) would otherwise be attributable only to "the page after the
            # last one logged". Stderr keeps stdout's parsed format unchanged.
            print(f"snap: starting {unit.stem}", file=sys.stderr, flush=True)
            record_done(*snap_one_page(unit.stem))
    print(f"{len(stale)} pages computed; {len(existing)} total in {out_path}")


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

    def pose_center(c):
        # The candidate's REFINED pose center, not its search center: two
        # searches from different hint centers converge to the same lock
        # (richmond p353's 6 ft and 10 ft twins came from hint clusters 250 m
        # apart), and twin-ness is a property of where the pose LANDED.
        a = c.get("world_affine")
        if a is None:
            return c["center"]
        w = record.get("width") or 0
        h = record.get("height") or 0
        return (
            a[0][0] * w / 2 + a[0][1] * h / 2 + a[0][2],
            a[1][0] * w / 2 + a[1][1] * h / 2 + a[1][2],
        )

    top_center = pose_center(top)
    for candidate in candidates[1:]:
        if candidate.get("select_score") is None:
            continue
        cand_center = pose_center(candidate)
        separation = haversine_m(
            top_center[1],
            top_center[0],
            cand_center[1],
            cand_center[0],
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


# Volume-level train/holdout split for the refinement-margin sweep (#153).
# The dev-4 volumes tuned every other constant, so they stay on the train
# side; NO-1896's truth is known-noisy, so it trains rather than judges;
# chicago balances the split at 6/6.
REFINE_SWEEP_TRAIN = {
    "chicago_il_1950_vol_1",
    "detroit_mich_1929_vol_11",
    "hudson_co_nj_1950_vol_9",
    "los_angeles_ca_1949_vol_14",
    "new_orleans_la_1896_vol_2",
    "washington_dc_1916_vol_2",
}
REFINE_SWEEP_MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
# Band-aware margins: below the incumbent-verification edge the incumbent is
# weakly supported (a low margin is safe); above it the incumbent is already
# well-verified and a higher bar protects it from churn. inf = never refine.
REFINE_SWEEP_BAND_EDGES = [0.25, 0.5, 0.75, 1.0]
REFINE_SWEEP_BAND_MARGINS = [0.0, 0.05, 0.1, 0.2, 0.4, math.inf]


def truth_item_land_weights(volume: Path) -> dict[str, float]:
    """Land-weighted area (m^2) for every truth item, split panels included.

    Unlike truth_land_weights (whole pages only, for the rescue sweeps), this
    keys every truth item by its own page key — the region-graded scorer
    grades split panels individually, so the refinement sweep needs their
    individual weights.
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
        key = source_id_to_page_key(
            item.get("target", {}).get("source", {}).get("id"), item.get("label", "")
        )
        land = polygon.area * land_fraction(polygon, tree)
        weights[key] = weights.get(key, 0.0) + land
    return weights


def refine_eligible_features(records: list[dict]) -> dict[str, dict]:
    """Per-target features for every fitted page refinement could ever adopt.

    Eligibility mirrors cmd_select's arbitrate branch with the margin removed:
    fitted, not claimed by arbitration, and carrying an agreeing top
    challenger with a verification head-to-head available. The sweep applies
    margin rules to these features in-process.
    """
    eligible: dict[str, dict] = {}
    for record in records:
        if record.get("fit_state") != "fitted":
            continue
        if arbitrate_challenge(record, PRODUCTION_ARBITRATE_GATE) is not None:
            continue
        adoption = refine_adoption(record, margin=-math.inf)
        if adoption is None:
            continue
        incumbent = record["incumbent"]
        top = record["candidates"][0]
        eligible[record["target"]] = {
            "incumbent_verification": incumbent["verification"],
            "challenger_verification": top["verification"],
            "incumbent_name": (incumbent.get("name") or {}).get("score") or 0.0,
            "challenger_name": (top.get("name") or {}).get("score") or 0.0,
            "disagreement_ft": adoption["disagreement_ft"],
            "incumbent_rmse_ft": incumbent.get("rmse_ft"),
            "challenger_rmse_ft": top.get("rmse_ft"),
        }
    return eligible


def build_hybrid_iiif(volume: Path, output: Path) -> None:
    """Build the osm-first hybrid IIIF for the volume's current sidecars."""
    import subprocess

    georef_glob = f"{volume / '*.georef-snap.json'},{volume / '*.georef.json'}"
    subprocess.run(
        [
            "mapsnap",
            "iiif",
            str(volume / "main.iiif.json"),
            georef_glob,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def grade_refine_variants(volume: Path, recompute: bool = False) -> dict:
    """Grade every truth item with refinement fully off and fully on.

    Materializes two sidecar variants (refine_margin +inf / -inf), builds the
    hybrid IIIF for each, and grades both through the real region-graded
    scorer (compare_pages) so the sweep is exactly faithful to `mapsnap
    score`. Restores the production selection afterwards. Cached in
    artifacts/osm_snap/refine_sweep.json.
    """
    from mapsnap.compare_iiif_georef import compare_pages

    cache_path = artifacts_dir(volume) / "refine_sweep.json"
    candidates_path = artifacts_dir(volume) / "candidates.jsonl"
    if (
        cache_path.exists()
        and not recompute
        and cache_path.stat().st_mtime >= candidates_path.stat().st_mtime
    ):
        return canonicalize_refine_keys(json.loads(cache_path.read_text()))

    records = load_candidates(volume)
    eligible = refine_eligible_features(records)
    graded: dict[str, dict[str, dict]] = {}
    try:
        for variant, margin in (("none", math.inf), ("all", -math.inf)):
            cmd_select(
                volume,
                "arbitrate",
                PRODUCTION_GATE_SCORE,
                PRODUCTION_GATE_MARGIN,
                PRODUCTION_ARBITRATE_GATE,
                refine_margin=margin,
            )
            cmd_materialize(volume, "arbitrate")
            variant_iiif = volume / f"refine-sweep-{variant}.iiif.json"
            build_hybrid_iiif(volume, variant_iiif)
            rows, missing = compare_pages(volume / "main.iiif.json", variant_iiif)
            variant_iiif.unlink()
            by_key: dict[str, dict] = {}
            for row in rows:
                by_key.setdefault(
                    row["page_key"],
                    {"gen_key": row["gen_page_key"], "rmse_ft": row["rmse_ft"]},
                )
            for row in missing:
                by_key.setdefault(row["page_key"], {"gen_key": None, "rmse_ft": None})
            graded[variant] = by_key
    finally:
        # Leave the volume's sidecars and selection in the production state.
        cmd_select(
            volume,
            "arbitrate",
            PRODUCTION_GATE_SCORE,
            PRODUCTION_GATE_MARGIN,
            PRODUCTION_ARBITRATE_GATE,
        )
        cmd_materialize(volume, "arbitrate")

    weights = truth_item_land_weights(volume)
    none_rows, all_rows = graded["none"], graded["all"]
    items = []
    total_land = 0.0
    for key in sorted(none_rows):
        weight = weights.get(key)
        if weight is None:
            continue
        total_land += weight
        none_row = none_rows[key]
        all_row = all_rows.get(key, none_row)
        items.append(
            {
                "key": key,
                "gen_key": none_row["gen_key"],
                "land_m2": weight,
                "rmse_none": none_row["rmse_ft"],
                "rmse_all": all_row["rmse_ft"],
            }
        )
    result = canonicalize_refine_keys(
        {
            "volume": volume.name,
            "eligible": eligible,
            "total_land_m2": total_land,
            "items": items,
        }
    )
    cache_path.write_text(json.dumps(result))
    return result


def canonicalize_refine_keys(result: dict) -> dict:
    """Join sidecar-cased targets with truth-cased gen keys, in place.

    Sidecar stems are lowercase for some volumes (chicago p101w) while the
    truth annotations carry the case (p101W); compare's gen_page_key uses the
    truth's casing, so without this remap those pages' adoptions silently
    fall out of every margin rule's outcome. Idempotent.
    """
    canonical = {target.lower(): target for target in result["eligible"]}
    for item in result["items"]:
        gen_key = item.get("gen_key")
        if gen_key is not None:
            item["gen_key"] = canonical.get(gen_key.lower(), gen_key)
    return result


def refine_bucket(rmse_ft: float | None) -> int:
    """Score-bucket value of one truth item: +1 good, -1 disaster, else 0."""
    if rmse_ft is None:
        return 0
    if rmse_ft <= 25.0:
        return 1
    if rmse_ft >= 200.0:
        return -1
    return 0


def refine_adopt_set(
    eligible: dict[str, dict],
    margin: float,
    name_parity: bool = False,
    band: tuple[float, float, float] | None = None,
) -> set[str]:
    """Targets a margin rule adopts.

    band=(edge, low, high) overrides margin: low applies below the
    incumbent-verification edge, high above it. name_parity additionally
    requires the challenger's name score to be at least the incumbent's.
    """
    adopted = set()
    for target, features in eligible.items():
        rule_margin = margin
        if band is not None:
            edge, low, high = band
            rule_margin = low if features["incumbent_verification"] < edge else high
        head_to_head = (
            features["challenger_verification"] - features["incumbent_verification"]
        )
        if head_to_head <= rule_margin:
            continue
        if name_parity and features["challenger_name"] < features["incumbent_name"]:
            continue
        adopted.add(target)
    return adopted


def refine_rule_outcome(
    volume_data: dict, adopted: set[str]
) -> tuple[float, float, int, int]:
    """(Δland_m2 vs no refinement, total land_m2, bucket gains, losses)."""
    delta = 0.0
    gains = losses = 0
    for item in volume_data["items"]:
        if item["gen_key"] not in adopted:
            continue
        before = refine_bucket(item["rmse_none"])
        after = refine_bucket(item["rmse_all"])
        if after == before:
            continue
        delta += (after - before) * item["land_m2"]
        if after > before:
            gains += 1
        else:
            losses += 1
    return delta, volume_data["total_land_m2"], gains, losses


def cmd_sweep_refine(volumes: list[Path], recompute: bool = False) -> None:
    """Sweep the refinement margin (#153) against the region-graded scorer."""
    data = [grade_refine_variants(volume, recompute) for volume in volumes]
    mismatched = sum(
        1
        for d in data
        for item in d["items"]
        if item["gen_key"] not in d["eligible"]
        and item["rmse_none"] != item["rmse_all"]
    )
    if mismatched:
        print(f"WARNING: {mismatched} non-eligible item(s) changed between variants")
    train = [d for d in data if d["volume"] in REFINE_SWEEP_TRAIN]
    holdout = [d for d in data if d["volume"] not in REFINE_SWEEP_TRAIN]
    n_eligible = sum(len(d["eligible"]) for d in data)
    print(
        f"{n_eligible} refinement-eligible pages across {len(data)} volume(s)"
        f" ({len(train)} train, {len(holdout)} holdout)"
    )

    def evaluate(
        subset: list[dict],
        margin: float,
        name_parity: bool = False,
        band: tuple[float, float, float] | None = None,
    ) -> tuple[float, int, int, int]:
        delta = land = 0.0
        adopted_total = gains = losses = 0
        for volume_data in subset:
            adopted = refine_adopt_set(
                volume_data["eligible"], margin, name_parity, band
            )
            adopted_total += len(adopted)
            vol_delta, vol_land, vol_gains, vol_losses = refine_rule_outcome(
                volume_data, adopted
            )
            delta += vol_delta
            land += vol_land
            gains += vol_gains
            losses += vol_losses
        return (delta / land if land else 0.0), adopted_total, gains, losses

    def print_row(
        label: str,
        margin: float,
        name_parity: bool = False,
        band: tuple[float, float, float] | None = None,
    ) -> None:
        cells = [f"  {label:<24}"]
        for subset, name in ((train, "train"), (holdout, "hold"), (data, "all")):
            net, adopted, gains, losses = evaluate(subset, margin, name_parity, band)
            cells.append(
                f"{name} {net * 100:+5.2f}% ({adopted:>3}a {gains:>3}g {losses:>2}l)"
            )
        print("  ".join(cells))

    print("== global margin sweep (a=adopted pages, g/l=bucket gains/losses) ==")
    for margin in REFINE_SWEEP_MARGINS:
        print_row(f"margin {margin:.2f}", margin)
    print("== + name parity (challenger name score >= incumbent's) ==")
    for margin in REFINE_SWEEP_MARGINS:
        print_row(f"margin {margin:.2f} +name", margin, name_parity=True)
    print("== band-aware margins, top 10 by train Δnet ==")
    combos = []
    for edge in REFINE_SWEEP_BAND_EDGES:
        for low in REFINE_SWEEP_BAND_MARGINS:
            for high in REFINE_SWEEP_BAND_MARGINS:
                band = (edge, low, high)
                train_net, _, _, train_losses = evaluate(train, 0.0, band=band)
                combos.append((train_net, -train_losses, band))
    combos.sort(reverse=True)
    for train_net, _, band in combos[:10]:
        edge, low, high = band
        print_row(f"edge {edge:.2f} lo {low:g} hi {high:g}", 0.0, band=band)


# The frozen production gates (dev-swept; `mapsnap snap` uses these): the
# per-page rescue score/margin gates, the volume-energy conservative elbow,
# and the arbitration score gate.
PRODUCTION_GATE_SCORE = 1.25
PRODUCTION_GATE_MARGIN = 0.25
PRODUCTION_ARBITRATE_GATE = 1.5

STAMP_RESCUE_SCORE = 0.7
"""Relaxed rescue bar for stamp-corroborated candidates.

A contradiction-demoted page's rescue candidate that lands its printed claim
back on the hinting neighbor's stamp carries external evidence the select
score cannot see — the neighbor's printed testimony pins the pose at the seam.
Measured true poses refused by the 1.25 bar: KC p551 at 0.77 (24 ft), GR p828
at 0.93 (23 ft). Stamp-INconsistent candidates are already implausible, so the
relaxed bar only ever admits poses the neighbors vouch for; the margin is
computed against the best non-corroborated rival (NO-1896 p125's true pose at
1.82 was margin-blocked by its own 16-degree twin)."""
VOLUME_MODE_GATE = 1.5  # the dev-chosen conservative elbow for the energy mode

# Arbitration: a snap candidate may replace a placed RANSAC fit only when it
# is a confident, unambiguous lock that clearly disagrees with the incumbent
# AND beats it on the shared evidence (geometry verification + name score).
ARBITRATE_MIN_DISAGREE_FT = 100.0
# ... and only when the incumbent is geometrically INDEFENSIBLE: across all 11
# truth-graded challenges (dev + holdout), every correct overturn had
# incumbent verification <= 0.05 and every wrong one >= 0.12. A plausible
# incumbent losing narrowly is the OSM-divergence trap (Chicago's 1950 core:
# a 400ft slide matches MODERN OSM better than the truth does), so arbitration
# is disaster recovery for fits OSM actively contradicts — not relitigation.
INCUMBENT_DEFENSIBLE_VERIFICATION = 0.1
# Refinement: adopting an AGREEING challenger that clearly wins the evidence
# head-to-head. RANSAC's mid-tier error is label-placement noise; the snap
# pose is chamfer-locked to the street grid, so when the two agree on the
# lock, the snap is simply the more precise estimator (25-50ft incumbents:
# challenger closer to truth 93% of the time, median 33ft -> 12ft). The
# margin protects already-good incumbents from churn. 0.05 is the
# `report --sweep-refine` elbow on the region-graded scorer (#153), 6/6
# volume train/holdout split: it beats the original dev-elbow 0.1 on BOTH
# sides (train +9.66 vs +9.40, holdout +8.39 vs +7.81) and creates no
# disasters (every loss at any margin is a <50ft threshold slide). Margin 0
# admits 7 train losses, so the protection is real but 0.1 was over-tight.
# Rejected by the same sweep: name parity (incumbent name scores come from
# the labels RANSAC fitted on, so the requirement is circular and costs ~3
# points everywhere) and band-aware margins (their edge over flat 0.05 is
# one train loss; zero holdout benefit for the extra shape).
REFINE_VER_MARGIN = 0.05
# ... and only when the incumbent is a real fit. The 93%-of-the-time
# calibration above was measured on multi-GCP RANSAC incumbents; a
# deferred/1-effective-GCP incumbent is a rung-guess free to swing about its
# single anchor, and the candidate search for fitted pages is LOCAL around
# that incumbent, so an "agreeing" challenger confirms the guess by
# construction rather than by evidence. Nashville p4 (#277): both A/B arms
# produced the identical half-scale deferred fit; one candidate roll had no
# agreeing pose, fell through to rung_flip, and landed 26 ft — the other
# rolled two agreeing half-scale poses, refine fired first, and blessed
# 406 ft. Weak incumbents must fall through to the calibrated rung_flip /
# keep-incumbent paths instead.
REFINE_MIN_EFFECTIVE_GCPS = 2
# ... and the challenger's own verification must be informative, not merely
# larger. Verification is inlier_frac + ncc_fine - chamfer/clamp; a negative
# value means the posed road skeleton lands further from OSM than the clamp
# allows, i.e. P(road) does not support this pose (nor, usually, any pose on
# that page). Comparing two negatives with a margin is arithmetic on noise:
# richmond p380 replaced a 16 ft fit (14 GCPs, name 8/9) with a 65 ft pose on
# -0.154 vs -0.249. Corpus-wide this gate touches exactly one adoption of 605
# (#291), and that one is the loss.
REFINE_MIN_VERIFICATION = 0.0


def refine_adoption(record: dict, margin: float = REFINE_VER_MARGIN) -> dict | None:
    """Adopt an agreeing challenger as a precision refinement, or None.

    The complement of arbitrate_challenge: same head-to-head evidence, but for
    challengers that AGREE with the incumbent (within the arbitration
    disagreement floor) and beat its verification by a clear margin. This is
    what reaches the 25-100ft mid-tier that arbitration structurally cannot
    touch (a correct challenger agrees with those incumbents). The margin
    override exists for the sweep harness (`report --sweep-refine`): -inf
    adopts every agreeing challenger, +inf adopts none.
    """
    incumbent = record.get("incumbent")
    candidates = record.get("candidates") or []
    if not incumbent or not candidates:
        return None
    # Refinement's premise — agreement means the pose is right and the snap is
    # merely more precise — requires an incumbent constrained enough to be
    # worth agreeing with (see REFINE_MIN_EFFECTIVE_GCPS). Records cached
    # before effective_gcps was recorded stay eligible.
    effective_gcps = incumbent.get("effective_gcps")
    if effective_gcps is not None and effective_gcps < REFINE_MIN_EFFECTIVE_GCPS:
        return None
    top = candidates[0]
    if top.get("select_score") is None:
        return None
    incumbent_verification = incumbent.get("verification")
    top_verification = top.get("verification")
    if incumbent_verification is None or top_verification is None:
        return None
    if top_verification <= REFINE_MIN_VERIFICATION:
        return None  # the challenger's own evidence is absent (#291)
    if top_verification <= incumbent_verification + margin:
        return None
    disagreement = grid_rmse_ft_between(
        np.array(incumbent["world_affine"]),
        np.array(top["world_affine"]),
        record["width"],
        record["height"],
    )
    if disagreement >= ARBITRATE_MIN_DISAGREE_FT:
        return None  # that far apart is arbitration territory, not refinement
    return {
        "target": record["target"],
        "chosen": 0,
        "reason": "refine",
        "refine": True,
        "select_score": top["select_score"],
        "disagreement_ft": round(disagreement, 1),
        "incumbent_verification": incumbent_verification,
        "challenger_verification": top_verification,
    }


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
    if margin is None or margin < PRODUCTION_GATE_MARGIN:
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
    if incumbent_verification >= INCUMBENT_DEFENSIBLE_VERIFICATION:
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


# Rung-flip arbitration: a fit published at HALF its true scale. Ten of the
# 2026-07-30 corpus's 39 disasters were exact half/double-scale fits with
# near-perfect rotation, and for several the correct pose already sat in the
# candidate list — trapped between refinement (needs <100 ft agreement; a rung
# variant disagrees by 200-700 ft) and arbitration (needs an indefensible
# incumbent; half-scale fits chamfer well on a gridiron and verify >= 0.1).
RUNG_UP_BAND = (1.7, 2.3)
"""Candidate/incumbent scale ratios treated as an up-rung challenge.

UP only, never down: verification carries a small-footprint bias. A half-scale
candidate explains 1/4 the ground and matches dense OSM cheaply — calibrating
over every rung-band candidate pair in the twelve truth volumes, every
would-be DOWN flip that passed any margin was a break (23-98 ft incumbents sent
to 340-11000 ft: NO-1951 p433/p434/p436, Chicago p61w, Detroit p21, Nashville
p44, LA p1499o/r), while every fix but one was an up flip. A doubled candidate
must explain 4x the ground, so verification preferring it is hard-won evidence.
"""
RUNG_VER_MARGIN = 0.1
RUNG_NAME_FLOOR = -0.10
RUNG_SELECT_MIN = 0.9


def printed_note_ratios(volume: Path, records: list[dict]) -> dict[str, float]:
    """expected/incumbent scale ratio per fitted page whose printed note read.

    Calibrates metres-per-pixel-per-printed-ft from the volume's own fitted
    note-bearing pages (the scan resolution, measured), then converts each
    page's note into an expected scale. See mapsnap.printed_scale.
    """
    from mapsnap.printed_scale import printed_scale_ft

    notes: dict[str, int] = {}
    inc_scale: dict[str, float] = {}
    for record in records:
        incumbent = record.get("incumbent") or {}
        affine = incumbent.get("world_affine")
        if not affine:
            continue
        note = printed_scale_ft(volume / f"{record['target']}.streets.json")
        if note is None:
            continue
        notes[record["target"]] = note[0]
        inc_scale[record["target"]] = math.hypot(affine[1][0], affine[1][1])
    samples = sorted(inc_scale[t] / notes[t] for t in notes)
    if len(samples) < 3:
        return {}
    unit = samples[len(samples) // 2]  # median deg-per-px per printed ft
    return {target: (unit * notes[target]) / inc_scale[target] for target in notes}


RUNG_NOTE_BAND = (0.80, 1.25)
"""How closely a scale must match the page's printed note to claim its authority."""


def rung_flip(record: dict, note_ratio: float | None = None) -> dict | None:
    """Adopt a double-scale candidate over a half-scale incumbent, or None.

    Calibrated offline over all 1228 rung-band candidate pairs in the twelve
    truth volumes: these gates flip five pages — four disasters to <=25 ft
    (Brooklyn p14 403->15, KC p556 289->13, Nashville p4 396->26, DC p153
    306->12), one disaster to a different disaster (NO-1896 p125), and break
    nothing. Family/scale priors were tried as the referee first and rejected:
    the family-rung prior endorses the wrong rung on exactly the contested
    pages (it was derived from the bad fits), and the volume median breaks
    legitimate second scale families (Nashville) and oversize sheets (LA
    p1499*). Direction + verification margin + name parity is the whole rule.
    """
    incumbent = record.get("incumbent")
    candidates = record.get("candidates") or []
    if not incumbent or not incumbent.get("world_affine") or not candidates:
        return None
    inc_affine = incumbent["world_affine"]
    inc_scale = math.hypot(inc_affine[1][0], inc_affine[1][1])
    incumbent_verification = incumbent.get("verification")
    if inc_scale <= 0 or incumbent_verification is None:
        return None
    incumbent_name = (incumbent.get("name") or {}).get("score")
    # The printed scale note (#196), where read, is authority rather than
    # inference: note_ratio is expected/incumbent scale. An incumbent that
    # contradicts its own page's note by ~a rung forfeits the up-only
    # protection, so a note-matching candidate may flip DOWN as well -- the
    # direction verification alone cannot referee (small-footprint bias).
    note_condemns = note_ratio is not None and not (
        1 / RUNG_NOTE_BAND[1] <= note_ratio <= 1 / RUNG_NOTE_BAND[0]
    )
    best_index = None
    best_margin = 0.0
    for index, candidate in enumerate(candidates):
        affine = candidate.get("world_affine")
        verification = candidate.get("verification")
        score = candidate.get("select_score")
        if not affine or verification is None or score is None:
            continue
        ratio = math.hypot(affine[1][0], affine[1][1]) / inc_scale
        in_up_band = RUNG_UP_BAND[0] <= ratio <= RUNG_UP_BAND[1]
        in_down_band = 1 / RUNG_UP_BAND[1] <= ratio <= 1 / RUNG_UP_BAND[0]
        note_endorses = (
            note_condemns
            and note_ratio is not None
            and RUNG_NOTE_BAND[0] <= ratio / note_ratio <= RUNG_NOTE_BAND[1]
        )
        if not (in_up_band or (in_down_band and note_endorses)):
            continue
        if score < RUNG_SELECT_MIN:
            continue
        margin = verification - incumbent_verification
        if margin < RUNG_VER_MARGIN:
            continue
        candidate_name = (candidate.get("name") or {}).get("score")
        if (
            incumbent_name is not None
            and candidate_name is not None
            and candidate_name - incumbent_name < RUNG_NAME_FLOOR
        ):
            continue
        if best_index is None or margin > best_margin:
            best_index, best_margin = index, margin
    if best_index is None:
        return None
    return {
        "target": record["target"],
        "chosen": best_index,
        "reason": "rung",
        "rung": True,
        "select_score": candidates[best_index].get("select_score"),
        "incumbent_verification": incumbent_verification,
        "challenger_verification": candidates[best_index].get("verification"),
    }


# Sheet-agreement gate for split panels. The GEOMETRY is exact: a panel jpg
# is a bbox crop of the base jpg (pure translation), so one panel's pose
# determines the whole base image's implied placement. The original intent
# was that siblings of one sheet must imply the SAME placement — but that
# premise assumes the sheet is one contiguous map cut into pieces, and
# measured against every fitted sibling pair in the 12 truth volumes, real
# Sanborn split pages are INSET COMPOSITES instead: separate neighborhood
# maps pasted onto one sheet of paper, whose implied sheets disagree by
# 233-2143 m. In practice, therefore, this gate acts as a conservative
# BLANKET BLOCK: a panel with a reliably-fitted sibling effectively has no
# agreeing candidates and is not rescued at all (that block is what stopped
# KC's four would-be panel disasters, e.g. p555__2 at 753 ft with select
# 2.58 — a score no threshold would have caught); accepted panels reach the
# iiif through the solo-score or mutual paths instead. The honest redesign —
# per-inset gating instead of the sheet fiction — is tracked in the PR's
# next steps; the behavior here is what the 12-volume scores validated.
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


SNAP_LOG_BEGIN = "==== mapsnap snap ===="
SNAP_LOG_END = "==== end mapsnap snap ===="


def append_snap_logs(
    volume: Path, records: list[dict], selections: list[dict], mode: str
) -> None:
    """Append each page's snap decision to its pN.txt georef log.

    The RANSAC georeferencer captures its per-page log to <stem>.txt; the
    debugger surfaces that file, so the snap channel's decision belongs there
    too. Idempotent: any previous snap section is replaced, not duplicated.
    """
    by_target = {s["target"]: s for s in selections}
    for record in records:
        stem = record["target"]
        choice = by_target.get(stem)
        lines = [
            SNAP_LOG_BEGIN,
            (
                f"mode: {mode}   status: {record['status']}   "
                f"fit_state: {record['fit_state']}"
            ),
        ]
        priors = record.get("priors", {}).get("rotation", [])
        if priors:
            lines.append(
                "rotation priors: "
                + ", ".join(f"{p['theta_deg']:.0f}deg({p['source']})" for p in priors)
            )
        for rank, candidate in enumerate((record.get("candidates") or [])[:3]):
            name = candidate.get("name") or {}
            lines.append(
                f"#{rank + 1}: select={candidate.get('select_score')} "
                f"ver={candidate.get('verification')} "
                f"theta={candidate['theta_deg']:.0f}deg({candidate['theta_source']}) "
                f"name={name.get('n_hits')}/{name.get('n_labels')} "
                f"center_dist={candidate['center_dist_m']:.0f}m"
                + (
                    f" rmse={candidate['rmse_ft']:.0f}ft(vs truth)"
                    if candidate.get("rmse_ft") is not None
                    else ""
                )
            )
        incumbent = record.get("incumbent")
        if incumbent:
            name = incumbent.get("name") or {}
            lines.append(
                f"incumbent: ver={incumbent.get('verification')} "
                f"name={name.get('n_hits')}/{name.get('n_labels')}"
                + (
                    f" rmse={incumbent['rmse_ft']:.0f}ft(vs truth)"
                    if incumbent.get("rmse_ft") is not None
                    else ""
                )
            )
        if choice is None:
            outcome = (
                "incumbent kept" if record["fit_state"] == "fitted" else "no entry"
            )
        elif choice.get("chosen") is None:
            outcome = f"abstain ({choice.get('reason')})"
        else:
            kind = (
                "challenge"
                if choice.get("challenge")
                else "refine"
                if choice.get("refine")
                else "rescue"
            )
            outcome = f"{kind}: candidate #{choice['chosen'] + 1} accepted"
        lines.append(f"outcome: {outcome}")
        lines.append(SNAP_LOG_END)

        path = volume / f"{stem}.txt"
        text = path.read_text() if path.exists() else ""
        if SNAP_LOG_BEGIN in text:
            head, _, rest = text.partition(SNAP_LOG_BEGIN)
            _, _, tail = rest.partition(SNAP_LOG_END)
            text = head + tail.lstrip("\n")
        if text and not text.endswith("\n"):
            text += "\n"
        path.write_text(text + "\n".join(lines) + "\n")


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
    refine_margin: float | None = None,
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
        note_ratios = printed_note_ratios(volume, records)
        challenged = refined = flipped = 0
        for record in records:
            if record.get("fit_state") != "fitted":
                continue
            challenge = arbitrate_challenge(record, arbitrate_gate)
            if challenge is not None:
                selections.append(challenge)
                challenged += 1
                continue
            refinement = refine_adoption(
                record, REFINE_VER_MARGIN if refine_margin is None else refine_margin
            )
            if refinement is not None:
                selections.append(refinement)
                refined += 1
                continue
            flip = rung_flip(record, note_ratio=note_ratios.get(record["target"]))
            if flip is not None:
                selections.append(flip)
                flipped += 1
        print(
            f"{challenged} challenges, {refined} refinements, "
            f"{flipped} rung flips accepted"
        )
    else:
        selections = select_argmax(rescue, gate_score, gate_margin, allowed)
    accepted = sum(1 for s in selections if s.get("chosen") is not None)
    with out_path.open("w") as handle:
        for choice in selections:
            handle.write(json.dumps(choice) + "\n")
    append_snap_logs(volume, records, selections, mode)
    print(f"{accepted}/{len(selections)} pages accepted -> {out_path}")


def stamp_corroborated(candidate: dict) -> bool:
    """Whether the hinting neighbors genuinely vouch for a candidate pose.

    Uses the MEDIAN partner separation: genuine hints agree, so a true pose
    satisfies essentially all of them, while junk hints (single-digit
    reciprocation) scatter across the volume and no wrong pose can satisfy
    most. Nashville p8's 325 ft candidate matched one of its four junk stamps
    at 66 m (the min) and was wrongly adopted; its median was far out.
    """
    from mapsnap.adjacency_gate import STAMP_REHOME_M

    median = candidate.get("stamp_median_m")
    return median is not None and median <= STAMP_REHOME_M


def uncorroborated_margin(record: dict) -> float:
    """Rank-1's select_score lead over the best rival the neighbors don't vouch for."""
    candidates = record.get("candidates") or []
    top_score = candidates[0]["select_score"]
    rivals = [
        c["select_score"]
        for c in candidates[1:]
        if c.get("select_score") is not None and not stamp_corroborated(c)
    ]
    return top_score - max(rivals) if rivals else math.inf


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
            corroborated = (
                record.get("fit_state") in RESCUE_STATES
                and score is not None
                and stamp_corroborated(top)
            )
            if corroborated:
                # The hinting neighbor's printed testimony pins this pose at
                # the seam: relax the absolute bar, and measure ambiguity only
                # against rivals the neighbors do NOT vouch for (a corroborated
                # twin is the same corridor, not a competing hypothesis).
                margin = uncorroborated_margin(record)
            effective_gate = STAMP_RESCUE_SCORE if corroborated else gate_score
            if score is None:
                choice["reason"] = "implausible"
            elif score < effective_gate:
                choice["reason"] = f"score {score:.2f} < {effective_gate}"
            elif margin is None or margin < gate_margin:
                choice["reason"] = f"margin {margin} < {gate_margin}"
            else:
                choice = {
                    "target": stem,
                    "chosen": 0,
                    "reason": "stamp-corroborated" if corroborated else "accepted",
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

    Panels additionally face the sheet-agreement machinery: options whose
    implied full-sheet placement disagrees with a reliable fitted sibling's
    are dropped up front, co-accepted sibling picks must imply the same sheet
    (the pairwise sheets_agree term), and an accepted panel with no reliable
    anchor needs a co-accepted sibling or a PANEL_SOLO_GATE score. NOTE:
    real split sheets are inset composites (see SHEET_AGREE_TOL_M), so in
    practice the first two conditions blanket-block rather than discriminate.
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

        # `component` is bound as a default rather than captured: every caller
        # below scores THIS component, and a later iteration must not rebind it.
        def total_energy(
            assign: dict[str, int | None], component: list[str] = component
        ) -> float:
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
        incumbent = record.get("incumbent")
        if incumbent is not None:
            if unit.truth is not None and unit.gen_affine is not None:
                incumbent["rmse_ft"] = round(
                    grid_rmse_ft_between(
                        unit.truth.affine_local,
                        unit.gen_affine,
                        unit.width,
                        unit.height,
                    ),
                    1,
                )
            else:
                incumbent.pop("rmse_ft", None)
    out_path = artifacts_dir(volume) / "candidates.jsonl"
    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(
        f"{volume.name}: {newly} split-truth pages attached, "
        f"{changed} records flipped has_truth"
    )


def osm_variant_path(volume: Path, stem: str) -> Path:
    return volume / f"{stem}.georef-snap.json"


def cmd_materialize(volume: Path, mode: str) -> None:
    """Write pN.georef-snap.json sidecars for the selection's accepted pages."""
    from mapsnap.edge_join_experiment import page_fit_state

    records = {r["target"]: r for r in load_candidates(volume)}
    selection_path = artifacts_dir(volume) / f"selection_{mode}.jsonl"
    if not selection_path.exists():
        sys.exit(f"{selection_path} missing; run `select --mode {mode}` first")
    # Remove every sidecar this channel owns before writing: a stale one from
    # a looser gate would silently keep scoring.
    for stale in volume.glob("p*.georef-snap.json"):
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
            # georef-snap sidecar can only be one written earlier this loop.
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
                "refine": bool(choice.get("refine")),
                "select_score": candidate.get("select_score"),
                # The margin selection actually gated on: rank-1's lead over
                # the best DISTINCT lock (raw rank1-rank2 is misleading when
                # the runner-up is a near-identical twin).
                "margin": (
                    None
                    if (margin := distinct_margin(record)) is None or math.isinf(margin)
                    else round(margin, 4)
                ),
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
    print(f"{written} pN.georef-snap.json sidecars written in {volume}")


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
    p_cand.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help="worker processes for the per-page matching pass (default: %(default)s)",
    )

    p_rep = sub.add_parser("report", help="ranking diagnostics vs truth")
    p_rep.add_argument("volume", type=Path, nargs="+")
    p_rep.add_argument(
        "--sweep", action="store_true", help="grid the gates, print simulated Δnet"
    )
    p_rep.add_argument(
        "--sweep-arbitrate",
        action="store_true",
        help="grid the arbitration gate over fitted-page challenges",
    )
    p_rep.add_argument(
        "--sweep-refine",
        action="store_true",
        help=(
            "sweep the refinement margin against the region-graded scorer "
            "(#153); accepts multiple volumes for the train/holdout split"
        ),
    )
    p_rep.add_argument(
        "--recompute",
        action="store_true",
        help="rebuild the cached refine-sweep gradings",
    )

    p_sel = sub.add_parser("select", help="pick candidates / abstain per page")
    p_sel.add_argument("volume", type=Path)
    p_sel.add_argument(
        "--mode",
        choices=["argmax", "volume", "union", "arbitrate"],
        default="argmax",
        help=(
            "argmax/volume: single rescue committee; union: both committees; "
            "arbitrate: union PLUS challenges and refinements of placed fits"
        ),
    )
    p_sel.add_argument("--gate-score", type=float, default=PRODUCTION_GATE_SCORE)
    p_sel.add_argument("--gate-margin", type=float, default=PRODUCTION_GATE_MARGIN)
    p_sel.add_argument(
        "--arbitrate-gate", type=float, default=PRODUCTION_ARBITRATE_GATE
    )

    p_mat = sub.add_parser("materialize", help="write pN.georef-snap.json sidecars")
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
            num_workers=args.num_workers,
        )
    elif args.command == "report":
        if args.sweep_refine:
            cmd_sweep_refine(args.volume, args.recompute)
        elif args.sweep_arbitrate:
            cmd_sweep_arbitrate(args.volume[0])
        elif args.sweep:
            cmd_sweep(args.volume[0])
        else:
            cmd_report(args.volume[0])
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
