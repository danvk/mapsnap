"""Truth-aware harness for the road-mask edge-join investigation.

Measures whether road-UNet masks support precise (<25 ft RMSE) edge-to-edge
joins between well-georeferenced "anchor" pages and their unmapped neighbors.
Everything here may read the volume's OIM truth (main.iiif.json); the matcher
itself lives in edge_join.py and is truth-free.

Subcommands:
    stats   volume-level truth statistics: anchor sets (truth-based vs
            truth-free), adjacency pair enumeration (detected + truth-derived),
            inter-page rotation distribution, overlap-strip geometry, and the
            coverage ceiling for the missing pages.
    infer   cache P(road) maps for every base page jpg under
            artifacts/edge_join/roadprob/.
    sanity  per-adjacent-pair seam contact sheets and strip statistics at
            truth poses (the ceiling any matcher could exploit).
    posegraph  measure every mutual-adjacency edge (multi-hypothesis) and
            jointly solve all page poses with robust priors; see
            edge_join_graph.py. `report --chain-source chain,posegraph`
            compares the hybrid (chain first, graph fallback).

Usage:
    uv run python -m mapsnap.edge_join_experiment stats data/washington_dc_1916_vol_2
    uv run python -m mapsnap.edge_join_experiment infer data/washington_dc_1916_vol_2
    uv run python -m mapsnap.edge_join_experiment sanity data/washington_dc_1916_vol_2
"""

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mapsnap import edge_join
from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    extract_gcps,
    fit_transform,
    haversine_ft,
    north_angle,
    sample_grid,
)
from mapsnap.keymap.align_page_region import (
    angle_wrap,
    image_neighbor_directions,
    load_adjacency,
)
from mapsnap.keymap.fit_keymap import page_number, project
from mapsnap.score_adjacency import truth_adjacent_pairs
from mapsnap.utils import jpeg_dimensions, source_id_to_page_key

GEOREF_VARIANTS = [
    "georef",
    "georef-nofit",
    "georef-misscale",
    "georef-1gcp",
    "georef-outlier",
]
METERS_PER_FOOT = 0.3048
TRUTH_ADJACENT_GAP_M = 30.0
ANCHOR_RMSE_FT = 25.0
ALLOWED_RELATIVE_ROTATIONS = (0.0, 90.0, -90.0)
# Printed-side direction gates. Measured on DC: joins <=50ft never fall below
# cos 0.92 (target-side) / 0.95 (anchor-side); wrong locks spread much lower.
DIRECTION_COS_MIN_TARGET = 0.85
DIRECTION_COS_MIN_ANCHOR = 0.90
RELATIVE_ROTATION_TOLERANCE_DEG = 6.0


@dataclass
class TruthFit:
    """A page's truth transform, expressed in local working-jpg pixel space."""

    affine_local: np.ndarray  # 2x3: local px -> (lon, lat)
    gcp_count: int
    transform_type: str


@dataclass
class PageUnit:
    """One base (unsplit) page of a volume with everything the harness needs."""

    stem: str
    number: int
    width: int
    height: int
    fit_state: str  # fitted | nofit | misscale | 1gcp | outlier | split | none
    truth: TruthFit | None
    split_truth: bool  # truth exists only as split items (out of v1 scope)
    gen_affine: np.ndarray | None  # local px -> (lon, lat)
    inlier_intersections: int
    inlier_streets: int
    keymap_centers: list[tuple[float, float]]
    keymap_radius_m: float
    keymap_regions: list[list[list[float]]] | None = None
    anchor_truth: bool = False
    anchor_free: bool = False
    rmse_ft: float | None = None  # generated-vs-truth RMSE


def scale_affine_to_local(
    affine_full: np.ndarray, source_width: float, local_width: float
) -> np.ndarray:
    """Rescale a full-resolution-pixel affine to local working-jpg pixels."""
    s = source_width / local_width
    return affine_full @ np.array([[s, 0, 0], [0, s, 0], [0, 0, 1.0]])


def apply_affine(affine: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Apply a 2x3 affine to one point."""
    return (
        affine[0, 0] * x + affine[0, 1] * y + affine[0, 2],
        affine[1, 0] * x + affine[1, 1] * y + affine[1, 2],
    )


def grid_rmse_ft_between(
    affine_a: np.ndarray, affine_b: np.ndarray, width: int, height: int
) -> float:
    """RMSE (ft) between two local-px affines over the standard 7x7 grid."""
    errors = []
    for x, y in sample_grid(width, height):
        lon_a, lat_a = apply_affine(affine_a, x, y)
        lon_b, lat_b = apply_affine(affine_b, x, y)
        errors.append(haversine_ft(lat_a, lon_a, lat_b, lon_b) ** 2)
    return math.sqrt(sum(errors) / len(errors))


def affine_scale_m_per_px(affine: np.ndarray) -> float:
    """Mean metres per pixel of a local-px -> lon/lat affine."""
    lat = affine[1, 2]
    kx = 111_320.0 * math.cos(math.radians(lat))
    ky = 110_540.0
    u = math.hypot(affine[0, 0] * kx, affine[1, 0] * ky)
    v = math.hypot(affine[0, 1] * kx, affine[1, 1] * ky)
    return (u + v) / 2


def load_truth_units(volume: Path) -> tuple[dict[str, dict], set[str]]:
    """(unsplit truth items by page key, page keys with split-only truth).

    Skeleton sheets ('s' suffix) with a full-color truth counterpart are
    dropped, mirroring make_iiif_georef and compare: they map the same ground
    as the full-color page.
    """
    data = json.loads((volume / "main.iiif.json").read_text())
    unsplit: dict[str, dict] = {}
    split_parents: set[str] = set()
    for item in data.get("items", []):
        key = source_id_to_page_key(
            item.get("target", {}).get("source", {}).get("id"), item.get("label", "")
        )
        if "__" in key:
            split_parents.add(key.split("__")[0])
        else:
            unsplit[key] = item
    for key in [k for k in unsplit if k.endswith("s") and k[:-1] in unsplit]:
        del unsplit[key]
    return unsplit, split_parents


def page_fit_state(volume: Path, stem: str) -> tuple[str, dict | None]:
    """(fit state, georef-variant JSON) for a base page stem."""
    for variant in GEOREF_VARIANTS:
        path = volume / f"{stem}.{variant}.json"
        if path.exists():
            state = variant.removeprefix("georef-") if "-" in variant else "fitted"
            return state, json.loads(path.read_text())
    # Split pieces (p239__1.georef*.json) mean the base page was split.
    if list(volume.glob(f"{stem}__*.georef*.json")):
        return "split", None
    return "none", None


def load_page_units(volume: Path) -> list[PageUnit]:
    """All base pages of the volume, with truth/generated fits attached."""
    from mapsnap.road_model import page_world_affine

    truth_by_key, split_truth_parents = load_truth_units(volume)
    units: list[PageUnit] = []
    for jpg in sorted(volume.glob("p*.jpg")):
        stem = jpg.stem
        if "__" in stem:
            continue
        number = page_number(stem)
        if number is None:
            continue
        width, height = jpeg_dimensions(jpg)
        state, georef = page_fit_state(volume, stem)

        truth: TruthFit | None = None
        truth_item = truth_by_key.get(stem)
        if truth_item is not None:
            source = truth_item["target"]["source"]
            affine_full = fit_transform(
                extract_gcps(truth_item), annotation_transform_type(truth_item)
            )
            truth = TruthFit(
                affine_local=scale_affine_to_local(affine_full, source["width"], width),
                gcp_count=len(extract_gcps(truth_item)),
                transform_type=annotation_transform_type(truth_item),
            )

        gen_affine = None
        inlier_int = inlier_str = 0
        keymap_centers: list[tuple[float, float]] = []
        keymap_radius = 0.0
        keymap_regions = None
        if georef is not None:
            if state == "fitted":
                gen_affine = page_world_affine(georef)
            inlier_int = sum(
                1 for i in georef.get("intersections", []) if i.get("inlier")
            )
            inlier_str = sum(1 for s in georef.get("streets", []) if s.get("inlier"))
            keymap = georef.get("keymap") or {}
            keymap_centers = [tuple(c) for c in keymap.get("centers", [])]
            keymap_radius = float(keymap.get("radius_m") or 0.0)
            keymap_regions = keymap.get("regions") or None

        unit = PageUnit(
            stem=stem,
            number=number,
            width=width,
            height=height,
            fit_state=state,
            truth=truth,
            split_truth=stem in split_truth_parents,
            gen_affine=gen_affine,
            inlier_intersections=inlier_int,
            inlier_streets=inlier_str,
            keymap_centers=keymap_centers,
            keymap_radius_m=keymap_radius,
            keymap_regions=keymap_regions,
        )
        if truth is not None and gen_affine is not None:
            unit.rmse_ft = grid_rmse_ft_between(
                truth.affine_local, gen_affine, width, height
            )
            unit.anchor_truth = unit.rmse_ft <= ANCHOR_RMSE_FT
        unit.anchor_free = state == "fitted" and inlier_int >= 3
        units.append(unit)
    return units


def page_rect_metres(
    unit: PageUnit, origin: tuple[float, float]
) -> list[tuple[float, float]]:
    """The page's full-rectangle truth footprint in the local metre frame."""
    assert unit.truth is not None
    corners_px = [(0, 0), (unit.width, 0), (unit.width, unit.height), (0, unit.height)]
    ring = []
    for x, y in corners_px:
        lon, lat = apply_affine(unit.truth.affine_local, x, y)
        ring.append(project(lon, lat, origin[0], origin[1]))
    return ring


def detected_pairs(volume: Path) -> set[frozenset[int]]:
    """Mutual adjacency edges as unordered page-number pairs, or empty if absent."""
    path = volume / "adjacency.json"
    if not path.exists():
        return set()
    doc = json.loads(path.read_text())
    pairs: set[frozenset[int]] = set()
    for a, b in doc.get("adjacency", []):
        na, nb = page_number(a), page_number(b)
        if na is not None and nb is not None and na != nb:
            pairs.add(frozenset((na, nb)))
    return pairs


def truth_pairs_by_number(volume: Path) -> set[frozenset[int]]:
    """Truth-derived adjacency: page footprints within TRUTH_ADJACENT_GAP_M."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from mapsnap.compare_iiif_georef import truth_polygons_by_page

    by_page = truth_polygons_by_page(volume / "main.iiif.json")
    all_pts = [pt for rings in by_page.values() for ring in rings for pt in ring]
    lon0 = sum(p[0] for p in all_pts) / len(all_pts)
    lat0 = sum(p[1] for p in all_pts) / len(all_pts)
    shapes = {}
    for number, rings in by_page.items():
        polys = []
        for ring in rings:
            pts = [project(lon, lat, lon0, lat0) for lon, lat in ring]
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                polys.append(poly)
        if polys:
            shapes[number] = unary_union(polys)
    return truth_adjacent_pairs(shapes, TRUTH_ADJACENT_GAP_M)


def pair_overlap_stats(a: PageUnit, b: PageUnit) -> dict | None:
    """Truth-pose overlap geometry for a pair of unsplit truth pages."""
    from shapely.geometry import Polygon

    assert a.truth is not None and b.truth is not None
    lon_a, lat_a = apply_affine(a.truth.affine_local, a.width / 2, a.height / 2)
    lon_b, lat_b = apply_affine(b.truth.affine_local, b.width / 2, b.height / 2)
    origin = ((lon_a + lon_b) / 2, (lat_a + lat_b) / 2)
    poly_a = Polygon(page_rect_metres(a, origin))
    poly_b = Polygon(page_rect_metres(b, origin))
    if not (poly_a.is_valid and poly_b.is_valid):
        return None
    inter = poly_a.intersection(poly_b)
    if inter.is_empty:
        strip_w = strip_len = area = 0.0
    else:
        area = inter.area
        rect = inter.minimum_rotated_rectangle
        exterior = getattr(rect, "exterior", None)
        if exterior is None:
            strip_w = strip_len = 0.0
        else:
            coords = list(exterior.coords)[:4]
            side1 = math.dist(coords[0], coords[1])
            side2 = math.dist(coords[1], coords[2])
            strip_w, strip_len = sorted([side1, side2])
    delta_theta = angle_wrap(
        north_angle(a.truth.affine_local) - north_angle(b.truth.affine_local)
    )
    gap = poly_a.distance(poly_b)
    return {
        "a": a.stem,
        "b": b.stem,
        "delta_theta_deg": round(delta_theta, 2),
        "overlap_area_m2": round(area, 1),
        "strip_width_m": round(strip_w, 1),
        "strip_length_m": round(strip_len, 1),
        "rect_gap_m": round(gap, 1),
    }


def coverage_report(
    units: list[PageUnit],
    pairs: set[frozenset[int]],
    truth_numbers: set[int],
) -> dict:
    """1-hop and multi-hop reachability of non-anchor truth pages from anchors."""
    anchors = {u.number for u in units if u.anchor_truth}
    unit_numbers = {u.number for u in units}
    # Targets: pages with truth (incl. split-only parents) but no truth-anchor fit.
    targets = truth_numbers - anchors
    neighbor_map: dict[int, set[int]] = {}
    for pair in pairs:
        x, y = tuple(pair)
        neighbor_map.setdefault(x, set()).add(y)
        neighbor_map.setdefault(y, set()).add(x)
    one_hop = {t for t in targets if neighbor_map.get(t, set()) & anchors}
    # BFS from anchors across targets.
    reached = set(anchors)
    frontier = set(anchors)
    hops = {}
    hop = 0
    while frontier:
        hop += 1
        nxt = set()
        for n in frontier:
            for m in neighbor_map.get(n, set()):
                if m not in reached:
                    reached.add(m)
                    hops[m] = hop
                    nxt.add(m)
        frontier = nxt
    reachable_targets = {t for t in targets if t in hops}
    return {
        "n_anchors": len(anchors),
        "n_targets": len(targets),
        "targets_one_hop": len(one_hop),
        "targets_reachable": len(reachable_targets),
        "hop_histogram": {
            h: sum(1 for t in reachable_targets if hops[t] == h)
            for h in sorted(set(hops[t] for t in reachable_targets))
        }
        if reachable_targets
        else {},
        "targets_without_units": len(targets - unit_numbers),
    }


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile of a non-empty list."""
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo = math.floor(pos)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (pos - lo)


def cmd_stats(volume: Path) -> None:
    """Phase 1: anchors, pairs, rotations, overlap strips, coverage ceiling."""
    units = load_page_units(volume)
    by_number = {u.number: u for u in units}
    truth_by_key, _ = load_truth_units(volume)
    truth_numbers = set()
    data = json.loads((volume / "main.iiif.json").read_text())
    for item in data.get("items", []):
        key = source_id_to_page_key(
            item.get("target", {}).get("source", {}).get("id"), item.get("label", "")
        )
        n = page_number(key.split("__")[0])
        if n is not None:
            truth_numbers.add(n)

    n_fitted = sum(1 for u in units if u.fit_state == "fitted")
    anchors = [u for u in units if u.anchor_truth]
    print(
        f"== {volume.name}: {len(units)} base pages, {len(truth_by_key)} unsplit truth items =="
    )
    print(
        f"fit states: fitted={n_fitted} "
        + " ".join(
            f"{s}={sum(1 for u in units if u.fit_state == s)}"
            for s in ["nofit", "misscale", "1gcp", "outlier", "split", "none"]
        )
    )
    print(f"truth-based anchors (RMSE<=25ft, unsplit): {len(anchors)}")

    # Truth-free anchor rule confusion matrix.
    for min_int in [2, 3, 4, 5]:
        free = {
            u.number
            for u in units
            if u.fit_state == "fitted" and u.inlier_intersections >= min_int
        }
        true_set = {u.number for u in anchors}
        measurable = {u.number for u in units if u.rmse_ft is not None}
        tp = len(free & true_set)
        fp = len((free & measurable) - true_set)
        fn = len(true_set - free)
        print(
            f"  truth-free rule n_int>={min_int}: {len(free)} selected;"
            f" vs truth-anchors: tp={tp} fp={fp} fn={fn}"
            f" precision={tp / max(1, tp + fp):.2f}"
        )

    scales = [affine_scale_m_per_px(u.truth.affine_local) for u in anchors if u.truth]
    if scales:
        med = statistics.median(scales)
        spread = [s / med - 1 for s in scales]
        print(
            f"volume scale: median {med:.4f} m/px;"
            f" spread p10 {percentile(spread, 0.1):+.1%} p90 {percentile(spread, 0.9):+.1%}"
        )

    rmses = sorted(u.rmse_ft for u in units if u.rmse_ft is not None)
    if rmses:
        print(
            f"generated-vs-truth RMSE (n={len(rmses)}):"
            f" median {percentile(rmses, 0.5):.0f}ft p90 {percentile(rmses, 0.9):.0f}ft"
        )

    detected = detected_pairs(volume)
    truthp = truth_pairs_by_number(volume)
    both_truth = {
        p for p in truthp if all(by_number.get(n) and by_number[n].truth for n in p)
    }
    print(
        f"\npairs: detected={len(detected) or '(no adjacency.json)'} truth-derived={len(truthp)}"
    )
    if detected:
        known = {p for p in detected if p <= truth_numbers}
        agree = known & truthp
        print(
            f"  detected edges with both-truth: {len(known)}; consistent with truth: {len(agree)}"
            f" ({len(agree) / max(1, len(known)):.0%})"
        )

    # Per-pair truth stats (unsplit pairs with truth on both sides).
    stats = []
    for pair in sorted(both_truth, key=sorted):
        x, y = sorted(pair)
        s = pair_overlap_stats(by_number[x], by_number[y])
        if s:
            stats.append(s)
    if stats:
        thetas = [abs(s["delta_theta_deg"]) for s in stats]
        widths = [s["strip_width_m"] for s in stats if s["overlap_area_m2"] > 0]
        print(f"\nper-pair truth stats over {len(stats)} unsplit truth pairs:")
        print(
            f"  |delta theta|: median {percentile(thetas, 0.5):.1f} deg,"
            f" p90 {percentile(thetas, 0.9):.1f}, max {max(thetas):.1f}"
        )
        n_overlap = sum(1 for s in stats if s["overlap_area_m2"] > 0)
        print(f"  full-rect overlap: {n_overlap}/{len(stats)} pairs overlap")
        if widths:
            print(
                f"  strip width (m): median {percentile(widths, 0.5):.0f},"
                f" p10 {percentile(widths, 0.1):.0f}, p90 {percentile(widths, 0.9):.0f}"
            )

    for name, pairs in [("detected", detected), ("truth-graph", truthp)]:
        if not pairs:
            continue
        cov = coverage_report(units, pairs, truth_numbers)
        print(f"\ncoverage via {name} graph: {json.dumps(cov)}")

    out_dir = volume / "artifacts" / "edge_join"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pairs.json").write_text(json.dumps(stats, indent=1))
    print(f"\nwrote {out_dir / 'pairs.json'}")


def cmd_infer(volume: Path) -> None:
    """Phase 2: cache P(road) maps for every base page as uint8 PNGs."""
    import torch  # noqa: F401  (import check before loading model)

    from mapsnap.keymap.number_model import select_device
    from mapsnap.road_model import load_model, predict_page

    out_dir = volume / "artifacts" / "edge_join" / "roadprob"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    model = load_model(Path("models/road_unet.pt"), device)
    jpgs = [p for p in sorted(volume.glob("p*.jpg")) if "__" not in p.stem]
    done = 0
    for jpg in jpgs:
        out = out_dir / f"{jpg.stem}.png"
        if out.exists():
            continue
        gray = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skipping unreadable {jpg}", file=sys.stderr)
            continue
        prob = predict_page(model, gray, device)
        cv2.imwrite(str(out), (prob * 255).round().astype(np.uint8))
        done += 1
        if done % 20 == 0:
            print(f"  {done} pages inferred…", file=sys.stderr)
    print(f"inferred {done} new pages; cache at {out_dir} ({len(jpgs)} total)")


def load_prob(volume: Path, stem: str) -> np.ndarray | None:
    """A cached P(road) map in [0,1], or None."""
    path = volume / "artifacts" / "edge_join" / "roadprob" / f"{stem}.png"
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if raw is None else raw.astype(np.float32) / 255.0


def render_to_frame(
    prob: np.ndarray,
    affine_local: np.ndarray,
    origin: tuple[float, float],
    frame_shape: tuple[int, int],
    res_m: float,
    frame_min: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a page raster into the local metre frame (y-down = north-flipped).

    Returns (warped, validity mask). The frame maps row = (y_max - y)/res,
    col = (x - x_min)/res where (x, y) are metres about origin.
    """
    lon0, lat0 = origin
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    # page px -> metres: p -> K @ (A(p) - origin), K = diag(kx, ky)
    a = affine_local
    m = np.array(
        [
            [a[0, 0] * kx, a[0, 1] * kx, (a[0, 2] - lon0) * kx],
            [a[1, 0] * ky, a[1, 1] * ky, (a[1, 2] - lat0) * ky],
        ]
    )
    # metres -> frame px (y flip): col = (x - xmin)/res, row = (ymax - y)/res
    x_min, y_max = frame_min[0], frame_min[1]
    to_frame = np.array(
        [[1 / res_m, 0, -x_min / res_m], [0, -1 / res_m, y_max / res_m]]
    )
    full = to_frame @ np.vstack([m, [0, 0, 1]])
    warped = cv2.warpAffine(prob, full, (frame_shape[1], frame_shape[0]))
    valid = cv2.warpAffine(np.ones_like(prob), full, (frame_shape[1], frame_shape[0]))
    return warped, valid > 0.5


def cmd_sanity(volume: Path, limit: int | None) -> None:
    """Phase 2: seam contact sheets + strip stats at truth poses."""
    from mapsnap.road_model import road_mask, road_skeleton, skeleton_junctions

    units = {u.number: u for u in load_page_units(volume)}
    truthp = truth_pairs_by_number(volume)
    pairs = sorted(
        (
            tuple(sorted(p))
            for p in truthp
            if all(units.get(n) and units[n].truth for n in p)
        ),
    )
    out_dir = volume / "artifacts" / "edge_join" / "contact"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = 1.0  # m/px
    records = []
    for x, y in pairs[: limit or len(pairs)]:
        ua, ub = units[x], units[y]
        prob_a, prob_b = load_prob(volume, ua.stem), load_prob(volume, ub.stem)
        if prob_a is None or prob_b is None:
            continue
        assert ua.truth and ub.truth
        lon_a, lat_a = apply_affine(ua.truth.affine_local, ua.width / 2, ua.height / 2)
        lon_b, lat_b = apply_affine(ub.truth.affine_local, ub.width / 2, ub.height / 2)
        origin = ((lon_a + lon_b) / 2, (lat_a + lat_b) / 2)
        rect_a = page_rect_metres(ua, origin)
        rect_b = page_rect_metres(ub, origin)
        xs = [p[0] for p in rect_a + rect_b]
        ys = [p[1] for p in rect_a + rect_b]
        frame_shape = (
            int((max(ys) - min(ys)) / res) + 1,
            int((max(xs) - min(xs)) / res) + 1,
        )
        if frame_shape[0] * frame_shape[1] > 4e7:
            continue
        frame_min = (min(xs), max(ys))
        wa, va = render_to_frame(
            prob_a, ua.truth.affine_local, origin, frame_shape, res, frame_min
        )
        wb, vb = render_to_frame(
            prob_b, ub.truth.affine_local, origin, frame_shape, res, frame_min
        )
        both = va & vb
        if not both.any():
            records.append({"a": ua.stem, "b": ub.stem, "overlap_px": 0})
            continue
        mask_a = road_mask(wa, min_area=500) & both
        mask_b = road_mask(wb, min_area=500) & both
        union = (mask_a | mask_b).sum()
        iou = float((mask_a & mask_b).sum() / union) if union else 0.0
        skel = road_skeleton(mask_a & mask_b)
        junctions = skeleton_junctions(road_skeleton(mask_a | mask_b))
        records.append(
            {
                "a": ua.stem,
                "b": ub.stem,
                "overlap_px": int(both.sum()),
                "strip_iou": round(iou, 3),
                "mean_p_a": round(float(wa[both].mean()), 3),
                "mean_p_b": round(float(wb[both].mean()), 3),
                "strip_skeleton_px": int(skel.sum()),
                "strip_junctions": int(len(junctions)),
            }
        )
        # Contact sheet: red = A, green = B, strip region brightened.
        vis = np.zeros((*frame_shape, 3), np.uint8)
        vis[..., 2] = (wa * 255).astype(np.uint8)
        vis[..., 1] = (wb * 255).astype(np.uint8)
        vis[~both] //= 2
        rows, cols = np.where(both)
        r0, r1 = max(rows.min() - 100, 0), min(rows.max() + 100, frame_shape[0])
        c0, c1 = max(cols.min() - 100, 0), min(cols.max() + 100, frame_shape[1])
        cv2.imwrite(str(out_dir / f"{ua.stem}_{ub.stem}.png"), vis[r0:r1, c0:c1])

    (out_dir.parent / "seam_stats.json").write_text(json.dumps(records, indent=1))
    with_overlap = [r for r in records if r.get("overlap_px", 0) > 0]
    ious = [r["strip_iou"] for r in with_overlap]
    juncs = [r["strip_junctions"] for r in with_overlap]
    print(f"seam stats over {len(records)} pairs ({len(with_overlap)} with overlap):")
    if ious:
        print(
            f"  strip IoU at truth pose: median {percentile(ious, 0.5):.2f},"
            f" p10 {percentile(ious, 0.1):.2f}, p90 {percentile(ious, 0.9):.2f}"
        )
        print(
            f"  strip junctions: median {percentile([float(j) for j in juncs], 0.5):.0f};"
            f" pairs with 0-1 junctions: {sum(1 for j in juncs if j <= 1)}/{len(juncs)}"
        )
    print(f"contact sheets in {out_dir}")


@dataclass
class FrameSpec:
    """A pair-local raster frame: equirectangular metres, north-up rows.

    col = (x - x_min) / res, row = (y_max - y) / res, where (x, y) are metres
    east/north of `origin`. Page images map into the raster without reflection.
    """

    origin: tuple[float, float]  # lon, lat
    x_min: float
    y_max: float
    res_m: float
    shape: tuple[int, int]  # rows, cols

    def metre_scales(self) -> tuple[float, float]:
        kx = 111_320.0 * math.cos(math.radians(self.origin[1]))
        return kx, 110_540.0

    def lonlat_to_raster(self, lon: float, lat: float) -> tuple[float, float]:
        kx, ky = self.metre_scales()
        x = (lon - self.origin[0]) * kx
        y = (lat - self.origin[1]) * ky
        return (x - self.x_min) / self.res_m, (self.y_max - y) / self.res_m

    def page_to_raster_affine(self, affine_local: np.ndarray) -> np.ndarray:
        """2x3 mapping page px -> raster px given a page px -> lon/lat affine."""
        kx, ky = self.metre_scales()
        a = affine_local
        metres = np.array(
            [
                [a[0, 0] * kx, a[0, 1] * kx, (a[0, 2] - self.origin[0]) * kx],
                [a[1, 0] * ky, a[1, 1] * ky, (a[1, 2] - self.origin[1]) * ky],
            ]
        )
        to_frame = np.array(
            [
                [1 / self.res_m, 0, -self.x_min / self.res_m],
                [0, -1 / self.res_m, self.y_max / self.res_m],
            ]
        )
        return edge_join.compose(to_frame, metres)

    def raster_pose_to_world_affine(self, pose: np.ndarray) -> np.ndarray:
        """2x3 mapping page px -> (lon, lat) given a page px -> raster px pose."""
        kx, ky = self.metre_scales()
        from_raster = np.array(
            [
                [self.res_m / kx, 0, self.origin[0] + self.x_min / kx],
                [0, -self.res_m / ky, self.origin[1] + self.y_max / ky],
            ]
        )
        return edge_join.compose(from_raster, pose)


def build_frame(
    anchor: PageUnit,
    anchor_affine: np.ndarray,
    init_lonlat: tuple[float, float],
    reach_m: float,
    res_m: float,
) -> FrameSpec:
    """A frame covering the anchor page plus a search disc around the init."""
    lon0, lat0 = apply_affine(anchor_affine, anchor.width / 2, anchor.height / 2)
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    xs, ys = [], []
    for u, v in [
        (0, 0),
        (anchor.width, 0),
        (anchor.width, anchor.height),
        (0, anchor.height),
    ]:
        lon, lat = apply_affine(anchor_affine, u, v)
        xs.append((lon - lon0) * kx)
        ys.append((lat - lat0) * ky)
    ix = (init_lonlat[0] - lon0) * kx
    iy = (init_lonlat[1] - lat0) * ky
    xs += [ix - reach_m, ix + reach_m]
    ys += [iy - reach_m, iy + reach_m]
    pad = 20.0
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad
    shape = (int((y_max - y_min) / res_m) + 1, int((x_max - x_min) / res_m) + 1)
    return FrameSpec((lon0, lat0), x_min, y_max, res_m, shape)


def run_join(
    volume: Path,
    anchor: PageUnit,
    target: PageUnit,
    anchor_affine: np.ndarray,
    init_centers: list[tuple[float, float]],
    radius_m: float,
    scale_m_per_px: float,
    params: "edge_join.MatchParams",
    expected_direction: tuple[float, float] | None = None,
    anchor_side_direction: tuple[float, float] | None = None,
    containment_regions: list[list[list[float]]] | None = None,
    exclusion_footprints: list[list[list[float]]] | None = None,
    allowed_rotations: tuple[float, ...] = ALLOWED_RELATIVE_ROTATIONS,
    debug_candidates: list | None = None,
) -> dict | None:
    """One anchor->target join attempt; returns a diagnostics record.

    All candidates from every init center compete; the verification-best wins.
    The record includes truth RMSE for the winner and for the best-possible
    candidate (to separate ranking failures from search failures).
    """
    prob_anchor = load_prob(volume, anchor.stem)
    prob_target = load_prob(volume, target.stem)
    if prob_anchor is None or prob_target is None or not init_centers:
        return None
    from mapsnap.road_model import road_mask, road_skeleton

    diag_m = math.hypot(target.width, target.height) * scale_m_per_px
    record: dict = {"anchor": anchor.stem, "target": target.stem}
    all_candidates: list[
        tuple[edge_join.JoinCandidate, FrameSpec, tuple[float, float]]
    ] = []
    direction_mode = expected_direction is not None
    plausibility_centers = list(init_centers)
    if direction_mode:
        # The printed-side prediction anchors the search next to the anchor
        # itself; keymap init centers remain as an absolute plausibility gate
        # (they break the 180-degree flip-and-swap-sides symmetry that the
        # content-relative printed-side check cannot).
        init_centers = [
            apply_affine(anchor_affine, anchor.width / 2, anchor.height / 2)
        ]
    for init in init_centers:
        reach = (diag_m + 280.0) if direction_mode else (radius_m + diag_m / 2)
        frame = build_frame(anchor, anchor_affine, init, reach, params.resolution_m)
        if frame.shape[0] * frame.shape[1] > 6e6:
            record["status"] = "frame_too_large"
            continue
        pose_anchor = frame.page_to_raster_affine(anchor_affine)
        anchor_center_raster = tuple(
            pose_anchor @ np.array([anchor.width / 2, anchor.height / 2, 1.0])
        )
        fixed, fixed_valid = edge_join.warp_page(prob_anchor, pose_anchor, frame.shape)
        search_center = frame.lonlat_to_raster(*init)
        raster_scale = scale_m_per_px / frame.res_m
        sigma_px = max(params.blur_sigma_m / frame.res_m, 0.5)
        fixed_blur = cv2.GaussianBlur(fixed * fixed_valid, (0, 0), sigma_px)
        thetas = edge_join.rotation_candidates(
            fixed, prob_target, params.jitter_deg, fixed_valid=fixed_valid
        )
        theta_anchor = math.degrees(math.atan2(-pose_anchor[1, 0], pose_anchor[0, 0]))
        for relative in allowed_rotations:
            seed = angle_wrap(theta_anchor + relative)
            if all(abs(angle_wrap(seed - t)) > 1.0 for t in thetas):
                thetas.append(seed)
        anchor_corner_offsets = (
            np.array(
                [
                    [0, 0, 1],
                    [anchor.width, 0, 1],
                    [anchor.width, anchor.height, 1],
                    [0, anchor.height, 1],
                ],
                dtype=float,
            )
            @ pose_anchor.T
            - anchor_center_raster
        )
        candidates = []
        for theta in thetas:
            if expected_direction is not None:
                # Predict the target's center: the printed number's side gives
                # the direction, the sheets' support extents give the distance
                # (minus the shared strip). This shrinks the search from the
                # keymap disc to a ~250 m window, before any keymap input.
                lin = cv2.getRotationMatrix2D((0.0, 0.0), theta, raster_scale)[:, :2]
                d = lin @ np.array(expected_direction)
                d /= max(float(np.linalg.norm(d)), 1e-9)
                support_anchor = float(np.abs(anchor_corner_offsets @ d).max())
                half = np.array([target.width / 2, target.height / 2])
                target_offsets = (
                    np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * half
                ) @ lin.T
                support_target = float(np.abs(target_offsets @ d).max())
                overlap_px = 70.0 / frame.res_m
                predicted = np.array(anchor_center_raster) - d * (
                    support_anchor + support_target - overlap_px
                )
                seed_center = (float(predicted[0]), float(predicted[1]))
                seed_radius = 250.0 / frame.res_m
            else:
                seed_center = search_center
                seed_radius = radius_m / frame.res_m
            for candidate in edge_join.match_at_rotation(
                fixed_blur,
                fixed_valid,
                prob_target,
                raster_scale,
                theta,
                params,
                seed_center,
                seed_radius,
            ):
                candidate.diagnostics["theta_anchor"] = theta_anchor
                candidates.append(candidate)
        if not candidates:
            continue
        mask = road_mask(fixed, min_area=500) & fixed_valid
        skeleton = road_skeleton(mask)
        distance = (
            cv2.distanceTransform((~skeleton).astype(np.uint8), cv2.DIST_L2, 5).astype(
                np.float32
            )
            * frame.res_m
        )
        distance = np.minimum(distance, 30.0)
        points = edge_join.skeleton_points(
            prob_target, params.mask_threshold, params.mask_min_area
        )
        # Score only skeleton points near the anchor (the shared strip).
        dilate_px = max(int(30.0 / frame.res_m), 1)
        near_anchor = cv2.dilate(
            fixed_valid.astype(np.uint8), np.ones((dilate_px, dilate_px), np.uint8)
        ).astype(bool)
        ranked = edge_join.refine_and_rank(
            candidates,
            distance,
            points,
            fixed_valid=fixed_valid,
            page_shape=prob_target.shape[:2],
            max_overlap_frac=params.max_overlap_frac,
            region=near_anchor,
            fixed_prob=fixed,
            target_prob=prob_target,
            fine_sigma_px=params.fine_sigma_m / frame.res_m,
            solve_scale=params.solve_scale,
        )
        all_candidates.extend((c, frame, anchor_center_raster) for c in ranked)

    if not all_candidates:
        record.setdefault("status", "no_candidates")
        return record
    # The printed-neighbor-side prior: a candidate placing the anchor on the
    # wrong side of the target (e.g. a 180-degree flip) is disqualified.
    if expected_direction is not None:
        for candidate, _, anchor_center in all_candidates:
            cosine = edge_join.direction_cosine(
                candidate.pose,
                prob_target.shape[:2],
                anchor_center,
                expected_direction,
            )
            candidate.diagnostics["direction_cos"] = round(cosine, 3)
            if cosine < DIRECTION_COS_MIN_TARGET:
                candidate.plausible = False
    # Relative-rotation prior: adjacent sheets in this volume differ by
    # 0 or +/-90 degrees (measured from truth: |dtheta| p90=90, max 92.8),
    # so near-180 relative rotations are implausible.
    for candidate, frame, _ in all_candidates:
        relative = angle_wrap(
            candidate.theta_deg
            - candidate.diagnostics.get("theta_anchor", candidate.theta_deg)
        )
        candidate.diagnostics["relative_rotation"] = round(relative, 1)
        if (
            min(abs(angle_wrap(relative - allowed)) for allowed in allowed_rotations)
            > RELATIVE_ROTATION_TOLERANCE_DEG
        ):
            candidate.plausible = False
    # Reciprocal printed-side prior: the ANCHOR's own claim of the target's
    # number pins the world direction to the target (the anchor's pose is
    # known), independent of the target's rotation. This breaks the joint
    # rotate-and-swing symmetry that the target-side check alone cannot: a
    # 90-degree-rotated target moved 90 degrees around the anchor keeps the
    # target-side cosine near 1, but lands on the wrong side of the anchor.
    if anchor_side_direction is not None:
        for candidate, frame, anchor_center in all_candidates:
            linear = frame.page_to_raster_affine(anchor_affine)[:, :2]
            implied = linear @ np.array(anchor_side_direction)
            center = candidate.pose @ np.array(
                [target.width / 2, target.height / 2, 1.0]
            )
            actual = np.array(center) - np.array(anchor_center)
            implied_norm = float(np.linalg.norm(implied))
            actual_norm = float(np.linalg.norm(actual))
            if implied_norm < 1e-9 or actual_norm < 1e-9:
                continue
            cosine = float(implied @ actual / (implied_norm * actual_norm))
            candidate.diagnostics["anchor_side_cos"] = round(cosine, 3)
            if cosine < DIRECTION_COS_MIN_ANCHOR:
                candidate.plausible = False
    # Keymap-region containment: the placed page footprint should mostly
    # fall inside the key map's segmented region for this page (schematic, so
    # the threshold is loose). Kills sideways and far-slid placements.
    if containment_regions:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        lon_r, lat_r = containment_regions[0][0][0], containment_regions[0][0][1]
        kxr = 111_320.0 * math.cos(math.radians(lat_r))
        kyr = 110_540.0

        def ring_metres(ring: list[list[float]]) -> list[tuple[float, float]]:
            return [((lon - lon_r) * kxr, (lat - lat_r) * kyr) for lon, lat in ring]

        region_poly = unary_union(
            [Polygon(ring_metres(r)).buffer(0) for r in containment_regions]
        ).buffer(60.0)
        for candidate, frame, _ in all_candidates:
            world = frame.raster_pose_to_world_affine(candidate.pose)
            corners = [
                apply_affine(world, x, y)
                for x, y in [
                    (0, 0),
                    (target.width, 0),
                    (target.width, target.height),
                    (0, target.height),
                ]
            ]
            footprint = Polygon(ring_metres([list(c) for c in corners]))
            if footprint.area > 0:
                contained = footprint.intersection(region_poly).area / footprint.area
                candidate.diagnostics["region_containment"] = round(contained, 3)
                if contained < 0.35:
                    candidate.plausible = False
    # Absolute-position gate: the target's center must land near a keymap
    # candidate center (or the study's init). Kills opposite-side placements.
    if plausibility_centers:
        kx = 111_320.0 * math.cos(math.radians(plausibility_centers[0][1]))
        ky = 110_540.0
        for candidate, frame, _ in all_candidates:
            world = frame.raster_pose_to_world_affine(candidate.pose)
            lon_c, lat_c = apply_affine(world, target.width / 2, target.height / 2)
            nearest = min(
                math.hypot((lon_c - lon) * kx, (lat_c - lat) * ky)
                for lon, lat in plausibility_centers
            )
            candidate.diagnostics["center_dist_m"] = round(nearest, 1)
            if nearest > radius_m:
                candidate.plausible = False
    # Posed-page exclusion: sheets only ever share a margin strip, so a
    # candidate overlapping any already-placed page by more than a strip is
    # physically impossible (e.g. a one-block slide onto a posed neighbor).
    if exclusion_footprints:
        lon_e, lat_e = exclusion_footprints[0][0]
        kxe = 111_320.0 * math.cos(math.radians(lat_e))
        kye = 110_540.0

        def ring_metres_e(ring: list[list[float]]) -> list[tuple[float, float]]:
            return [((lon - lon_e) * kxe, (lat - lat_e) * kye) for lon, lat in ring]

        from shapely.geometry import Polygon as ShPolygon

        exclusions = [
            ShPolygon(ring_metres_e(ring)).buffer(0) for ring in exclusion_footprints
        ]
        for candidate, frame, _ in all_candidates:
            if not candidate.plausible:
                continue
            world = frame.raster_pose_to_world_affine(candidate.pose)
            corners = [
                list(apply_affine(world, x, y))
                for x, y in [
                    (0, 0),
                    (target.width, 0),
                    (target.width, target.height),
                    (0, target.height),
                ]
            ]
            fp = ShPolygon(ring_metres_e(corners))
            if fp.area <= 0:
                continue
            worst = max(fp.intersection(ex).area / fp.area for ex in exclusions)
            candidate.diagnostics["posed_overlap"] = round(worst, 3)
            if worst > 0.42:
                candidate.plausible = False

    # Rank with a keymap-containment bonus: the slid-one-block lock and the
    # true pose score within a few hundredths on strip agreement alone, but
    # the true pose sits squarely inside the page's keymap region.
    def ranking_score(candidate: "edge_join.JoinCandidate") -> float:
        bonus = 0.3 * candidate.diagnostics.get("region_containment", 0.0)
        return candidate.verification_score() + bonus

    all_candidates.sort(key=lambda cf: -ranking_score(cf[0]))
    if debug_candidates is not None:
        debug_candidates.extend(all_candidates)
    best, best_frame, _ = all_candidates[0]
    world = best_frame.raster_pose_to_world_affine(best.pose)
    record.update(
        status="ok",
        n_candidates=len(all_candidates),
        theta_deg=round(best.theta_deg, 2),
        ncc=round(best.ncc, 3),
        ncc_fine=round(best.ncc_fine, 3),
        scale_adjust=round(best.scale_adjust, 4),
        verification=round(best.verification_score(), 3),
        chamfer_mean_m=round(best.chamfer_mean_m, 2),
        inlier_frac=round(best.inlier_frac, 3),
        n_points=best.n_points,
        overlap_frac=round(best.overlap_frac, 3),
        direction_cos=best.diagnostics.get("direction_cos"),
        anchor_side_cos=best.diagnostics.get("anchor_side_cos"),
        jtj_eig_ratio=round(best.jtj_eig_ratio, 5),
        world_affine=[list(row) for row in world],
    )
    # Top plausible candidates as alternates: on self-similar grids the true
    # pose is often rank 2 behind an aliased slide, and a pose-graph solver
    # can re-pick the alternate that is globally consistent.
    alternates: list[dict] = []
    for candidate, frame, _ in all_candidates:
        if not candidate.plausible:
            continue
        w = frame.raster_pose_to_world_affine(candidate.pose)
        center = apply_affine(w, target.width / 2, target.height / 2)
        kx = 111_320.0 * math.cos(math.radians(center[1]))
        near_existing = any(
            math.hypot((center[0] - c[0]) * kx, (center[1] - c[1]) * 110_540.0) < 15.0
            for c in (a["center"] for a in alternates)
        )
        if near_existing:
            continue
        alternate = {
            "world_affine": [list(row) for row in w],
            "verification": round(candidate.verification_score(), 3),
            "theta_deg": round(candidate.theta_deg, 2),
            "center": list(center),
        }
        if target.truth is not None:
            alternate["rmse_ft"] = round(
                grid_rmse_ft_between(
                    w, target.truth.affine_local, target.width, target.height
                ),
                1,
            )
        alternates.append(alternate)
        if len(alternates) == 5:
            break
    record["alternates"] = alternates
    if target.truth is not None:
        record["rmse_ft"] = round(
            grid_rmse_ft_between(
                world, target.truth.affine_local, target.width, target.height
            ),
            1,
        )
        rmses = []
        for candidate, frame, _ in all_candidates:
            w = frame.raster_pose_to_world_affine(candidate.pose)
            rmses.append(
                grid_rmse_ft_between(
                    w, target.truth.affine_local, target.width, target.height
                )
            )
        best_idx = int(np.argmin(rmses))
        record["best_possible_rmse_ft"] = round(rmses[best_idx], 1)
        record["best_possible_rank"] = best_idx
    return record


def volume_median_scale(units: list[PageUnit]) -> float:
    """Median truth-anchor scale (metres per page pixel)."""
    scales = [
        affine_scale_m_per_px(u.truth.affine_local)
        for u in units
        if u.anchor_truth and u.truth
    ]
    return statistics.median(scales)


def summarize_joins(records: list[dict], label: str) -> None:
    """Print RMSE percentiles and threshold rates for join records."""
    ok = [r for r in records if r.get("status") == "ok" and "rmse_ft" in r]
    print(f"\n== {label}: {len(records)} attempts, {len(ok)} scored ==")
    if not ok:
        return
    rmses = [r["rmse_ft"] for r in ok]
    print(
        f"  winner RMSE ft: median {percentile(rmses, 0.5):.0f},"
        f" p90 {percentile(rmses, 0.9):.0f}, max {max(rmses):.0f}"
    )
    for threshold in [15, 25, 50, 100]:
        n = sum(1 for r in rmses if r <= threshold)
        print(f"    <={threshold}ft: {n}/{len(rmses)} ({n / len(rmses):.0%})")
    possible = [r["best_possible_rmse_ft"] for r in ok]
    n25 = sum(1 for r in possible if r <= 25)
    print(
        f"  best-possible RMSE <=25ft: {n25}/{len(possible)}"
        f" (ranking losses: {n25 - sum(1 for r in rmses if r <= 25)})"
    )
    wrong = [r for r in ok if r["rmse_ft"] > 100 and r["inlier_frac"] > 0.6]
    print(f"  wrong-lock (RMSE>100ft with inliers>0.6): {len(wrong)}/{len(ok)}")


def cmd_perturb(
    volume: Path, limit: int | None, seed: int, draws: int, solve_scale: bool
) -> None:
    """Phase 3a: anchor<->anchor joins from perturbed inits (precision floor)."""
    import time as time_mod

    units = load_page_units(volume)
    by_number = {u.number: u for u in units}
    truthp = truth_pairs_by_number(volume)
    anchor_pairs = []
    for p in sorted(truthp, key=sorted):
        x, y = sorted(p)
        ux, uy = by_number.get(x), by_number.get(y)
        if not (ux and uy and ux.anchor_truth and uy.anchor_truth):
            continue
        overlap = pair_overlap_stats(ux, uy)
        # Corner-adjacent pages share no strip; they are unjoinable by design.
        if overlap is None or overlap["overlap_area_m2"] < 10000:
            continue
        anchor_pairs.append((x, y))
    rng = np.random.default_rng(seed)
    params = edge_join.MatchParams(solve_scale=solve_scale)
    scale = volume_median_scale(units)
    radius = 570.0
    adjacency = load_adjacency(volume)
    out_path = volume / "artifacts" / "edge_join" / "perturb.jsonl"
    records = []
    start = time_mod.time()
    directed_pairs = [(x, y) for x, y in anchor_pairs] + [
        (y, x) for x, y in anchor_pairs
    ]
    for x, y in directed_pairs[: limit or len(directed_pairs)]:
        anchor, target = by_number[x], by_number[y]
        assert anchor.truth and target.truth
        for _ in range(draws):
            # Init: target's truth center displaced uniformly within the disc.
            angle = rng.uniform(0, 2 * math.pi)
            dist = radius * math.sqrt(rng.uniform(0, 1))
            lon_c, lat_c = apply_affine(
                target.truth.affine_local, target.width / 2, target.height / 2
            )
            kx, ky = 111_320.0 * math.cos(math.radians(lat_c)), 110_540.0
            init = (
                lon_c + dist * math.cos(angle) / kx,
                lat_c + dist * math.sin(angle) / ky,
            )
            directions = image_neighbor_directions(adjacency, target.stem)
            expected = (directions.get(anchor.number) or (None, 0.0))[0]
            anchor_dirs = image_neighbor_directions(adjacency, anchor.stem)
            anchor_side = (anchor_dirs.get(target.number) or (None, 0.0))[0]
            rec = run_join(
                volume,
                anchor,
                target,
                anchor.truth.affine_local,
                [init],
                radius,
                scale,
                params,
                expected_direction=expected,
                anchor_side_direction=anchor_side,
            )
            if rec is None:
                continue
            rec["init_offset_m"] = round(dist, 1)
            records.append(rec)
        if len(records) % 20 < draws:
            elapsed = time_mod.time() - start
            print(f"  {len(records)} attempts, {elapsed:.0f}s elapsed", file=sys.stderr)
    out_path.write_text("\n".join(json.dumps(r) for r in records))
    summarize_joins(records, f"perturbation study ({len(anchor_pairs)} anchor pairs)")
    print(f"wrote {out_path}")


def cmd_match(
    volume: Path, limit: int | None, anchor_pose: str, solve_scale: bool
) -> None:
    """Phase 3b: real anchor->target joins over the detected adjacency graph."""
    units = load_page_units(volume)
    by_number = {u.number: u for u in units}
    pairs = detected_pairs(volume)
    params = edge_join.MatchParams(solve_scale=solve_scale)
    scale = volume_median_scale(units)
    adjacency = load_adjacency(volume)
    allowed_rotations = volume_relative_rotations(units, pairs)
    attempts: list[tuple[PageUnit, PageUnit]] = []
    for pair in sorted(pairs, key=sorted):
        for a, t in [tuple(sorted(pair)), tuple(sorted(pair))[::-1]]:
            anchor, target = by_number.get(a), by_number.get(t)
            if not anchor or not target or not anchor.anchor_truth:
                continue
            if target.anchor_truth or target.fit_state == "split":
                continue
            attempts.append((anchor, target))
    records = []
    anchor_footprints: list[list[list[float]]] = []
    for unit in units:
        if unit.anchor_truth:
            pose = (
                unit.truth.affine_local
                if anchor_pose == "truth" and unit.truth
                else unit.gen_affine
            )
            if pose is not None:
                anchor_footprints.append(
                    [
                        list(apply_affine(pose, x, y))
                        for x, y in [
                            (0, 0),
                            (unit.width, 0),
                            (unit.width, unit.height),
                            (0, unit.height),
                        ]
                    ]
                )
    for anchor, target in attempts[: limit or len(attempts)]:
        affine = (
            anchor.truth.affine_local
            if anchor_pose == "truth" and anchor.truth
            else anchor.gen_affine
        )
        if affine is None:
            continue
        inits = target.keymap_centers
        radius = target.keymap_radius_m or 570.0
        if not inits:
            # No keymap hint: search around the anchor itself with a wide disc.
            inits = [apply_affine(affine, anchor.width / 2, anchor.height / 2)]
            radius = 800.0
        directions = image_neighbor_directions(adjacency, target.stem)
        expected = (directions.get(anchor.number) or (None, 0.0))[0]
        anchor_dirs = image_neighbor_directions(adjacency, anchor.stem)
        anchor_side = (anchor_dirs.get(target.number) or (None, 0.0))[0]
        rec = run_join(
            volume,
            anchor,
            target,
            affine,
            inits,
            radius,
            scale,
            params,
            expected_direction=expected,
            anchor_side_direction=anchor_side,
            containment_regions=target.keymap_regions,
            exclusion_footprints=anchor_footprints,
            allowed_rotations=allowed_rotations,
        )
        if rec is None:
            continue
        rec["anchor_pose"] = anchor_pose
        rec["target_state"] = target.fit_state
        records.append(rec)
        status = rec.get("status")
        rmse = rec.get("rmse_ft", "n/a")
        print(
            f"  {rec['anchor']}->{rec['target']}: {status} rmse={rmse}", file=sys.stderr
        )
    out_path = volume / "artifacts" / "edge_join" / f"joins_{anchor_pose}.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in records))
    summarize_joins(records, f"real joins (anchor@{anchor_pose})")
    # Cross-neighbor agreement: targets joined from 2+ anchors.
    by_target: dict[str, list[dict]] = {}
    for r in records:
        if r.get("status") == "ok":
            by_target.setdefault(r["target"], []).append(r)
    multi = {t: rs for t, rs in by_target.items() if len(rs) >= 2}
    agreements = []
    for t, rs in multi.items():
        unit = next(u for u in units if u.stem == t)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a = np.array(rs[i]["world_affine"])
                b = np.array(rs[j]["world_affine"])
                agreements.append(grid_rmse_ft_between(a, b, unit.width, unit.height))
    if agreements:
        print(
            f"  cross-neighbor agreement over {len(multi)} multi-anchor targets:"
            f" median {percentile(agreements, 0.5):.0f}ft p90 {percentile(agreements, 0.9):.0f}ft"
        )
    print(f"wrote {out_path}")


def cmd_chain(
    volume: Path,
    anchor_pose: str,
    min_verification: float,
    max_rounds: int,
    solve_scale: bool,
    seeds: str = "truth",
) -> None:
    """Chaining + fusion: verification-gated joins become anchors themselves.

    Round k joins every eligible neighbor of the pages posed so far; a target
    joined from several anchors gets a fused pose (verification-weighted corner
    average). Newly posed pages anchor round k+1.
    """
    units = load_page_units(volume)
    by_number = {u.number: u for u in units}
    pairs = detected_pairs(volume)
    neighbor_map: dict[int, set[int]] = {}
    for pair in pairs:
        x, y = tuple(pair)
        neighbor_map.setdefault(x, set()).add(y)
        neighbor_map.setdefault(y, set()).add(x)
    params = edge_join.MatchParams(solve_scale=solve_scale)
    scale = volume_median_scale(units)
    adjacency = load_adjacency(volume)
    allowed_rotations = volume_relative_rotations(units, pairs)
    print(
        f"allowed relative rotations (measured): {allowed_rotations}", file=sys.stderr
    )

    posed: dict[int, tuple[np.ndarray, int]] = {}

    def footprint_ring(unit: PageUnit, affine: np.ndarray) -> list[list[float]]:
        return [
            list(apply_affine(affine, x, y))
            for x, y in [
                (0, 0),
                (unit.width, 0),
                (unit.width, unit.height),
                (0, unit.height),
            ]
        ]

    posed_footprints: list[list[list[float]]] = []
    for unit in units:
        # "truth" seeds = RANSAC fits within 25ft of truth (study framing);
        # "inliers" seeds = the truth-free rule (>=3 inlier intersections).
        is_seed = (
            unit.anchor_truth
            if seeds == "truth"
            else unit.fit_state == "fitted" and unit.inlier_intersections >= 3
        )
        if is_seed:
            affine = (
                unit.truth.affine_local
                if anchor_pose == "truth" and unit.truth
                else unit.gen_affine
            )
            if affine is not None:
                posed[unit.number] = (affine, 0)
                posed_footprints.append(footprint_ring(unit, affine))

    accepted: list[dict] = []
    for round_index in range(1, max_rounds + 1):
        proposals: dict[int, list[dict]] = {}
        frontier = list(posed.items())
        for anchor_number, (anchor_affine, _) in frontier:
            for target_number in sorted(neighbor_map.get(anchor_number, ())):
                if target_number in posed:
                    continue
                anchor = by_number.get(anchor_number)
                target = by_number.get(target_number)
                if not anchor or not target or target.fit_state == "split":
                    continue
                inits = target.keymap_centers or [
                    apply_affine(anchor_affine, anchor.width / 2, anchor.height / 2)
                ]
                radius = target.keymap_radius_m or 800.0
                directions = image_neighbor_directions(adjacency, target.stem)
                expected = (directions.get(anchor.number) or (None, 0.0))[0]
                anchor_dirs = image_neighbor_directions(adjacency, anchor.stem)
                anchor_side = (anchor_dirs.get(target.number) or (None, 0.0))[0]
                record = run_join(
                    volume,
                    anchor,
                    target,
                    anchor_affine,
                    inits,
                    radius,
                    scale,
                    params,
                    expected_direction=expected,
                    anchor_side_direction=anchor_side,
                    containment_regions=target.keymap_regions,
                    exclusion_footprints=posed_footprints,
                    allowed_rotations=allowed_rotations,
                )
                if record is None or record.get("status") != "ok":
                    continue
                if record["verification"] < min_verification:
                    continue
                record["hop"] = round_index
                proposals.setdefault(target_number, []).append(record)
        if not proposals:
            break
        round_rmses = []
        for target_number, records in sorted(proposals.items()):
            target = by_number[target_number]
            affines = [np.array(r["world_affine"]) for r in records]
            weights = [max(r["verification"], 0.01) for r in records]
            fused = fuse_affines(affines, weights, target.width, target.height)
            merged = dict(max(records, key=lambda r: r["verification"]))
            merged["world_affine"] = [list(row) for row in fused]
            merged["n_anchors_fused"] = len(records)
            merged["contributors"] = [r["anchor"] for r in records]
            merged["target_state"] = target.fit_state
            if target.truth is not None:
                merged["rmse_ft"] = round(
                    grid_rmse_ft_between(
                        fused, target.truth.affine_local, target.width, target.height
                    ),
                    1,
                )
                round_rmses.append(merged["rmse_ft"])
            else:
                merged.pop("rmse_ft", None)
            accepted.append(merged)
            posed[target_number] = (fused, round_index)
            posed_footprints.append(footprint_ring(target, fused))
        stats = ""
        if round_rmses:
            stats = (
                f"; rmse median {percentile(round_rmses, 0.5):.0f}ft"
                f" max {max(round_rmses):.0f}ft"
            )
        print(f"round {round_index}: posed {len(proposals)} new pages{stats}")

    suffix = "" if seeds == "truth" else f"_{seeds}"
    out_path = volume / "artifacts" / "edge_join" / f"chain{suffix}.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in accepted))
    scored = [r["rmse_ft"] for r in accepted if "rmse_ft" in r]
    print(
        f"\n== chain (gate {min_verification}, anchor@{anchor_pose}, seeds={seeds}) =="
    )
    print(
        f"posed {len(accepted)} pages beyond the {sum(1 for _, (_, h) in posed.items() if h == 0)} seed anchors"
    )
    if scored:
        for threshold in [25, 50, 100]:
            n = sum(1 for r in scored if r <= threshold)
            print(f"  <={threshold}ft: {n}/{len(scored)} ({n / len(scored):.0%})")
        print(
            f"  rmse median {percentile(scored, 0.5):.0f}ft"
            f" p90 {percentile(scored, 0.9):.0f}ft max {max(scored):.0f}ft"
        )
    fused_multi = [r for r in accepted if r.get("n_anchors_fused", 1) >= 2]
    print(f"  multi-anchor fusions: {len(fused_multi)}")
    by_state: dict[str, int] = {}
    for r in accepted:
        by_state[r["target_state"]] = by_state.get(r["target_state"], 0) + 1
    print(f"  by previous state: {by_state}")
    print(f"wrote {out_path}")


def volume_relative_rotations(
    units: list[PageUnit], pairs: set[frozenset[int]]
) -> tuple[float, ...]:
    """Allowed neighbor rotation offsets, measured from the volume's own fits.

    Truth-free: relative north angles of detected-adjacent RANSAC-fitted pages,
    clustered to 6 degrees; singleton observations are kept too (a volume like
    Brooklyn 1939 mixes many grid orientations). The {0, +/-90} lattice
    defaults are always included.
    """
    by_number = {u.number: u for u in units}
    deltas: list[float] = []
    for pair in pairs:
        x, y = tuple(pair)
        a, b = by_number.get(x), by_number.get(y)
        if not a or not b or a.gen_affine is None or b.gen_affine is None:
            continue
        if a.fit_state != "fitted" or b.fit_state != "fitted":
            continue
        delta = angle_wrap(north_angle(a.gen_affine) - north_angle(b.gen_affine))
        deltas.extend([delta, -delta])
    allowed = {0.0, 90.0, -90.0}
    for delta in sorted(deltas):
        if min(abs(angle_wrap(delta - existing)) for existing in allowed) > 6.0:
            allowed.add(round(delta, 1))
    return tuple(sorted(allowed))


def fuse_affines(
    affines: list[np.ndarray],
    weights: list[float],
    width: int,
    height: int,
) -> np.ndarray:
    """Fuse several page->world affines into one via weighted corner averaging.

    Each affine's images of the page corners are averaged (weighted), and the
    2x3 affine best mapping the corners to the averages is refit — exact when
    the inputs agree, least-squares otherwise.
    """
    corners_px = [(0, 0), (width, 0), (width, height), (0, height)]
    total = sum(weights)
    mean_corners = []
    for x, y in corners_px:
        lon = sum(w * apply_affine(a, x, y)[0] for a, w in zip(affines, weights))
        lat = sum(w * apply_affine(a, x, y)[1] for a, w in zip(affines, weights))
        mean_corners.append((lon / total, lat / total))
    rows = np.array([[x, y, 1.0] for x, y in corners_px])
    solution, _, _, _ = np.linalg.lstsq(rows, np.array(mean_corners), rcond=None)
    return solution.T


def neighbor_variant_path(volume: Path, stem: str) -> Path:
    """Path of a page's materialized neighbor-fit variant georef file."""
    return volume / f"{stem}.georef-neighbor.json"


def materialize_records(
    volume: Path, records: list[dict], units_by_stem: dict[str, PageUnit]
) -> list[dict]:
    """Write pN.georef-neighbor.json files (best record per target).

    The files use the standard georef shape (width/height/corners plus empty
    streets/intersections and the page's keymap block) so the debugger renders
    them; join diagnostics ride along under "neighbor_join". Returns the rows
    written, for reporting.
    """
    best_by_target: dict[str, dict] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        stem = record["target"]
        current = best_by_target.get(stem)
        if current is None or record["verification"] > current["verification"]:
            best_by_target[stem] = record
    rows = []
    for stem, record in sorted(best_by_target.items()):
        unit = units_by_stem[stem]
        affine = np.array(record["world_affine"])
        corners = [
            list(apply_affine(affine, x, y))
            for x, y in [
                (0, 0),
                (unit.width, 0),
                (unit.width, unit.height),
                (0, unit.height),
            ]
        ]
        _, georef = page_fit_state(volume, stem)
        doc: dict = {
            "width": unit.width,
            "height": unit.height,
            "corners": corners,
            "streets": [],
            "intersections": [],
            "neighbor_join": {
                "anchor": record["anchor"],
                "hop": record.get("hop", 1),
                "n_anchors": record.get("n_anchors_fused", 1),
                "verification": record["verification"],
                "inlier_frac": record["inlier_frac"],
                "ncc_fine": record["ncc_fine"],
                "theta_deg": record["theta_deg"],
                "previous_state": unit.fit_state,
                "rmse_ft": record.get("rmse_ft"),
            },
        }
        if georef and georef.get("keymap"):
            doc["keymap"] = georef["keymap"]
        neighbor_variant_path(volume, stem).write_text(json.dumps(doc, indent=2))
        rows.append(
            {
                "stem": stem,
                "previous_state": unit.fit_state,
                "rmse_ft": record.get("rmse_ft"),
                "verification": record["verification"],
                "hop": record.get("hop", 1),
            }
        )
    return rows


def load_placement_records(volume: Path, sources: str) -> dict[str, dict]:
    """Best record per target from comma-separated jsonl basenames.

    Earlier sources win: "chain,posegraph" prefers the gated greedy chain's
    placement and falls back to the pose graph for pages it refused.
    """
    by_target: dict[str, dict] = {}
    for source in sources.split(","):
        path = volume / "artifacts" / "edge_join" / f"{source.strip()}.jsonl"
        if not path.exists():
            print(f"note: no {path.name}", file=sys.stderr)
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                by_target.setdefault(record["target"], record)
    return by_target


def cmd_report(volume: Path, chain_source: str = "chain") -> None:
    """Compare RANSAC-only, augmented, and replace policies on one denominator.

    Policies (page level, splits excluded from neighbor placement):
      ransac-all  every accepted RANSAC fit (any quality)
      augment     RANSAC everywhere it fit; neighbor fits ONLY for pages
                  RANSAC could not fit (the production intent)
      replace     seed anchors keep RANSAC; chain placements override
                  non-anchor RANSAC fits (the study framing)
    """
    units = load_page_units(volume)
    chained = load_placement_records(volume, chain_source)

    def placement(unit: PageUnit, policy: str):
        chain_rec = chained.get(unit.stem)
        chain_affine = (
            np.array(chain_rec["world_affine"]) if chain_rec is not None else None
        )
        ransac = unit.gen_affine if unit.fit_state == "fitted" else None
        if policy == "ransac-all":
            return ("ransac", ransac)
        if policy == "augment":
            if ransac is not None:
                return ("ransac", ransac)
            return ("neighbor", chain_affine)
        if policy == "replace":
            if unit.anchor_truth and ransac is not None:
                return ("ransac", ransac)
            return ("neighbor", chain_affine)
        raise ValueError(policy)

    truth_units = [u for u in units if u.truth is not None]
    n_split_truth = sum(1 for u in units if u.split_truth and u.truth is None)
    total = len(truth_units) + n_split_truth
    print(
        f"== {volume.name}: {total} truth pages"
        f" ({n_split_truth} split-only, out of scope) — chain={chain_source} =="
    )
    header = (
        f"{'policy':<12} {'placed':>8} {'median':>7} {'mean':>6} {'max':>6}"
        f" {'<=25':>5} {'<=50':>5} {'<=100':>6}"
    )
    print(header)
    print("-" * len(header))
    for policy in ["ransac-all", "augment", "replace"]:
        rmses = []
        for unit in truth_units:
            _, affine = placement(unit, policy)
            if affine is None or unit.truth is None:
                continue
            rmses.append(
                grid_rmse_ft_between(
                    affine, unit.truth.affine_local, unit.width, unit.height
                )
            )
        rmses.sort()
        n = len(rmses)
        buckets = [sum(1 for r in rmses if r <= t) for t in (25, 50, 100)]
        print(
            f"{policy:<12} {n:>4}/{total:<3} {percentile(rmses, 0.5):>6.0f}f"
            f" {sum(rmses) / n:>5.0f}f {max(rmses):>5.0f}f"
            f" {buckets[0]:>5} {buckets[1]:>5} {buckets[2]:>6}"
        )


GRAPH_MIN_VERIFICATION = 0.5
SINGLETON_MIN_VERIFICATION = 1.2


def measurement_sigmas(verification: float, synthetic: bool) -> tuple[float, float]:
    """(sigma_pos_m, sigma_theta_rad) for one join measurement.

    Calibrated on DC: verification>=1.3 joins are ~5m-class; sub-gate joins
    enter with loose sigmas and rely on the solver's Huber loss to be outvoted
    when wrong. Synthetic-anchor measurements (both pages unfitted, anchor
    posed at its keymap guess) searched a worse frame, so they get 1.5x.
    """
    if verification >= 1.3:
        pos, theta = 5.0, 0.6
    elif verification >= 1.1:
        pos, theta = 9.0, 1.2
    elif verification >= 0.9:
        pos, theta = 18.0, 2.5
    else:
        pos, theta = 35.0, 5.0
    factor = 1.5 if synthetic else 1.0
    return pos * factor, math.radians(theta * factor)


def build_volume_frame(units: list[PageUnit], scale_m_per_px: float):
    """A VolumeFrame centred on the volume's fitted (or keymap) pages."""
    from mapsnap.edge_join_graph import VolumeFrame

    lons, lats = [], []
    for unit in units:
        if unit.fit_state == "fitted" and unit.gen_affine is not None:
            lon, lat = apply_affine(unit.gen_affine, unit.width / 2, unit.height / 2)
        elif unit.keymap_centers:
            lon, lat = unit.keymap_centers[0]
        else:
            continue
        lons.append(lon)
        lats.append(lat)
    return VolumeFrame(statistics.mean(lons), statistics.mean(lats), scale_m_per_px)


def median_fitted_theta(vframe, units: list[PageUnit]) -> float:
    """Circular mean pose rotation of the volume's fitted pages."""
    sines = cosines = 0.0
    for unit in units:
        if unit.fit_state == "fitted" and unit.gen_affine is not None:
            theta = vframe.affine_to_pose(unit.gen_affine)[2]
            sines += math.sin(theta)
            cosines += math.cos(theta)
    return math.atan2(sines, cosines)


def synthetic_anchor_affine(
    vframe, unit: PageUnit, center_lonlat: tuple[float, float], theta: float
) -> np.ndarray:
    """A pose guess for an unfitted anchor: keymap centre + volume rotation.

    Its absolute error cancels in the relative measurement; it only needs to
    be close enough for the matcher's search window and direction gates.
    """
    xc, yc = vframe.lonlat_to_xy(*center_lonlat)
    scale = vframe.scale_m_per_px
    c, s = math.cos(theta), math.sin(theta)
    ox = scale * (c * unit.width / 2 - s * unit.height / 2)
    oy = scale * (s * unit.width / 2 + c * unit.height / 2)
    return vframe.pose_to_affine(xc - ox, yc - oy, theta)


def collect_graph_measurements(
    volume: Path, units: list[PageUnit], limit: int | None = None
) -> list[dict]:
    """Run the matcher over every measurable mutual-adjacency edge.

    Direction policy per edge: both fitted -> one run anchored on the better
    fit (the reverse adds little); one fitted -> anchored on the fitted page;
    neither fitted -> both directions with synthetic keymap-posed anchors
    (these island edges are what the pose graph adds over greedy chaining).
    No verification gate here — sub-gate joins are kept as loose evidence.
    """
    by_number = {u.number: u for u in units}
    pairs = detected_pairs(volume)
    params = edge_join.MatchParams()
    scale = volume_median_scale(units)
    adjacency = load_adjacency(volume)
    allowed_rotations = volume_relative_rotations(units, pairs)
    vframe = build_volume_frame(units, scale)
    theta_syn = median_fitted_theta(vframe, units)

    footprints_by_number: dict[int, list[list[float]]] = {}
    for unit in units:
        if unit.anchor_free and unit.gen_affine is not None:
            footprints_by_number[unit.number] = [
                list(apply_affine(unit.gen_affine, x, y))
                for x, y in [
                    (0, 0),
                    (unit.width, 0),
                    (unit.width, unit.height),
                    (0, unit.height),
                ]
            ]

    def is_fitted(unit: PageUnit) -> bool:
        return unit.fit_state == "fitted" and unit.gen_affine is not None

    directed: list[tuple[PageUnit, PageUnit]] = []
    for pair in sorted(pairs, key=sorted):
        x, y = sorted(pair)
        ux, uy = by_number.get(x), by_number.get(y)
        if not ux or not uy or "split" in (ux.fit_state, uy.fit_state):
            continue
        if is_fitted(ux) and is_fitted(uy):
            better = ux if ux.inlier_intersections >= uy.inlier_intersections else uy
            other = uy if better is ux else ux
            directed.append((better, other))
        elif is_fitted(ux):
            directed.append((ux, uy))
        elif is_fitted(uy):
            directed.append((uy, ux))
        else:
            directed.append((ux, uy))
            directed.append((uy, ux))

    records: list[dict] = []
    for anchor, target in directed[: limit or len(directed)]:
        synthetic = not is_fitted(anchor)
        if synthetic:
            if not anchor.keymap_centers:
                continue
            anchor_affine = synthetic_anchor_affine(
                vframe, anchor, anchor.keymap_centers[0], theta_syn
            )
        else:
            anchor_affine = anchor.gen_affine
            assert anchor_affine is not None
        inits = target.keymap_centers or [
            apply_affine(anchor_affine, anchor.width / 2, anchor.height / 2)
        ]
        radius = target.keymap_radius_m or 800.0
        if synthetic:
            # The anchor's own keymap error adds to the target's; widen.
            radius = max(radius, 900.0)
        directions = image_neighbor_directions(adjacency, target.stem)
        expected = (directions.get(anchor.number) or (None, 0.0))[0]
        anchor_dirs = image_neighbor_directions(adjacency, anchor.stem)
        anchor_side = (anchor_dirs.get(target.number) or (None, 0.0))[0]
        exclusion = None
        if not synthetic:
            exclusion = [
                fp
                for n, fp in footprints_by_number.items()
                if n not in (anchor.number, target.number)
            ] or None
        record = run_join(
            volume,
            anchor,
            target,
            anchor_affine,
            inits,
            radius,
            scale,
            params,
            expected_direction=expected,
            anchor_side_direction=anchor_side,
            containment_regions=None if synthetic else target.keymap_regions,
            exclusion_footprints=exclusion,
            allowed_rotations=allowed_rotations,
        )
        if record is None:
            continue
        record["synthetic_anchor"] = synthetic
        record["anchor_state"] = anchor.fit_state
        record["target_state"] = target.fit_state
        record["anchor_affine"] = [list(row) for row in anchor_affine]
        records.append(record)
        print(
            f"  [{len(records)}/{len(directed)}] {record['anchor']}->{record['target']}"
            f" {record.get('status')} ver={record.get('verification', 'n/a')}"
            f" rmse={record.get('rmse_ft', 'n/a')}{' (syn)' if synthetic else ''}",
            file=sys.stderr,
        )
    return records


def keymap_position_prior(
    unit: PageUnit, vframe, init_xy: tuple[float, float]
) -> tuple[float, float, float] | None:
    """(x, y, sigma_m) position evidence from the page's keymap block.

    A segmented region centroid is trusted more than a bare centre; with
    several candidate hypotheses the one nearest the initialized pose is used,
    at a widened sigma.
    """
    if unit.keymap_regions:
        from shapely.geometry import Polygon

        centroids = []
        for ring in unit.keymap_regions:
            poly = Polygon(ring).buffer(0)
            if not poly.is_empty and poly.area > 0:
                centroids.append(vframe.lonlat_to_xy(poly.centroid.x, poly.centroid.y))
        if centroids:
            x, y = min(
                centroids,
                key=lambda p: math.hypot(p[0] - init_xy[0], p[1] - init_xy[1]),
            )
            return x, y, 150.0 if len(centroids) == 1 else 250.0
    if unit.keymap_centers:
        candidates = [vframe.lonlat_to_xy(lon, lat) for lon, lat in unit.keymap_centers]
        x, y = min(
            candidates, key=lambda p: math.hypot(p[0] - init_xy[0], p[1] - init_xy[1])
        )
        return x, y, 250.0 if len(candidates) == 1 else 350.0
    return None


def cmd_posegraph(volume: Path, remeasure: bool, limit: int | None) -> None:
    """Global pose-graph solve over all edge-join measurements.

    Replaces the greedy chain's sequential accept/reject with one robust joint
    optimization: every mutual edge contributes a relative measurement
    (including sub-gate ones), RANSAC fits and keymap locations contribute
    absolute priors, and the Huber loss lets consistent evidence outvote
    wrong locks. Writes posegraph_all.jsonl (every grounded node, for
    `report --chain-source posegraph_all`) and posegraph.jsonl (previously
    unfitted nodes only, for `materialize --source posegraph`).
    """
    from mapsnap.edge_join_graph import (
        AbsolutePrior,
        EdgeHypotheses,
        RelativeMeasurement,
        solve_pose_graph_hypotheses,
        spanning_tree_initialization,
    )

    units = load_page_units(volume)
    by_stem = {u.stem: u for u in units}
    out_dir = volume / "artifacts" / "edge_join"
    meas_path = out_dir / "measurements.jsonl"
    if meas_path.exists() and not remeasure:
        records = [
            json.loads(line) for line in meas_path.read_text().splitlines() if line
        ]
        print(f"loaded {len(records)} cached measurements from {meas_path}")
    else:
        records = collect_graph_measurements(volume, units, limit)
        meas_path.write_text("\n".join(json.dumps(r) for r in records))
        print(f"wrote {len(records)} measurements to {meas_path}")

    kept = [
        r
        for r in records
        if r.get("status") == "ok"
        and r.get("verification", 0.0) >= GRAPH_MIN_VERIFICATION
    ]
    print(f"{len(kept)} measurements at verification >= {GRAPH_MIN_VERIFICATION}")

    scale = volume_median_scale(units)
    vframe = build_volume_frame(units, scale)
    theta_syn = median_fitted_theta(vframe, units)

    def is_fitted(unit: PageUnit) -> bool:
        return unit.fit_state == "fitted" and unit.gen_affine is not None

    node_stems = sorted(
        {u.stem for u in units if is_fitted(u)}
        | {r["anchor"] for r in kept}
        | {r["target"] for r in kept},
        key=lambda s: by_stem[s].number,
    )
    index = {stem: i for i, stem in enumerate(node_stems)}

    edges = []
    incident: dict[str, list[dict]] = {}
    for r in kept:
        pose_a = vframe.affine_to_pose(np.array(r["anchor_affine"]))
        candidates = []
        # Winner first, then the recorded alternates (deduped against it):
        # the EM solver may re-pick a lower-ranked candidate that is globally
        # consistent (aliased slides often outrank the true pose locally).
        alternates = r.get("alternates") or [
            {"world_affine": r["world_affine"], "verification": r["verification"]}
        ]
        for alternate in alternates:
            pose_t = vframe.affine_to_pose(np.array(alternate["world_affine"]))
            dx, dy, dtheta = vframe.relative(pose_a, pose_t)
            sigma_pos, sigma_theta = measurement_sigmas(
                alternate["verification"], r.get("synthetic_anchor", False)
            )
            candidates.append(
                RelativeMeasurement(
                    index[r["anchor"]],
                    index[r["target"]],
                    dx,
                    dy,
                    dtheta,
                    sigma_pos_m=sigma_pos,
                    sigma_theta_rad=sigma_theta,
                )
            )
        edges.append(EdgeHypotheses(index[r["anchor"]], index[r["target"]], candidates))
        incident.setdefault(r["target"], []).append(r)
        incident.setdefault(r["anchor"], []).append(r)
    winners = [e.candidates[0] for e in edges]

    # Grounding: connected components of the measurement graph. A component
    # with no fitted page floats on keymap priors alone (position ~150-350m
    # class); components with neither are unplaceable and dropped.
    parent = list(range(len(node_stems)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for m in winners:
        parent[find(m.a)] = find(m.b)
    grounded_roots = {find(index[u.stem]) for u in units if is_fitted(u)}
    grounded = {s for s in node_stems if find(index[s]) in grounded_roots}
    floating = [
        s for s in node_stems if s not in grounded and by_stem[s].keymap_centers
    ]
    dropped = [
        s for s in node_stems if s not in grounded and not by_stem[s].keymap_centers
    ]
    print(
        f"graph: {len(node_stems)} nodes, {len(edges)} edges;"
        f" {len(grounded)} grounded, {len(floating)} floating-on-keymap,"
        f" {len(dropped)} unplaceable"
    )

    initial_known = {
        index[u.stem]: vframe.affine_to_pose(u.gen_affine)
        for u in units
        if is_fitted(u) and u.gen_affine is not None and u.stem in index
    }
    fallback = {}
    for stem in node_stems:
        unit = by_stem[stem]
        if unit.keymap_centers:
            x, y = vframe.lonlat_to_xy(*unit.keymap_centers[0])
            fallback[index[stem]] = (x, y, theta_syn)
    initial = spanning_tree_initialization(
        len(node_stems), initial_known, winners, fallback
    )

    priors = []
    for stem in node_stems:
        unit = by_stem[stem]
        i = index[stem]
        if is_fitted(unit) and unit.gen_affine is not None:
            x, y, theta = vframe.affine_to_pose(unit.gen_affine)
            # Tight tiers (validated on DC): letting anchors drift was the
            # dominant error source, and the Huber loss still lets the rare
            # high-inlier-but-wrong fit be outvoted by its neighbors.
            if unit.inlier_intersections >= 5:
                sigma_pos, sigma_theta = 1.5, 0.15
            elif unit.inlier_intersections >= 3:
                sigma_pos, sigma_theta = 3.0, 0.3
            else:
                # Low-inlier fits are where RANSAC catastrophes live, but
                # loosening this tier (tried 12m and 25m on DC) lets whole
                # weakly-anchored clusters slide onto aliased grid poses;
                # 6m was the best global compromise.
                sigma_pos, sigma_theta = 6.0, 0.6
            priors.append(
                AbsolutePrior(
                    i,
                    x,
                    y,
                    sigma_pos_m=sigma_pos,
                    theta=theta,
                    sigma_theta_rad=math.radians(sigma_theta),
                )
            )
        keymap_prior = keymap_position_prior(unit, vframe, tuple(initial[i, :2]))
        if keymap_prior is not None:
            x, y, sigma_pos = keymap_prior
            priors.append(AbsolutePrior(i, x, y, sigma_pos_m=sigma_pos))

    solved, assignment, active, diagnostics = solve_pose_graph_hypotheses(
        initial, edges, priors
    )
    reassigned = sum(1 for a in assignment if a != 0)
    print(f"solver: {diagnostics}; {reassigned} edges re-assigned to an alternate")
    # Rebuild incidence from surviving edges only: a page whose every edge was
    # trimmed has no measurement support and must not be written as placed.
    incident = {}
    for i, r in enumerate(kept):
        if not active[i]:
            continue
        incident.setdefault(r["target"], []).append(r)
        incident.setdefault(r["anchor"], []).append(r)

    # Evaluate every node against truth at its graph pose.
    graph_rmse: dict[str, float] = {}
    for stem in node_stems:
        unit = by_stem[stem]
        if unit.truth is None:
            continue
        affine = vframe.pose_to_affine(*solved[index[stem]])
        graph_rmse[stem] = grid_rmse_ft_between(
            affine, unit.truth.affine_local, unit.width, unit.height
        )
    anchor_sanity = [graph_rmse[s] for s in graph_rmse if by_stem[s].anchor_truth]
    if anchor_sanity:
        print(
            f"graph pose on {len(anchor_sanity)} truth-anchor pages:"
            f" median {percentile(anchor_sanity, 0.5):.0f}ft"
            f" max {max(anchor_sanity):.0f}ft"
        )

    def supported(stem: str) -> bool:
        """Enough measurement support to trust the page's graph pose.

        A page held by a single measurement has no redundancy for the solver
        to exploit, so it must meet the chain's verification gate on its own;
        multi-measurement pages are protected by trimming and re-assignment.
        (DC+Detroit: this kills four 300-1000ft singletons at the cost of one
        38ft placement.)
        """
        records = incident.get(stem, [])
        if not records:
            return False
        if len(records) == 1:
            return records[0]["verification"] >= SINGLETON_MIN_VERIFICATION
        return True

    newly = [s for s in node_stems if not is_fitted(by_stem[s]) and s in grounded]
    print(f"\npreviously-unfitted pages placed by the graph ({len(newly)}):")
    for stem in newly:
        rmse = graph_rmse.get(stem)
        n_meas = len(incident.get(stem, []))
        best = max((r["verification"] for r in incident.get(stem, [])), default=0.0)
        rmse_str = f"{rmse:7.1f}ft" if rmse is not None else " no truth"
        note = "" if supported(stem) else "  (unsupported, not written)"
        print(
            f"  {stem:>6} was {by_stem[stem].fit_state:<9} rmse {rmse_str}"
            f"  measurements {n_meas}  best-ver {best:.2f}{note}"
        )
    for stem in floating:
        print(f"  {stem:>6} floating on keymap only (not written)")

    def graph_record(stem: str) -> dict:
        unit = by_stem[stem]
        pose = solved[index[stem]]
        affine = vframe.pose_to_affine(*pose)
        incident_records = incident.get(stem, [])
        best = max(incident_records, key=lambda r: r["verification"])
        record = {
            "target": stem,
            "status": "ok",
            "anchor": best["anchor"] if best["target"] == stem else best["target"],
            "hop": 0,
            "n_anchors_fused": len(incident_records),
            "contributors": sorted(
                {
                    r["anchor"] if r["target"] == stem else r["target"]
                    for r in incident_records
                }
            ),
            "verification": best["verification"],
            "inlier_frac": best["inlier_frac"],
            "ncc_fine": best["ncc_fine"],
            "theta_deg": round(math.degrees(pose[2]), 2),
            "target_state": unit.fit_state,
            "world_affine": [list(row) for row in affine],
        }
        if stem in graph_rmse:
            record["rmse_ft"] = round(graph_rmse[stem], 1)
        return record

    all_records = [
        graph_record(s)
        for s in sorted(grounded, key=lambda s: by_stem[s].number)
        if incident.get(s) and (is_fitted(by_stem[s]) or supported(s))
    ]
    (out_dir / "posegraph_all.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_records)
    )
    new_records = [r for r in all_records if not is_fitted(by_stem[r["target"]])]
    (out_dir / "posegraph.jsonl").write_text(
        "\n".join(json.dumps(r) for r in new_records)
    )
    print(
        f"\nwrote {len(all_records)} records to posegraph_all.jsonl,"
        f" {len(new_records)} previously-unfitted to posegraph.jsonl"
    )
    print()
    cmd_report(volume, "posegraph_all")


def cmd_materialize(volume: Path, source: str) -> None:
    """Write neighbor-fit variant georef files and print inspection lists."""
    units_by_stem = {u.stem: u for u in load_page_units(volume)}
    records = list(load_placement_records(volume, source).values())
    rows = materialize_records(volume, records, units_by_stem)
    print(
        f"wrote {len(rows)} {volume.name}/pN.georef-neighbor.json files from {source}"
    )

    newly = [r for r in rows if r["previous_state"] != "fitted"]
    print(f"\nnot previously georeferenced ({len(newly)}):")
    for r in sorted(
        newly, key=lambda r: r["rmse_ft"] if r["rmse_ft"] is not None else 1e9
    ):
        rmse = f"{r['rmse_ft']:.0f}ft" if r["rmse_ft"] is not None else "no truth"
        print(
            f"  {r['stem']:>6} was {r['previous_state']:<9} rmse {rmse:>9}"
            f"  ver {r['verification']:.2f}  hop {r['hop']}"
        )
    scored = [r for r in rows if r["rmse_ft"] is not None]
    scored.sort(key=lambda r: r["rmse_ft"])
    print("\nbest fits:")
    for r in scored[:8]:
        print(
            f"  {r['stem']:>6} rmse {r['rmse_ft']:7.1f}ft ver {r['verification']:.2f} (was {r['previous_state']})"
        )
    print("\nworst fits:")
    for r in scored[-8:][::-1]:
        print(
            f"  {r['stem']:>6} rmse {r['rmse_ft']:7.1f}ft ver {r['verification']:.2f} (was {r['previous_state']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "stats",
            "infer",
            "sanity",
            "perturb",
            "match",
            "chain",
            "posegraph",
            "materialize",
            "report",
        ],
    )
    parser.add_argument("volume", type=Path)
    parser.add_argument("--limit", type=int, help="only first N pairs/attempts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--draws", type=int, default=2, help="perturb: inits per pair")
    parser.add_argument(
        "--anchor-pose", choices=["truth", "generated"], default="truth"
    )
    parser.add_argument("--solve-scale", action="store_true")
    parser.add_argument(
        "--min-verification", type=float, default=1.2, help="chain: acceptance gate"
    )
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument(
        "--source", default="joins_generated", help="materialize: jsonl basename"
    )
    parser.add_argument("--seeds", choices=["truth", "inliers"], default="truth")
    parser.add_argument("--chain-source", default="chain", help="report: chain file")
    parser.add_argument(
        "--remeasure",
        action="store_true",
        help="posegraph: ignore cached measurements.jsonl",
    )
    args = parser.parse_args()
    if args.command == "stats":
        cmd_stats(args.volume)
    elif args.command == "infer":
        cmd_infer(args.volume)
    elif args.command == "sanity":
        cmd_sanity(args.volume, args.limit)
    elif args.command == "perturb":
        cmd_perturb(args.volume, args.limit, args.seed, args.draws, args.solve_scale)
    elif args.command == "match":
        cmd_match(args.volume, args.limit, args.anchor_pose, args.solve_scale)
    elif args.command == "chain":
        cmd_chain(
            args.volume,
            args.anchor_pose,
            args.min_verification,
            args.max_rounds,
            args.solve_scale,
            args.seeds,
        )
    elif args.command == "posegraph":
        cmd_posegraph(args.volume, args.remeasure, args.limit)
    elif args.command == "materialize":
        cmd_materialize(args.volume, args.source)
    elif args.command == "report":
        cmd_report(args.volume, args.chain_source)


if __name__ == "__main__":
    main()
