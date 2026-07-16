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

from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    extract_gcps,
    fit_transform,
    haversine_ft,
    north_angle,
    sample_grid,
)
from mapsnap.keymap.align_page_region import angle_wrap
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
    """(unsplit truth items by page key, page keys with split-only truth)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stats", "infer", "sanity"])
    parser.add_argument("volume", type=Path)
    parser.add_argument("--limit", type=int, help="sanity: only first N pairs")
    args = parser.parse_args()
    if args.command == "stats":
        cmd_stats(args.volume)
    elif args.command == "infer":
        cmd_infer(args.volume)
    elif args.command == "sanity":
        cmd_sanity(args.volume, args.limit)


if __name__ == "__main__":
    main()
