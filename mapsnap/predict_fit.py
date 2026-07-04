"""Predict-then-verify georeferencing prototype: fit pages from key-map priors.

The production fitter is match-then-fit: label texts are matched to street names,
co-detected street pairs yield intersection GCPs, and RANSAC fits a similarity over
those. Pages fail when they can't assemble two distinct trustworthy intersections;
one-intersection pages defer to an assigned-scale fit.

This prototype inverts the order. A georeferenced key map implies a prior similarity
transform for every placed page:

  - translation: the page number's detection center(s) on the key map;
  - scale: the volume's median fitted scale x the page's region multiplier
    (half-scale sheets have ~2x-area regions);
  - rotation: voted, not assumed — each label's pixel direction against each
    same-name candidate street's bearing votes for theta = bearing + label_dir
    (mod pi); correct matches pile onto a mode, wrong candidates scatter.

Association then becomes trivial (project a label, take same-name segments within a
shrinking tolerance), and the fit minimizes *point-to-line* residuals: a label
constrains the fit by lying on its street, no intersections required. A page with
two street labels and no usable crossing — nofit today — is over-determined here.

Verification is refuse-first: a fit is only emitted if enough independent labels
agree (count, distinct names, normal-direction span, residual RMS, scale and
position within the prior's tolerance). Everything else stays unfit.

Standalone CLI; writes georef.json-compatible files to --out and, with --truth,
prints a three-way per-page comparison (prototype vs existing pipeline vs truth).

    uv run python -m mapsnap.predict_fit data/detroit_mich_1929_vol_11 \
        --keymap data/detroit_mich_1929_vol_11/raw/p0.keymap.json \
        --out /tmp/proto --truth data/detroit_mich_1929_vol_11/main.iiif.json
"""

import argparse
import dataclasses
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    annotations_by_source,
    extract_gcps,
    fit_transform,
    haversine_ft,
    sample_grid,
)
from mapsnap.georef_from_labels import (
    _FT_PER_DEG_LAT,
    image_size,
    is_bare_letter,
    is_number_only,
    is_split_page,
    label_features,
    LabelFeature,
    region_prior_px_per_ft,
    region_relative_scale,
)
from mapsnap.keymap.locate import KeymapLocator, page_number
from mapsnap.streets import build_block_index
from mapsnap.utils import image_stem, list_pages

# Association tolerance schedule (ft): coarse enough for the position prior's real
# error on the first pass (key-map detections sit up to ~650 ft from the true page
# center on Detroit), tight enough to shed wrong-name coincidences by the last.
TOLERANCE_SCHEDULE_FT = [1500.0, 900.0, 500.0, 250.0]

# A candidate segment must roughly agree with the label's predicted world direction.
DIRECTION_TOLERANCE_RAD = math.radians(30.0)

# Verification thresholds (see verify()).
INLIER_DIST_FT = 100.0
MIN_INLIERS = 3
MIN_DISTINCT_NAMES = 2
MIN_NORMAL_SPAN_RAD = math.radians(30.0)
MAX_INLIER_RMS_FT = 60.0
# Verified fits within 10% of the scale prior measured 12 ft median / 74 ft max truth
# error on Detroit; the only fits beyond 20% off-prior were both wrong (166, 243 ft),
# having latched onto wrong parallel branches. The prior is region-adjusted, so a
# genuine half-scale sheet does not stress this band.
SCALE_PRIOR_BAND = (0.85, 1.2)
MAX_CENTER_DRIFT_RADII = 1.5
MIN_LABEL_SPREAD_FRAC = 0.25

FT_PER_M = 3.280839895


@dataclass
class Frame:
    """Local planar frame in feet centered on a key-map detection."""

    lon0: float
    lat0: float
    cos_phi: float

    @classmethod
    def at(cls, lon: float, lat: float) -> "Frame":
        return cls(lon, lat, math.cos(math.radians(lat)))

    def to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        return (
            (lon - self.lon0) * _FT_PER_DEG_LAT * self.cos_phi,
            (lat - self.lat0) * _FT_PER_DEG_LAT,
        )

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.lon0 + x / (_FT_PER_DEG_LAT * self.cos_phi),
            self.lat0 + y / _FT_PER_DEG_LAT,
        )


@dataclass
class Similarity:
    """Page-to-frame similarity with the image y-flip baked in.

    x = p*u + q*v + tx ;  y = q*u - p*v + ty. An image direction delta maps to world
    angle theta_r - delta where theta_r = atan2(q, p); scale (ft/px) = hypot(p, q).
    """

    p: float
    q: float
    tx: float
    ty: float

    @classmethod
    def from_prior(
        cls, scale_ft_per_px: float, theta: float, center_px: tuple[float, float]
    ) -> "Similarity":
        """The transform with the given scale/rotation mapping center_px to (0, 0)."""
        p = scale_ft_per_px * math.cos(theta)
        q = scale_ft_per_px * math.sin(theta)
        u, v = center_px
        return cls(p, q, -(p * u + q * v), -(q * u - p * v))

    def apply(self, u: float, v: float) -> tuple[float, float]:
        return (
            self.p * u + self.q * v + self.tx,
            self.q * u - self.p * v + self.ty,
        )

    def scale_ft_per_px(self) -> float:
        return math.hypot(self.p, self.q)

    def rotation(self) -> float:
        return math.atan2(self.q, self.p)


@dataclass
class Match:
    """A label associated with the nearest point on a same-name street segment."""

    label: LabelFeature
    confidence: float
    point_xy: tuple[float, float]  # nearest polyline point, frame ft
    seg_dir: float  # segment bearing at that point, radians
    dist_ft: float  # label-projection to point distance


def polyline_nearest(
    coords_xy: np.ndarray, point: tuple[float, float]
) -> tuple[float, tuple[float, float], float]:
    """(distance, nearest point, segment bearing) from ``point`` to a polyline."""
    px, py = point
    best = (float("inf"), (0.0, 0.0), 0.0)
    for i in range(len(coords_xy) - 1):
        ax, ay = coords_xy[i]
        bx, by = coords_xy[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0:
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
        nx, ny = ax + t * dx, ay + t * dy
        d = math.hypot(px - nx, py - ny)
        if d < best[0]:
            best = (d, (nx, ny), math.atan2(dy, dx))
    return best


def angle_diff_mod_pi(a: float, b: float) -> float:
    """Smallest absolute difference between two undirected angles (mod pi)."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def vote_rotation(
    labels: list[tuple[LabelFeature, float]],
    blocks_xy: dict[str, list[np.ndarray]],
    bins: int = 36,
) -> list[float]:
    """Rotation candidates (mod pi) from label-direction x candidate-bearing votes.

    Every (label, same-name segment) pair votes theta = bearing + label_dir (mod pi),
    weighted by confidence and segment length. Correct matches concentrate on the true
    rotation; wrong-candidate votes spread. Returns up to two histogram peaks.
    """
    hist = np.zeros(bins)
    for label, confidence in labels:
        for coords in blocks_xy.get(label.text, []):
            for i in range(len(coords) - 1):
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]
                seg_len = math.hypot(dx, dy)
                if seg_len < 100.0:
                    continue
                theta = (math.atan2(dy, dx) + label.dir_pix) % math.pi
                weight = confidence * min(1.0, seg_len / 300.0)
                hist[int(theta / math.pi * bins) % bins] += weight
    if not hist.any():
        return []
    smoothed = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    order = np.argsort(smoothed)[::-1]
    peaks: list[float] = []
    for idx in order:
        theta = (idx + 0.5) * math.pi / bins
        if all(angle_diff_mod_pi(theta, p) > math.radians(15) for p in peaks):
            peaks.append(theta)
        if len(peaks) == 2:
            break
    return peaks


def associate(
    transform: Similarity,
    labels: list[tuple[LabelFeature, float]],
    blocks_xy: dict[str, list[np.ndarray]],
    tolerance_ft: float,
) -> list[Match]:
    """Match each label to the nearest same-name segment within tolerance."""
    matches = []
    for label, confidence in labels:
        proj = transform.apply(*label.center)
        predicted_dir = transform.rotation() - label.dir_pix
        best: Match | None = None
        for coords in blocks_xy.get(label.text, []):
            dist, point, seg_dir = polyline_nearest(coords, proj)
            if dist >= tolerance_ft:
                continue
            if angle_diff_mod_pi(predicted_dir, seg_dir) > DIRECTION_TOLERANCE_RAD:
                continue
            if best is None or dist < best.dist_ft:
                best = Match(label, confidence, point, seg_dir, dist)
        if best is not None:
            matches.append(best)
    return matches


# Lever arm (ft) converting a label's angular direction error into a comparable
# distance residual, and the fractional-scale-deviation lever for the scale prior.
DIRECTION_LEVER_FT = 250.0
SCALE_CLAMP_BAND = (0.7, 1.45)


def solve_wls(
    matches: list[Match],
    center_px: tuple[float, float],
    position_sigma_ft: float,
    prior_scale_ft_per_px: float,
    perp_sigma_ft: float = 50.0,
) -> Similarity | None:
    """Weighted least squares for (p, q, tx, ty) from point-to-line residuals.

    Each match contributes two rows: the label's projection must lie on its segment's
    line (residual along the segment normal), and the label's mapped direction must
    stay parallel to the segment (linear in p, q). With only point-to-line rows, a few
    labels in two direction families leave a near-degenerate grow-and-rotate family
    that least squares runs down (p85: scale 0.66 -> 2.34 ft/px in one step); the
    direction rows pin rotation and a scale-prior row plus post-solve clamp pin scale.
    Two weak rows anchor the page center to the key-map location.
    """
    rows, rhs, weights = [], [], []
    for m in matches:
        nx, ny = -math.sin(m.seg_dir), math.cos(m.seg_dir)
        u, v = m.label.center
        weight = (
            m.confidence
            * min(1.0, 2 * perp_sigma_ft / max(m.dist_ft, 1.0))
            / perp_sigma_ft
        )
        rows.append([nx * u - ny * v, nx * v + ny * u, nx, ny])
        rhs.append(nx * m.point_xy[0] + ny * m.point_xy[1])
        weights.append(weight)
        # Direction row: the world image-direction (p cos d + q sin d, q cos d - p sin d)
        # must be parallel to the segment -> its normal component is zero:
        # -p sin(phi + d) + q cos(phi + d) = 0, scaled by a lever arm so an angular
        # error costs like a distance. Normalized by the prior scale so the row's
        # magnitude is comparable to the perp rows regardless of image resolution.
        phi_d = m.seg_dir + m.label.dir_pix
        lever = DIRECTION_LEVER_FT / prior_scale_ft_per_px
        rows.append([-math.sin(phi_d) * lever, math.cos(phi_d) * lever, 0.0, 0.0])
        rhs.append(0.0)
        weights.append(weight)
    # Weak anchor: page center -> frame origin. Coefficients on (p, q, tx, ty):
    # x = p*u + q*v + tx -> [u, v, 1, 0];  y = q*u - p*v + ty -> [-v, u, 0, 1].
    u, v = center_px
    for row in ([u, v, 1.0, 0.0], [-v, u, 0.0, 1.0]):
        rows.append(row)
        rhs.append(0.0)
        weights.append(1.0 / position_sigma_ft)
    A = np.array(rows) * np.array(weights)[:, None]
    b = np.array(rhs) * np.array(weights)
    try:
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    fitted = Similarity(*solution)
    # Hard clamp: Sanborn scales are quantized and the prior is region-adjusted, so a
    # solution outside the band is a degeneracy, not evidence. Renormalize (p, q).
    scale = fitted.scale_ft_per_px()
    low = SCALE_CLAMP_BAND[0] * prior_scale_ft_per_px
    high = SCALE_CLAMP_BAND[1] * prior_scale_ft_per_px
    if scale > 0 and not (low <= scale <= high):
        factor = max(low, min(high, scale)) / scale
        fitted = Similarity(fitted.p * factor, fitted.q * factor, fitted.tx, fitted.ty)
    return fitted


@dataclass
class FitResult:
    transform: Similarity | None
    verified: bool
    reason: str
    inliers: list[Match]
    rms_ft: float = float("nan")


def verify(
    transform: Similarity,
    matches: list[Match],
    prior_scale: float,
    radius_ft: float,
    image_wh: tuple[int, int],
) -> FitResult:
    """Accept the fit only if independent evidence agrees; name the failure if not."""
    inliers = [m for m in matches if m.dist_ft < INLIER_DIST_FT]
    if len(inliers) < MIN_INLIERS:
        return FitResult(transform, False, f"{len(inliers)} inliers", inliers)
    names = {m.label.text for m in inliers}
    if len(names) < MIN_DISTINCT_NAMES:
        return FitResult(transform, False, f"{len(names)} distinct names", inliers)
    normals = [m.seg_dir for m in inliers]
    span = max(angle_diff_mod_pi(a, b) for a in normals for b in normals)
    if span < MIN_NORMAL_SPAN_RAD:
        return FitResult(
            transform, False, f"normal span {math.degrees(span):.0f} deg", inliers
        )
    rms = math.sqrt(sum(m.dist_ft**2 for m in inliers) / len(inliers))
    if rms > MAX_INLIER_RMS_FT:
        return FitResult(transform, False, f"rms {rms:.0f} ft", inliers, rms)
    ratio = transform.scale_ft_per_px() / prior_scale
    if not (SCALE_PRIOR_BAND[0] <= ratio <= SCALE_PRIOR_BAND[1]):
        return FitResult(transform, False, f"scale {ratio:.2f}x prior", inliers, rms)
    w, h = image_wh
    cx, cy = transform.apply(w / 2, h / 2)
    if math.hypot(cx, cy) > MAX_CENTER_DRIFT_RADII * radius_ft:
        return FitResult(transform, False, "center drift", inliers, rms)
    xs = [m.label.center for m in inliers]
    spread = max(
        math.hypot(a[0] - b[0], a[1] - b[1]) for a in xs for b in xs
    ) / math.hypot(w, h)
    if spread < MIN_LABEL_SPREAD_FRAC:
        return FitResult(transform, False, f"spread {spread:.2f}", inliers, rms)
    return FitResult(transform, True, "ok", inliers, rms)


def fit_page(
    labels: list[tuple[LabelFeature, float]],
    blocks_xy: dict[str, list[np.ndarray]],
    image_wh: tuple[int, int],
    prior_scale_ft_per_px: float,
    radius_ft: float,
) -> FitResult:
    """Try each voted rotation (both half-turns); keep the best verified fit."""
    peaks = vote_rotation(labels, blocks_xy)
    if not peaks:
        return FitResult(None, False, "no rotation votes", [])
    center_px = (image_wh[0] / 2, image_wh[1] / 2)
    best = FitResult(None, False, "no candidate converged", [])
    final_tol = TOLERANCE_SCHEDULE_FT[-1]

    def tight_score(transform: Similarity) -> tuple[int, float]:
        """(match count, -mean distance) under the tightest tolerance."""
        tight = associate(transform, labels, blocks_xy, final_tol)
        mean = sum(m.dist_ft for m in tight) / len(tight) if tight else float("inf")
        return (len(tight), -mean)

    for peak in peaks:
        for theta in (peak, peak + math.pi):
            transform = Similarity.from_prior(prior_scale_ft_per_px, theta, center_px)
            # A wrong association in a coarse round can drag the solve away from a
            # good start (p5: 4 matches -> 2 -> 0), so keep the transform that scores
            # best at the tightest tolerance rather than blindly chaining rounds.
            best_transform, best_score = transform, tight_score(transform)
            for tol in TOLERANCE_SCHEDULE_FT:
                matches = associate(transform, labels, blocks_xy, tol)
                if len(matches) < 2:
                    break
                solved = solve_wls(
                    matches,
                    center_px,
                    position_sigma_ft=radius_ft,
                    prior_scale_ft_per_px=prior_scale_ft_per_px,
                )
                if solved is None:
                    break
                transform = solved
                score = tight_score(transform)
                if score > best_score:
                    best_transform, best_score = transform, score
            if best_score[0] < 2:
                continue
            # Re-associate at the final tolerance for verification.
            matches = associate(best_transform, labels, blocks_xy, final_tol)
            result = verify(
                best_transform, matches, prior_scale_ft_per_px, radius_ft, image_wh
            )
            better_key = (
                result.verified,
                len(result.inliers),
                -result.rms_ft if result.rms_ft == result.rms_ft else 0,
            )
            best_key = (
                best.verified,
                len(best.inliers),
                -best.rms_ft if best.rms_ft == best.rms_ft else 0,
            )
            if better_key > best_key:
                best = result
    return best


def load_labels(streets_path: str) -> list[tuple[LabelFeature, float]]:
    """Usable (LabelFeature, confidence) pairs from a streets.json."""
    doc = json.load(open(streets_path))
    raw = [
        d
        for d in doc.get("streets", [])
        if d.get("confidence", 0) >= 0.3
        and not d.get("hint")
        and not is_number_only(d.get("text", ""))
        and not is_bare_letter(d.get("text", ""))
    ]
    feats = label_features(raw)
    return [(f, float(d["confidence"])) for f, d in zip(feats, raw)]


def corners_lonlat(
    transform: Similarity, frame: Frame, w: int, h: int
) -> list[list[float]]:
    return [
        list(frame.to_lonlat(*transform.apply(u, v)))
        for u, v in ((0, 0), (w, 0), (w, h), (0, h))
    ]


def corners_to_affine(corners: list[list[float]], w: int, h: int) -> np.ndarray:
    """2x3 affine (lon, lat) = A @ (u, v, 1) from a georef.json corner quad."""
    src = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    dst = np.array(corners, dtype=float)
    coeffs, *_ = np.linalg.lstsq(src, dst, rcond=None)
    return coeffs.T


def rmse_vs_truth(
    affine: np.ndarray, truth_item: dict, local_wh: tuple[int, int]
) -> float:
    """RMSE (ft) between a local-pixel affine and a truth annotation over a 7x7 grid."""
    gcps = extract_gcps(truth_item)
    truth_affine = fit_transform(gcps, annotation_transform_type(truth_item))
    source = truth_item["target"]["source"]
    sx = source["width"] / local_wh[0]
    sy = source["height"] / local_wh[1]
    total = 0.0
    points = sample_grid(local_wh[0], local_wh[1], 7)
    for u, v in points:
        lon_a, lat_a = affine @ np.array([u, v, 1.0])
        lon_b, lat_b = truth_affine @ np.array([u * sx, v * sy, 1.0])
        total += haversine_ft(lat_a, lon_a, lat_b, lon_b) ** 2
    return math.sqrt(total / len(points))


def truth_by_stem(truth_path: Path) -> dict[str, dict]:
    """Unsplit truth annotations keyed by lowercase page stem (pN, pNw, pNn...).

    Split panels ("p73 [1]"-style labels) are skipped: their GCP pixels live in
    full-sheet coordinates while the local image is a crop, so a resolution ratio
    can't relate them (the official compare resolves this by polygon IOU).
    """
    out: dict[str, dict] = {}
    for _, items in annotations_by_source(truth_path).items():
        for item in items:
            label = str(item.get("label", ""))
            m = re.search(r"\bp(\d+[A-Za-z]?)$", label.strip())
            if m:
                out[f"p{m.group(1).lower()}"] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", type=Path)
    parser.add_argument("--keymap", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--pages", help="Comma-separated stems to restrict to.")
    args = parser.parse_args()

    locator = KeymapLocator.from_keymaps([Path(k) for k in args.keymap])
    wide = dataclasses.replace(locator, radius_m=locator.radius_m * 2)
    radius_ft = locator.radius_m * FT_PER_M
    geojson = json.load(open(args.volume / "centerlines.geojson"))

    # Volume scale bootstrap: median px/ft of the existing pipeline fits.
    scales = []
    for path in sorted(args.volume.glob("p*.georef.json")):
        doc = json.load(open(path))
        if "corners" not in doc:
            continue
        (ax, ay), (bx, by) = doc["corners"][0], doc["corners"][1]
        mid = math.radians((ay + by) / 2)
        dist_ft = math.hypot((bx - ax) * math.cos(mid), by - ay) * _FT_PER_DEG_LAT
        if dist_ft > 0:
            scales.append(doc["width"] / dist_ft)
    median_px_per_ft = float(np.median(scales))
    print(f"Volume scale bootstrap: {median_px_per_ft:.3f} px/ft ({len(scales)} fits)")

    # Median region prior for the half-scale multiplier.
    region_priors = []
    pages = [str(p) for p in list_pages(args.volume)]
    for image_path in pages:
        entry = locator.page_keymap(page_number(image_stem(image_path)))
        if entry and entry.get("regions"):
            prior = region_prior_px_per_ft(
                entry,
                *image_size(image_path),
                split_page=is_split_page(image_stem(image_path)),
            )
            if prior:
                region_priors.append(prior)
    median_region_prior = (
        float(np.median(region_priors)) if len(region_priors) >= 5 else None
    )

    only = set(args.pages.split(",")) if args.pages else None
    os.makedirs(args.out, exist_ok=True)
    truth = truth_by_stem(args.truth) if args.truth else {}

    print(
        f"{'page':10s} {'labels':>6s} {'inliers':>7s} {'rms_ft':>6s} "
        f"{'scale_r':>7s} {'verified':>8s}  {'proto_rmse':>10s} {'pipeline':>9s}  reason"
    )
    results = []
    summary_rows: list[dict] = []
    for image_path in pages:
        stem = image_stem(image_path)
        if only and stem not in only:
            continue
        number = page_number(stem)
        entry = locator.page_keymap(number)
        if entry is None:  # the key map doesn't place this page (or it IS the key map)
            continue
        streets_path = str(args.volume / f"{stem}.streets.json")
        if not os.path.exists(streets_path):
            continue
        labels = load_labels(streets_path)
        w, h = image_size(image_path)

        multiplier = 1.0
        if median_region_prior:
            prior = region_prior_px_per_ft(entry, w, h, split_page=is_split_page(stem))
            if prior:
                multiplier = region_relative_scale(prior, median_region_prior)
        prior_scale_ft_per_px = 1.0 / (median_px_per_ft * multiplier)

        best = FitResult(None, False, "no centers", [])
        best_frame = None
        for lon, lat in entry.get("centers", [[entry["lon"], entry["lat"]]]):
            frame = Frame.at(lon, lat)
            blocks_xy: dict[str, list[np.ndarray]] = {}
            restricted = wide.restricted_features(number, geojson["features"]) or []
            for name, blocks in build_block_index(
                {"type": "FeatureCollection", "features": restricted}
            ).items():
                blocks_xy[name] = [
                    np.array([frame.to_xy(lon_, lat_) for lon_, lat_ in blk.coords])
                    for blk in blocks
                ]
            result = fit_page(
                labels, blocks_xy, (w, h), prior_scale_ft_per_px, radius_ft
            )
            key = (result.verified, len(result.inliers))
            if best.transform is None or key > (best.verified, len(best.inliers)):
                best, best_frame = result, frame

        proto_rmse = pipeline_rmse = float("nan")
        if best.verified and best.transform and best_frame:
            corners = corners_lonlat(best.transform, best_frame, w, h)
            out_doc = {
                "width": w,
                "height": h,
                "corners": corners,
                "streets": [
                    {"text": m.label.raw_text, "inlier": True} for m in best.inliers
                ],
                "intersections": [],
                "keymap": entry,
                "prototype": {
                    "rms_ft": round(best.rms_ft, 1),
                    "inliers": len(best.inliers),
                    "scale_ratio": round(
                        best.transform.scale_ft_per_px()
                        * median_px_per_ft
                        * multiplier,
                        3,
                    ),
                },
            }
            json.dump(out_doc, open(args.out / f"{stem}.georef.json", "w"))
            if stem in truth:
                proto_rmse = rmse_vs_truth(
                    corners_to_affine(corners, w, h), truth[stem], (w, h)
                )
        if stem in truth:
            existing = args.volume / f"{stem}.georef.json"
            if existing.exists():
                doc = json.load(open(existing))
                if "corners" in doc:
                    pipeline_rmse = rmse_vs_truth(
                        corners_to_affine(doc["corners"], w, h), truth[stem], (w, h)
                    )
        scale_ratio = (
            best.transform.scale_ft_per_px() / prior_scale_ft_per_px
            if best.transform
            else float("nan")
        )
        print(
            f"{stem:10s} {len(labels):6d} {len(best.inliers):7d} {best.rms_ft:6.0f} "
            f"{scale_ratio:7.2f} {str(best.verified):>8s}  {proto_rmse:10.1f} {pipeline_rmse:9.1f}  {best.reason}"
        )
        results.append((stem, best.verified, proto_rmse, pipeline_rmse))
        summary_rows.append(
            {
                "stem": stem,
                "labels": len(labels),
                "inliers": len(best.inliers),
                "rms_ft": None if best.rms_ft != best.rms_ft else round(best.rms_ft, 1),
                "verified": best.verified,
                "reason": best.reason,
                "proto_rmse": None
                if proto_rmse != proto_rmse
                else round(proto_rmse, 1),
                "pipeline_rmse": None
                if pipeline_rmse != pipeline_rmse
                else round(pipeline_rmse, 1),
            }
        )

    json.dump(summary_rows, open(args.out / "summary.json", "w"), indent=1)
    verified = [r for r in results if r[1]]
    print(f"\nVerified fits: {len(verified)}/{len(results)}")
    both = [(p, x) for _, v, p, x in results if v and p == p and x == x]
    if both:
        protos = [p for p, _ in both]
        pipes = [x for _, x in both]
        print(
            f"Pages with truth + both fits ({len(both)}): "
            f"proto median {float(np.median(protos)):.1f} ft vs pipeline {float(np.median(pipes)):.1f} ft"
        )


if __name__ == "__main__":
    main()
