"""Georeference pages by matching their P(road) map to the key map's (#211).

A key map draws the volume's whole street network in one georeferenced sheet.
Every page is a rectangle somewhere on it, so a page whose own street OCR failed
can still be placed: extract P(road) for the page and for the key map, and slide
the page's roads over the key map's until they line up.

The search is over one scale and one rotation -- an isotropic similarity --
which is only correct in a frame where a correctly-placed page really is one.
Key-map georeferences are not that frame: Detroit's is 10.6% anisotropic, so a
page in its right place comes out visibly stretched in key-map pixels and the
right answer sits *outside* the search space. Measured on Detroit's seven
never-otherwise-fitted pages, the best isotropic pose was still 25.5 ft from
truth (median) against 52.6 ft actually achieved -- about half the error was the
search space being the wrong shape.

So the crop is first warped into a frame where the constraint holds. Polar-
decompose the key-map model's local linear part ``L = R S`` (S symmetric
positive definite) and warp by ``W = S / sqrt(det S)``::

    p_world = L p_keymap,  p_warp = W p_keymap
      =>  p_world = L W^-1 p_warp = sqrt(det S) * R p_warp

a pure rotation and scalar, exactly the family being searched. W has unit
determinant, so the crop is neither rescaled nor flipped, and the pose is
carried back through ``W^-1`` afterwards. No search dimension is added.

The key-map model is a thin-plate spline through the sheet's own inlier
intersections, so it has no single linear part; the Jacobian is taken at each
page's own region centroid and the spatial variation is followed to first order.
That local tangent is worth a little on its own (p75 142 -> 110 ft) but the
stretch is what moves the median: 60.4 -> 44.0 ft over 93 Detroit pages, with
pages within 25 ft going 5 -> 21.
"""

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mapsnap.keymap.road_prob import MIN_KEYMAP_INLIERS
from mapsnap.road_model import page_world_affine

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQUATOR = 111_320.0

SEARCH_SLACK_PX = 600
"""How far outside the page's expected footprint the crop reaches. The region
centroid is a good but not exact guess at where the page sits."""

PRIOR_SIGMA_PX = 260.0
"""Width of the soft prior pulling the match toward the page's key-map region.
Wide enough not to veto a real offset, tight enough to break grid aliasing."""

SCALE_LADDER = (0.94, 1.0, 1.06)
THETA_JITTER_DEG = (-3.0, 0.0, 3.0)
"""Search rungs around the volume's median page scale and rotation. The jitter
is deliberately narrow: it is a refinement of a prior, not a search for the
rotation. Pages far off the volume median (Detroit has a few at 90 degrees)
need a better prior, not a wider ladder -- widening it admits aliases."""

PEAK_SUPPRESS_PX = 70
"""Radius blanked around the winning peak before measuring the runner-up, so
the margin compares distinct placements rather than one peak against itself."""

TPS_SMOOTHING = 1e6
"""Thin-plate spline regularization. The intersections carry real residuals, so
an interpolating spline would chase them; this keeps the sheet-scale warp."""

STRETCH_PAD = 1.12
"""Crop padding for the unit-determinant stretch, which lengthens one axis."""


@dataclass(frozen=True)
class SnapTarget:
    """Where a page is expected on the key map, and at what scale and rotation.

    The scale and rotation are the volume's medians over its already-fitted
    pages: pages in one volume are printed at one scale and near one rotation,
    which is what makes a three-rung ladder enough.
    """

    centre: tuple[float, float]
    m_per_px: float
    theta_deg: float


@dataclass(frozen=True)
class Match:
    """One page placed on the key map."""

    to_keymap: np.ndarray
    score: float
    margin: float
    anisotropy_pct: float


def thin_plate_spline(
    source: np.ndarray, destination: np.ndarray, smoothing: float = TPS_SMOOTHING
) -> Callable[[np.ndarray], np.ndarray]:
    """Smoothed thin-plate spline mapping `source` points onto `destination`."""
    count = len(source)
    squared = ((source[:, None, :] - source[None, :, :]) ** 2).sum(-1)
    kernel = np.where(
        squared > 0, squared * 0.5 * np.log(np.maximum(squared, 1e-12)), 0.0
    ) + smoothing * np.eye(count)
    poly = np.column_stack([np.ones(count), source])
    system = np.zeros((count + 3, count + 3))
    system[:count, :count] = kernel
    system[:count, count:] = poly
    system[count:, :count] = poly.T
    rhs = np.zeros((count + 3, 2))
    rhs[:count] = destination
    solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
    weights, affine_part = solution[:count], solution[count:]

    def evaluate(query: np.ndarray) -> np.ndarray:
        squared_q = ((query[:, None, :] - source[None, :, :]) ** 2).sum(-1)
        kernel_q = np.where(
            squared_q > 0, squared_q * 0.5 * np.log(np.maximum(squared_q, 1e-12)), 0.0
        )
        return (
            kernel_q @ weights
            + np.column_stack([np.ones(len(query)), query]) @ affine_part
        )

    return evaluate


def keymap_model(
    georef: dict, min_gcps: int = MIN_KEYMAP_INLIERS
) -> Callable[[np.ndarray], np.ndarray]:
    """Key-map pixel -> lon/lat, as a spline through the sheet's own GCPs.

    A key map's shipped affine is a global fit; its residuals against the
    sheet's inlier intersections run to ~100 ft on Detroit, and that error is
    the floor under every page placed through it. A spline through those same
    intersections follows the sheet. Falls back to the corner affine when the
    sheet has too few of them to constrain a warp.
    """
    inliers = [i for i in georef.get("intersections", []) if i.get("inlier")]
    affine = page_world_affine(georef)
    if len(inliers) < min_gcps:
        return lambda query: (
            np.column_stack([query[:, 0], query[:, 1], np.ones(len(query))]) @ affine.T
        )

    pixels = np.array([[i["x"], i["y"]] for i in inliers], float)
    lonlat = np.array([[i["lon"], i["lat"]] for i in inliers], float)
    origin = lonlat[0]
    lon_scale = M_PER_DEG_LON_EQUATOR * math.cos(
        math.radians(float(lonlat[:, 1].mean()))
    )
    metres = np.column_stack(
        [
            (lonlat[:, 0] - origin[0]) * lon_scale,
            (lonlat[:, 1] - origin[1]) * M_PER_DEG_LAT,
        ]
    )
    spline = thin_plate_spline(pixels, metres)

    def to_world(query: np.ndarray) -> np.ndarray:
        out = spline(query)
        return np.column_stack(
            [out[:, 0] / lon_scale + origin[0], out[:, 1] / M_PER_DEG_LAT + origin[1]]
        )

    return to_world


def local_tangent(
    model: Callable[[np.ndarray], np.ndarray],
    centre: tuple[float, float],
    step: float = 200.0,
) -> np.ndarray:
    """2x3 key-map-px -> lon/lat affine tangent to `model` at `centre`."""
    probe = np.array(
        [centre, [centre[0] + step, centre[1]], [centre[0], centre[1] + step]], float
    )
    world = model(probe)
    along_x = (world[1] - world[0]) / step
    along_y = (world[2] - world[0]) / step
    return np.array(
        [[along_x[0], along_y[0], world[0, 0]], [along_x[1], along_y[1], world[0, 1]]]
    )


def affine_m_per_px(affine: np.ndarray) -> float:
    """Ground metres per pixel implied by a pixel -> lon/lat affine."""
    return math.hypot(affine[1, 0], affine[1, 1]) * M_PER_DEG_LAT


def affine_theta_deg(affine: np.ndarray) -> float:
    """Rotation in degrees of a pixel -> lon/lat affine."""
    return math.degrees(math.atan2(-affine[1, 0], affine[0, 0]))


def linear_part_metres(affine: np.ndarray, latitude: float) -> np.ndarray:
    """The 2x2 linear part of a pixel -> lon/lat affine, in metres per pixel."""
    lon_scale = M_PER_DEG_LON_EQUATOR * math.cos(math.radians(latitude))
    return np.array(
        [
            [affine[0, 0] * lon_scale, affine[0, 1] * lon_scale],
            [affine[1, 0] * M_PER_DEG_LAT, affine[1, 1] * M_PER_DEG_LAT],
        ]
    )


def unit_stretch(linear: np.ndarray) -> tuple[np.ndarray, float]:
    """The shape distortion in `linear`, as a unit-determinant stretch.

    Polar decomposition splits a linear map into a rotation and a symmetric
    stretch, ``L = R S``. Only S is the problem: the matcher searches rotations
    already, but not stretches. Normalizing S to unit determinant isolates the
    shape without touching overall scale, so warping by it neither resizes nor
    flips what it is applied to.

    Returns (stretch, anisotropy percent).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(linear.T @ linear)
    stretch = (
        eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12))) @ eigenvectors.T
    )
    determinant = float(np.linalg.det(stretch))
    singular = np.linalg.svd(linear, compute_uv=False)
    anisotropy = 100.0 * (singular[0] / max(singular[1], 1e-12) - 1.0)
    return stretch / math.sqrt(max(determinant, 1e-12)), float(anisotropy)


def as_3x3(matrix: np.ndarray) -> np.ndarray:
    """Promote a 2x3 affine to a 3x3 homogeneous matrix."""
    return np.vstack([matrix, [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class StretchedCrop:
    """A key-map crop warped into the frame where a correct page is isotropic."""

    image: np.ndarray
    zero_centred: np.ndarray
    centre: tuple[float, float]
    m_per_px: float
    theta_deg: float
    to_keymap: np.ndarray
    anisotropy_pct: float


def stretched_crop(
    probability: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    target: SnapTarget,
    page_shape: tuple[int, int],
) -> StretchedCrop | None:
    """Cut the key map around where a page belongs and undo the sheet's stretch.

    Returns None when the region falls off the sheet or is too small to match.
    """
    height, width = page_shape
    centre_x, centre_y = target.centre
    tangent = local_tangent(model, target.centre)
    stretch, anisotropy = unit_stretch(
        linear_part_metres(tangent, latitude=float(tangent[1, 2]))
    )

    nominal = affine_m_per_px(tangent)
    if nominal <= 0:
        return None
    expected = target.m_per_px / nominal
    span_x, span_y = int(width * expected), int(height * expected)
    x0 = max(0, int(centre_x - span_x / 2 - SEARCH_SLACK_PX))
    x1 = min(probability.shape[1], int(centre_x + span_x / 2 + SEARCH_SLACK_PX))
    y0 = max(0, int(centre_y - span_y / 2 - SEARCH_SLACK_PX))
    y1 = min(probability.shape[0], int(centre_y + span_y / 2 + SEARCH_SLACK_PX))
    crop = probability[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape) < 60:
        return None

    crop_h, crop_w = crop.shape
    out_w, out_h = int(crop_w * STRETCH_PAD), int(crop_h * STRETCH_PAD)
    source_centre = np.array([crop_w / 2.0, crop_h / 2.0])
    target_centre = np.array([out_w / 2.0, out_h / 2.0])
    warp = np.column_stack([stretch, target_centre - stretch @ source_centre])
    warped = cv2.warpAffine(crop, warp, (out_w, out_h))

    crop_to_keymap = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0]])
    to_keymap = (as_3x3(crop_to_keymap) @ np.linalg.inv(as_3x3(warp)))[:2, :]
    warped_georef = (as_3x3(tangent) @ as_3x3(to_keymap))[:2, :]

    blurred = cv2.GaussianBlur(warped, (0, 0), 2.0)
    moved_centre = warp[:, :2] @ np.array([centre_x - x0, centre_y - y0]) + warp[:, 2]
    return StretchedCrop(
        image=warped,
        zero_centred=blurred - cv2.blur(blurred, (91, 91)),
        centre=(float(moved_centre[0]), float(moved_centre[1])),
        m_per_px=affine_m_per_px(warped_georef),
        theta_deg=affine_theta_deg(warped_georef),
        to_keymap=to_keymap,
        anisotropy_pct=anisotropy,
    )


def match_page(
    crop: StretchedCrop, probability: np.ndarray, target: SnapTarget
) -> Match | None:
    """Slide a page's P(road) over a stretched key-map crop; best pose or None."""
    height, width = probability.shape
    if crop.m_per_px <= 0:
        return None
    base = target.m_per_px / crop.m_per_px
    best: Match | None = None

    for rung in SCALE_LADDER:
        factor = base * rung
        template_w, template_h = int(width * factor), int(height * factor)
        if template_w < 8 or template_h < 8:
            continue
        resized = cv2.GaussianBlur(
            cv2.resize(
                probability, (template_w, template_h), interpolation=cv2.INTER_AREA
            ),
            (0, 0),
            2.0,
        )
        for jitter in THETA_JITTER_DEG:
            theta = target.theta_deg - crop.theta_deg + jitter
            rotation = cv2.getRotationMatrix2D(
                (template_w / 2.0, template_h / 2.0), theta, 1.0
            )
            rotated = cv2.warpAffine(resized, rotation, (template_w, template_h))
            mass = float(rotated.sum())
            if mass < 1e-3:
                continue
            if (
                rotated.shape[0] >= crop.image.shape[0] - 2
                or rotated.shape[1] >= crop.image.shape[1] - 2
            ):
                continue

            response = (
                cv2.matchTemplate(crop.zero_centred, rotated, cv2.TM_CCORR) / mass
            )
            rows, cols = np.mgrid[0 : response.shape[0], 0 : response.shape[1]]
            offset_x = cols + template_w / 2.0 - crop.centre[0]
            offset_y = rows + template_h / 2.0 - crop.centre[1]
            weighted = response * np.exp(
                -0.5 * (offset_x**2 + offset_y**2) / PRIOR_SIGMA_PX**2
            )
            _, peak, _, location = cv2.minMaxLoc(weighted)
            if not math.isfinite(peak) or (best is not None and peak <= best.score):
                continue

            suppressed = weighted.copy()
            y0 = max(0, location[1] - PEAK_SUPPRESS_PX)
            y1 = min(suppressed.shape[0], location[1] + PEAK_SUPPRESS_PX)
            x0 = max(0, location[0] - PEAK_SUPPRESS_PX)
            x1 = min(suppressed.shape[1], location[0] + PEAK_SUPPRESS_PX)
            suppressed[y0:y1, x0:x1] = -1e9
            runner_up = float(suppressed.max())

            in_crop = (
                np.array([[1.0, 0, location[0]], [0, 1.0, location[1]], [0, 0, 1.0]])
                @ as_3x3(rotation)
                @ np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1.0]])
            )
            best = Match(
                to_keymap=(as_3x3(crop.to_keymap) @ in_crop)[:2, :],
                score=float(peak),
                margin=float((peak - runner_up) / max(abs(peak), 1e-9)),
                anisotropy_pct=crop.anisotropy_pct,
            )
    return best


def page_world_affine_from_match(
    match: Match,
    model: Callable[[np.ndarray], np.ndarray],
    page_shape: tuple[int, int],
) -> np.ndarray:
    """2x3 page-px -> lon/lat affine for a matched page.

    The key-map model is a spline, so the composition is not affine; over one
    page footprint it is near enough that a least-squares affine is what a
    downstream consumer effectively sees, and georef sidecars store corners.
    """
    height, width = page_shape
    grid_x, grid_y = np.meshgrid(np.linspace(0, width, 6), np.linspace(0, height, 6))
    page_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    keymap_points = np.column_stack(
        [
            match.to_keymap[0, 0] * page_points[:, 0]
            + match.to_keymap[0, 1] * page_points[:, 1]
            + match.to_keymap[0, 2],
            match.to_keymap[1, 0] * page_points[:, 0]
            + match.to_keymap[1, 1] * page_points[:, 1]
            + match.to_keymap[1, 2],
        ]
    )
    design = np.column_stack(
        [page_points[:, 0], page_points[:, 1], np.ones(len(page_points))]
    )
    coefficients, *_ = np.linalg.lstsq(design, model(keymap_points), rcond=None)
    return coefficients.T


def affine_corners(
    affine: np.ndarray, page_shape: tuple[int, int]
) -> list[list[float]]:
    """The four page corners in lon/lat, in georef sidecar order."""
    height, width = page_shape
    return [
        [
            float(affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]),
            float(affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]),
        ]
        for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]


def volume_pose_medians(volume: Path) -> tuple[float, float] | None:
    """Median scale and rotation over a volume's already-fitted pages."""
    scales, thetas = [], []
    for page in sorted(volume.glob("p*.jpg")):
        if "__" in page.stem:
            continue
        for channel in ("georef-streets", "georef-osm", "georef"):
            path = volume / f"{page.stem}.{channel}.json"
            if path.exists():
                affine = page_world_affine(json.loads(path.read_text()))
                scales.append(affine_m_per_px(affine))
                thetas.append(affine_theta_deg(affine))
                break
    if not scales:
        return None
    return float(np.median(scales)), float(np.median(thetas))


def snap_volume(volume: Path, output_dir: Path, min_margin: float = 0.0) -> list[dict]:
    """Place every page that has a key-map region and a P(road) map.

    Writes one georef sidecar per placed page into `output_dir` and returns a
    row per page for reporting.
    """
    from mapsnap.edge_join_experiment import load_prob

    medians = volume_pose_medians(volume)
    if medians is None:
        print(f"{volume.name}: no fitted pages to take a scale from", file=sys.stderr)
        return []
    volume_m_per_px, volume_theta = medians

    rows: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for georef_path in sorted((volume / "raw").glob("*.georef.json")):
        if ".truth." in georef_path.name:
            continue
        stem = georef_path.name[: -len(".georef.json")]
        regions_path = volume / "raw" / f"{stem}.regions.panels.json"
        prob_path = volume / "raw" / f"{stem}.roadprob.png"
        if not regions_path.exists() or not prob_path.exists():
            continue

        keymap_image = cv2.imread(str(prob_path), cv2.IMREAD_GRAYSCALE)
        if keymap_image is None:
            print(f"{stem}: unreadable P(road) map, skipping", file=sys.stderr)
            continue

        georef = json.loads(georef_path.read_text())
        model = keymap_model(georef)
        keymap_prob = keymap_image.astype(np.float32) / 255.0
        regions = json.loads(regions_path.read_text())
        labels = [str(label) for label in regions["labels"]]

        for page in sorted(volume.glob("p*.jpg")):
            if "__" in page.stem or page.stem[1:] not in labels:
                continue
            probability = load_prob(volume, page.stem)
            if probability is None:
                continue
            panel = np.asarray(regions["panels"][labels.index(page.stem[1:])], float)
            target = SnapTarget(
                centre=(float(panel[:, 0].mean()), float(panel[:, 1].mean())),
                m_per_px=volume_m_per_px,
                theta_deg=volume_theta,
            )
            crop = stretched_crop(keymap_prob, model, target, probability.shape)
            if crop is None:
                continue
            match = match_page(crop, probability, target)
            if match is None or match.margin < min_margin:
                continue

            affine = page_world_affine_from_match(match, model, probability.shape)
            (output_dir / f"{page.stem}.georef.json").write_text(
                json.dumps(
                    {
                        "width": probability.shape[1],
                        "height": probability.shape[0],
                        "corners": affine_corners(affine, probability.shape),
                        "intersections": [],
                        "source": f"keymap-snap from {stem} (#211)",
                        "match_margin": round(match.margin, 3),
                        "keymap_anisotropy_pct": round(match.anisotropy_pct, 2),
                    },
                    indent=2,
                )
            )
            rows.append(
                {
                    "stem": page.stem,
                    "keymap": stem,
                    "margin": match.margin,
                    "anisotropy_pct": match.anisotropy_pct,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place pages by matching their P(road) map to the key map's."
    )
    parser.add_argument("volume", type=Path, help="Volume directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write georef sidecars (default: <volume>/kmsnap)",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Discard matches whose peak barely beats the runner-up",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.volume / "kmsnap")
    rows = snap_volume(args.volume, output_dir, min_margin=args.min_margin)
    if not rows:
        print("No pages placed.", file=sys.stderr)
        return
    anisotropy = float(np.median([row["anisotropy_pct"] for row in rows]))
    margins = sorted(row["margin"] for row in rows)
    print(
        f"Placed {len(rows)} page(s) into {output_dir}. "
        f"Median match margin {margins[len(margins) // 2]:.3f}; "
        f"median key-map anisotropy {anisotropy:.1f}% (corrected)."
    )


if __name__ == "__main__":
    main()
