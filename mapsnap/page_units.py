"""A volume's pages as the fit channels see them: PageUnit and its loaders.

Each page (or split panel) with its image size, the RANSAC fit state and pose
the pipeline published for it, runner-up poses, and -- when the volume has
OIM truth -- the truth pose, used only for diagnostics. Shared by snap,
street-solve and reconcile, with a few volume-level helpers: P(road) map
loading, detected adjacency pairs, key-map region adjacency, and pose
comparison in feet.
"""

import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mapsnap import sidecar
from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    extract_gcps,
    fit_transform,
    redundant_skeleton_keys,
    sample_grid,
)
from mapsnap.keymap.fit_keymap import page_number
from mapsnap.keymap.records import recorded_keymap_keys
from mapsnap.utils import (
    FEET_PER_METER,
    haversine_m,
    jpeg_dimensions,
    source_id_to_page_key,
    source_images,
)

GEOREF_VARIANTS = [
    "georef",
    "georef-nofit",
    "georef-misscale",
    "georef-1gcp",
    "georef-outlier",
]


ANCHOR_RMSE_FT = 25.0


@dataclass
class TruthFit:
    """A page's truth transform, expressed in local working-jpg pixel space."""

    affine_local: np.ndarray  # 2x3: local px -> (lon, lat)
    gcp_count: int
    transform_type: str


def runner_up_affines_of(georef: dict | None, width: int, height: int) -> list:
    """Affines for a fitted sidecar's runner_up_poses entries (empty if none).

    Shared by both PageUnit loaders (this module's and snap_volume's
    panel loader) so the runner-up channel cannot fall out of sync between
    base pages and split panels.
    """
    from mapsnap.road_model import page_world_affine

    if not georef:
        return []
    affines = []
    for pose in georef.get("runner_up_poses") or []:
        try:
            affines.append(
                page_world_affine(
                    {"corners": pose["corners"], "width": width, "height": height}
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return affines


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
    # A pose its channel produced and DECLINED (misscale/outlier/contradicted
    # sidecar with corners). Never an incumbent; #315 uses it only to seed
    # snap's search, because a demoted pose is measurably a good init (refine
    # fixed richmond p353's 3.06x scale error from one).
    demoted_affine: np.ndarray | None = None
    # For one half of a sheet scanned as two (pNL/pNR) whose other half is
    # fitted: that half's pose, shifted across the gutter. Seeds snap's rescue
    # (see snap_volume.attach_half_sheet_seeds).
    sibling_affine: np.ndarray | None = None
    # Clustered candidate-GCP world positions from a FAILED fit (#335):
    # location evidence snap's rescue uses as search centers, recorded in the
    # nofit sidecar. Empty for fitted pages.
    gcp_hints: list[tuple[float, float]] = field(default_factory=list)
    # Distinct near-tie poses RANSAC scored but did not pick (#340): consumed
    # by snap as extra challenge seeds when the incumbent verifies poorly.
    runner_up_affines: list[np.ndarray] = field(default_factory=list)
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
        errors.append((haversine_m(lat_a, lon_a, lat_b, lon_b) * FEET_PER_METER) ** 2)
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

    Skeleton sheets ('s' suffix) with a full-color truth counterpart map the
    same ground, so only one of the pair is kept: the skeleton when it alone
    has a georef fit, the full-color page otherwise (matching compare_pages
    and make_iiif_georef).
    """
    truth_path = volume / "main.iiif.json"
    if not truth_path.exists():
        # No truth data: production placement still works, all truth-derived
        # diagnostics (rmse_ft, anchor_truth) are simply absent.
        return {}, set()
    data = json.loads(truth_path.read_text())
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
    fitted = {k for k in unsplit if (volume / f"{k}.georef.json").exists()}
    for key in redundant_skeleton_keys(set(unsplit), fitted):
        del unsplit[key]
    return unsplit, split_parents


# Verdicts that GEOREF_VARIANTS never listed, so a page carrying one used to
# fall through page_fit_state to ("none", None): rescue-eligible, and denied the
# key-map centers in its own sidecar. Preserved here deliberately -- collapsing
# the variants into one file must not quietly change what snap searches (#270
# phase 3). Withholding a key-map prior from a page demoted FOR being far from
# its key map is worth revisiting, but as its own measured change.
LEGACY_UNLISTED_VERDICTS = ("keymap-outlier", "contradicted")


def page_fit_state(volume: Path, stem: str) -> tuple[str, dict | None]:
    """(fit state, georef JSON) for a base page stem.

    The state is the verdict recorded in ``p<stem>.georef.json``; "fitted"
    means georef stands behind the pose. Older volumes encoded the verdict as a
    filename suffix instead, so those are still recognized.
    """
    path = volume / f"{stem}.georef.json"
    if path.exists():
        doc = json.loads(path.read_text())
        state = sidecar.status(doc)
        if state in LEGACY_UNLISTED_VERDICTS:
            return "none", None
        return ("fitted" if state == sidecar.VALID else state), doc
    for variant in GEOREF_VARIANTS[1:]:  # legacy renamed-aside layout
        path = volume / f"{stem}.{variant}.json"
        if path.exists():
            return variant.removeprefix("georef-"), json.loads(path.read_text())
    # Split pieces (p239__1.georef*.json) mean the base page was split.
    if list(volume.glob(f"{stem}__*.georef*.json")):
        return "split", None
    return "none", None


def load_page_units(volume: Path) -> list[PageUnit]:
    """All base pages of the volume, with truth/generated fits attached."""
    from mapsnap.road_model import page_world_affine

    truth_by_key, split_truth_parents = load_truth_units(volume)
    # A recorded key map is georeferenced by the key-map chain, never as a page (#542).
    keymaps = recorded_keymap_keys(volume)
    units: list[PageUnit] = []
    for jpg in source_images(volume):
        stem = jpg.stem
        if "__" in stem or stem in keymaps:
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
        demoted_affine = None
        if georef is not None:
            if state == "fitted":
                gen_affine = page_world_affine(georef)
            else:
                # #315: a declined pose (misscale/outlier/contradicted) still
                # anchors snap's rescue search; nofit-style docs have no
                # corners and stay None.
                try:
                    demoted_affine = page_world_affine(georef)
                except (KeyError, TypeError, ValueError):
                    demoted_affine = None
            inlier_int = sum(
                1 for i in georef.get("intersections", []) if i.get("inlier")
            )
            inlier_str = sum(1 for s in georef.get("streets", []) if s.get("inlier"))
            keymap = georef.get("keymap") or {}
            keymap_centers = [tuple(c) for c in keymap.get("centers", [])]
            keymap_radius = float(keymap.get("radius_m") or 0.0)
            keymap_regions = keymap.get("regions") or None
            gcp_hints = [tuple(h) for h in georef.get("gcp_hints") or []]
        else:
            gcp_hints = []

        unit = PageUnit(
            stem=stem,
            number=number,
            width=width,
            height=height,
            fit_state=state,
            truth=truth,
            split_truth=stem in split_truth_parents,
            gen_affine=gen_affine,
            demoted_affine=demoted_affine,
            gcp_hints=gcp_hints,
            inlier_intersections=inlier_int,
            inlier_streets=inlier_str,
            keymap_centers=keymap_centers,
            keymap_radius_m=keymap_radius,
            keymap_regions=keymap_regions,
            runner_up_affines=runner_up_affines_of(georef, width, height),
        )
        if truth is not None and gen_affine is not None:
            unit.rmse_ft = grid_rmse_ft_between(
                truth.affine_local, gen_affine, width, height
            )
            unit.anchor_truth = unit.rmse_ft <= ANCHOR_RMSE_FT
        unit.anchor_free = state == "fitted" and inlier_int >= 3
        units.append(unit)
    return units


def adjacency_number_pairs(volume: Path, doc_key: str) -> set[frozenset[int]]:
    """An adjacency.json edge list as unordered page-number pairs, or empty if absent."""
    path = volume / "adjacency.json"
    if not path.exists():
        return set()
    doc = json.loads(path.read_text())
    pairs: set[frozenset[int]] = set()
    for a, b in doc.get(doc_key, []):
        na, nb = page_number(a), page_number(b)
        if na is not None and nb is not None and na != nb:
            pairs.add(frozenset((na, nb)))
    return pairs


def detected_pairs(volume: Path) -> set[frozenset[int]]:
    """Mutual adjacency edges as unordered page-number pairs, or empty if absent."""
    return adjacency_number_pairs(volume, "adjacency")


def keymap_region_adjacency(
    volume: Path, gap_m: float = 40.0
) -> tuple[set[frozenset[int]], dict[int, tuple[float, float]]]:
    """Candidate adjacency pairs from key-map region proximity, and region centroids.

    A key map draws one colored block per page; two pages whose blocks nearly
    touch (segmented-region polygon distance <= ``gap_m``) are candidate
    neighbors. Unlike printed-number adjacency this needs no OCR and covers
    every page the key map places, so it is far denser (LA: ~58% of truth pairs
    at 40m vs the printed graph's ~21%). Returns the pair set and each page
    number's region centroid (world lon/lat), used to derive the anchor-side
    direction the printed number would otherwise supply.
    """
    from shapely.geometry import Polygon
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import unary_union

    from mapsnap.keymap.locate import KeymapLocator, usable_keymaps

    keymaps = usable_keymaps(volume / "raw")
    if not keymaps:
        return set(), {}
    regions = KeymapLocator.from_keymaps(keymaps).regions_by_number()
    if not regions:
        return set(), {}
    all_pts = [pt for rings in regions.values() for ring in rings for pt in ring]
    lon0 = sum(p[0] for p in all_pts) / len(all_pts)
    lat0 = sum(p[1] for p in all_pts) / len(all_pts)
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    shapes: dict[int, BaseGeometry] = {}
    centroids: dict[int, tuple[float, float]] = {}
    for number, rings in regions.items():
        polys = [
            Polygon(
                [((lon - lon0) * kx, (lat - lat0) * ky) for lon, lat in ring]
            ).buffer(0)
            for ring in rings
            if len(ring) >= 3
        ]
        polys = [p for p in polys if not p.is_empty]
        if not polys:
            continue
        shape = unary_union(polys)
        centroid = shape.centroid
        # Every ring above can be non-empty and their union still have no
        # centroid (on Linux x86 one Bonham key-map region did; the same inputs
        # pass on macOS, so the degenerate case is numerical). One page's region
        # is not worth failing fit for the volume (#514): skip it, as a page
        # with no usable rings is skipped.
        if shape.is_empty or centroid.is_empty:
            continue
        shapes[number] = shape
        centroids[number] = (lon0 + centroid.x / kx, lat0 + centroid.y / ky)
    numbers = sorted(shapes)
    pairs: set[frozenset[int]] = set()
    for i, a in enumerate(numbers):
        for b in numbers[i + 1 :]:
            if shapes[a].distance(shapes[b]) <= gap_m:
                pairs.add(frozenset((a, b)))
    return pairs, centroids


def load_prob(volume: Path, stem: str) -> np.ndarray | None:
    """A cached P(road) map in [0,1], or None.

    Thin wrapper over :func:`mapsnap.roadprob.load_roadprob` keyed the way this
    module's callers hold a page (volume plus stem) rather than by image path.
    """
    from mapsnap.roadprob import load_roadprob

    return load_roadprob(volume / f"{stem}.jpg")


# Median metres per pixel over the 2,540 fitted pages of the truth volumes
# (p10 0.174, p90 0.312) -- the 50 ft-to-the-inch rung at a typical 25% scan.
# Used only when a volume offers no fitted page at all, so that such a volume
# still runs to completion and records its abstentions instead of crashing.
FALLBACK_M_PER_PX = 0.203


def fitted_scales(units: list[PageUnit]) -> list[float]:
    """Metres per pixel for each unit the pipeline actually fitted."""
    return [
        affine_scale_m_per_px(u.gen_affine)
        for u in units
        if u.fit_state == "fitted" and u.gen_affine is not None
    ]


def volume_median_scale(
    units: list[PageUnit], panels: list[PageUnit] | None = None
) -> float:
    """Median RANSAC-fit scale (metres per page pixel).

    Deliberately truth-free: the volume scale must come from the pipeline's
    own fits so the matcher and pose graph run identically on volumes without
    ground truth.

    Base pages first, panels only if no base page was fitted. A panel is a crop
    of its parent at the same pixel scale, so the two are directly comparable --
    measured across the truth volumes they agree to the third decimal on all but
    the volumes with genuine half-scale sheets. Preferring base pages therefore
    leaves every volume that has one unchanged, while a volume whose only sheet
    is split still has a scale: Gardiner NY 1913 is one sheet cut into two
    panels, so every base page is ``split``, none is ``fitted``, and the median
    was taken over an empty list.
    """
    scales = fitted_scales(units)
    if not scales and panels:
        scales = fitted_scales(panels)
    if not scales:
        print(
            "No fitted page in this volume; assuming the corpus-median scale "
            f"{FALLBACK_M_PER_PX} m/px.",
            file=sys.stderr,
        )
        return FALLBACK_M_PER_PX
    return statistics.median(scales)
