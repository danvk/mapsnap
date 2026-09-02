"""Synthetic-fixture tests for the geometry-first OSM matcher.

Convention bugs (cv2 theta sign, y-down frames, lon/lat anisotropy) are the
costliest failure class here, so every prior rung and the frame math get a
round-trip test against a synthetic world with known geometry before any
real-data use.
"""

import math

import cv2
import numpy as np
import pytest

from mapsnap.edge_join import MatchParams
from mapsnap.feature_index import FeatureIndex
from mapsnap.georef_from_labels import LabelFeature
from mapsnap.osm_snap import (
    CHAMFER_CLAMP_M,
    NAME_MISS_MIN_LABELS,
    W_CONTAIN,
    W_NAME,
    W_PRIOR,
    NameAlignment,
    PageContext,
    RotationPrior,
    ScalePrior,
    SnapCandidate,
    adjacency_keymap_rotations,
    affine_theta_deg,
    calibrated_radius_m,
    cluster_rotation,
    confident_theta_deg,
    dedupe_thetas,
    directed_prior_residual_sigma,
    frame_around,
    frame_thetas,
    label_osm_rotations,
    merge_candidates,
    name_alignment,
    name_evidence,
    name_evidence_of,
    osm_distance_m,
    osm_rasters,
    page_scale_priors,
    pose_theta_deg,
    selection_score,
    snap_page,
    wrap_deg,
)
from mapsnap.streets import build_block_index

LON0, LAT0 = -74.0, 40.0
KX = 111_320.0 * math.cos(math.radians(LAT0))
KY = 110_540.0


def lonlat(x_m: float, y_north_m: float) -> list[float]:
    """World lon/lat of a point given in metres east/north of the test origin."""
    return [LON0 + x_m / KX, LAT0 + y_north_m / KY]


def street(name: str, points_m: list[tuple[float, float]]) -> dict:
    """A GeoJSON street feature from metre coordinates."""
    return {
        "properties": {"street_name": name},
        "geometry": {
            "type": "LineString",
            "coordinates": [lonlat(x, y) for x, y in points_m],
        },
    }


# An irregular grid: no run of gaps repeats reversed anywhere, so neither a
# translation nor a 180-degree flip can re-align the visible corridors.
VERTICALS_M = [-520, -400, -230, -60, 20, 160, 340, 500]
HORIZONTALS_M = [-460, -310, -150, -30, 60, 220, 380, 530]


def grid_features() -> list[dict]:
    features = [
        street(f"V{i} STREET", [(x, -600), (x, 600)]) for i, x in enumerate(VERTICALS_M)
    ]
    features += [
        street(f"H{i} STREET", [(-600, y), (600, y)])
        for i, y in enumerate(HORIZONTALS_M)
    ]
    # A dead-end street: content asymmetry like a real city, so a flip cannot
    # even partially correlate.
    features.append(street("STUB STREET", [(-600, 140), (40, 140)]))
    return features


def grid_index() -> FeatureIndex:
    """The synthetic grid, spatially indexed as the matcher consumes it."""
    return FeatureIndex(grid_features())


def extract_page(
    world: np.ndarray, pose: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Sample a 'page' image from a raster given the page->raster pose."""
    inverse = cv2.invertAffineTransform(pose)
    return cv2.warpAffine(world, inverse, size)


def test_frame_roundtrip() -> None:
    frame = frame_around((LON0, LAT0), half_m=500.0)
    theta = math.radians(20.0)
    scale_deg = 0.6 / KX  # ~0.6 m/px expressed in degrees of longitude
    affine = np.array(
        [
            [scale_deg * math.cos(theta), scale_deg * math.sin(theta), LON0 + 1e-4],
            [scale_deg * math.sin(theta), -scale_deg * math.cos(theta), LAT0 - 1e-4],
        ]
    )
    pose = frame.page_to_raster_affine(affine)
    back = frame.raster_pose_to_world_affine(pose)
    assert np.allclose(affine, back, atol=1e-12)
    # A page point mapped by the affine then projected into the frame must land
    # where the composed pose puts it.
    lon = affine[0, 0] * 40 + affine[0, 1] * 70 + affine[0, 2]
    lat = affine[1, 0] * 40 + affine[1, 1] * 70 + affine[1, 2]
    col, row = frame.lonlat_to_raster(lon, lat)
    via_pose = pose @ np.array([40.0, 70.0, 1.0])
    assert math.hypot(col - via_pose[0], row - via_pose[1]) < 1e-6


def test_osm_raster_geometry() -> None:
    frame = frame_around((LON0, LAT0), half_m=200.0)
    prob, valid, skeleton = osm_rasters(
        frame, FeatureIndex([street("A", [(0, -300), (0, 300)])])
    )
    assert valid.all()
    rows, cols = frame.shape
    center_col = cols // 2
    # The stroked corridor covers ~12 m (6 px) about the center column.
    assert prob[rows // 2, center_col] == 1.0
    assert prob[rows // 2, center_col - 2] == 1.0
    assert prob[rows // 2, center_col + 20] == 0.0
    # The skeleton is a single-pixel centerline.
    skeleton_cols = np.nonzero(skeleton[rows // 2])[0]
    assert len(skeleton_cols) <= 2
    assert abs(skeleton_cols.mean() - center_col) <= 1.5
    distance = osm_distance_m(skeleton)
    assert distance[rows // 2, center_col] <= 2.0
    assert distance[rows // 2, 5] == CHAMFER_CLAMP_M


def test_pose_theta_roundtrip() -> None:
    frame = frame_around((LON0, LAT0), half_m=500.0)
    for theta in [-135.0, -45.0, 0.0, 30.0, 90.0]:
        pose = cv2.getRotationMatrix2D((0.0, 0.0), theta, 0.4)
        pose[:, 2] += [250.0, 250.0]
        assert abs(wrap_deg(pose_theta_deg(pose) - theta)) < 1e-9
        world = frame.raster_pose_to_world_affine(pose)
        assert abs(wrap_deg(affine_theta_deg(world, frame) - theta)) < 1e-9


def test_dedupe_thetas_keeps_first_of_near_duplicates() -> None:
    priors = [
        RotationPrior(10.0, 4.0, "label-pair-exact"),
        RotationPrior(11.5, 8.0, "ransac-neighbor"),
        RotationPrior(-170.0, 4.0, "label-osm-mod180"),
        RotationPrior(100.0, 4.0, "mask-mod90"),
    ]
    kept = dedupe_thetas(priors)
    assert [p.source for p in kept] == [
        "label-pair-exact",
        "label-osm-mod180",
        "mask-mod90",
    ]


def test_confident_theta_needs_agreeing_label_pairs() -> None:
    def exact(theta: float) -> RotationPrior:
        return RotationPrior(theta, 4.0, "label-pair-exact")

    # A lone pair is a hypothesis, not corroboration.
    assert confident_theta_deg([exact(20.0)]) is None
    # Two pairs that agree pin the rotation.
    assert confident_theta_deg([exact(20.0), exact(23.0)]) == 20.0
    # One dissenting pair means the matcher still has to arbitrate.
    assert confident_theta_deg([exact(20.0), exact(23.0), exact(-60.0)]) is None
    # Other rungs never confer confidence, however many agree.
    assert (
        confident_theta_deg(
            [
                RotationPrior(20.0, 6.0, "ransac-neighbor"),
                RotationPrior(20.5, 12.0, "adjacency-keymap"),
                RotationPrior(20.0, 4.0, "label-osm-mod180"),
            ]
        )
        is None
    )
    # Agreement is measured mod 360: a pair reading -179 and one reading 179
    # are 2 degrees apart, not 358.
    assert confident_theta_deg([exact(179.0), exact(-179.0)]) == 179.0


def test_frame_thetas_prunes_only_under_confidence() -> None:
    page, _, _ = make_world_and_page(25.0)
    frame = frame_around((LON0, LAT0), half_m=1200.0)
    osm_prob, valid, _ = osm_rasters(frame, grid_index())
    params = MatchParams(mask_min_area=200)

    def context(priors: list[RotationPrior]) -> PageContext:
        return PageContext(
            stem="p1",
            number=1,
            width=300,
            height=420,
            prob=page,
            search_centers=[(LON0, LAT0)],
            radius_m=300.0,
            rotation_priors=priors,
            scale_priors=[ScalePrior(1.0, 0.05, "volume-median")],
        )

    agreeing = [
        RotationPrior(25.0, 4.0, "label-pair-exact"),
        RotationPrior(26.0, 4.0, "label-pair-exact"),
        RotationPrior(-70.0, 12.0, "adjacency-keymap"),
    ]
    ctx = context(agreeing)
    pruned = frame_thetas(ctx, confident_theta_deg(agreeing), (osm_prob, valid), params)
    assert [p.source for p in pruned] == ["label-pair-exact"]

    # The same ladder with the pairs disagreeing keeps everything and appends
    # the mask sweep.
    disagreeing = [
        RotationPrior(25.0, 4.0, "label-pair-exact"),
        RotationPrior(-40.0, 4.0, "label-pair-exact"),
        RotationPrior(-70.0, 12.0, "adjacency-keymap"),
    ]
    ctx = context(disagreeing)
    full = frame_thetas(
        ctx, confident_theta_deg(disagreeing), (osm_prob, valid), params
    )
    assert len(full) > len(pruned)
    assert "mask-mod90" in {p.source for p in full}


def test_cluster_rotation_rejects_outlier() -> None:
    theta, inliers = cluster_rotation([(20.0, 1.0), (24.0, 1.0), (150.0, 1.0)])
    assert inliers == 2
    assert abs(theta - 22.0) < 1.0


def test_label_osm_rotations_single_label_gives_both_flips() -> None:
    features = [street("V4 STREET", [(30, -600), (30, 600)])]
    block_index = build_block_index({"type": "FeatureCollection", "features": features})
    # A north-up page: a label along the north-south street reads vertically.
    label = LabelFeature(
        raw_text="V4",
        text="V4 STREET",
        center=(180.0, 210.0),
        dir_pix=math.pi / 2,
        long_side=80.0,
        short_side=12.0,
    )
    priors = label_osm_rotations([label], block_index, (LON0, LAT0))
    sources = {p.source for p in priors}
    assert sources == {"label-osm-mod180"}
    thetas = sorted(wrap_deg(p.theta_deg) for p in priors)
    assert abs(thetas[0] - (-180.0)) < 1e-6 or abs(thetas[0] - 0.0) < 1e-6
    assert len({round(t % 180.0, 3) for t in thetas}) == 1


def test_label_pair_resolves_flip() -> None:
    features = [
        street("AAA STREET", [(-100, -600), (-100, 600)]),
        street("BBB STREET", [(100, -600), (100, 600)]),
    ]
    block_index = build_block_index({"type": "FeatureCollection", "features": features})
    # North-up page, 1 m/px, centered on the origin: AAA appears at page
    # x=50, BBB at x=250 (page center x=150).
    label_a = LabelFeature(
        raw_text="AAA",
        text="AAA STREET",
        center=(50.0, 210.0),
        dir_pix=math.pi / 2,
        long_side=80.0,
        short_side=12.0,
    )
    label_b = LabelFeature(
        raw_text="BBB",
        text="BBB STREET",
        center=(250.0, 190.0),
        dir_pix=math.pi / 2,
        long_side=80.0,
        short_side=12.0,
    )
    priors = label_osm_rotations([label_a, label_b], block_index, (LON0, LAT0))
    exact = [p for p in priors if p.source == "label-pair-exact"]
    assert exact, "two labels on distinct parallel streets must resolve the flip"
    assert all(abs(wrap_deg(p.theta_deg)) < 5.0 for p in exact)


def test_adjacency_keymap_rotations() -> None:
    centroids = {
        10: (LON0, LAT0 + 400 / KY),  # north neighbor
        11: (LON0 + 400 / KX, LAT0),  # east neighbor
    }
    # North-up page: the north neighbor's number prints on the top margin, the
    # east neighbor's on the right margin.
    image_directions = {
        10: ((0.0, -1.0), 0.9),
        11: ((1.0, 0.0), 0.8),
    }
    priors = adjacency_keymap_rotations(image_directions, centroids, (LON0, LAT0))
    assert len(priors) == 1
    assert abs(wrap_deg(priors[0].theta_deg)) < 1.0

    # A page rotated cv2 +90: north is now toward page +x (right margin).
    rotated_directions = {
        10: ((1.0, 0.0), 0.9),
        11: ((0.0, 1.0), 0.8),
    }
    priors = adjacency_keymap_rotations(rotated_directions, centroids, (LON0, LAT0))
    assert len(priors) == 1
    assert abs(wrap_deg(priors[0].theta_deg - 90.0)) < 1.0


def test_page_scale_priors_family_rung() -> None:
    # Region area implying ~2x the median scale adds a family rung.
    side_m = 600.0
    ring = [lonlat(0, 0), lonlat(side_m, 0), lonlat(side_m, side_m), lonlat(0, side_m)]
    priors = page_scale_priors(1.0, [ring], 300, 300)
    assert [p.source for p in priors] == ["volume-median", "family-rung"]
    assert abs(priors[1].m_per_px - 2.0) < 1e-6
    # A region matching the median adds nothing.
    side_m = 300.0
    ring = [lonlat(0, 0), lonlat(side_m, 0), lonlat(side_m, side_m), lonlat(0, side_m)]
    priors = page_scale_priors(1.0, [ring], 300, 300)
    assert [p.source for p in priors] == ["volume-median"]


def test_calibrated_radius() -> None:
    assert calibrated_radius_m([50.0] * 3, 600.0) == (600.0, "locator")
    radius, source = calibrated_radius_m([40.0, 55.0, 60.0, 70.0, 90.0] * 2, 600.0)
    assert source == "calibrated"
    assert 150.0 <= radius <= 200.0
    # A wild p90 clamps to the locator radius; a tiny one to the floor.
    assert calibrated_radius_m([2000.0] * 10, 600.0)[0] == 600.0
    assert calibrated_radius_m([1.0] * 10, 600.0)[0] == 150.0


def test_name_alignment_correct_beats_shifted_and_renamed_scores_zero() -> None:
    features = [
        street("AAA STREET", [(-100, -600), (-100, 600)]),
        street("BBB STREET", [(100, -600), (100, 600)]),
    ]
    block_index = build_block_index({"type": "FeatureCollection", "features": features})
    labels = [
        LabelFeature("AAA", "AAA STREET", (50.0, 210.0), math.pi / 2, 80.0, 12.0),
        LabelFeature("BBB", "BBB STREET", (250.0, 190.0), math.pi / 2, 80.0, 12.0),
    ]
    # North-up 1 m/px affine centered on the origin (page 300x420).
    correct = np.array(
        [[1.0 / KX, 0.0, LON0 - 150.0 / KX], [0.0, -1.0 / KY, LAT0 + 210.0 / KY]]
    )
    shifted = correct.copy()
    shifted[0, 2] += 80.0 / KX  # slide 80 m east: labels miss their streets
    good = name_alignment(labels, block_index, correct)
    bad = name_alignment(labels, block_index, shifted)
    assert good.n_hits == 2
    assert good.score > bad.score
    # A renamed street matches nothing: score 0, never negative.
    renamed = name_alignment(
        [LabelFeature("ZZZ", "ZZZ STREET", (50.0, 210.0), math.pi / 2, 80.0, 12.0)],
        block_index,
        correct,
    )
    assert renamed.score == 0.0
    assert renamed.n_hits == 0


def make_world_and_page(
    theta_deg: float, page_size: tuple[int, int] = (300, 420)
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """(page P(road), truth page->lonlat affine, page-center lonlat)."""
    world_frame = frame_around((LON0, LAT0), half_m=1200.0)
    world_prob, _, _ = osm_rasters(world_frame, grid_index())
    width, height = page_size
    center_m = (20.0, 30.0)  # metres east/north of the origin
    center_px = world_frame.lonlat_to_raster(*lonlat(*center_m))
    # 1 m/px page scale -> 0.5 raster px per page px at the 2 m frame.
    pose = cv2.getRotationMatrix2D((width / 2, height / 2), theta_deg, 0.5)
    moved = pose @ np.array([width / 2, height / 2, 1.0])
    pose[:, 2] += [center_px[0] - moved[0], center_px[1] - moved[1]]
    page = extract_page(world_prob, pose, page_size)
    affine = world_frame.raster_pose_to_world_affine(pose)
    lon_c = affine[0, 0] * width / 2 + affine[0, 1] * height / 2 + affine[0, 2]
    lat_c = affine[1, 0] * width / 2 + affine[1, 1] * height / 2 + affine[1, 2]
    return page, affine, (lon_c, lat_c)


def affine_corner_error_m(a: np.ndarray, b: np.ndarray, size: tuple[int, int]) -> float:
    """Max corner displacement (metres) between two page->lonlat affines."""
    width, height = size
    worst = 0.0
    for x, y in [(0, 0), (width, 0), (width, height), (0, height)]:
        lon_a = a[0, 0] * x + a[0, 1] * y + a[0, 2]
        lat_a = a[1, 0] * x + a[1, 1] * y + a[1, 2]
        lon_b = b[0, 0] * x + b[0, 1] * y + b[0, 2]
        lat_b = b[1, 0] * x + b[1, 1] * y + b[1, 2]
        worst = max(worst, math.hypot((lon_a - lon_b) * KX, (lat_a - lat_b) * KY))
    return worst


def test_page_context_memos_match_a_fresh_computation() -> None:
    """The per-page memos must answer exactly what recomputing would."""
    from mapsnap.edge_join import dominant_orientation_deg, skeleton_points

    page, _, _ = make_world_and_page(25.0)
    ctx = PageContext(
        stem="p1",
        number=1,
        width=300,
        height=420,
        prob=page,
        search_centers=[(LON0, LAT0)],
        radius_m=300.0,
        rotation_priors=[],
        scale_priors=[ScalePrior(1.0, 0.05, "volume-median")],
    )
    params = MatchParams(mask_min_area=200)
    expected = skeleton_points(page, params.mask_threshold, params.mask_min_area)
    assert np.array_equal(ctx.road_points(params), expected)
    assert ctx.road_points(params) is ctx.road_points(params)  # memoized
    # A different mask setting is a different question, not a cache hit.
    coarse = MatchParams(mask_min_area=5000)
    assert ctx.road_points(coarse) is not ctx.road_points(params)
    assert ctx.road_orientation_deg() == dominant_orientation_deg(page)


def test_rotation_candidates_target_dir_matches_measuring_it() -> None:
    from mapsnap.edge_join import dominant_orientation_deg, rotation_candidates

    page, _, _ = make_world_and_page(25.0)
    frame = frame_around((LON0, LAT0), half_m=1200.0)
    world_prob, _, _ = osm_rasters(frame, grid_index())
    supplied = rotation_candidates(
        world_prob, page, target_dir=dominant_orientation_deg(page)
    )
    assert supplied == rotation_candidates(world_prob, page)


def test_snap_recovers_synthetic_pose() -> None:
    theta_true = 25.0
    page, truth_affine, center = make_world_and_page(theta_true)
    # Init 100 m off the true center with no rotation priors: the mask-mod-90
    # sweep alone must propose the right rotation and the irregular grid must
    # disambiguate the translation.
    init = (center[0] + 60.0 / KX, center[1] - 80.0 / KY)
    ctx = PageContext(
        stem="p1",
        number=1,
        width=300,
        height=420,
        prob=page,
        search_centers=[init],
        radius_m=300.0,
        rotation_priors=[],
        scale_priors=[ScalePrior(1.0, 0.05, "volume-median")],
    )
    params = MatchParams(
        min_overlap_m2=30_000.0, max_overlap_frac=1.0, top_k=8, mask_min_area=200
    )
    candidates = snap_page(ctx, grid_index(), params)
    assert candidates, "the matcher must produce candidates"
    best = candidates[0]
    assert best.plausible
    assert abs(wrap_deg(best.theta_deg - theta_true)) < 1.5
    assert affine_corner_error_m(best.world_affine, truth_affine, (300, 420)) < 8.0
    assert best.theta_source == "mask-mod90"


def test_snap_flipped_prior_recovered_by_mask_sweep() -> None:
    # A wrong 180-degree prior must not poison the result: the mask sweep
    # appends the true rotation and the content correlation picks it.
    theta_true = 25.0
    page, truth_affine, center = make_world_and_page(theta_true)
    ctx = PageContext(
        stem="p1",
        number=1,
        width=300,
        height=420,
        prob=page,
        search_centers=[center],
        radius_m=250.0,
        rotation_priors=[RotationPrior(theta_true - 180.0, 4.0, "adjacency-keymap")],
        scale_priors=[ScalePrior(1.0, 0.05, "volume-median")],
    )
    params = MatchParams(
        min_overlap_m2=30_000.0, max_overlap_frac=1.0, top_k=8, mask_min_area=200
    )
    candidates = snap_page(ctx, grid_index(), params)
    assert candidates
    best = candidates[0]
    assert abs(wrap_deg(best.theta_deg - theta_true)) < 1.5
    assert affine_corner_error_m(best.world_affine, truth_affine, (300, 420)) < 8.0


def test_evaluate_pose_truth_beats_shifted() -> None:
    from mapsnap.osm_snap import evaluate_pose

    page, truth_affine, center = make_world_and_page(25.0)
    ctx = PageContext(
        stem="p1",
        number=1,
        width=300,
        height=420,
        prob=page,
        search_centers=[center],
        radius_m=250.0,
        rotation_priors=[],
        scale_priors=[ScalePrior(1.0, 0.05, "volume-median")],
    )
    good = evaluate_pose(ctx, grid_index(), truth_affine)
    shifted_affine = truth_affine.copy()
    shifted_affine[0, 2] += 60.0 / KX  # slide 60 m east
    bad = evaluate_pose(ctx, grid_index(), shifted_affine)
    assert good is not None and bad is not None
    assert good["verification"] > bad["verification"]
    assert good["inlier_frac"] > bad["inlier_frac"]


def make_candidate(select: float, lon: float, theta: float = 0.0) -> SnapCandidate:
    # verification carries the whole select_score here (no bonuses attached).
    return SnapCandidate(
        world_affine=np.zeros((2, 3)),
        center=(lon, LAT0),
        theta_deg=theta,
        theta_source="mask-mod90",
        scale_m_per_px=1.0,
        scale_source="volume-median",
        scale_adjust=1.0,
        ncc=0.5,
        ncc_fine=0.3,
        chamfer_mean_m=5.0,
        inlier_frac=0.5,
        n_points=1000,
        jtj_eig_ratio=0.1,
        overlap_frac=0.0,
        refine_shift_m=3.0,
        center_dist_m=50.0,
        verification=select,
    )


def test_merge_candidates_dedupes_same_lock() -> None:
    near = 10.0 / KX  # 10 m apart: the same lock found from two centers
    a = make_candidate(1.5, LON0)
    b = make_candidate(1.2, LON0 + near)
    c = make_candidate(1.0, LON0 + 500.0 / KX)
    kept = merge_candidates([b, a, c], top_k=8)
    assert [k.verification for k in kept] == [1.5, 1.0]
    # The same location at a different rotation is a distinct lock.
    d = make_candidate(1.1, LON0 + near, theta=90.0)
    kept = merge_candidates([a, d], top_k=8)
    assert len(kept) == 2


def test_selection_score_matches_the_candidate_property():
    candidate = SnapCandidate(
        world_affine=np.eye(2, 3),
        center=(0.0, 0.0),
        theta_deg=0.0,
        theta_source="t",
        scale_m_per_px=1.0,
        scale_source="s",
        scale_adjust=1.0,
        ncc=0.0,
        ncc_fine=0.0,
        chamfer_mean_m=0.0,
        inlier_frac=0.0,
        n_points=0,
        jtj_eig_ratio=1.0,
        overlap_frac=1.0,
        refine_shift_m=0.0,
        center_dist_m=0.0,
        verification=0.5,
        region_containment=0.8,
        prior_theta_residual_sigma=0.5,
        name=NameAlignment(score=0.4, n_labels=5, n_hits=2),
    )
    # The candidate ranks on SIGNED name evidence (3 of 5 labels unmatched),
    # not on name_alignment's reward-only score.
    evidence = name_evidence(0.4, 5, 2)
    assert candidate.select_score() == selection_score(0.5, evidence, 0.8, 0.5)
    assert selection_score(0.5, evidence, 0.8, 0.5) == pytest.approx(
        0.5 + W_NAME * evidence + W_CONTAIN * 0.8 + W_PRIOR * 0.5
    )
    assert selection_score(-math.inf, 0.4, 0.8, 0.5) == -math.inf
    # Absent bonuses add nothing, exactly as the property behaves.
    assert selection_score(0.5, None, None, None) == 0.5


def test_directed_prior_residual_ignores_the_mod180_rung():
    priors = [
        RotationPrior(10.0, 4.0, "label-pair-exact"),
        RotationPrior(-170.0, 4.0, "label-osm-mod180"),
    ]
    # 12 deg is half a sigma from the directed prior; the mod-180 entry that
    # would match a 180-flip exactly does not count.
    assert directed_prior_residual_sigma(priors, 12.0) == pytest.approx(0.5)
    assert directed_prior_residual_sigma(priors, -170.0) == pytest.approx(45.0)
    assert directed_prior_residual_sigma(priors[1:], 12.0) is None


def test_name_evidence_charges_for_labels_that_match_nothing():
    # richmond p311's published disaster: 9 eligible labels, not one hit. The
    # reward-only score cannot distinguish it from a page with no labels.
    assert name_evidence(0.0, 9, 0) == pytest.approx(-0.5 * 9 / 11)
    assert NameAlignment(score=0.0, n_labels=9, n_hits=0).evidence < 0.0
    # The 18.8 ft pose on the same page hits 5 of 9 and stays positive.
    truth = NameAlignment(score=0.4099, n_labels=9, n_hits=5)
    assert truth.evidence > 0.0
    # ... and now outranks the alias, which it lost to by 0.046 before.
    alias = NameAlignment(score=0.0, n_labels=9, n_hits=0)
    assert selection_score(0.2288, truth.evidence, 0.786, 0.0) > selection_score(
        0.8249, alias.evidence, 0.612, 0.88
    )


def test_name_evidence_is_inert_without_enough_labels():
    # One or two labels say nothing either way, so the reward-only value stands.
    for n_labels in range(NAME_MISS_MIN_LABELS):
        assert name_evidence(0.1, n_labels, 0) == 0.1
    # A page with no eligible labels is unchanged at zero, as before.
    assert name_evidence(0.0, 0, 0) == 0.0
    # A pose that matches every label pays nothing.
    assert name_evidence(0.7, 5, 5) == 0.7


def test_name_evidence_charges_the_tail_only_not_a_per_miss_slope():
    """Partial agreement is untouched; only matching *nothing* is contradiction.

    The corpus statistic behind this term is about the zero-hit tail (0.8% of
    accurate poses vs 43.3% of >=500 ft poses). It says nothing about 1-of-5
    being proportionally worse than 3-of-5, and charging per miss regressed
    real pages: brooklyn p9's correct pose hits 1 of 5 while the 399 ft alias
    hits 4 of 5, and nashville p24's correct pose hits 4 of 7 and must keep
    its score to clear the absolute production gate.
    """
    for n_hits in range(1, 6):
        assert name_evidence(0.3, 5, n_hits) == 0.3
    assert name_evidence(0.142, 5, 1) == 0.142  # brooklyn p9's truth pose
    assert name_evidence(0.4235, 7, 4) == 0.4235  # nashville p24's truth pose
    # Only the zero-hit pose pays, and it pays more the more the page said.
    assert name_evidence(0.0, 3, 0) == pytest.approx(-0.5 * 3 / 5)
    assert name_evidence(0.0, 9, 0) < name_evidence(0.0, 3, 0)


def test_name_evidence_of_recomputes_for_pre_375_records():
    fresh = {"score": 0.4, "evidence": 0.1857, "n_labels": 5, "n_hits": 2}
    assert name_evidence_of(fresh) == pytest.approx(0.1857)
    legacy = {"score": 0.4, "n_labels": 5, "n_hits": 2}
    assert name_evidence_of(legacy) == pytest.approx(name_evidence(0.4, 5, 2))
    assert name_evidence_of(None) is None
    assert name_evidence_of({}) is None
    # A record with a score but no counts keeps the reward-only value.
    assert name_evidence_of({"score": 0.3}) == pytest.approx(0.3)
