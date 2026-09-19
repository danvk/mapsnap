import math

import cv2
import numpy as np

from mapsnap.georef_from_labels import LabelFeature
from mapsnap.road_continuity import (
    Frame,
    PlacedPage,
    Placement,
    PlacementOptions,
    Prior,
    RoadLine,
    TargetPage,
    WorldIndex,
    attach_label_names,
    content_box,
    continuity_score,
    corner_rmse_ft,
    extract_road_lines,
    orientation_modes,
    place_page,
    pose_affine,
    rotation_of,
    scale_of,
    world_lines,
)


def synthetic_road_map(
    size: tuple[int, int],
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> np.ndarray:
    """A P(road) map with the given road segments drawn 18 px wide and softened."""
    width, height = size
    image = np.zeros((height, width), np.float32)
    for (x0, y0), (x1, y1) in segments:
        cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, 18)
    return cv2.GaussianBlur(image, (0, 0), 2.0)


def test_extract_road_lines_finds_crossing_streets_with_their_angles() -> None:
    prob = synthetic_road_map(
        (800, 1000),
        [((100, 300), (700, 300)), ((400, 50), (400, 950)), ((100, 900), (700, 600))],
    )
    lines = extract_road_lines(prob)
    angles = sorted(round(math.degrees(line.angle)) % 180 for line in lines)
    assert len(lines) == 3, angles
    assert angles[0] == 0 and angles[1] in (90,) and 150 <= angles[2] <= 155
    horizontal = next(
        line for line in lines if round(math.degrees(line.angle)) % 180 == 0
    )
    assert 560 <= horizontal.length <= 640
    assert abs(horizontal.start[1] - 300) < 3 and abs(horizontal.end[1] - 300) < 3


def test_extract_road_lines_splits_a_street_that_stops_mid_page() -> None:
    prob = synthetic_road_map(
        (800, 1000), [((50, 500), (350, 500)), ((650, 500), (780, 500))]
    )
    lines = extract_road_lines(prob)
    # The 130 px stub is below the 300 px floor; the 300 px run survives alone.
    assert len(lines) == 1
    assert 280 <= lines[0].length <= 330


def test_orientation_modes_two_grids() -> None:
    angles = [math.radians(a) for a in [10, 11, 10, 100, 101, 55, 55, 56]]
    weights = [1.0] * 8
    modes = [int(math.degrees(m)) for m in orientation_modes(angles, weights)]
    assert modes == [10, 55]


def test_orientation_modes_second_mode_needs_weight() -> None:
    angles = [math.radians(a) for a in [10] * 20 + [55]]
    modes = orientation_modes(angles, [1.0] * 21)
    assert [int(math.degrees(m)) for m in modes] == [10]


def feature(text: str, centre: tuple[float, float], dir_pix: float) -> LabelFeature:
    return LabelFeature(
        raw_text=text,
        text=text,
        center=centre,
        dir_pix=dir_pix,
        long_side=80.0,
        short_side=20.0,
    )


def test_attach_label_names_names_the_line_the_label_sits_on() -> None:
    lines = [
        RoadLine(start=np.array([0.0, 300.0]), end=np.array([600.0, 300.0]), angle=0.0)
    ]
    attach_label_names(
        lines,
        [
            feature("MAIN STREET", (300.0, 305.0), 0.05),
            feature("FAR STREET", (300.0, 380.0), 0.0),
            feature("CROSS STREET", (300.0, 300.0), math.pi / 2),
        ],
    )
    assert lines[0].names == {"MAIN STREET"}


def test_world_lines_extend_only_past_a_sheet_edge() -> None:
    page = PlacedPage(
        stem="p1",
        affine=pose_affine(np.array([0.0, 0.0]), 0.0, 0.2),
        width=1000,
        height=1200,
        lines=[
            RoadLine(
                start=np.array([10.0, 500.0]), end=np.array([990.0, 500.0]), angle=0.0
            ),
            RoadLine(
                start=np.array([300.0, 100.0]),
                end=np.array([300.0, 700.0]),
                angle=math.pi / 2,
            ),
        ],
    )
    through, stub = world_lines([page], extension_m=450.0)
    assert through.extend_before == 450.0 and through.extend_after == 450.0
    assert stub.extend_before == 0.0 and stub.extend_after == 0.0
    assert abs(through.length - 980 * 0.2) < 1e-6


def test_continuity_score_rewards_alignment_and_name_matches() -> None:
    anchor = PlacedPage(
        stem="a",
        affine=pose_affine(np.array([0.0, 0.0]), 0.0, 1.0),
        width=200,
        height=200,
        lines=[
            RoadLine(
                start=np.array([0.0, 50.0]),
                end=np.array([200.0, 50.0]),
                angle=0.0,
                names={"OAK STREET"},
            )
        ],
    )
    lines = world_lines([anchor], extension_m=300.0)
    target = [
        RoadLine(
            start=np.array([0.0, 0.0]),
            end=np.array([150.0, 0.0]),
            angle=0.0,
            names={"OAK STREET"},
        )
    ]
    aligned = pose_affine(
        np.array([250.0, 50.0]), 0.0, 1.0
    )  # continues OAK past the sheet edge
    shifted = pose_affine(np.array([250.0, 90.0]), 0.0, 1.0)  # 40 m off the line
    score_aligned, names = continuity_score(aligned, target, WorldIndex(lines, target))
    score_shifted, _ = continuity_score(shifted, target, WorldIndex(lines, target))
    assert names == {"OAK STREET"}
    assert score_aligned > 3.5 and score_shifted < 0.1
    conflicting = [
        RoadLine(
            start=np.array([0.0, 0.0]),
            end=np.array([150.0, 0.0]),
            angle=0.0,
            names={"ELM STREET"},
        )
    ]
    score_conflict, _ = continuity_score(
        aligned, conflicting, WorldIndex(lines, conflicting)
    )
    assert score_conflict == 0.0


def grid_lines(
    origin: tuple[float, float],
    size: tuple[int, int],
    spacing: float,
    names: dict[str, str],
) -> list[RoadLine]:
    """Full-height verticals and full-width horizontals of a block grid, named by world index."""
    width, height = size
    lines = []
    x = -(origin[0] % spacing)
    while x < width:
        if x >= 0:
            column = round((origin[0] + x) / spacing)
            lines.append(
                RoadLine(
                    start=np.array([x, 0.0]),
                    end=np.array([x, float(height)]),
                    angle=math.pi / 2,
                    names={names[f"v{column}"]} if f"v{column}" in names else set(),
                )
            )
        x += spacing
    y = -(origin[1] % spacing)
    while y < height:
        if y >= 0:
            row = round((origin[1] + y) / spacing)
            lines.append(
                RoadLine(
                    start=np.array([0.0, y]),
                    end=np.array([float(width), y]),
                    angle=0.0,
                    names={names[f"h{row}"]} if f"h{row}" in names else set(),
                )
            )
        y += spacing
    return lines


def test_place_page_recovers_a_page_among_four_placed_neighbours() -> None:
    # A 400 px block grid in a 0.2 m/px world. The target sheet sits in the middle of a
    # cross of placed sheets: left/right share its east-west streets, top/bottom its
    # north-south avenues, all named, and every street runs off the sheet edge so it is
    # extrapolated into the target. The target is rotated 3 degrees.
    scale, spacing = 0.2, 400.0
    names = {f"v{i}": f"AVENUE {i}" for i in range(40)} | {
        f"h{i}": f"STREET {i}" for i in range(-20, 40)
    }
    width, height = 1600, 1900
    neighbours = []
    for stem, (px, py) in {
        "left": (-1500, 0),
        "right": (1500, 0),
        "top": (0, -1800),
        "bottom": (0, 1800),
    }.items():
        world_origin = (1500.0 + px, 2000.0 + py)  # sheet's (0,0) in world pixels
        neighbours.append(
            PlacedPage(
                stem,
                pose_affine(
                    np.array([world_origin[0] * scale, world_origin[1] * scale]),
                    0.0,
                    scale,
                ),
                width,
                height,
                grid_lines(world_origin, (width, height), spacing, names),
            )
        )
    nominal_origin = (1500.0, 2000.0)
    nominal = pose_affine(
        np.array([nominal_origin[0] * scale, nominal_origin[1] * scale]), 0.0, scale
    )
    true_rotation = math.radians(3.0)
    truth = pose_affine(nominal[:, 2] + np.array([7.0, 11.0]), true_rotation, scale)
    inverse = cv2.invertAffineTransform(truth)
    target_lines = []
    for line in grid_lines(nominal_origin, (width, height), spacing, names):
        a = inverse @ np.append(nominal @ np.append(line.start, 1.0), 1.0)
        b = inverse @ np.append(nominal @ np.append(line.end, 1.0), 1.0)
        target_lines.append(
            RoadLine(
                start=a,
                end=b,
                angle=math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi,
                names=set(line.names),
            )
        )
    target = TargetPage(stem="mid", width=width, height=height, lines=target_lines)
    prior = Prior(
        centre=truth @ np.array([width / 2, height / 2, 1.0]) + np.array([25.0, -20.0]),
        radius_m=300.0,
    )
    options = PlacementOptions(scale_m_per_px=scale, prior_sigma_m=60.0, max_seeds=12)
    placement = place_page(target, neighbours, prior, options, truth=truth)
    best = placement.best
    assert best is not None and best.rmse_ft is not None
    assert best.rmse_ft < 15.0, [
        (round(c.rmse_ft or -1), round(c.score, 1)) for c in placement.candidates[:5]
    ]
    assert abs(best.rotation_deg - 3.0) < 0.5
    assert placement.accepted()
    assert best.names & {"AVENUE 4", "AVENUE 5", "STREET 5", "STREET 6"}


def test_frame_round_trips_an_affine() -> None:
    frame = Frame(-83.0, 42.4)
    lonlat = np.array([[1.0e-6, 2.0e-7, -83.01], [3.0e-7, -9.0e-7, 42.41]])
    back = frame.affine_to_lonlat(frame.affine_to_xy(lonlat))
    assert np.allclose(back, lonlat)


def test_pose_affine_rotation_and_scale_round_trip() -> None:
    affine = pose_affine(np.array([10.0, -5.0]), math.radians(30.0), 0.25)
    assert abs(math.degrees(rotation_of(affine)) - 30.0) < 1e-9
    assert abs(scale_of(affine) - 0.25) < 1e-12
    assert corner_rmse_ft((100, 100), affine, affine) == 0.0


def test_content_box_ignores_blank_margins() -> None:
    prob = np.zeros((400, 300), np.float32)
    prob[100:300, 50:250] = 1.0
    assert content_box(prob, shrink_px=10) == (60, 110, 239, 289)


def test_placement_margin_and_gate() -> None:
    def candidate(score: float, tx: float):
        from mapsnap.road_continuity import Candidate

        return Candidate(
            affine=pose_affine(np.array([tx, 0.0]), 0.0, 0.2),
            rotation_deg=0.0,
            vote=1.0,
            score=score,
            names=set(),
        )

    placement = Placement(
        stem="p",
        candidates=[candidate(20.0, 0.0), candidate(19.0, 5.0), candidate(10.0, 100.0)],
        anchors=[],
        target_lines=5,
        named_lines=2,
    )
    assert placement.margin == 2.0  # the 5 m neighbour is the same pose, not a rival
    assert placement.accepted()
    weak = Placement(
        stem="p",
        candidates=[candidate(8.0, 0.0)],
        anchors=[],
        target_lines=5,
        named_lines=2,
    )
    assert not weak.accepted()
