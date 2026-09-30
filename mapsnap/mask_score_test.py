import json
from pathlib import Path

import pytest
from shapely.geometry import LineString, box

from mapsnap.mask_score import (
    GroundMask,
    ImageAgreement,
    LocalFrame,
    agreement_summary,
    defect_summary,
    enclosed_gaps,
    gap_areas,
    ground_masks,
    image_agreements,
    image_defect_shares,
    image_key,
    is_invalid,
    off_street_seam_lengths,
    overlap_areas,
    reference_outline,
    seam_street_share,
    selector_points,
    sheet_equal,
    sheet_equal_mean,
    trusted_truth_masks,
)

# Near the equator a degree of longitude and of latitude are the same length, so
# this pixel -> (lon, lat) map is a similarity: 1 px = 1e-5 degrees = 1.11 m.
DEG_PER_PX = 1e-5


def to_lonlat(x: float, y: float, x0: float = 0.0) -> list[float]:
    return [(x0 + x) * DEG_PER_PX, -y * DEG_PER_PX]


def annotation(
    key: str,
    mask: list[tuple[float, float]],
    *,
    split: int | None = None,
    x0: float = 0.0,
) -> dict:
    """A georeference annotation on a 1000 px sheet placed x0 px east of the origin."""
    points = " ".join(f"{x},{y}" for x, y in mask)
    gcps = [(100.0, 100.0), (900.0, 900.0)]
    label = f"Town | 1900 | {key}" + (f" [{split}]" if split else "")
    return {
        "type": "Annotation",
        "label": label,
        "target": {
            "source": {
                "id": f"https://example.org/iiif/1900-{key[1:].zfill(4)}/info.json",
                "type": "ImageService2",
                "width": 1000,
                "height": 1000,
            },
            "selector": {
                "type": "SvgSelector",
                "value": f'<svg><polygon points="{points}" /></svg>',
            },
        },
        "body": {
            "type": "FeatureCollection",
            "transformation": {"type": "helmert"},
            "features": [
                {
                    "type": "Feature",
                    "properties": {"type": "gcp", "resourceCoords": [x, y]},
                    "geometry": {"type": "Point", "coordinates": to_lonlat(x, y, x0)},
                }
                for x, y in gcps
            ],
        },
    }


SHEET = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0), (0.0, 0.0)]
INNER = [(50.0, 50.0), (950.0, 50.0), (950.0, 950.0), (50.0, 950.0), (50.0, 50.0)]


def test_selector_points_reads_the_svg_polygon():
    assert selector_points(annotation("p1", INNER))[:2] == [(50.0, 50.0), (950.0, 50.0)]
    assert selector_points({"target": {}}) == []


def test_image_key_includes_the_split_panel():
    assert image_key(annotation("p12", SHEET)) == "p12"
    assert image_key(annotation("p12", SHEET, split=2)) == "p12__2"


def test_trusted_truth_masks_drop_a_selector_missing_its_gcps():
    # A mask in the crop's frame (OIM#402) sits away from the GCPs it should hold.
    shifted = [(500.0, 0.0), (990.0, 0.0), (990.0, 400.0), (500.0, 400.0)]
    masks = trusted_truth_masks([annotation("p1", INNER), annotation("p2", shifted)])
    assert set(masks) == {"p1"}


def test_reference_outline_is_the_oim_panel_for_a_split(tmp_path: Path):
    panels = {
        "width": 100,
        "height": 100,
        "panels": [[[0, 0], [50, 0], [50, 100], [0, 100]]],
    }
    (tmp_path / "p3.panels.json").write_text(json.dumps(panels))
    source = {"width": 1000, "height": 1000}
    assert reference_outline("p3__1", source, tmp_path).area == pytest.approx(500_000)
    assert reference_outline("p3", source, tmp_path).area == pytest.approx(1_000_000)


def test_image_agreements_score_iou_and_flag_unclipped_truth():
    half = [(0.0, 0.0), (500.0, 0.0), (500.0, 1000.0), (0.0, 1000.0)]
    ours = [annotation("p1", INNER), annotation("p2", half)]
    truth = [annotation("p1", INNER), annotation("p2", SHEET)]
    rows = {row.key: row for row in image_agreements(ours, truth)}
    assert rows["p1"].iou == pytest.approx(1.0)
    assert rows["p1"].clipped
    # OIM left p2 as its whole sheet: scored, but not "clipped".
    assert rows["p2"].iou == pytest.approx(0.5)
    assert not rows["p2"].clipped
    summary = agreement_summary(ours, truth)
    assert summary["iou"] == pytest.approx(0.75)
    assert summary["iou_clipped"] == pytest.approx(1.0)


def test_sheet_equal_mean_gives_a_split_sheet_one_weight():
    rows = [
        ImageAgreement("p1", 1.0, True),
        ImageAgreement("p2__1", 0.0, True),
        ImageAgreement("p2__2", 1.0, True),
    ]
    assert sheet_equal_mean(rows) == pytest.approx(0.75)
    assert sheet_equal_mean([]) is None


def test_ground_masks_follow_each_annotations_transform():
    masks, frame = ground_masks(
        [annotation("p1", SHEET), annotation("p2", SHEET, split=1, x0=1000)]
    )
    assert frame is not None
    assert [(mask.index, mask.key) for mask in masks] == [(0, "p1"), (1, "p2__1")]
    assert [mask.polygon.area for mask in masks] == pytest.approx(
        [1.11195**2 * 1e6] * 2, rel=1e-3
    )
    assert masks[0].polygon.intersection(masks[1].polygon).area == pytest.approx(
        0, abs=1
    )


def test_defect_summary_measures_overlap_and_enclosed_gaps():
    # Two sheets side by side, each masked 10% past the shared edge: they overlap.
    wide = [(0.0, 0.0), (1100.0, 0.0), (1100.0, 1000.0), (0.0, 1000.0)]
    overlapping = defect_summary(
        [annotation("p1", wide), annotation("p2", SHEET, x0=1000)]
    )
    assert overlapping["overlap"] == pytest.approx(0.1 / 2.0, rel=0.01)
    assert overlapping["gaps"] == pytest.approx(0)
    assert overlapping["invalid"] == 0
    # A ring of masks around an unmasked middle. The middle is a gap only where
    # a sheet's scan covers it: p2 and p4 are strips masked out of whole sheets.
    ring = [annotation("p1", SHEET, x0=x0) for x0 in (0, 1000, 2000)]
    top = [(0.0, 0.0), (1000.0, 0.0), (1000.0, 100.0), (0.0, 100.0)]
    bottom = [(0.0, 900.0), (1000.0, 900.0), (1000.0, 1000.0), (0.0, 1000.0)]
    ring[1] = annotation("p2", top, x0=1000)
    ring.append(annotation("p4", bottom, x0=1000))
    holed = defect_summary(ring)
    assert holed["gaps"] == pytest.approx(800_000 / 2_200_000, rel=0.01)


def test_defect_summary_counts_self_intersecting_selectors():
    bowtie = [(0.0, 0.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 1000.0), (0.0, 0.0)]
    assert defect_summary([annotation("p1", bowtie)])["invalid"] == 1


def test_seam_street_share_finds_a_seam_along_a_street():
    left, right = box(0, 0, 100, 100), box(100, 0, 200, 100)
    union = left.union(right)
    street = LineString([(100, -50), (100, 150)])
    elsewhere = LineString([(50, -50), (50, 150)])
    length, share = seam_street_share([left, right], union, [street])
    # The shared edge once, less the outline margin at either end.
    assert length == pytest.approx(90, rel=0.05)
    assert share == pytest.approx(1.0)
    assert seam_street_share([left, right], union, [elsewhere])[1] == pytest.approx(0.0)
    assert seam_street_share([left], left, [street]) == (0.0, None)


def test_local_frame_is_metres_from_its_origin():
    frame = LocalFrame(0.0, 0.0)
    assert frame.to_m(0.001, 0.001) == pytest.approx((111.195, 111.195))


def test_overlap_areas_split_each_overlap_between_the_pair():
    left, right, far = box(0, 0, 100, 100), box(80, 0, 180, 100), box(500, 0, 600, 100)
    assert overlap_areas([left, right, far]) == pytest.approx([1000, 1000, 0])


def test_gap_areas_go_to_the_masks_bordering_the_gap():
    # A 100 x 100 hole: two tall masks either side, two short ones above and below.
    west, east = box(0, 0, 100, 300), box(200, 0, 300, 300)
    north, south = box(100, 200, 200, 300), box(100, 0, 200, 100)
    areas = gap_areas([west, east, north, south], box(100, 100, 200, 200))
    assert sum(areas) == pytest.approx(10_000, rel=0.01)
    assert areas[0] == pytest.approx(areas[1], rel=0.05)
    assert areas[0] == pytest.approx(2_500, rel=0.1)


def test_enclosed_gaps_count_only_ground_a_scan_covers():
    # Four masks ring a 100 x 100 hole; only the west scan reaches into it.
    def mask(key: str, polygon, scan=None) -> GroundMask:
        return GroundMask(0, key, polygon, scan or polygon)

    ring = [
        mask("p1", box(0, 0, 100, 300), scan=box(0, 0, 150, 300)),
        mask("p2", box(200, 0, 300, 300)),
        mask("p3", box(100, 200, 200, 300)),
        mask("p4", box(100, 0, 200, 100)),
    ]
    assert enclosed_gaps(ring).area == pytest.approx(50 * 100)


def test_off_street_seam_lengths_skip_seams_on_streets_and_the_outline():
    left, right = box(0, 0, 100, 100), box(100, 0, 200, 100)
    union = left.union(right)
    on_street = LineString([(100, -50), (100, 150)]).buffer(10)
    assert off_street_seam_lengths([left, right], union, on_street) == pytest.approx(
        [0, 0]
    )
    nowhere = LineString([(1000, 0), (1000, 1)]).buffer(10)
    assert off_street_seam_lengths([left, right], union, nowhere) == pytest.approx(
        [90, 90], rel=0.01
    )


def test_image_defect_shares_charge_overlap_seams_and_failures():
    left = GroundMask(0, "p1", box(0, 0, 100, 100), box(0, 0, 100, 100))
    right = GroundMask(1, "p2", box(80, 0, 180, 100), box(80, 0, 180, 100))
    # 2000 m^2 of overlap, 1000 to each 10,000 m^2 page.
    shares = dict(image_defect_shares([left, right], None, {}))
    assert shares == pytest.approx({"p1": 0.1, "p2": 0.1})
    # A page Allmaps fails half the time is wrong half the time.
    shares = dict(image_defect_shares([left, right], None, {1: 0.5}))
    assert shares["p2"] == pytest.approx(0.5 + 0.5 * 0.1)
    # A seam off the streets costs a 5 m strip along each side.
    touching = GroundMask(1, "p2", box(100, 0, 200, 100), box(100, 0, 200, 100))
    nowhere = LineString([(1000, 0), (1000, 1)]).buffer(10)
    shares = dict(image_defect_shares([left, touching], nowhere, {}))
    assert shares["p1"] == pytest.approx(90 * 5 / 10_000, rel=0.01)


def test_sheet_equal_gives_a_split_sheet_one_weight():
    assert sheet_equal([("p1", 0.0), ("p2__1", 1.0), ("p2__2", 0.0)]) == 0.25
    assert sheet_equal([]) is None


def test_is_invalid_catches_self_intersections():
    bowtie = [(0.0, 0.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 1000.0), (0.0, 0.0)]
    assert is_invalid(annotation("p1", bowtie))
    assert not is_invalid(annotation("p1", SHEET))


def test_defect_summary_charges_invalid_selectors_in_full():
    bowtie = [(0.0, 0.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 1000.0), (0.0, 0.0)]
    items = [annotation("p1", SHEET), annotation("p2", bowtie, x0=5000)]
    summary = defect_summary(items)
    assert summary["defect_share"] == pytest.approx(0.5, abs=0.01)
    assert summary["worst"][0][0] == "p2"
