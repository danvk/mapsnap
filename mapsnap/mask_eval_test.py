import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from mapsnap.mask_eval import (
    ImageFrame,
    fit_similarity,
    gcp_pairs,
    image_footprint,
    no_masks,
    oim_sidecar,
    split_index,
    voronoi_masks,
    write_oim_sidecars,
)

DEG_PER_PX = 1e-5  # near the equator: a similarity, 1 px = 1.11 m


def oim_item(
    key: str, gcps: list[tuple[float, float]], split: int | None = None
) -> dict:
    """An OIM annotation on a 400 x 300 canvas; GCP ground positions are px * 1e-5 deg."""
    label = f"Town | 1900 | {key}" + (f" [{split}]" if split else "")
    return {
        "label": label,
        "target": {
            "source": {
                "id": f"https://example.org/iiif/1900-{key[1:].zfill(4)}/info.json",
                "type": "ImageService2",
                "width": 400,
                "height": 300,
            },
            "selector": {"type": "SvgSelector", "value": "<svg></svg>"},
        },
        "body": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"resourceCoords": [x, y]},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [x * DEG_PER_PX, -y * DEG_PER_PX],
                    },
                }
                for x, y in gcps
            ],
        },
    }


def test_split_index_reads_the_panel_number():
    assert split_index("Town | 1900 | p3 [2]") == 2
    assert split_index("Town | 1900 | p3") is None


def test_gcp_pairs_pair_pixels_with_ground():
    pairs = gcp_pairs(oim_item("p1", [(10, 20)]))
    assert pairs == [((10, 20), (10 * DEG_PER_PX, -20 * DEG_PER_PX))]


def test_fit_similarity_recovers_a_rotated_scaled_pose():
    pixels = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], float)
    # 2 m per px, rotated 30 degrees, near the equator.
    theta = np.radians(30)
    m_per_deg = 111_195.0
    ground = [
        (
            2 * (x * np.cos(theta) + y * np.sin(theta)) / m_per_deg,
            2 * (x * np.sin(theta) - y * np.cos(theta)) / m_per_deg,
        )
        for x, y in pixels
    ]
    transform = fit_similarity(pixels, np.array(ground))
    for (x, y), expected in zip(pixels, ground):
        assert transform(x, y) == pytest.approx(expected, abs=1e-9)


def test_oim_sidecar_writes_corners_and_the_two_farthest_gcps():
    pairs = gcp_pairs(oim_item("p1", [(40, 40), (200, 150), (360, 260)]))
    doc = oim_sidecar(pairs, ImageFrame(0.25, (0.0, 0.0), (100, 75)))
    assert doc is not None
    assert (doc["width"], doc["height"]) == (100, 75)
    # Corners of the quarter-scale image land on the canvas's corners.
    assert doc["corners"][2] == pytest.approx([400 * DEG_PER_PX, -300 * DEG_PER_PX])
    initials = [(i["x"], i["y"]) for i in doc["intersections"] if i["initial"]]
    assert sorted(initials) == [(10.0, 10.0), (90.0, 65.0)]


def test_oim_sidecar_needs_two_distinct_gcps():
    frame = ImageFrame(0.25, (0.0, 0.0), (100, 75))
    assert oim_sidecar(gcp_pairs(oim_item("p1", [(40, 40)])), frame) is None
    assert oim_sidecar(gcp_pairs(oim_item("p1", [(40, 40), (40, 41)])), frame) is None


def test_oim_sidecar_places_a_panel_in_its_crop():
    pairs = gcp_pairs(oim_item("p2", [(240, 40), (360, 260)], split=2))
    doc = oim_sidecar(pairs, ImageFrame(0.25, (50.0, 0.0), (50, 75)))
    assert doc is not None
    # The crop's top-left pixel is canvas (200, 0).
    assert doc["corners"][0] == pytest.approx([200 * DEG_PER_PX, 0.0], abs=1e-12)


def write_volume(volume: Path) -> None:
    """p1 whole; p2 split by OIM into left and right halves."""
    volume.mkdir()
    for sheet in ("p1", "p2"):
        Image.new("RGB", (100, 75), "white").save(volume / f"{sheet}.jpg")
    items = [
        oim_item("p1", [(40, 40), (360, 260)]),
        oim_item("p2", [(40, 40), (160, 260)], split=1),
        oim_item("p2", [(240, 40), (360, 260)], split=2),
        oim_item("p9", [(40, 40), (360, 260)]),  # no image
    ]
    (volume / "main.iiif.json").write_text(json.dumps({"items": items}))
    (volume / "oim").mkdir()
    panels = {
        "width": 400,
        "height": 300,
        "panels": [
            [[0, 0], [200, 0], [200, 300], [0, 300]],
            [[200, 0], [400, 0], [400, 300], [200, 300]],
        ],
    }
    (volume / "oim" / "p2.panels.json").write_text(json.dumps(panels))


def test_write_oim_sidecars_cuts_oim_panels_in_the_work_dir(tmp_path: Path):
    volume, work = tmp_path / "vol", tmp_path / "work"
    write_volume(volume)
    skipped = write_oim_sidecars(volume, work)
    assert skipped == ["p9 (no image)"]
    written = sorted(p.name for p in work.glob("*.georef-final.json"))
    assert written == [
        "p1.georef-final.json",
        "p2__1.georef-final.json",
        "p2__2.georef-final.json",
    ]
    # The crops and panels.json are the work dir's, never the volume's.
    assert (work / "p2__2.jpg").exists() and (work / "p2.panels.json").exists()
    assert not (volume / "p2__2.jpg").exists()
    panel = json.loads((work / "p2__2.georef-final.json").read_text())
    assert panel["corners"][0] == pytest.approx([200 * DEG_PER_PX, 0.0], abs=1e-12)


def georef(x0: float, width: float = 100) -> dict:
    """A north-up 100 x 100 px page, 1 px = 1e-5 deg, x0 px east of the origin."""
    corners = [
        [x0 * DEG_PER_PX, 0.0],
        [(x0 + width) * DEG_PER_PX, 0.0],
        [(x0 + width) * DEG_PER_PX, -100 * DEG_PER_PX],
        [x0 * DEG_PER_PX, -100 * DEG_PER_PX],
    ]
    return {"width": width, "height": 100, "corners": corners}


def test_image_footprint_is_the_scan():
    footprint = image_footprint(georef(0), Path("p1.jpg"))
    assert footprint.bounds == pytest.approx(
        (0, -100 * DEG_PER_PX, 100 * DEG_PER_PX, 0)
    )


def test_voronoi_masks_split_an_overlap_at_the_midline():
    masks = voronoi_masks([georef(0), georef(50)])
    assert all(mask is not None for mask in masks)
    left, right = (m for m in masks if m is not None)
    # Centroids at 50 and 100 px: the overlap (50..100) splits at 75, so each
    # page keeps three quarters of its 100 x 100 px scan.
    page_area = (100 * DEG_PER_PX) ** 2
    assert left.area == pytest.approx(0.75 * page_area, rel=1e-4)
    assert right.area == pytest.approx(0.75 * page_area, rel=1e-4)
    assert left.bounds[2] == pytest.approx(75 * DEG_PER_PX, rel=1e-4)
    assert right.bounds[0] == pytest.approx(75 * DEG_PER_PX, rel=1e-4)
    assert left.intersection(right).area == pytest.approx(0, abs=1e-15)


def test_voronoi_masks_leave_a_lone_page_whole():
    (mask,) = voronoi_masks([georef(0)])
    assert isinstance(mask, Polygon)
    assert mask.area == pytest.approx((100 * DEG_PER_PX) ** 2)


def test_no_masks_masks_nothing():
    assert no_masks([georef(0), georef(50)], None) == [None, None]
