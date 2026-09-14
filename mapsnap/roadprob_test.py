"""Tests for the P(road) sidecar cache (mapsnap.roadprob)."""

from pathlib import Path

import cv2
import numpy as np

from mapsnap.roadprob import (
    QUALITY,
    crop_to_panel,
    derive_panel_roadprob,
    legacy_roadprob_path,
    load_roadprob,
    panel_rings,
    pending_images,
    roadprob_path,
    save_roadprob,
)


def test_sidecar_sits_beside_the_image_and_the_legacy_path_under_artifacts() -> None:
    image = Path("data/vol/p12__2.jpg")
    assert roadprob_path(image) == Path("data/vol/p12__2.roadprob.jpg")
    assert roadprob_path(str(image)) == Path("data/vol/p12__2.roadprob.jpg")
    assert legacy_roadprob_path(image) == Path(
        "data/vol/artifacts/edge_join/roadprob/p12__2.png"
    )


def test_save_then_load_round_trips_within_jpeg_error(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    # Smooth, like a real P(road) map: JPEG is hostile to per-pixel noise.
    probability = cv2.GaussianBlur(
        rng.random((64, 96), dtype=np.float32), (0, 0), sigmaX=4
    )
    probability = (probability - probability.min()) / np.ptp(probability)
    path = tmp_path / "p1.roadprob.jpg"
    save_roadprob(path, probability)
    loaded = load_roadprob(tmp_path / "p1.jpg")
    assert loaded is not None
    assert loaded.dtype == np.float32
    assert loaded.shape == probability.shape
    assert float(np.abs(loaded - probability).mean()) < 0.02


def test_save_clips_out_of_range_values(tmp_path: Path) -> None:
    save_roadprob(tmp_path / "p1.roadprob.jpg", np.array([[-3.0, 0.5, 4.0]] * 8))
    loaded = load_roadprob(tmp_path / "p1.jpg")
    assert loaded is not None
    assert loaded.min() >= 0.0 and loaded.max() <= 1.0
    assert loaded[0, 0] < 0.1 and loaded[0, 2] > 0.9


def test_load_falls_back_to_the_legacy_png_then_gives_up(tmp_path: Path) -> None:
    image = tmp_path / "p7.jpg"
    assert load_roadprob(image) is None
    legacy = legacy_roadprob_path(image)
    legacy.parent.mkdir(parents=True)
    cv2.imwrite(str(legacy), np.full((4, 5), 128, np.uint8))
    loaded = load_roadprob(image)
    assert loaded is not None
    assert loaded.shape == (4, 5)
    assert abs(float(loaded[0, 0]) - 128 / 255) < 1e-6
    # The sidecar wins once it exists.
    save_roadprob(roadprob_path(image), np.zeros((4, 5), np.float32))
    assert float(load_roadprob(image).max()) == 0.0  # type: ignore[union-attr]


def test_crop_to_panel_takes_the_bounding_box_and_zeroes_outside_the_ring() -> None:
    probability = np.ones((10, 10), np.float32)
    triangle = [(2.0, 2.0), (8.0, 2.0), (2.0, 8.0), (2.0, 2.0)]
    cropped = crop_to_panel(probability, triangle)
    assert cropped.shape == (6, 6)
    assert cropped[0, 0] == 1.0  # inside the triangle
    assert cropped[-1, -1] == 0.0  # bounding box, outside the ring
    assert probability.min() == 1.0  # the parent map is not modified


def test_derive_panel_roadprob_writes_one_map_per_panel(tmp_path: Path) -> None:
    image = tmp_path / "p3.jpg"
    rings = [
        [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)],
        [(10.0, 0.0), (30.0, 0.0), (30.0, 20.0), (10.0, 20.0)],
    ]
    assert derive_panel_roadprob(image, rings, "p3") == []  # no parent map yet

    save_roadprob(roadprob_path(image), np.ones((20, 30), np.float32))
    written = derive_panel_roadprob(image, rings, "p3")
    assert [path.name for path in written] == [
        "p3__1.roadprob.jpg",
        "p3__2.roadprob.jpg",
    ]
    first = load_roadprob(tmp_path / "p3__1.jpg")
    second = load_roadprob(tmp_path / "p3__2.jpg")
    assert first is not None and second is not None
    assert first.shape == (20, 10)
    assert second.shape == (20, 20)
    assert first.min() > 0.9


def test_panel_rings_reads_the_split_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "p4.jpg"
    assert panel_rings(image) == []
    (tmp_path / "p4.panels.json").write_text(
        '{"image": "p4.jpg", "width": 4, "height": 4, '
        '"panels": [[[0, 0], [2, 0], [2, 4], [0, 4]]]}'
    )
    assert panel_rings(image) == [[(0, 0), (2, 0), (2, 4), (0, 4)]]


def test_pending_images_skips_fresh_maps_only_when_resuming(tmp_path: Path) -> None:
    image = tmp_path / "p1.jpg"
    image.write_bytes(b"jpg")
    images = [str(image)]
    assert pending_images(images, resume=False) == images
    assert pending_images(images, resume=True) == images  # no map yet

    save_roadprob(roadprob_path(image), np.zeros((2, 2), np.float32))
    assert pending_images(images, resume=True) == []
    assert pending_images(images, resume=False) == images

    # An image touched after its map needs a fresh one.
    stale = roadprob_path(image).stat().st_mtime
    import os

    os.utime(image, (stale + 10, stale + 10))
    assert pending_images(images, resume=True) == images


def test_quality_is_the_measured_setting() -> None:
    assert QUALITY == 90


def test_derived_map_lands_in_the_panel_jpeg_s_exact_pixel_frame(
    tmp_path: Path,
) -> None:
    """The crop must match split's own, including its fractional-bound rounding."""
    from shapely.geometry import Polygon

    from mapsnap.split import write_panels

    image = tmp_path / "p5.jpg"
    cv2.imwrite(str(image), np.full((40, 60, 3), 200, np.uint8))
    save_roadprob(roadprob_path(image), np.ones((40, 60), np.float32))
    # Fractional bounds: split truncates the low corner and rounds the high one.
    panels = [
        Polygon([(0.0, 0.0), (25.6, 0.0), (25.6, 40.0), (0.0, 40.0)]),
        Polygon([(25.6, 0.4), (60.0, 0.4), (60.0, 40.0), (25.6, 40.0)]),
    ]
    written = write_panels(image, panels, "p5")
    for index, panel_image in enumerate(written, start=1):
        panel = cv2.imread(str(panel_image), cv2.IMREAD_GRAYSCALE)
        derived = load_roadprob(tmp_path / f"p5__{index}.jpg")
        assert panel is not None and derived is not None
        assert derived.shape == panel.shape
