import json
import os
from pathlib import Path

import pytest

from mapsnap.craft import expand_images, pending_images, process_pending
from mapsnap.detect_text import boxes_path, craft_hint, missing_boxes, require_boxes


def touch(path: Path, mtime: float | None = None) -> Path:
    path.write_text("{}")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_expand_images_globs_and_dedupes(tmp_path):
    for name in ("p2.jpg", "p10.jpg", "p1.jpg"):
        (tmp_path / name).write_bytes(b"")
    got = expand_images([str(tmp_path / "p*.jpg"), str(tmp_path / "p1.jpg")])
    assert [Path(p).name for p in got] == ["p1.jpg", "p10.jpg", "p2.jpg"]
    # A literal path is passed through even before it exists (checked later).
    assert expand_images([str(tmp_path / "absent.jpg")]) == [
        str(tmp_path / "absent.jpg")
    ]


def test_pending_images_resume_skips_only_current_boxes(tmp_path):
    image = tmp_path / "p1.jpg"
    image.write_bytes(b"")
    stale = tmp_path / "p2.jpg"
    stale.write_bytes(b"")
    images = [str(image), str(stale)]
    assert pending_images(images, resume=False) == images
    # No boxes at all -> pending.
    assert pending_images(images, resume=True) == images
    # Fresh boxes -> skipped; boxes older than the image -> still pending.
    touch(Path(boxes_path(str(image))), mtime=os.path.getmtime(image) + 10)
    touch(Path(boxes_path(str(stale))), mtime=os.path.getmtime(stale) - 10)
    assert pending_images(images, resume=True) == [str(stale)]


def test_missing_boxes_and_hint(tmp_path):
    have = tmp_path / "p1.jpg"
    lack = tmp_path / "p2.jpg"
    for image in (have, lack):
        image.write_bytes(b"")
    Path(boxes_path(str(have))).write_text(json.dumps({"boxes": []}))
    assert missing_boxes([str(have), str(lack)]) == [str(lack)]
    hint = craft_hint([str(lack)])
    assert hint == f"mapsnap craft '{tmp_path}/p*.jpg'"


def test_require_boxes_exits_with_the_craft_command(tmp_path):
    image = tmp_path / "p1.jpg"
    image.write_bytes(b"")
    with pytest.raises(SystemExit) as excinfo:
        require_boxes([str(image)])
    message = str(excinfo.value)
    assert "p1.jpg" in message and "mapsnap craft" in message
    # With boxes present it is a no-op.
    Path(boxes_path(str(image))).write_text(json.dumps({"boxes": []}))
    require_boxes([str(image)])


def test_volume_craft_images_covers_panels_and_split_parents(tmp_path):
    # ocr reads panels; adjacency reads the parent sheet. Both need boxes.
    for name in ("p1.jpg", "p2.jpg", "p2__1.jpg", "p2__2.jpg"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "p2.panels.json").write_text(json.dumps({"panels": [[], []]}))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "p0.jpg").write_bytes(b"")
    from mapsnap.craft import volume_craft_images

    stems = {Path(p).stem for p in volume_craft_images(tmp_path, ["p0"])}
    assert {"p1", "p2", "p2__1", "p2__2", "p0"} <= stems


def test_process_pending_detects_parents_before_deriving_panels(tmp_path):
    # A split sheet whose parent has NO boxes yet (deleted before a re-run):
    # the parent must be detected first so the panel can be derived from it,
    # and the panel must then not be detected on its own.
    from PIL import Image

    parent = tmp_path / "p12.jpg"
    Image.new("RGB", (200, 100), "white").save(parent)
    panel = tmp_path / "p12__1.jpg"
    Image.new("RGB", (100, 100), "white").save(panel)
    (tmp_path / "p12.panels.json").write_text(
        json.dumps(
            {
                "image": "p12.jpg",
                "width": 200,
                "height": 100,
                "panels": [
                    [[0, 0], [100, 0], [100, 100], [0, 100]],
                    [[100, 0], [200, 0], [200, 100], [100, 100]],
                ],
            }
        )
    )
    detected: list[str] = []

    def detect(image: str) -> None:
        detected.append(Path(image).name)
        Path(image).with_suffix(".boxes.json").write_text(
            json.dumps({"width": 200, "height": 100, "boxes": [], "command": []})
        )

    derived, count = process_pending([str(panel), str(parent)], detect)
    assert detected == ["p12.jpg"]
    assert derived == 1 and count == 1
    assert (tmp_path / "p12__1.boxes.json").exists()
