import json
import os
from pathlib import Path

import pytest

from mapsnap.craft import expand_images, pending_images
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
