"""Tests for the street-recognizer dataset build and decode helpers."""

import json

import numpy as np
from PIL import Image

from mapsnap.train_street_recognizer import (
    HOLDOUT_VOLUMES,
    extract_crop,
    greedy_decode,
    load_harvest_pairs,
)


def write_harvest(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def record(
    cls="INLIER",
    vol="chicago_il_1950_vol_1",
    text="MAIN",
    label="MAIN",
    conf=0.9,
    poly=None,
):
    return {
        "cls": cls,
        "vol": vol,
        "stem": "p1",
        "text": text,
        "label": label,
        "conf": conf,
        "poly": poly or [[10, 20], [30, 20], [30, 28], [10, 28]],
        "angle": 0,
    }


def test_load_harvest_pairs_agreement_and_holdout(tmp_path):
    records = [
        record(),  # kept
        record(
            text="MA1N", label="MAIN", poly=[[100, 20], [130, 20], [130, 28], [100, 28]]
        ),  # disagreement: dropped
        record(
            cls="LOWCONF", poly=[[200, 20], [230, 20], [230, 28], [200, 28]]
        ),  # kept
        record(vol=HOLDOUT_VOLUMES[0]),  # holdout: dropped
        record(cls="WRONGREAD", text="XX", label="YY"),  # junk pool
    ]
    harvest = tmp_path / "h.jsonl"
    write_harvest(harvest, records)
    pairs, junk = load_harvest_pairs(harvest)
    assert len(pairs) == 2
    assert len(junk) == 1
    assert all(r["text"] == r["label"] for r in pairs)


def test_load_harvest_pairs_confidence_floor(tmp_path):
    # The sub-floor tail is where the 180-flips and forced junk matches live
    # (OLIVE at conf 0.0000); they must not become training pairs.
    harvest = tmp_path / "h.jsonl"
    write_harvest(harvest, [record(conf=0.0132), record(conf=0.15)])
    pairs, _ = load_harvest_pairs(harvest)
    assert len(pairs) == 1
    assert pairs[0]["conf"] == 0.15


def test_load_harvest_pairs_dedupes_rotation_twins(tmp_path):
    # Same box read at 90 and 270 both position-match as inliers; keep the
    # confident one.
    twin_a = record(conf=0.95)
    twin_b = record(conf=0.4, poly=[[11, 21], [31, 21], [31, 29], [11, 29]])
    far = record(poly=[[300, 20], [330, 20], [330, 28], [300, 28]])
    harvest = tmp_path / "h.jsonl"
    write_harvest(harvest, [twin_b, twin_a, far])
    pairs, _ = load_harvest_pairs(harvest)
    assert len(pairs) == 2
    assert {p["conf"] for p in pairs} == {0.95, 0.9}


def test_load_harvest_pairs_include_volumes_overrides_holdout(tmp_path):
    harvest = tmp_path / "h.jsonl"
    write_harvest(harvest, [record(vol=HOLDOUT_VOLUMES[0])])
    pairs, _ = load_harvest_pairs(harvest, include_volumes=(HOLDOUT_VOLUMES[0],))
    assert len(pairs) == 1


def test_extract_crop_angle_rotation(tmp_path):
    # A 40x30 page: white with a dark 4x8 bar at (10..14, 20..28).
    page = np.full((30, 40), 255, dtype=np.uint8)
    page[20:28, 10:14] = 0
    vol_dir = tmp_path / "vol"
    vol_dir.mkdir()
    Image.fromarray(page).save(vol_dir / "p1.jpg")
    rec = {
        "vol": "vol",
        "stem": "p1",
        "poly": [[10, 20], [13, 20], [13, 27], [10, 27]],
        "angle": 90,
    }
    crop = extract_crop(tmp_path, rec, {})
    # 4x8 vertical bar rotated 90deg -> 8x4 (wider than tall), mostly dark.
    assert crop.shape == (4, 8)
    assert (crop < 128).mean() > 0.8


def test_greedy_decode_collapses_repeats_and_blanks():
    import torch

    character = ["[blank]", "A", "B"]
    # Timesteps: A A blank A B B -> "AAB"
    steps = [1, 1, 0, 1, 2, 2]
    logits = torch.full((1, len(steps), 3), -10.0)
    for t, idx in enumerate(steps):
        logits[0, t, idx] = 10.0
    assert greedy_decode(logits, character) == ["AAB"]
