"""Tests for the cutline UNet trainer."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mapsnap.cutline_model import INPUT_SIZE
from mapsnap.train_cutline_unet import (
    augment,
    cutline_loss,
    cutline_mask,
    prepare_page,
    read_image,
    read_manifest,
    to_tensors,
    training_split,
)


def test_training_split_keeps_the_test_fold_and_holds_out_whole_volumes():
    assert training_split({"fold": "test", "item": "sanborn00001_001"}) == "test"
    splits = {
        training_split({"fold": "train", "item": f"sanborn{i:05d}_001"})
        for i in range(200)
    }
    assert splits == {"train", "val"}
    # Every page of a volume lands in the same split.
    row = {"fold": "train", "item": "sanborn04023_023"}
    assert training_split(row) == training_split(dict(row))


def test_cutline_mask_draws_the_line_in_the_letterboxed_frame():
    # A vertical cut down the middle of a 2000 x 1000 (canvas) page, whose image
    # is 1000 x 500: the letterbox scales the long side to INPUT_SIZE.
    cutlines = {"width": 2000, "height": 1000, "cutlines": [[[1000, 0], [1000, 1000]]]}
    mask = cutline_mask(cutlines, (1000, 500))
    column = INPUT_SIZE // 2
    assert mask[: INPUT_SIZE // 2, column].all()
    assert not mask[:, : column - 5].any()
    assert not mask[INPUT_SIZE // 2 + 5 :, :].any()  # below the page: letterbox padding


def test_prepare_page_writes_an_image_and_an_empty_mask_for_an_unsplit_page(
    tmp_path: Path,
):
    benchmark = tmp_path / "benchmark"
    (benchmark / "images").mkdir(parents=True)
    cv2.imwrite(
        str(benchmark / "images" / "a__p1.jpg"), np.zeros((50, 40, 3), np.uint8)
    )
    (benchmark / "manifest.tsv").write_text(
        "image\titem\tlabel\tfold\nimages/a__p1.jpg\ta\tunsplit\ttrain\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    prepare_page(read_manifest(benchmark)[0], benchmark, out)
    image = read_image(out / "a__p1.jpg")
    mask = read_image(out / "a__p1.mask.png", cv2.IMREAD_GRAYSCALE)
    assert image.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert mask.shape == (INPUT_SIZE, INPUT_SIZE) and not mask.any()


def test_prepare_page_rasterizes_a_split_pages_cutlines(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    (benchmark / "images").mkdir(parents=True)
    (benchmark / "labels").mkdir()
    cv2.imwrite(
        str(benchmark / "images" / "a__p2.jpg"), np.zeros((40, 40, 3), np.uint8)
    )
    (benchmark / "labels" / "a__p2.cutlines.json").write_text(
        json.dumps({"width": 40, "height": 40, "cutlines": [[[0, 20], [40, 20]]]})
    )
    (benchmark / "manifest.tsv").write_text(
        "image\titem\tlabel\tfold\nimages/a__p2.jpg\ta\tsplit\ttrain\n"
    )
    prepare_page(read_manifest(benchmark)[0], benchmark, tmp_path)
    mask = read_image(tmp_path / "a__p2.mask.png", cv2.IMREAD_GRAYSCALE)
    assert mask[INPUT_SIZE // 2].any()


def test_augment_flips_the_mask_with_the_image():
    image = np.zeros((4, 4, 3), np.uint8)
    image[0, 0] = 255
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 0] = 255
    for seed in range(8):
        out_image, out_mask = augment(image, mask, np.random.default_rng(seed))
        brightest = np.unravel_index(out_image[..., 0].argmax(), (4, 4))
        assert out_mask[brightest] == 255


def test_cutline_loss_prefers_the_right_lines():
    mask = np.zeros((32, 32), np.uint8)
    mask[:, 16] = 255
    _, target = to_tensors(
        [(np.zeros((32, 32, 3), np.uint8), mask)], torch.device("cpu")
    )
    right = torch.where(target > 0, 8.0, -8.0)
    empty = torch.full_like(target, -8.0)
    assert cutline_loss(right, target) < cutline_loss(empty, target)
