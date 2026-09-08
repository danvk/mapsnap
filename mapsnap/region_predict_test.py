"""Tests for the per-page P(region) map writer."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mapsnap.region_predict import region_prob_path, write_region_maps


class HalfModel(torch.nn.Module):
    """Stub UNet: zero logits everywhere, so P(region) is 0.5 at every pixel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])


def write_page(path: Path, width: int, height: int) -> None:
    cv2.imwrite(str(path), np.full((height, width, 3), 200, np.uint8))


def test_writes_maps_at_page_resolution_and_keeps_existing(tmp_path: Path) -> None:
    write_page(tmp_path / "p7.jpg", 40, 30)
    write_page(tmp_path / "p7__1.jpg", 20, 30)
    model, device = HalfModel(), torch.device("cpu")
    images = sorted(tmp_path.glob("p*.jpg"))

    written = write_region_maps(tmp_path, images, model=model, device=device)

    assert written == [
        region_prob_path(tmp_path, "p7"),
        region_prob_path(tmp_path, "p7__1"),
    ]
    prob = cv2.imread(str(region_prob_path(tmp_path, "p7")), cv2.IMREAD_GRAYSCALE)
    assert prob is not None
    assert prob.shape == (30, 40)
    assert np.all(prob == 127)  # sigmoid(0) * 255, truncated
    panel = cv2.imread(str(region_prob_path(tmp_path, "p7__1")), cv2.IMREAD_GRAYSCALE)
    assert panel is not None
    assert panel.shape == (30, 20)

    # A second pass rewrites nothing; --force redoes everything.
    assert write_region_maps(tmp_path, images, model=model, device=device) == []
    redone = write_region_maps(tmp_path, images, model=model, device=device, force=True)
    assert len(redone) == 2


def test_skips_unreadable_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "p9.jpg").write_bytes(b"not a jpeg")
    written = write_region_maps(
        tmp_path, [tmp_path / "p9.jpg"], model=HalfModel(), device=torch.device("cpu")
    )
    assert written == []
    assert not region_prob_path(tmp_path, "p9").exists()
    assert "skip (unreadable)" in capsys.readouterr().err
