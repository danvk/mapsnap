"""Learned dividing lines for the splitter (#83): a whole-page UNet that draws cutlines.

The model is trained on OldInsuranceMaps volunteers' cutlines (8.5k split pages
whose cuts follow printed ink) and draws P(cutline) per pixel over a page
letterboxed to INPUT_SIZE. The splitter uses it in place of its own line
detection and divider filtering: the predicted lines are thinned to centerlines,
cut into straight segments, and handed to split.finalize_panels, which closes
near-miss ends, polygonizes and assembles panels exactly as it does for the
classical detector's dividers.

On the cutline benchmark's test fold (760 pages from held-out volumes) this
gets the right panel count on 85.5% of split pages (tuned classical: 76.3%) and
leaves 99.5% of unsplit pages whole (98.5%).

Enabled by MODEL_ENV_VAR (the weights' path, or "default" for MODEL_PATH), so a
corpus A/B runs one build with and without it.
"""

import os
from functools import cache
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cutline_unet.pt"
MODEL_ENV_VAR = "MAPSNAP_CUTLINE_MODEL"
INPUT_SIZE = 768  # letterbox side the model was trained at
LINE_THRESHOLD = 0.3  # P(cutline) at or above this is line
# Hough on the predicted centerlines: the model's lines have small breaks
# (a compass rose, a gap at a junction) that a 90 px bridge closes; 15 px
# left a quarter of split pages uncut on the benchmark.
HOUGH_THRESHOLD = 15
HOUGH_MAX_GAP_PX = 90
HOUGH_MIN_LEN_FRAC = 0.03  # of the shorter side of the cropped page


def enabled_model_path() -> Path | None:
    """The cutline model's weights if MODEL_ENV_VAR turns it on, else None."""
    value = os.environ.get(MODEL_ENV_VAR, "")
    if not value:
        return None
    return MODEL_PATH if value == "default" else Path(value)


@cache
def load_cutline_model(path: Path):
    """(model, device): the UNet with these weights, on the best available device.

    Checkpoints from the first training run carry a retired page-level head
    (``page.*``); its weights are dropped.
    """
    import torch

    from mapsnap.keymap.number_model import select_device
    from mapsnap.road_model import UNet

    device = select_device()
    state = torch.load(path, map_location=device)
    base = state["enc1.block.0.weight"].shape[0]
    model = UNet(base=base, in_channels=3, norm="group").to(device)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("page.")})
    model.eval()
    return model, device


def letterbox(rgb: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize preserving aspect onto a white size x size canvas, anchored top-left."""
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((size, size, 3), 255, np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas


def cutline_probability(rgb: np.ndarray, path: Path = MODEL_PATH) -> np.ndarray:
    """P(cutline) as uint8 (0-255) at the page's own resolution.

    ``rgb`` is the full page, in RGB channel order (the model was trained on
    OpenCV-decoded BGR, so the channels are swapped back here).
    """
    import torch

    model, device = load_cutline_model(path)
    boxed = letterbox(rgb)[:, :, ::-1].astype(np.float32)  # RGB -> BGR, as trained
    boxed = (boxed - boxed.mean()) / (boxed.std() + 1e-6)
    tensor = torch.from_numpy(boxed.transpose(2, 0, 1).copy()[None]).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    letterboxed = (prob * 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    scale = INPUT_SIZE / max(height, width)
    crop = letterboxed[: max(1, round(height * scale)), : max(1, round(width * scale))]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def cutline_segments(prob: np.ndarray, border: int) -> np.ndarray:
    """Straight segments along the predicted lines, in the splitter's cropped frame.

    ``prob`` is cutline_probability's full-page map; ``border`` is the margin the
    splitter crops off each edge before it works.
    """
    from skimage.morphology import skeletonize

    height, width = prob.shape
    lines = (
        prob[border : height - border, border : width - border] >= LINE_THRESHOLD * 255
    )
    skeleton = skeletonize(lines).astype(np.uint8) * 255
    found = cv2.HoughLinesP(
        skeleton,
        1,
        np.pi / 360,
        threshold=HOUGH_THRESHOLD,
        minLineLength=int(HOUGH_MIN_LEN_FRAC * min(lines.shape)),
        maxLineGap=HOUGH_MAX_GAP_PX,
    )
    if found is None:
        return np.zeros((0, 4))
    return found[:, 0, :].astype(float)
