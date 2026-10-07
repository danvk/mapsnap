"""Learned dividing lines for the splitter (#83): a whole-page UNet that draws cutlines.

The splitter's default way of finding the lines that divide a sheet into
panels. The model is trained on OldInsuranceMaps volunteers' cutlines (split
pages whose cuts follow printed ink; mapsnap.train_cutline_unet) and draws
P(cutline) per pixel over a page letterboxed to INPUT_SIZE. The splitter uses
it in place of the classical line detection and divider filtering: the
predicted lines are thinned to centerlines, cut into straight segments, and
handed to split.finalize_panels, which closes near-miss ends, polygonizes and
assembles panels as it does for the classical detector's dividers; panel
boundaries the network never drew are then dissolved (merge_unsupported).

On the cutline benchmark's test fold (760 pages from held-out volumes) this
gets the right panel count on 87.4% of split pages (tuned classical: 76.3%)
and leaves 99.2% of unsplit pages whole (98.5%).

``MAPSNAP_SPLITTER=classical`` selects the classical detector instead, so an
A/B runs one build both ways; ``MAPSNAP_CUTLINE_WEIGHTS`` points at other weights.
"""

import os
from functools import cache
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cutline_unet.pt"
SPLITTER_ENV_VAR = "MAPSNAP_SPLITTER"  # "classical" for the classical detector
WEIGHTS_ENV_VAR = "MAPSNAP_CUTLINE_WEIGHTS"  # other weights than MODEL_PATH
INPUT_SIZE = 768  # letterbox side the model was trained at
LINE_THRESHOLD = 0.3  # P(cutline) at or above this is line
# Hough on the predicted centerlines: the model's lines have small breaks
# (a compass rose, a gap at a junction) that a 120 px bridge closes; 15 px
# left a quarter of split pages uncut on the benchmark (#568).
HOUGH_THRESHOLD = 15
HOUGH_MAX_GAP_PX = 120
HOUGH_MIN_LEN_FRAC = 0.03  # of the shorter side of the cropped page
# Line closing (split.finalize_panels) for the model's lines, tuned on the
# benchmark's held-out training volumes (#568): overshoot junctions by this
# share of the short side, then dissolve any panel boundary that predicted
# lines (P >= LINE_THRESHOLD, within MERGE_SUPPORT_RADIUS_PX) cover less than
# MERGE_MIN_SUPPORT of. The overshoot closes near-misses; the support check
# undoes the closures the network never drew -- Covington 1909 p1's 3% sliver
# between two cutlines, which the key-map rules then refused the whole cut for.
OVERSHOOT_FRAC = 0.14
MERGE_MIN_SUPPORT = 0.3
MERGE_SUPPORT_RADIUS_PX = 5


def cutline_model_path() -> Path | None:
    """The cutline model's weights, or None when MAPSNAP_SPLITTER selects the classical detector."""
    splitter = os.environ.get(SPLITTER_ENV_VAR, "")
    if splitter == "classical":
        return None
    if splitter not in ("", "cutline"):
        raise ValueError(
            f"{SPLITTER_ENV_VAR}={splitter!r}: expected 'cutline' (default) or 'classical'"
        )
    weights = os.environ.get(WEIGHTS_ENV_VAR, "")
    return Path(weights) if weights else MODEL_PATH


@cache
def load_cutline_model(path: Path):
    """(model, device): the UNet with these weights, on the best available device."""
    import torch

    from mapsnap.keymap.number_model import select_device
    from mapsnap.road_model import UNet

    device = select_device()
    state = torch.load(path, map_location=device)
    base = state["enc1.block.0.weight"].shape[0]
    model = UNet(base=base, in_channels=3, norm="group").to(device)
    model.load_state_dict(state)
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


def boundary_support(shared, near_line: np.ndarray, step: float = 4.0) -> float:
    """Share of a (multi)line's length that lies on True pixels of ``near_line``."""
    length = shared.length
    if length <= 0:
        return 1.0
    height, width = near_line.shape
    hits = samples = 0
    for k in range(int(length // step) + 1):
        point = shared.interpolate(min(k * step, length))
        x, y = round(point.x), round(point.y)
        if 0 <= x < width and 0 <= y < height:
            samples += 1
            hits += bool(near_line[y, x])
    return hits / samples if samples else 1.0


def merge_unsupported(panels: list, prob: np.ndarray) -> list:
    """Merge neighbouring panels whose shared boundary the predicted lines do not cover.

    ``panels`` are full-frame polygons and ``prob`` is cutline_probability's
    map. The weakest-supported boundary under MERGE_MIN_SUPPORT is dissolved
    first, repeatedly, until every boundary left is drawn.
    """
    if len(panels) < 2:
        return panels
    r = MERGE_SUPPORT_RADIUS_PX
    near_line = (
        cv2.dilate(
            (prob >= LINE_THRESHOLD * 255).astype(np.uint8),
            np.ones((2 * r + 1, 2 * r + 1), np.uint8),
        )
        > 0
    )
    panels = list(panels)
    while len(panels) > 1:
        weakest: tuple[float, int, int] | None = None
        for i in range(len(panels)):
            for j in range(i + 1, len(panels)):
                shared = panels[i].boundary.intersection(panels[j].boundary)
                if shared.length < 20:  # touching at a point, or barely
                    continue
                support = boundary_support(shared, near_line)
                if support < MERGE_MIN_SUPPORT and (
                    weakest is None or support < weakest[0]
                ):
                    weakest = (support, i, j)
        if weakest is None:
            break
        _, i, j = weakest
        merged = panels[i].union(panels[j])
        if merged.geom_type != "Polygon":
            merged = merged.buffer(0)
        if merged.geom_type != "Polygon":
            break
        panels = [p for k, p in enumerate(panels) if k not in (i, j)] + [merged]
    return panels
