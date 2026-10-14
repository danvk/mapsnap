"""Find printed adjacent-sheet numbers with a trained UNet, for the adjacency step's reader.

``mapsnap adjacency`` reads digits in CRAFT's boxes, and CRAFT never boxes many
of the margin numbers a sheet prints for its neighbours: big thin single-stroke
numerals and the hollow outline font of older volumes. This UNet
(``models/adjacency_unet.pt``, trained by ``mapsnap.train_adjacency_detector``)
marks those numbers directly on the mirror's 25% page images, and its
components replace CRAFT's boxes as the reader's input.

Components cover the whole printed number with a little margin, and EasyOCR
reads stray ink near a box's edge as an extra digit ("25" -> "251"). So each
component is read at three sizes (READ_SIZES) and the most confident read that
resolves to a page key is kept.

Against the hand labels (100-page corpus sample; Hudson vol 9, LA vol 14 and
Nashville 1a), digits read near a labelled number: 64% -> 84% and 74% -> 85%,
with ~0.4 unmatched reads per page instead of ~9.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from mapsnap import page_adjacency
from mapsnap.road_model import UNet, predict_page

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "adjacency_unet.pt"
BASE_WIDTH = 16  # UNet base channels the shipped weights were trained with
THRESHOLD = 0.5  # probability above which a pixel belongs to a number
MIN_AREA = 60  # px; smaller components are specks
# Each component is read grown by 15% of its height per side, bare, and shrunk by 10%.
READ_SIZES = (0.15, 0.0, -0.1)
# EasyOCR divides by a box's width and height, so a read box must keep some of each.
MIN_READ_PX = 2

Component = tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels


def build_detector(device: torch.device | str = "cpu") -> UNet:
    """An untrained detector UNet of the shipped architecture, in eval mode on ``device``."""
    model = UNet(base=BASE_WIDTH, in_channels=1, norm="group").to(device)
    model.eval()
    return model


def load_detector(path: Path = MODEL_PATH, device: torch.device | str = "cpu") -> UNet:
    """The trained detector from ``path``, in eval mode on ``device``."""
    model = build_detector(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def number_components(
    probability: np.ndarray, threshold: float = THRESHOLD, min_area: int = MIN_AREA
) -> list[Component]:
    """Bounding boxes of the connected regions of a probability map above ``threshold``."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (probability >= threshold).astype(np.uint8)
    )
    return [
        (int(x), int(y), int(x + width), int(y + height))
        for x, y, width, height, area in stats[1:count]
        if area >= min_area
    ]


def resized_box(
    component: Component, factor: float, image_size: tuple[int, int]
) -> list[int]:
    """A component grown (factor > 0) or shrunk by ``factor`` x its height per side.

    Returned as EasyOCR's horizontal-box form [x1, x2, y1, y2], clamped to the
    image's (width, height). A shrink takes at most a quarter of each dimension
    per side: a lone "1" is narrower than a fifth of its height, and shrinking
    its sides by a tenth of the height turned the box inside out, which crashed
    EasyOCR (a division by a zero width) and with it the whole item
    (Schenectady 1914).
    """
    x0, y0, x1, y1 = component
    width, height = image_size
    pad = factor * (y1 - y0)
    pad_x = max(pad, -(x1 - x0) / 4)
    pad_y = max(pad, -(y1 - y0) / 4)
    return [
        int(max(0, x0 - pad_x)),
        int(min(width, x1 + pad_x)),
        int(max(0, y0 - pad_y)),
        int(min(height, y1 + pad_y)),
    ]


def readable(box: list[int]) -> bool:
    """Whether a horizontal box [x1, x2, y1, y2] is big enough for EasyOCR to crop and read."""
    x1, x2, y1, y2 = box
    return x2 - x1 >= MIN_READ_PX and y2 - y1 >= MIN_READ_PX


def detection_centre(detection: dict) -> tuple[float, float]:
    """Centre of a digit detection's polygon."""
    xs = [p[0] for p in detection["polygon"]]
    ys = [p[1] for p in detection["polygon"]]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def best_reads(components: list[Component], reads: list[list[dict]]) -> list[dict]:
    """Per component, the most confident detection from any read size that lies within it.

    ``reads`` holds one digit_detections list per read size. A detection belongs
    to a component when its centre falls inside the component grown by the
    largest read size.
    """
    margin = max(READ_SIZES)
    kept = []
    for x0, y0, x1, y1 in components:
        pad = margin * (y1 - y0)
        inside = [
            detection
            for detections in reads
            for detection in detections
            if x0 - pad <= detection_centre(detection)[0] <= x1 + pad
            and y0 - pad <= detection_centre(detection)[1] <= y1 + pad
        ]
        if inside:
            kept.append(max(inside, key=lambda d: d["confidence"]))
    return kept


def detector_digit_detections(
    model: UNet, device: torch.device | str = "cpu"
) -> Callable[..., list[dict]]:
    """A drop-in for page_adjacency.digit_detections that reads detector components.

    The returned function takes digit_detections' arguments and ignores
    ``craft_boxes``: it predicts the page's number components, reads each at
    READ_SIZES through the existing reader, and keeps each component's most
    confident valid read.
    """

    def detections(
        image_path: Path,
        reader,
        valid_keys: set[str],
        craft_boxes: tuple[list, list] | None = None,
        units: list | None = None,
    ) -> list[dict]:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"cannot read image {image_path}")
        height, width = gray.shape
        components = number_components(predict_page(model, gray, device))
        reads = [
            page_adjacency.digit_detections(
                image_path,
                reader,
                valid_keys,
                (
                    [
                        box
                        for c in components
                        if readable(box := resized_box(c, factor, (width, height)))
                    ],
                    [],
                ),
                units=units,
            )
            for factor in READ_SIZES
        ]
        return best_reads(components, reads)

    return detections
