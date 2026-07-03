"""A tiny CRNN that reads a page number's digit string from a crop around its center.

The CNN localizer (mapsnap.keymap.detect_numbers_cnn) already finds page-number centers at ~99%
recall, so this recognizer is fed a fixed crop around each center and reads the 1-3 digit
string directly — no CRAFT boxing, no EasyOCR. Architecture: a small conv stack collapses
the crop's height to a width-sequence, a BiLSTM adds context, and a linear head emits
per-step class logits decoded with CTC.

Crops are taken at the image's native resolution from a fixed box sized in working-scale
units (so a page number is a roughly constant fraction of the crop regardless of scan
DPI; see keymap_patches.working_scale), then resized to CRNN_HEIGHT x CRNN_WIDTH.
"""

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms

from mapsnap.keymap.keymap_patches import NUMBER_MAX_H_FULL, SCALE

# Crop input size fed to the CRNN (grayscale).
CRNN_HEIGHT = 48
CRNN_WIDTH = 96

# Fixed crop box around the candidate center, in working-scale px (mapped to the image's
# own scale per call). Wide enough for a 3-digit number (~70px at working scale) with
# margin, tight enough to mostly exclude neighboring blocks.
BOX_HALF_W_WORKING = 55
BOX_HALF_H_WORKING = 30

# Classes: index 0 is the CTC blank; 1..10 map to digits '0'..'9'.
BLANK_INDEX = 0
NUM_CLASSES = 11

IMAGENET_GRAY_MEAN = [0.5]
IMAGENET_GRAY_STD = [0.5]


def encode_text(text: str) -> list[int]:
    """Encode a digit string to CTC class indices (digit d -> d + 1)."""
    return [int(c) + 1 for c in text if c.isdigit()]


def ctc_greedy_decode(indices: list[int]) -> str:
    """Collapse a per-timestep argmax index sequence to a digit string (CTC rule)."""
    out: list[str] = []
    prev = -1
    for idx in indices:
        if idx != prev and idx != BLANK_INDEX:
            out.append(str(idx - 1))
        prev = idx
    return "".join(out)


def strip_crop_box(
    width: int,
    height: int,
    cx: float,
    cy: float,
    factor: float,
    *,
    half_w_working: float = BOX_HALF_W_WORKING,
) -> tuple[int, int, int, int]:
    """Clamped (x0, y0, x1, y1) of the strip's source region around (cx, cy) in image coords.

    Box is sized in working-scale units and converted to the image's own scale with
    ``factor`` (the localizer's working_scale), so a number fills a consistent fraction. A
    tighter ``half_w_working`` un-squishes a multi-digit number the default width would blur.
    """
    half_w = round(half_w_working / factor)
    half_h = round(BOX_HALF_H_WORKING / factor)
    x0 = max(0, round(cx) - half_w)
    y0 = max(0, round(cy) - half_h)
    x1 = min(width, round(cx) + half_w)
    y1 = min(height, round(cy) + half_h)
    return x0, y0, x1, y1


def number_strip(
    image: np.ndarray,
    cx: float,
    cy: float,
    factor: float,
    *,
    half_w_working: float = BOX_HALF_W_WORKING,
) -> np.ndarray:
    """Grayscale CRNN_HEIGHT x CRNN_WIDTH crop centered on (cx, cy) in image coords."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = strip_crop_box(
        width, height, cx, cy, factor, half_w_working=half_w_working
    )
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return np.full((CRNN_HEIGHT, CRNN_WIDTH), 255, dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (CRNN_WIDTH, CRNN_HEIGHT), interpolation=cv2.INTER_AREA)


def eval_transform() -> transforms.Compose:
    """Grayscale strip -> normalized 1xHxW tensor (no augmentation)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_GRAY_MEAN, IMAGENET_GRAY_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    """Augmenting transform for the recognizer (no flips, digits upright).

    Translation is deliberately large: at inference the crop is centered on the CNN
    candidate, which is off the true center by up to ~a localizer stride, so the CRNN must
    read off-center digits. Training only on well-centered crops leaves it brittle.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=4, translate=(0.15, 0.15), scale=(0.85, 1.15)
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_GRAY_MEAN, IMAGENET_GRAY_STD),
        ]
    )


def _conv_block(in_ch: int, out_ch: int, pool: tuple[int, int]) -> nn.Sequential:
    """Conv(3x3) + BN + ReLU + MaxPool(pool)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class CRNN(nn.Module):
    """Conv stack -> BiLSTM -> per-timestep class logits (log-softmax for CTC).

    For a 48x96 grayscale input the conv stack yields a length-24 width sequence with the
    height collapsed to 1; the BiLSTM and linear head produce (T=24, N, NUM_CLASSES).
    """

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_block(1, 32, (2, 2)),  # 48x96 -> 24x48
            _conv_block(32, 64, (2, 2)),  # -> 12x24
            _conv_block(64, 128, (2, 1)),  # -> 6x24
            _conv_block(128, 128, (2, 1)),  # -> 3x24
            _conv_block(128, 128, (3, 1)),  # -> 1x24
        )
        self.rnn = nn.LSTM(128, 64, bidirectional=True, batch_first=False)
        self.head = nn.Linear(128, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)  # (N, 128, 1, T)
        features = features.squeeze(2).permute(2, 0, 1)  # (T, N, 128)
        seq, _ = self.rnn(features)  # (T, N, 128)
        logits = self.head(seq)  # (T, N, NUM_CLASSES)
        return logits.log_softmax(dim=2)


def build_crnn() -> CRNN:
    """Construct the CRNN recognizer."""
    return CRNN()


def decode_batch(log_probs: torch.Tensor) -> list[str]:
    """Greedy-CTC decode a (T, N, C) log-prob batch into N digit strings."""
    best = log_probs.argmax(dim=2).permute(1, 0).cpu().numpy()  # (N, T)
    return [ctc_greedy_decode(row.tolist()) for row in best]


def greedy_paths(log_probs: torch.Tensor) -> list[list[int]]:
    """Per-sample raw argmax index path (length T) for a (T, N, C) log-prob batch."""
    best = log_probs.argmax(dim=2).permute(1, 0).cpu().numpy()  # (N, T)
    return [row.tolist() for row in best]


# A blank run of at least this many timesteps separates two numbers (within a number, digits
# are <=2-3 blanks apart; between adjacent numbers the whitespace is ~7+ blanks).
GAP_STEPS = 4


def central_group(path: list[int], *, gap: int = GAP_STEPS) -> tuple[int, int] | None:
    """(first, last) span of the firing cluster nearest the strip center, or None if all blank.

    The CRNN crop is wide enough to catch a neighboring number; the path then has two firing
    clusters separated by a wide blank run. The candidate is centered on its own number, so
    the cluster nearest the center timestep is the one to keep — both for decoding and for
    boxing — which drops the neighbor's digits. Clusters are split on blank runs >= ``gap``.
    """
    clusters: list[tuple[int, int]] = []
    start: int | None = None
    last = 0
    blanks = 0
    for t, idx in enumerate(path):
        if idx != BLANK_INDEX:
            if start is None:
                start = t
            last = t
            blanks = 0
        elif start is not None:
            blanks += 1
            if blanks >= gap:
                clusters.append((start, last))
                start = None
    if start is not None:
        clusters.append((start, last))
    if not clusters:
        return None

    center = (len(path) - 1) / 2

    def distance(cluster: tuple[int, int]) -> float:
        lo, hi = cluster
        if lo <= center <= hi:
            return 0.0
        return min(abs(lo - center), abs(hi - center))

    return min(clusters, key=distance)


def ink_row_center(ink_per_row: np.ndarray) -> float | None:
    """Ink-weighted mean row of a per-row ink count (the digits' vertical center), or None.

    A weighted centroid is robust to the speckle and shadows on a colored/ornate block that
    defeat thresholded band-finding: stray dark pixels barely move the center of mass.
    """
    total = float(ink_per_row.sum())
    if total <= 0:
        return None
    return float((np.arange(len(ink_per_row)) * ink_per_row).sum() / total)


# Half-height (strip rows) of a page number: it is ~NUMBER_MAX_H_FULL px tall in a crop
# 2*BOX_HALF_H_WORKING working px tall, scaled into the CRNN_HEIGHT strip.
NUMBER_HALF_H_STRIP = (
    (NUMBER_MAX_H_FULL * SCALE / 2) / BOX_HALF_H_WORKING * (CRNN_HEIGHT / 2)
)


def locate_number(
    strip: np.ndarray,
    span: tuple[int, int],
    steps: int,
    crop_box: tuple[int, int, int, int],
    *,
    pad_frac: float = 0.15,
) -> list[list[int]]:
    """Tight box (image coords) around one number, given its CTC timestep ``span``.

    Horizontal extent comes from the firing timesteps in ``span`` — each timestep maps to a
    known column of the strip (``steps`` total), giving a precise, well-centered span.
    Vertical position comes from the ink centroid within those columns, with a number-sized
    height around it (robust: fragile band-finding under-/over-segments ornate digits).
    ``crop_box`` is the strip's source region (x0, y0, x1, y1) in image coords (see
    strip_crop_box).
    """
    first_t, last_t = span
    height, width = strip.shape
    cell = width / steps
    sx_lo = first_t * cell
    sx_hi = (last_t + 1) * cell

    col0 = max(0, int(sx_lo))
    col1 = min(width, int(round(sx_hi)))
    sub = strip[:, col0:col1] if col1 > col0 else strip
    _, ink = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    center = ink_row_center(ink.sum(axis=1) / 255)
    if center is None:
        sy_lo, sy_hi = 0.0, float(height)
    else:
        half = NUMBER_HALF_H_STRIP * (1 + pad_frac)
        sy_lo = max(0.0, center - half)
        sy_hi = min(float(height), center + half)

    pad_x = pad_frac * (sx_hi - sx_lo)
    sx_lo = max(0.0, sx_lo - pad_x)
    sx_hi = min(float(width), sx_hi + pad_x)

    x0, y0, x1, y1 = crop_box
    crop_w = x1 - x0
    crop_h = y1 - y0
    ix_lo = round(x0 + sx_lo / width * crop_w)
    ix_hi = round(x0 + sx_hi / width * crop_w)
    iy_lo = round(y0 + sy_lo / height * crop_h)
    iy_hi = round(y0 + sy_hi / height * crop_h)
    return [[ix_lo, iy_lo], [ix_hi, iy_lo], [ix_hi, iy_hi], [ix_lo, iy_hi]]
