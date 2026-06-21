"""A tiny CRNN that reads a page number's digit string from a crop around its center.

The CNN localizer (mapsnap.detect_numbers_cnn) already finds page-number centers at ~99%
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


def number_strip(image: np.ndarray, cx: float, cy: float, factor: float) -> np.ndarray:
    """Grayscale CRNN_HEIGHT x CRNN_WIDTH crop centered on (cx, cy) in image coords.

    The box is sized in working-scale units and converted to the image's own scale with
    ``factor`` (the localizer's working_scale), so the number fills a consistent fraction.
    """
    half_w = round(BOX_HALF_W_WORKING / factor)
    half_h = round(BOX_HALF_H_WORKING / factor)
    height, width = image.shape[:2]
    x0 = max(0, round(cx) - half_w)
    y0 = max(0, round(cy) - half_h)
    x1 = min(width, round(cx) + half_w)
    y1 = min(height, round(cy) + half_h)
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
