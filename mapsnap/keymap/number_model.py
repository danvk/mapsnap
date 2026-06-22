"""Shared model definition for the page-number localizer (trainer + inference).

A tiny MobileNetV3-small with a single-logit head: given a PATCH_SIZE patch (at the
working SCALE), it predicts whether a bold page number is centered in it. Kept in one
place so training and inference build the identical architecture and preprocessing.
"""

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ImageNet normalization (the pretrained backbone expects it).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def select_device() -> torch.device:
    """Best available torch device: MPS (Apple), else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(pretrained: bool = False) -> nn.Module:
    """MobileNetV3-small with the classifier replaced by a single logit."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    head = model.classifier[-1]
    assert isinstance(head, nn.Linear)
    model.classifier[-1] = nn.Linear(head.in_features, 1)
    return model


def eval_transform() -> transforms.Compose:
    """Deterministic patch → normalized tensor transform (no augmentation)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    """Augmenting transform: small affine + color jitter (paper-color robustness).

    No horizontal/vertical flips — digits are not flip-invariant. Rotation is kept small
    because page numbers are printed upright.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=5, translate=(0.08, 0.08), scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve for binary scores/labels.

    Ranks by descending score and integrates precision over recall increments. Returns
    0.0 if there are no positives.
    """
    order = np.argsort(-scores, kind="stable")
    labels = labels[order].astype(np.int64)
    total_positives = int(labels.sum())
    if total_positives == 0:
        return 0.0
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_positives
    # Sum precision over each positive recall step (recall jumps by 1/total_positives).
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))
