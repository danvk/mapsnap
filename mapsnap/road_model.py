"""Road-corridor segmentation for Sanborn pages: a small UNet with free auto-labels.

Street corridors are the geometry drawn consistently on *both* renderings of a page seam
(each sheet draws its own blocks densely but its neighbor's sketchily — appearance differs,
corridors don't), which makes them the substrate for page-to-page geometric matching. Color
and ink heuristics are brittle across volumes, so this trains a segmentation model instead —
and the training labels are free: every already-georeferenced page maps OSM centerlines into
its own pixel space, giving thousands of (image, road-mask) pairs with no manual annotation.
The labels are weak (the modern street network differs from the 1900s one here and there),
which a segmentation loss tolerates.

The model is a UNet: a convolutional encoder-decoder with skip connections, the standard
architecture when the output is an image-sized mask that needs both context (is this white
strip a street or a courtyard? — decided by structure hundreds of pixels away) and precise
localization (the corridor's edges), see ``UNet``.

Train with mapsnap.train_road_unet; predict a full page with :func:`predict_page`. The model's
raw output is a per-pixel road *probability* heat map; :func:`road_mask` / :func:`road_skeleton` /
:func:`skeleton_junctions` derive the thresholded mask, 1-px centerlines, and junction points from
it. Run this module as a script to visualize any page (``--mode heatmap|segments|both``).
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

# Rasterized road width for training labels, in metres: Sanborn residential streets run
# ~15-20 m between block faces; drawing the OSM centerline at 12 m keeps the label inside
# the corridor even where the fit is a few metres off.
ROAD_WIDTH_M = 12.0

# Patch size the model trains on; must be divisible by 2**4 (four poolings).
PATCH = 256


def page_world_affine(georef: dict) -> np.ndarray:
    """2x3 affine mapping page pixels to (lon, lat), from the georef corner quad."""
    corners = np.asarray(georef["corners"], dtype=np.float64)  # TL, TR, BR, BL
    width, height = georef["width"], georef["height"]
    u = (corners[1] - corners[0]) / width
    v = (corners[3] - corners[0]) / height
    return np.array([[u[0], v[0], corners[0][0]], [u[1], v[1], corners[0][1]]])


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine, as a 2x3 affine."""
    square = np.vstack([matrix, [0.0, 0.0, 1.0]])
    return np.linalg.inv(square)[:2]


def page_scale_m_per_px(georef: dict) -> float:
    """Approximate metres per pixel of a georeferenced page (mean of the two axes)."""
    corners = np.asarray(georef["corners"], dtype=np.float64)
    lat = float(corners[:, 1].mean())
    kx = 111_320.0 * math.cos(math.radians(lat))
    ky = 110_540.0
    top = (corners[1] - corners[0]) * (kx, ky)
    left = (corners[3] - corners[0]) * (kx, ky)
    return (
        np.linalg.norm(top) / georef["width"] + np.linalg.norm(left) / georef["height"]
    ) / 2


def rasterize_road_mask(
    georef: dict, features: list[dict], width_m: float = ROAD_WIDTH_M
) -> np.ndarray:
    """Auto-label: OSM centerlines drawn into the page's pixel space as a road mask.

    Projects every (Multi)LineString vertex through the inverse of the page's georef
    transform and strokes the polylines at ``width_m`` (converted through the page's own
    scale). Returns a uint8 mask (0/255) of shape (height, width).
    """
    world_to_px = invert_affine(page_world_affine(georef))
    thickness = max(3, round(width_m / page_scale_m_per_px(georef)))
    mask = np.zeros((georef["height"], georef["width"]), np.uint8)
    for feature in features:
        geometry = feature.get("geometry", {})
        kind = geometry.get("type")
        lines = (
            [geometry["coordinates"]]
            if kind == "LineString"
            else geometry.get("coordinates", [])
            if kind == "MultiLineString"
            else []
        )
        for line in lines:
            pts = np.asarray(line, dtype=np.float64)
            px = pts @ world_to_px[:, :2].T + world_to_px[:, 2]
            # Skip lines entirely outside a generous page margin.
            if (
                px[:, 0].max() < -200
                or px[:, 0].min() > georef["width"] + 200
                or px[:, 1].max() < -200
                or px[:, 1].min() > georef["height"] + 200
            ):
                continue
            cv2.polylines(
                mask,
                [px.round().astype(np.int32)],
                isClosed=False,
                color=255,
                thickness=thickness,
            )
    return mask


class DoubleConv(nn.Module):
    """Two 3x3 conv + batch-norm + ReLU blocks, the UNet's basic unit."""

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """A small UNet: grayscale page patch in, per-pixel road logit out.

    Encoder halves resolution four times (so the deepest features see ~16x16 patch cells =
    broad context: block structure, corridor continuity); the decoder doubles it back,
    concatenating the encoder's same-resolution features at each step (skip connections) so
    the output regains pixel-precise edges the downsampling destroyed.
    """

    def __init__(self, base: int = 24):
        super().__init__()
        self.enc1 = DoubleConv(1, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = DoubleConv(base * 16 + base * 8, base * 8)
        self.dec3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.dec1 = DoubleConv(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.head(d1)


def normalize_patch(gray: np.ndarray) -> np.ndarray:
    """Grayscale uint8 -> float32 in roughly [-1, 1] (paper is bright, ink dark)."""
    return (gray.astype(np.float32) / 255.0 - 0.5) / 0.5


@torch.no_grad()
def predict_page(
    model: nn.Module,
    gray: np.ndarray,
    device,
    *,
    tile: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Road probability for a full page, predicted in overlapping tiles.

    Tiles of ``tile`` px with ``overlap`` px margins are predicted independently and
    averaged where they overlap (a Hann-like weight avoids seams). Returns float32
    probabilities in [0, 1] with the page's shape.
    """
    model.eval()
    height, width = gray.shape
    probabilities = np.zeros((height, width), np.float32)
    weights = np.full((height, width), 1e-6, np.float32)
    ramp = np.minimum(np.linspace(0, 1, tile), np.linspace(1, 0, tile))
    window = np.clip(np.outer(ramp, ramp).astype(np.float32) * 16, 0.05, 1.0)
    stride = tile - overlap
    ys = list(range(0, max(1, height - tile + 1), stride))
    xs = list(range(0, max(1, width - tile + 1), stride))
    if ys[-1] != max(0, height - tile):
        ys.append(max(0, height - tile))
    if xs[-1] != max(0, width - tile):
        xs.append(max(0, width - tile))
    for y0 in ys:
        for x0 in xs:
            patch = gray[y0 : y0 + tile, x0 : x0 + tile]
            ph, pw = patch.shape
            padded = np.pad(
                patch, ((0, tile - ph), (0, tile - pw)), constant_values=210
            )
            tensor = (
                torch.from_numpy(normalize_patch(padded))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()[:ph, :pw]
            probabilities[y0 : y0 + ph, x0 : x0 + pw] += prob * window[:ph, :pw]
            weights[y0 : y0 + ph, x0 : x0 + pw] += window[:ph, :pw]
    return probabilities / weights


# The shipped checkpoint, anchored to the repo so commands work from any CWD.
ROAD_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "road_unet.pt"


def load_model(model_path: Path, device) -> "UNet":
    """Load a trained road UNet from a checkpoint, moved to ``device`` and set to eval mode.

    The channel width is inferred from the checkpoint so differently-sized
    models (--base at training time) load transparently.
    """
    state_dict = torch.load(str(model_path), map_location=device)
    model = UNet(base=int(state_dict["enc1.block.0.weight"].shape[0]))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def effective_gcp_count(georef: dict) -> int:
    """Distinct physical intersections among a fitted page's inliers.

    OSM name variants (SOUTH CAPITOL ST SE / SW / bare) expand one physical
    intersection into several inlier records, so raw inlier counts overstate
    how constrained a fit is; cluster by pixel distance instead. Pages with
    fewer than ~3 of these are fragile fits whose OSM projection may not
    match the drawn streets.
    """
    clusters: list[tuple[float, float]] = []
    for intersection in georef.get("intersections", []):
        if not intersection.get("inlier"):
            continue
        x, y = intersection["x"], intersection["y"]
        if all(math.hypot(x - cx, y - cy) >= 60 for cx, cy in clusters):
            clusters.append((x, y))
    return len(clusters)


def road_mask(
    probabilities: np.ndarray, *, threshold: float = 0.5, min_area: int = 2000
) -> np.ndarray:
    """Binarize the probability heat map and drop specks smaller than ``min_area`` px.

    Returns a uint8 (0/255) mask; the small-blob filter removes isolated false positives
    (stray colored blocks, labels) that would otherwise fragment the skeleton.
    """
    binary = (probabilities > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(int(count), dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > min_area
    return keep[np.asarray(labels, dtype=np.intp)].astype(np.uint8) * 255


def road_skeleton(mask: np.ndarray) -> np.ndarray:
    """Thin a road mask to a 1-pixel-wide centerline skeleton (boolean array)."""
    from skimage.morphology import skeletonize

    return skeletonize(mask > 0)


def skeleton_junctions(skeleton: np.ndarray) -> np.ndarray:
    """Junction points (>=3-way skeleton pixels), clustered to one (x, y) per intersection.

    A skeleton pixel with three or more skeleton neighbors is a branch point; adjacent such
    pixels are merged via connected components. Returns an (N, 2) float array of (x, y) centroids.
    """
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    junction_pixels = skeleton & (neighbors >= 4)  # self + 3 neighbors
    _, _, _, centroids = cv2.connectedComponentsWithStats(
        junction_pixels.astype(np.uint8), connectivity=8
    )
    return np.asarray(centroids)[1:]


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    """Draw an outlined caption in the top-left of a BGR image (in place); return it."""
    outline: tuple[int, int, int] = (0, 0, 0)
    fill: tuple[int, int, int] = (0, 255, 255)
    for color, thickness in ((outline, 4), (fill, 2)):
        cv2.putText(
            image, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness
        )
    return image


def heatmap_overlay(
    gray: np.ndarray, probabilities: np.ndarray, *, max_alpha: float = 0.75
) -> np.ndarray:
    """The raw model output visualized: a JET colormap of P(road), alpha-weighted by P itself.

    Low-probability areas stay as the page (alpha ~ 0), so the page shows through where the model
    is unsure and glows where it is confident — the honest continuous output, before any threshold.
    """
    color = cv2.applyColorMap((probabilities * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha = (probabilities * max_alpha)[:, :, None]
    return (base * (1 - alpha) + color * alpha).astype(np.uint8)


def segment_overlay(
    gray: np.ndarray, mask: np.ndarray, skeleton: np.ndarray, junctions: np.ndarray
) -> np.ndarray:
    """The thresholded view: red mask + blue centerline skeleton + green junction circles."""
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(
        np.uint8
    )
    ys, xs = np.nonzero(skeleton)
    overlay[ys, xs] = (255, 0, 0)
    for cx, cy in junctions:
        cv2.circle(overlay, (int(cx), int(cy)), 12, (0, 255, 0), 3)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize road-UNet output on a page image."
    )
    parser.add_argument(
        "images", nargs="+", type=Path, help="Page image(s) to segment."
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/road_unet.pt"), help="Checkpoint."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Directory for the PNGs."
    )
    parser.add_argument(
        "--mode",
        choices=["heatmap", "segments", "both"],
        default="both",
        help="heatmap = raw P(road); segments = thresholded mask/skeleton/junctions.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=2000)
    parser.add_argument(
        "--downscale",
        type=int,
        default=3,
        help="Shrink the written PNG by this factor.",
    )
    args = parser.parse_args()

    from mapsnap.keymap.number_model import select_device

    device = select_device()
    print(f"device: {device}", file=sys.stderr)
    model = load_model(args.model, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skip (unreadable): {image_path}", file=sys.stderr)
            continue
        probabilities = predict_page(model, gray, device)
        panels = []
        if args.mode in ("heatmap", "both"):
            panels.append(label_panel(heatmap_overlay(gray, probabilities), "P(road)"))
        if args.mode in ("segments", "both"):
            mask = road_mask(
                probabilities, threshold=args.threshold, min_area=args.min_area
            )
            junctions = skeleton_junctions(road_skeleton(mask))
            panels.append(
                label_panel(
                    segment_overlay(gray, mask, road_skeleton(mask), junctions),
                    f"{len(junctions)} junctions",
                )
            )
            print(f"{image_path.name}: {len(junctions)} junctions")
        canvas = panels[0] if len(panels) == 1 else np.hstack(panels)
        if args.downscale > 1:
            canvas = cv2.resize(
                canvas,
                (canvas.shape[1] // args.downscale, canvas.shape[0] // args.downscale),
            )
        out_path = args.out_dir / f"{image_path.parent.name}_{image_path.stem}.road.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
