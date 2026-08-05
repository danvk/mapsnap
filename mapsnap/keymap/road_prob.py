"""P(road) maps for key-map sheets, via a key-map-specific road UNet (#211).

A key map's roads defeat both existing extractors. The page road UNet is
style-blind off its training distribution: a key map's streets are a few times
narrower in pixels and drawn over pastel page-region fills it has never seen.
Classical morphology (CLOSE(nonpaper) - nonpaper) is structurally biased the
other way: it detects paper-coloured corridors, and a key-map road is drawn the
same way at the same width whether it crosses plain paper or a pastel fill --
only the background changes -- so the fill-background roads score as nonpaper
and vanish. Two rounds of that approach failed visual inspection.

What key maps do have is the same free supervision that trained the page model:
every sheet carries an exact-affine georeference (``raw/<stem>.georef.json``)
and its volume has ``centerlines.geojson``, so OSM centerlines rendered through
the affine label roads wherever they actually are, over paper and fill alike.
This module supplies everything around that idea except the training loop
(``mapsnap.keymap.train_road_prob``):

  * the sheet inventory (``keymap_sheets``),
  * the per-sheet drawn stroke width, measured from the drawing itself
    (``measure_stroke_px``) -- road width on a key map is a stylistic paper
    width, near-constant in pixels across sheets scanned at similar DPI, so no
    ground-metre constant can be right at more than one sheet,
  * soft training labels with an ignore mask where OSM coverage ends
    (``sheet_label``),
  * colour tiled inference (``predict_sheet``) writing ``raw/<stem>.roadprob.png``,
  * evaluation that respects the ~30-40 px georef error in the labels:
    buffered completeness/correctness at a tolerance (``buffered_scores``),
    stratified by background so "found the roads over paper but not the roads
    over fills" is visible as a number, not just to the eye.

Output PNGs live next to the sheet in raw/ (uint8, probability * 255). They
must NOT go in artifacts/edge_join/roadprob/: champaign's key map is stem p1,
which would collide with page p1's probability map there.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mapsnap.road_model import (
    invert_affine,
    page_scale_m_per_px,
    page_world_affine,
    rasterize_road_mask,
)

MIN_KEYMAP_INLIERS = 25
"""Sheets with fewer inlier street intersections than this are excluded from
training and evaluation: their affine is too weakly constrained to paint labels
in the right place (queens has 3 inliers, asheville 10; every other sheet has
33+)."""

STROKE_CLAMP_PX = (15, 64)
STROKE_FALLBACK_PX = 33
"""Bounds and fallback for the measured drawn road width. The fallback is the
Detroit-ish typical width; the clamp guards against a degenerate paper split."""

LABEL_SOFT_SIGMA_PX = 10.0
"""Gaussian blur applied to the rendered label. The key-map georef is an affine
fit with ~30-40 px of local error, so a hard-edged label at the drawn stroke
width penalizes a model for finding the road exactly where it is drawn. Soft
targets (BCE accepts them directly) let the loss forgive that misalignment."""

BUFFER_TOLERANCE_PX = 40
"""Evaluation tolerance: a predicted road within this of a label counts as
correct, and a label within this of a prediction counts as found. Roughly one
stroke width, which is also the median georef misalignment -- plain IoU at that
misalignment scores a PERFECT extraction near zero, so it cannot be the gate."""

FILL_SATURATION = 40
"""HSV saturation above which a pixel is a pastel page-region fill rather than
paper, for stratifying completeness by background."""


@dataclass(frozen=True)
class KeymapSheet:
    """One georeferenced key-map sheet and the files that describe it."""

    volume: Path
    stem: str

    @property
    def image_path(self) -> Path:
        return self.volume / "raw" / f"{self.stem}.jpg"

    @property
    def georef_path(self) -> Path:
        return self.volume / "raw" / f"{self.stem}.georef.json"

    @property
    def roadprob_path(self) -> Path:
        return self.volume / "raw" / f"{self.stem}.roadprob.png"

    @property
    def centerlines_path(self) -> Path:
        return self.volume / "centerlines.geojson"

    def georef(self) -> dict:
        return json.loads(self.georef_path.read_text())

    def key(self) -> str:
        return f"{self.volume.name}_{self.stem}"


def keymap_inliers(georef: dict) -> int:
    """Inlier street intersections behind a key map's affine fit."""
    return sum(
        1
        for intersection in georef.get("intersections", [])
        if intersection.get("inlier")
    )


def keymap_sheets(
    data_dir: Path, min_inliers: int = MIN_KEYMAP_INLIERS
) -> list[KeymapSheet]:
    """Every usable georeferenced key-map sheet under ``data_dir``.

    Usable means: a georef sidecar in raw/ with a same-stem key-map jpg, volume
    centerlines to render labels from, and enough inliers behind the affine to
    trust where those labels land.
    """
    sheets = []
    for georef_path in sorted(data_dir.glob("*/raw/*.georef.json")):
        if ".truth." in georef_path.name:
            continue
        stem = georef_path.name[: -len(".georef.json")]
        volume = georef_path.parent.parent
        sheet = KeymapSheet(volume, stem)
        if not sheet.image_path.exists() or not sheet.centerlines_path.exists():
            continue
        if not (volume / "raw" / f"{stem}.keymap.json").exists():
            continue
        if keymap_inliers(sheet.georef()) < min_inliers:
            continue
        sheets.append(sheet)
    return sheets


def paper_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Bright, unsaturated pixels -- the sheet's paper, split per sheet.

    The saturation threshold is Otsu on the bright pixels rather than a
    constant: pale washes (Detroit's are S 20-48) pass any fixed test that
    also admits aged paper. Ported from the #211 classical experiments, where
    this split was the one piece that survived scrutiny.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    bright = value > max(120, float(np.percentile(value, 35)))
    samples = saturation[bright]
    if samples.size == 0:
        return np.zeros(saturation.shape, np.uint8)
    threshold, _ = cv2.threshold(
        samples.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = float(np.clip(threshold, 12, 60))
    return (bright & (saturation < threshold)).astype(np.uint8)


def measure_stroke_px(image_bgr: np.ndarray) -> int:
    """The sheet's drawn road width in pixels, measured from the drawing.

    Roads are corridors of paper between structures, so the distance transform
    of the paper mask peaks at half the corridor width along road centrelines;
    the median of its plausible band is a robust half-width. Measured over
    paper-background roads only, but the drawn width is uniform sheet-wide, so
    it holds for the fill-background roads as well.
    """
    paper = paper_mask(image_bgr)
    distance = cv2.distanceTransform(paper, cv2.DIST_L2, 3)
    samples = distance[(distance > 2) & (distance < 45)]
    if samples.size < 1000:
        return STROKE_FALLBACK_PX
    stroke = 2.0 * float(np.median(samples))
    return int(np.clip(round(stroke), *STROKE_CLAMP_PX))


def centerlines_bbox(features: list[dict]) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) over every centerline vertex."""
    lon_min = lat_min = math.inf
    lon_max = lat_max = -math.inf
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
            lon_min = min(lon_min, pts[:, 0].min())
            lon_max = max(lon_max, pts[:, 0].max())
            lat_min = min(lat_min, pts[:, 1].min())
            lat_max = max(lat_max, pts[:, 1].max())
    return lon_min, lat_min, lon_max, lat_max


def coverage_mask(georef: dict, features: list[dict]) -> np.ndarray:
    """1 where the sheet is inside the OSM download's bbox, 0 where it is not.

    A sheet can extend past its volume's centerlines (Miami runs ~1 km past
    the bbox edge): out there a rendered label says "no road" about ground the
    label knows nothing about, which is wrong supervision. Loss is masked to
    zero outside this, and evaluation ignores it too.
    """
    lon_min, lat_min, lon_max, lat_max = centerlines_bbox(features)
    world_to_px = invert_affine(page_world_affine(georef))
    corners_world = np.array(
        [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
        ]
    )
    corners_px = corners_world @ world_to_px[:, :2].T + world_to_px[:, 2]
    mask = np.zeros((georef["height"], georef["width"]), np.uint8)
    cv2.fillPoly(mask, [corners_px.round().astype(np.int32)], color=1)
    return mask


def within_px(mask: np.ndarray, radius: int) -> np.ndarray:
    """Boolean: within ``radius`` pixels of a nonzero cell of ``mask``.

    A morphological dilation with an r-px disc, but computed via a distance
    transform: cv2.dilate is O(area * r^2) and takes minutes at the radii used
    here on a 50-megapixel sheet, while the distance transform is one linear
    pass.
    """
    inverted = (mask == 0).astype(np.uint8)
    distance = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
    return distance <= radius


MAPPED_EXTENT_MARGIN_PX = 400
"""How far beyond the outermost page region the drawn map is trusted to extend
(peripheral streets are drawn about one page-block past the last region)."""


def mapped_extent_mask(
    sheet: KeymapSheet, margin_px: int = MAPPED_EXTENT_MARGIN_PX
) -> np.ndarray:
    """1 where the sheet actually depicts the city, 0 over furniture.

    The affine maps EVERY sheet pixel to some ground point, and OSM has roads
    at many of them -- so a naive label paints roads across the title block,
    the correction record and the volume-index inset, none of which draw that
    ground. (The first Detroit label render showed exactly this.) The drawn
    map's extent is the union of the page-region polygons dilated a block's
    worth -- regions sit a fraction of that apart, so the map fuses into one
    component -- keeping only the LARGEST connected component: a convex hull
    bridged to stray "regions" the segmenter finds inside the volume-index
    inset (Detroit has three), and the inset cluster sits far enough from the
    map to stay a separate component here. Detached true map continuations
    would lose supervision too, which is the safe direction: they are ignored,
    not labelled wrongly. Falls back to all-ones when regions are missing.
    """
    georef = sheet.georef()
    shape = (georef["height"], georef["width"])
    regions_path = sheet.volume / "raw" / f"{sheet.stem}.regions.panels.json"
    if not regions_path.exists():
        return np.ones(shape, np.uint8)
    try:
        panels = json.loads(regions_path.read_text()).get("panels", [])
    except (OSError, ValueError):
        return np.ones(shape, np.uint8)
    polygons = [
        np.asarray(panel, dtype=np.int32) for panel in panels if len(panel) >= 3
    ]
    if not polygons:
        return np.ones(shape, np.uint8)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, polygons, color=1)
    grown = within_px(mask, margin_px).astype(np.uint8)
    count, components = cv2.connectedComponents(grown)
    if count <= 2:
        return grown
    sizes = np.bincount(components.ravel())
    sizes[0] = 0  # background
    return (components == int(sizes.argmax())).astype(np.uint8)


def sheet_label(
    sheet: KeymapSheet,
    features: list[dict],
    stroke_px: int,
    soft_sigma: float = LABEL_SOFT_SIGMA_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """(soft label in [0,1] float32, valid mask uint8) for one sheet.

    The valid mask is where BOTH sides of the supervision are trustworthy: the
    sheet depicts the ground (mapped extent) and OSM covers it (centerlines
    bbox). Outside it neither a positive nor a negative label means anything,
    so training loss and evaluation are confined to it.
    """
    georef = sheet.georef()
    mask = rasterize_road_mask(georef, features, width_px=stroke_px)
    label = mask.astype(np.float32) / 255.0
    if soft_sigma > 0:
        kernel = int(soft_sigma * 4) | 1
        label = cv2.GaussianBlur(label, (kernel, kernel), soft_sigma)
        peak = label.max()
        if peak > 0:
            label /= peak
    valid = coverage_mask(georef, features) & mapped_extent_mask(sheet)
    return label, valid


def normalize_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Colour uint8 HWC -> float32 CHW in roughly [-1, 1], matching the page model's range."""
    scaled = (image_bgr.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(scaled.transpose(2, 0, 1))


def predict_sheet(
    model,
    image_bgr: np.ndarray,
    device,
    *,
    tile: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Road probability for a full colour sheet, predicted in overlapping tiles.

    The colour twin of road_model.predict_page: Hann-blended overlapping tiles,
    paper-coloured padding at the edges, float32 probabilities in [0, 1].
    """
    import torch

    model.eval()
    height, width = image_bgr.shape[:2]
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
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                patch = image_bgr[y0 : y0 + tile, x0 : x0 + tile]
                ph, pw = patch.shape[:2]
                padded = np.pad(
                    patch,
                    ((0, tile - ph), (0, tile - pw), (0, 0)),
                    constant_values=210,
                )
                tensor = torch.from_numpy(normalize_bgr(padded)).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()[:ph, :pw]
                probabilities[y0 : y0 + ph, x0 : x0 + pw] += prob * window[:ph, :pw]
                weights[y0 : y0 + ph, x0 : x0 + pw] += window[:ph, :pw]
    return probabilities / weights


def buffered_scores(
    probability: np.ndarray,
    label: np.ndarray,
    valid: np.ndarray,
    image_bgr: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
    tolerance_px: int = BUFFER_TOLERANCE_PX,
) -> dict:
    """Buffered road-extraction metrics, tolerant of the labels' georef error.

    correctness   share of predicted road within ``tolerance_px`` of a label road
    completeness  share of label road within ``tolerance_px`` of predicted road
    iou           plain intersection-over-union, reported but NOT a gate: at the
                  labels' own misalignment it scores a perfect extraction near 0

    With ``image_bgr``, completeness is also stratified by background --
    ``completeness_paper`` vs ``completeness_fill`` -- because roads are drawn
    identically over both and any gap between the strata is background
    sensitivity, the failure mode the classical extractor could not escape.
    """
    predicted = (probability >= threshold) & (valid > 0)
    labelled = (label >= 0.5) & (valid > 0)
    label_zone = within_px(labelled.astype(np.uint8), tolerance_px)
    predicted_zone = within_px(predicted.astype(np.uint8), tolerance_px)

    predicted_count = int(predicted.sum())
    labelled_count = int(labelled.sum())
    found = labelled & predicted_zone
    scores = {
        "correctness": float((predicted & label_zone).sum() / max(1, predicted_count)),
        "completeness": float(found.sum() / max(1, labelled_count)),
        "iou": float(
            (predicted & labelled).sum() / max(1, (predicted | labelled).sum())
        ),
        "predicted_px": predicted_count,
        "label_px": labelled_count,
    }
    if image_bgr is not None:
        saturation = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        fill = saturation >= FILL_SATURATION
        for name, stratum in (("fill", fill), ("paper", ~fill)):
            stratum_label = labelled & stratum
            total = int(stratum_label.sum())
            scores[f"completeness_{name}"] = float(
                (stratum_label & predicted_zone).sum() / max(1, total)
            )
            scores[f"label_px_{name}"] = total
    return scores


def overlay_render(image_bgr: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """The sheet with P(road) burned in red over it, for eyeballing.

    Both prior extraction failures were caught by looking, not by a number, so
    every claim about a sheet ships with one of these.
    """
    heat = (np.clip(probability, 0, 1) * 255).astype(np.uint8)
    overlay = image_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = (heat.astype(np.float32) / 255.0 * 0.65)[:, :, None]
    return (overlay * (1 - alpha) + red * alpha).astype(np.uint8)


def load_centerlines(volume: Path) -> list[dict]:
    """The volume's OSM centerline features."""
    return json.loads((volume / "centerlines.geojson").read_text())["features"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict or inspect P(road) for key-map sheets."
    )
    parser.add_argument(
        "command",
        choices=["widths", "labels", "predict"],
        help=(
            "widths: print the per-sheet measured stroke table. labels: write "
            "label overlay renders for sanity-checking. predict: run the model "
            "and write raw/<stem>.roadprob.png + an overlay render."
        ),
    )
    parser.add_argument("--data", default="data", help="Data directory.")
    parser.add_argument(
        "--model",
        default="models/keymap_road_unet.pt",
        help="Checkpoint for `predict`.",
    )
    parser.add_argument(
        "--sheets",
        nargs="*",
        metavar="VOL/STEM",
        help="Restrict to these sheets, e.g. detroit_mich_1929_vol_11/p0__1.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for overlay renders (default: alongside).",
    )
    args = parser.parse_args()

    sheets = keymap_sheets(Path(args.data))
    if args.sheets:
        wanted = set(args.sheets)
        sheets = [s for s in sheets if f"{s.volume.name}/{s.stem}" in wanted]
    if not sheets:
        sys.exit("No matching key-map sheets.")

    def read_sheet(sheet: KeymapSheet) -> np.ndarray:
        image = cv2.imread(str(sheet.image_path))
        if image is None:
            sys.exit(f"Could not read {sheet.image_path}")
        return image

    if args.command == "widths":
        print(f"{'sheet':44} {'m/px':>6} {'stroke_px':>9} {'inliers':>8}")
        for sheet in sheets:
            georef = sheet.georef()
            stroke = measure_stroke_px(read_sheet(sheet))
            print(
                f"{sheet.key():44} {page_scale_m_per_px(georef):>6.2f} "
                f"{stroke:>9} {keymap_inliers(georef):>8}"
            )
        return

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "labels":
        for sheet in sheets:
            image = read_sheet(sheet)
            stroke = measure_stroke_px(image)
            label, valid = sheet_label(sheet, load_centerlines(sheet.volume), stroke)
            render = overlay_render(image, label * (valid > 0).astype(np.float32))
            name = (
                f"{sheet.key()}.labelcheck.jpg"
                if out_dir
                else f"{sheet.stem}.labelcheck.jpg"
            )
            path = (out_dir or sheet.volume / "raw") / name
            cv2.imwrite(str(path), render, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{sheet.key()}: stroke {stroke}px -> {path}")
        return

    from mapsnap.keymap.number_model import select_device
    from mapsnap.road_model import load_model

    device = select_device()
    model = load_model(Path(args.model), device)
    for sheet in sheets:
        image = read_sheet(sheet)
        probability = predict_sheet(model, image, device)
        # The model was never supervised on furniture (the loss is masked to
        # the mapped extent), so over legends and title blocks it free-runs --
        # Detroit's KEY box comes out solid road. The artifact only claims to
        # be P(road) of the MAP, so confine it to the same extent.
        probability *= mapped_extent_mask(sheet)
        cv2.imwrite(str(sheet.roadprob_path), (probability * 255).astype(np.uint8))
        render = overlay_render(image, probability)
        name = (
            f"{sheet.key()}.roadprob-overlay.jpg"
            if out_dir
            else f"{sheet.stem}.roadprob-overlay.jpg"
        )
        path = (out_dir or sheet.volume / "raw") / name
        cv2.imwrite(str(path), render, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{sheet.key()} -> {sheet.roadprob_path} (+ {path.name})")


if __name__ == "__main__":
    main()
