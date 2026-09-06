#!/usr/bin/env python3
"""Render the figures for docs/HOW-IT-WORKS.md from a real pipeline run.

Every figure is drawn from sidecars and archives on disk -- nothing is mocked
up. The walkthrough follows Columbus, Ohio 1951 vol. 3 (tag 2026-09-03) and
borrows Richmond 1925 vol. 3 p311 for the snap-rescue story. Output goes to
images/how-it-works/; run from the repo root:

    uv run python scripts/how_it_works_figures.py [--tag TAG] [--only NAME ...]
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "images" / "how-it-works"
VOLUME = "columbus_oh_1951_vol_3"
PAGE = "p220"
SPLIT_PAGE = "p209"
RESCUE_VOLUME, RESCUE_PAGE = "richmond_va_1925_vol_3", "p311"
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
MAX_WIDTH = 1600

GOOD_FT, DISASTER_FT = 25.0, 200.0
GREEN, RED, BLUE, ORANGE, GRAY, BLACK = (
    (40, 160, 60),
    (220, 50, 50),
    (40, 90, 220),
    (240, 140, 20),
    (130, 130, 130),
    (20, 20, 20),
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def page_image(volume: str, stem: str) -> Image.Image:
    return Image.open(DATA / volume / f"{stem}.jpg").convert("RGB")


def save(image: Image.Image, name: str) -> None:
    """Write a figure, downscaled to MAX_WIDTH, as JPEG (scans) or PNG (charts)."""
    OUT.mkdir(parents=True, exist_ok=True)
    if image.width > MAX_WIDTH:
        image = image.resize(
            (MAX_WIDTH, round(image.height * MAX_WIDTH / image.width)),
            Image.Resampling.LANCZOS,
        )
    path = OUT / name
    if path.suffix == ".png":
        image.save(path, optimize=True)
    else:
        image.convert("RGB").save(path, quality=82, optimize=True)
    print(f"wrote {path.relative_to(ROOT)} {image.size}")


def caption(image: Image.Image, text: str, size: int = 34) -> Image.Image:
    """Return ``image`` with a caption strip added along the top, wrapped to fit."""
    import textwrap

    per_line = max(30, int(image.width / (size * 0.47)))
    text = "\n".join(
        line
        for paragraph in text.split("\n")
        for line in textwrap.wrap(paragraph, per_line)
    )
    strip = 20 + size * (text.count("\n") + 1) + 8 * text.count("\n")
    out = Image.new("RGB", (image.width, image.height + strip), "white")
    ImageDraw.Draw(out).multiline_text((16, 10), text, fill=BLACK, font=font(size))
    out.paste(image, (0, strip))
    return out


def side_by_side(images: list[Image.Image], gap: int = 24) -> Image.Image:
    height = max(image.height for image in images)
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width + gap
    return out


def label(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color, size=22
):
    """Text with a white halo so it reads over map ink."""
    f = font(size)
    x, y = xy
    for dx, dy in (
        (-2, 0),
        (2, 0),
        (0, -2),
        (0, 2),
        (-2, -2),
        (2, 2),
        (-2, 2),
        (2, -2),
    ):
        draw.text((x + dx, y + dy), text, fill="white", font=f)
    draw.text((x, y), text, fill=color, font=f)


def affine_from_points(pixels: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Least-squares 2x3 affine mapping pixel (x, y) -> world (lon, lat)."""
    design = np.column_stack([pixels, np.ones(len(pixels))])
    solution, *_ = np.linalg.lstsq(design, world, rcond=None)
    return solution.T


def apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ affine[:, :2].T + affine[:, 2]


def invert_affine(affine: np.ndarray) -> np.ndarray:
    full = np.vstack([affine, [0.0, 0.0, 1.0]])
    return np.linalg.inv(full)[:2]


def corners_affine(corners: list, width: int, height: int) -> np.ndarray:
    pixels = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    return affine_from_points(pixels, np.array(corners, float))


class World:
    """A local metre frame around ``center`` (lon, lat), drawn at ``m_per_px``."""

    def __init__(
        self, center: tuple[float, float], size: tuple[int, int], m_per_px: float
    ):
        self.lon0, self.lat0 = center
        self.width, self.height = size
        self.m_per_px = m_per_px
        self.kx = 111_320.0 * math.cos(math.radians(self.lat0))
        self.ky = 110_540.0

    def to_px(self, lonlat: np.ndarray) -> np.ndarray:
        x = (lonlat[:, 0] - self.lon0) * self.kx / self.m_per_px + self.width / 2
        y = self.height / 2 - (lonlat[:, 1] - self.lat0) * self.ky / self.m_per_px
        return np.column_stack([x, y])

    def draw_osm(
        self, draw: ImageDraw.ImageDraw, volume: str, color=(200, 200, 200), width=2
    ):
        margin_lon = self.width * self.m_per_px / self.kx
        margin_lat = self.height * self.m_per_px / self.ky
        for feature in load_json(DATA / volume / "centerlines.geojson")["features"]:
            coords = np.array(feature["geometry"]["coordinates"], float)
            if (
                coords[:, 0].max() < self.lon0 - margin_lon
                or coords[:, 0].min() > self.lon0 + margin_lon
                or coords[:, 1].max() < self.lat0 - margin_lat
                or coords[:, 1].min() > self.lat0 + margin_lat
            ):
                continue
            draw.line([tuple(p) for p in self.to_px(coords)], fill=color, width=width)

    def scale_bar(self, draw: ImageDraw.ImageDraw, metres: float = 200.0):
        px = metres / self.m_per_px
        x0, y0 = 30, self.height - 40
        draw.line([(x0, y0), (x0 + px, y0)], fill=BLACK, width=5)
        label(draw, (x0, y0 - 34), f"{metres:.0f} m", BLACK, 24)


# --------------------------------------------------------------------------- figures


def fig_sheet() -> None:
    page = page_image(VOLUME, PAGE)
    keymap = page_image(VOLUME, "p0")
    raw = load_json(DATA / VOLUME / f"{PAGE}.streets.json")
    page = caption(
        page,
        f"{VOLUME} {PAGE}: {raw['width']}x{raw['height']} px working copy\n"
        f"(the scan is {raw['width'] * 4}x{raw['height'] * 4}); no coordinates anywhere on it",
    )
    keymap = caption(
        keymap, "p0, the key map: every sheet's number drawn over its footprint"
    )
    save(side_by_side([page, keymap]), "01-sheet.jpg")


def fig_split() -> None:
    image = page_image(VOLUME, SPLIT_PAGE)
    panels = load_json(DATA / VOLUME / f"{SPLIT_PAGE}.panels.json")
    draw = ImageDraw.Draw(image, "RGBA")
    colors = [
        (40, 90, 220, 70),
        (220, 50, 50, 70),
        (40, 160, 60, 70),
        (240, 140, 20, 70),
    ]
    for index, ring in enumerate(panels["panels"]):
        pts = [tuple(p) for p in ring]
        draw.polygon(pts, fill=colors[index % len(colors)], outline=BLACK, width=6)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        label(draw, (cx - 60, cy - 30), f"{SPLIT_PAGE}__{index + 1}", BLACK, 48)
    save(
        caption(
            image,
            f"{SPLIT_PAGE}: one scan, {len(panels['panels'])} panels, each its own page from here on",
        ),
        "02-split.jpg",
    )


def fig_craft() -> None:
    image = page_image(VOLUME, PAGE)
    boxes = load_json(DATA / VOLUME / f"{PAGE}.boxes.json")
    draw = ImageDraw.Draw(image)
    counts = []
    for entry in boxes["boxes"]:
        angle = entry["angle"]
        counts.append(
            f"{len(entry['horizontal_list']) + len(entry['free_list'])} at {angle}°"
        )
        if angle != 0:
            continue
        for x0, x1, y0, y1 in entry["horizontal_list"]:
            draw.rectangle([x0, y0, x1, y1], outline=BLUE, width=3)
        for quad in entry["free_list"]:
            draw.polygon([tuple(p) for p in quad], outline=ORANGE, width=3)
    save(
        caption(
            image,
            f"CRAFT text boxes on {PAGE}: {', '.join(counts)} (the 0° pass drawn; blue axis-aligned, orange free-form)",
        ),
        "03-craft.jpg",
    )


def fig_ocr() -> None:
    image = page_image(VOLUME, PAGE)
    reads = load_json(DATA / VOLUME / f"{PAGE}.streets.json")["streets"]
    draw = ImageDraw.Draw(image)
    confident = [r for r in reads if r["confidence"] >= 0.5]
    weak = [r for r in reads if r["confidence"] < 0.5]
    for read in weak:
        draw.polygon(
            [tuple(p) for p in read["polygon"]], outline=(200, 200, 200), width=2
        )
    for read in confident:
        pts = [tuple(p) for p in read["polygon"]]
        draw.polygon(pts, outline=GREEN, width=4)
        x, y = min(p[0] for p in pts), min(p[1] for p in pts)
        label(draw, (x, y - 26), f"{read['text']} {read['confidence']:.2f}", GREEN, 22)
    save(
        caption(
            image,
            f"Vocabulary-constrained reads on {PAGE}: {len(confident)} at confidence ≥ 0.5 (green, named), "
            f"{len(weak)} weaker (gray). Every read is a street name from OSM.",
        ),
        "04-ocr.jpg",
    )


def fig_georef() -> None:
    image = page_image(VOLUME, PAGE)
    georef = load_json(DATA / VOLUME / f"{PAGE}.georef.json")
    width, height = georef["width"], georef["height"]
    draw = ImageDraw.Draw(image)
    # OSM centerlines pulled into the page frame through the fitted pose.
    affine = corners_affine(georef["corners"], width, height)
    inverse = invert_affine(affine)
    corners = np.array(georef["corners"], float)
    lon_lo, lon_hi = corners[:, 0].min(), corners[:, 0].max()
    lat_lo, lat_hi = corners[:, 1].min(), corners[:, 1].max()
    for feature in load_json(DATA / VOLUME / "centerlines.geojson")["features"]:
        coords = np.array(feature["geometry"]["coordinates"], float)
        if coords[:, 0].max() < lon_lo or coords[:, 0].min() > lon_hi:
            continue
        if coords[:, 1].max() < lat_lo or coords[:, 1].min() > lat_hi:
            continue
        pts = apply_affine(inverse, coords)
        draw.line([tuple(p) for p in pts], fill=(80, 160, 255), width=5)
    diagonal = math.hypot(width, height)
    for street in georef["streets"]:
        color = GREEN if street.get("inlier") else GRAY
        x, y, dx, dy = street["x"], street["y"], street["dir_x"], street["dir_y"]
        draw.line(
            [
                (x - dx * diagonal, y - dy * diagonal),
                (x + dx * diagonal, y + dy * diagonal),
            ],
            fill=color,
            width=3,
        )
        label(draw, (x + 8, y + 8), street["street"], color, 24)
    for gcp in georef["intersections"]:
        x, y = gcp["x"], gcp["y"]
        color = GREEN if gcp.get("inlier") else RED
        radius = 22 if gcp.get("initial") else 14
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], outline=color, width=5
        )
    inliers = sum(1 for g in georef["intersections"] if g.get("inlier"))
    save(
        caption(
            image,
            f"RANSAC on {PAGE}: label axes (green inlier, gray outlier), {len(georef['intersections'])} candidate "
            f"intersections ({inliers} inliers, big rings = the seed pair), OSM centerlines drawn through the fitted pose in blue",
        ),
        "05-georef.jpg",
    )


def fig_keymap() -> None:
    image = page_image(VOLUME, "p0")
    raw = DATA / VOLUME / "raw"
    reads = load_json(raw / "p0.keymap.json")
    regions = load_json(raw / "p0.regions.panels.json")
    scale = image.width / reads["width"]
    draw = ImageDraw.Draw(image, "RGBA")
    rng = np.random.default_rng(3)
    for text, ring in zip(regions["labels"], regions["panels"]):
        color = tuple(int(v) for v in rng.integers(60, 230, 3))
        pts = [(x * scale, y * scale) for x, y in ring]
        fill = (*color, 150) if text == PAGE[1:] else (*color, 60)
        draw.polygon(pts, fill=fill, outline=(*color, 255), width=2)
    inset_path = raw / "p0.inset.panels.json"
    if inset_path.exists():
        for ring in load_json(inset_path)["panels"]:
            pts = [(x * scale, y * scale) for x, y in ring]
            draw.polygon(
                pts, fill=(220, 50, 50, 60), outline=(220, 50, 50, 255), width=6
            )
    for read in reads["streets"]:
        if read.get("inset"):
            continue
        pts = [(x * scale, y * scale) for x, y in read["polygon"]]
        draw.polygon(pts, outline=(0, 0, 0, 255), width=2)
        x, y = min(p[0] for p in pts), min(p[1] for p in pts)
        label(draw, (x, y - 18), read["text"], BLACK, 16)
    # The page's search centre, through the key map's own georef.
    georef = load_json(raw / "p0.georef.json")
    page_georef = load_json(DATA / VOLUME / f"{PAGE}.georef.json")
    keymap_entry = page_georef.get("keymap") or {}
    if georef.get("corners") and keymap_entry.get("centers"):
        inverse = invert_affine(
            corners_affine(georef["corners"], georef["width"], georef["height"])
        )
        centers = (
            apply_affine(inverse, np.array(keymap_entry["centers"], float)) * scale
        )
        m_per_raw_px = _keymap_m_per_px(georef)
        radius = keymap_entry["radius_m"] / m_per_raw_px * scale
        for cx, cy in centers:
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=(220, 50, 50, 255),
                width=6,
            )
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(220, 50, 50, 255))
    n_inset = sum(1 for r in reads["streets"] if r.get("inset"))
    save(
        caption(
            image,
            f"The key map: {len(reads['streets']) - n_inset} page-number reads (boxed), {len(regions['panels'])} segmented "
            f"page regions (tinted; {PAGE}'s solid), the volume-index inset masked in red ({n_inset} reads inside it "
            f"ignored), and {PAGE}'s search circle ({keymap_entry.get('radius_m', 0):.0f} m).",
        ),
        "06-keymap.jpg",
    )


def _keymap_m_per_px(georef: dict) -> float:
    corners = np.array(georef["corners"], float)
    lat = corners[:, 1].mean()
    kx, ky = 111_320.0 * math.cos(math.radians(lat)), 110_540.0
    top = math.hypot(
        (corners[1, 0] - corners[0, 0]) * kx, (corners[1, 1] - corners[0, 1]) * ky
    )
    return top / georef["width"]


def fig_adjacency() -> None:
    adjacency = load_json(DATA / VOLUME / "adjacency.json")
    # Left: the page's margin numbers, claims in green.
    image = page_image(VOLUME, PAGE)
    draw = ImageDraw.Draw(image)
    entry = adjacency["pages"][PAGE]
    for det in entry["detections"]:
        pts = [tuple(p) for p in det["polygon"]]
        color = GREEN if det.get("claim") else (200, 200, 200)
        draw.polygon(pts, outline=color, width=5 if det.get("claim") else 2)
        if det.get("claim"):
            x, y = min(p[0] for p in pts), max(p[1] for p in pts)
            label(draw, (x, y + 4), f"sheet {det['key']} ({det['edge']})", GREEN, 26)
    claims = [d["key"] for d in entry["detections"] if d.get("claim")]
    left = caption(
        image,
        f"{PAGE}'s margin reads: claims {', '.join(claims)} (green) survive the height floor and edge band",
    )
    # Right: the mutual-claim graph over the key map's page regions.
    keymap = page_image(VOLUME, "p0")
    regions = load_json(DATA / VOLUME / "raw" / "p0.regions.panels.json")
    scale = keymap.width / regions["width"]
    centroids: dict[str, tuple[float, float]] = {}
    for text, ring in zip(regions["labels"], regions["panels"]):
        pts = np.array(ring, float) * scale
        centroids.setdefault(
            f"p{text}", (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        )
    draw = ImageDraw.Draw(keymap, "RGBA")
    parent = lambda stem: stem.split("__")[0]
    for a, b in adjacency.get("one_sided", []):
        if parent(a) in centroids and parent(b) in centroids:
            draw.line(
                [centroids[parent(a)], centroids[parent(b)]],
                fill=(240, 140, 20, 140),
                width=3,
            )
    mutual = 0
    for a, b in adjacency["adjacency"]:
        if parent(a) in centroids and parent(b) in centroids:
            draw.line(
                [centroids[parent(a)], centroids[parent(b)]],
                fill=(40, 160, 60, 220),
                width=6,
            )
            mutual += 1
    for x, y in centroids.values():
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(0, 0, 0, 255))
    right = caption(
        keymap,
        f"The volume's adjacency graph on the key map: {len(adjacency['adjacency'])} mutual edges (green), "
        f"{len(adjacency.get('one_sided', []))} one-sided (orange)",
    )
    save(side_by_side([left, right]), "07-adjacency.jpg")


def fig_unet() -> None:
    import torch

    from mapsnap.road_model import ROAD_MODEL_PATH, load_model, predict_page

    image = page_image(VOLUME, PAGE)
    gray = np.array(image.convert("L"))
    model = load_model(ROAD_MODEL_PATH, torch.device("cpu"))
    prob = predict_page(model, gray, torch.device("cpu"))
    heat = Image.fromarray((255 * (1 - prob)).astype(np.uint8)).convert("RGB")
    left = caption(image, f"{PAGE} as the UNet sees it (grayscale)")
    right = caption(heat, "P(road): the road-corridor UNet's output, dark = road")
    # Key map: the shipped P(road) raster for the raw sheet.
    keymap_prob = Image.open(DATA / VOLUME / "raw" / "p0.roadprob.png").convert("L")
    keymap_prob = keymap_prob.resize(image.size, Image.Resampling.LANCZOS)
    keymap = caption(
        Image.fromarray(255 - np.array(keymap_prob)).convert("RGB"),
        "The same model on the key map",
    )
    save(side_by_side([left, right, keymap]), "08-unet.jpg")


def fig_snap() -> None:
    volume, page = RESCUE_VOLUME, RESCUE_PAGE
    archive = DATA / volume / "artifacts" / "2026-09-03"
    record = next(
        json.loads(line)
        for line in (archive / "osm_snap" / "candidates.jsonl").read_text().splitlines()
        if json.loads(line)["target"] == page
    )
    width, height = record["width"], record["height"]
    page_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    # Frame the figure around the truth pose.
    truth = load_json(DATA / volume / "main.iiif.json")
    truth_item = next(
        item
        for item in truth["items"]
        if re.search(rf"[-_]0*{page[1:]}(?:[/_.]|$)", item["target"]["source"]["id"])
    )
    truth_pixels = np.array(
        [f["properties"]["resourceCoords"] for f in truth_item["body"]["features"]],
        float,
    )
    truth_world = np.array(
        [f["geometry"]["coordinates"] for f in truth_item["body"]["features"]], float
    )
    truth_scale = width / truth_item["target"]["source"]["width"]
    truth_affine = affine_from_points(truth_pixels * truth_scale, truth_world)
    truth_ring = apply_affine(truth_affine, page_corners)
    center = tuple(truth_ring.mean(axis=0))
    world = World(center, (1500, 1500), 1.0)
    image = Image.new("RGB", (world.width, world.height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    world.draw_osm(draw, volume)
    for lon, lat in record["search"]["centers"]:
        cx, cy = world.to_px(np.array([[lon, lat]]))[0]
        radius = record["search"]["radius_m"] / world.m_per_px
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=(220, 50, 50, 160),
            width=3,
        )
    incumbent = record.get("incumbent") or {}
    if incumbent.get("world_affine"):
        ring = world.to_px(
            apply_affine(np.array(incumbent["world_affine"]), page_corners)
        )
        draw.polygon([tuple(p) for p in ring], outline=(220, 50, 50, 255), width=5)
        label(
            draw,
            tuple(ring[0]),
            f"RANSAC incumbent {incumbent.get('rmse_ft', 0):.0f} ft",
            RED,
            24,
        )
    ranked = record["candidates"]
    for rank, candidate in reversed(list(enumerate(ranked, 1))):
        ring = world.to_px(
            apply_affine(np.array(candidate["world_affine"]), page_corners)
        )
        color = (40, 160, 60, 255) if rank == 1 else (40, 90, 220, 110)
        draw.polygon(
            [tuple(p) for p in ring], outline=color, width=6 if rank == 1 else 3
        )
        if rank <= 3:
            label(
                draw,
                tuple(ring[2]),
                f"#{rank} select {candidate['select_score']:.2f}, {candidate.get('rmse_ft', 0):.0f} ft",
                GREEN if rank == 1 else BLUE,
                24,
            )
    ring = world.to_px(truth_ring)
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        for t in np.linspace(0, 1, 24)[:-1:2]:
            draw.line(
                [tuple(a + (b - a) * t), tuple(a + (b - a) * (t + 1 / 24))],
                fill=BLACK,
                width=4,
            )
    world.scale_bar(draw)
    save(
        caption(
            image,
            f"snap on {volume} {page}: the search circle (red), the RANSAC pose it challenged (red box), "
            f"the top-8 candidates (blue, winner green) and the hand-placed truth (black dashes)",
        ),
        "09-snap.jpg",
    )


def fig_iiif(tag: str) -> None:
    doc = load_json(DATA / VOLUME / f"{tag}.iiif.json")
    rings = []
    masks = []
    for item in doc["items"]:
        source = item["target"]["source"]
        features = item["body"]["features"]
        pixels = np.array([f["properties"]["resourceCoords"] for f in features], float)
        world_pts = np.array([f["geometry"]["coordinates"] for f in features], float)
        affine = affine_from_points(pixels, world_pts)
        w, h = source["width"], source["height"]
        rings.append(
            apply_affine(affine, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
        )
        svg = item["target"].get("selector", {}).get("value", "")
        match = re.search(r'points="([^"]+)"', svg)
        if match:
            pts = np.array(
                [[float(v) for v in pair.split(",")] for pair in match[1].split()]
            )
            masks.append(apply_affine(affine, pts))
    all_pts = np.vstack(rings)
    center = (float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean()))
    kx = 111_320.0 * math.cos(math.radians(center[1]))
    span_m = max(
        (all_pts[:, 0].max() - all_pts[:, 0].min()) * kx,
        (all_pts[:, 1].max() - all_pts[:, 1].min()) * 110_540,
    )
    world = World(center, (1600, 1600), span_m * 1.05 / 1600)
    image = Image.new("RGB", (world.width, world.height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    world.draw_osm(draw, VOLUME, color=(215, 215, 215), width=1)
    rng = np.random.default_rng(7)
    for ring, mask in zip(rings, masks):
        color = tuple(int(v) for v in rng.integers(40, 220, 3))
        draw.polygon(
            [tuple(p) for p in world.to_px(ring)], outline=(*color, 90), width=1
        )
        draw.polygon(
            [tuple(p) for p in world.to_px(mask)],
            fill=(*color, 110),
            outline=(*color, 255),
            width=2,
        )
    world.scale_bar(draw, 1000)
    save(
        caption(
            image,
            f"The published IIIF annotation page for {VOLUME}: {len(doc['items'])} placed pages, each clipped to its "
            f"block-based mask (filled); the full page footprints are the faint outlines",
        ),
        "10-iiif.jpg",
    )


def fig_score(tag: str) -> None:
    manifest = load_json(DATA / VOLUME / "artifacts" / tag / "manifest.json")
    per_page = manifest["metrics"]["truth"]["per_page"]
    footer = ""
    for line in (DATA / VOLUME / f"{tag}.txt").read_text().splitlines()[::-1]:
        if line.startswith("Score:"):
            footer = line
            break
    total = (
        manifest["metrics"]["truth"]["score"]["n_pages"]
        if "score" in manifest["metrics"]["truth"]
        else len(per_page)
    )
    values = sorted(per_page.values())
    unplaced = max(0, total - len(values))
    width, height = 1600, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 90, width - 30, 70, height - 110
    bar_w = (right - left) / (len(values) + unplaced)
    log_max = math.log10(3000)
    for i, rmse in enumerate(values + [None] * unplaced):
        x0 = left + i * bar_w
        if rmse is None:
            draw.rectangle(
                [x0, bottom - 12, x0 + bar_w - 1, bottom], fill=(180, 180, 180)
            )
            continue
        bar_h = (bottom - top) * min(1.0, math.log10(max(rmse, 1.0)) / log_max)
        color = GREEN if rmse <= GOOD_FT else (RED if rmse >= DISASTER_FT else ORANGE)
        draw.rectangle([x0, bottom - bar_h, x0 + bar_w - 1, bottom], fill=color)
    for ft, text in ((25, "25 ft"), (200, "200 ft"), (1000, "1000 ft")):
        y = bottom - (bottom - top) * math.log10(ft) / log_max
        draw.line([(left, y), (right, y)], fill=(120, 120, 120), width=1)
        label(draw, (8, y - 12), text, BLACK, 20)
    label(
        draw,
        (left, bottom + 10),
        "pages, sorted by rmse against the hand-placed truth (log scale); gray = left unplaced",
        BLACK,
        22,
    )
    label(draw, (left, bottom + 44), footer, BLACK, 22)
    label(
        draw,
        (left, 14),
        f"{VOLUME}, run {tag}: green ≤ 25 ft, orange in between, red ≥ 200 ft",
        BLACK,
        26,
    )
    save(image, "11-score.png")


FIGURES = {
    "sheet": fig_sheet,
    "split": fig_split,
    "craft": fig_craft,
    "ocr": fig_ocr,
    "georef": fig_georef,
    "keymap": fig_keymap,
    "adjacency": fig_adjacency,
    "unet": fig_unet,
    "snap": fig_snap,
    "iiif": fig_iiif,
    "score": fig_score,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--tag", default="2026-09-03", help="archived run to draw from")
    parser.add_argument(
        "--only", nargs="*", choices=sorted(FIGURES), help="figures to render"
    )
    args = parser.parse_args()
    for name in args.only or FIGURES:
        function = FIGURES[name]
        if name in ("iiif", "score"):
            function(args.tag)
        else:
            function()


if __name__ == "__main__":
    main()
