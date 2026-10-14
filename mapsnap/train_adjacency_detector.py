"""Train the adjacency-number detector (mapsnap.adjacency_detector) on OIM-confirmed reads.

Labels cost nothing by hand. OIM's georeferencing predicts which neighbour numbers a
sheet prints and along which stretch of its outline (mapsnap.oim_neighbours);
corpus-v1's adjacency step already read digits on every sheet. A read whose page key
is a predicted neighbour's number, near that neighbour's stretch and of a plausible
height for its volume, is a positive box. Stretches whose neighbour no read confirms,
and outline with no OIM neighbour at all (where explicit zeros and references to other
volumes sit), are ignored rather than taken as negatives -- they are where CRAFT's
misses are. On the hand-labelled dev volumes 95% of positive boxes are real printed
references and 2.7% of printed references fall in negative territory.

A workspace holds ``manifest.json`` (one entry per page: item, page key, mirror image
URL, size, OIM content outline and neighbour stretches in image pixels, the digit reads
that could confirm a neighbour, and boxes adopted by self-training), ``images/`` and
``masks/``. The shipped weights' manifest is ``models/adjacency_unet.pages.json.gz``.

    uv run python -m mapsnap.train_adjacency_detector select WORK      # needs OIM exports + S3
    uv run python -m mapsnap.train_adjacency_detector fetch WORK       # 25% page images from S3
    uv run python -m mapsnap.train_adjacency_detector labels WORK
    uv run python -m mapsnap.train_adjacency_detector train WORK --output models/adjacency_unet.pt
    uv run python -m mapsnap.train_adjacency_detector adopt WORK --weights models/adjacency_unet.pt

To retrain from the shipped manifest instead of re-selecting, gunzip it to
``WORK/manifest.json`` and start at ``fetch``. ``adopt`` is one round of self-training:
the detector's lone plausible component on each unconfirmed stretch, unless the reader
reads it as another number, becomes a positive (97.5% correct on the dev volumes);
``labels --adopted`` then uses those boxes. One such round did not improve the
hand-label scores, so the shipped weights are round 0. 12,000 steps take about five
hours on an M2's GPU.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from torch import nn

from mapsnap.adjacency_detector import (
    MODEL_PATH,
    Component,
    build_detector,
    load_detector,
    number_components,
)
from mapsnap.keymap.number_model import select_device
from mapsnap.oim_neighbours import OimPage, load_pages, predicted_neighbours
from mapsnap.page_adjacency import MIN_HEIGHT
from mapsnap.road_model import UNet, normalize_patch, predict_page

REPO = Path(__file__).resolve().parent.parent
MAPSNAP_DOCUMENTS = Path.home() / "Documents" / "mapsnap"
BUCKET = "s3://mapsnap-sanborn"
AWS = ["aws", "--profile", "mapsnap"]

# Volumes are drawn per decade roughly in proportion to the 100-page corpus sample.
DECADE_QUOTA = {
    1880: 25,
    1890: 35,
    1900: 45,
    1910: 40,
    1920: 30,
    1930: 8,
    1940: 9,
    1950: 40,
    1960: 6,
}
PAGES_PER_VOLUME = 18
# City series (sanbornNNNNN) of the hand-labelled dev volumes: Abbeville 1924,
# Hudson vol 9, LA vol 14, Nashville 1a. The corpus sample's series come from its
# sources file. Training never sees a city with adjacency truth.
DEV_TRUTH_SERIES = {"sanborn00001", "sanborn05511", "sanborn00656", "sanborn08356"}
SAMPLE_SOURCES = REPO / "testdata" / "adjacency_truth" / "corpus-100.sources.json"

NEAR = (
    0.15  # a confirming read lies within this share of the image width of its stretch
)
IGNORE_BAND = 0.20  # ignore band half-width around unconfirmed outline, share of width
HEIGHT_BAND = (0.6, 1.8)  # confirmed reads within this factor of the volume's median
DEFAULT_ADOPT_HEIGHTS = (28.0, 160.0)  # px, for volumes with too few confirmed reads
IGNORE = 255  # mask value: no loss

TILE = 512
POS_WEIGHT = 5.0


# ---------------------------------------------------------------- select


def truth_series() -> set[str]:
    """City series of every adjacency-truth page, excluded from training."""
    sources = json.loads(SAMPLE_SOURCES.read_text())["pages"]
    return {s["item"].split("_")[0] for s in sources.values()} | DEV_TRUTH_SERIES


def oim_exports(documents: Path) -> dict[str, Path]:
    """Item -> OIM truth export, from ``oim/`` and ``oim-editions/``."""
    found = {p.parent.name: p for p in (documents / "oim").glob("*/main.iiif.json")}
    for path in (documents / "oim-editions").glob("*/main.export.iiif.json"):
        found.setdefault(path.parent.name, path)
    return found


def norm(text: str) -> str:
    """Comparable page-number text."""
    return "".join((text or "").split()).upper()


def page_entry(
    page: OimPage, neighbours: dict[str, LineString], adjacency_page: dict, image: str
) -> dict:
    """A manifest entry: OIM geometry scaled to the mirror image, plus the reads that matter.

    Only reads of a neighbour's number are kept; the rest can never confirm anything.
    """
    sx = adjacency_page["width"] / page.width
    sy = adjacency_page["height"] / page.height
    numbers = set(neighbours)
    return {
        "image": image,
        "width": adjacency_page["width"],
        "height": adjacency_page["height"],
        "selector": [
            [round(x * sx, 1), round(y * sy, 1)]
            for x, y in page.selector.exterior.coords
        ],
        "neighbours": {
            number: [[round(x * sx, 1), round(y * sy, 1)] for x, y in line.coords]
            for number, line in neighbours.items()
        },
        "detections": [
            {"key": d["key"], "polygon": d["polygon"]}
            for d in adjacency_page.get("detections") or []
            if norm(d.get("key") or "") in numbers
        ],
        "adopted": [],
    }


def volume_entries(export: Path, adjacency: dict, mirror: str) -> list[dict]:
    """Manifest entries for a volume's unsplit, non-key-map pages with OIM neighbours."""
    oim_pages = load_pages(export)
    sheets = Counter(p.key.split("__")[0] for p in oim_pages)
    mirror_keys = {k.lower(): k for k in adjacency["pages"]}
    entries = []
    for page in oim_pages:
        key = mirror_keys.get(page.key.lower())
        if key is None or "__" in page.key or sheets[page.key] > 1 or key == "p0":
            continue
        neighbours = predicted_neighbours(page, oim_pages)
        if neighbours:
            entry = page_entry(
                page, neighbours, adjacency["pages"][key], f"{mirror}/{key}.jpg"
            )
            entries.append({"key": key, **entry})
    return entries


def fetch_adjacency(mirror: str, path: Path) -> dict | None:
    """corpus-v1's adjacency.json for an item (cached at ``path``), or None."""
    if not path.exists():
        command = [*AWS, "s3", "cp", f"{mirror}/runs/corpus-v1/adjacency.json"]
        if subprocess.run([*command, str(path), "--quiet"], check=False).returncode:
            return None
    return json.loads(path.read_text())


def cmd_select(args: argparse.Namespace) -> None:
    work: Path = args.work
    cache = work / "adjacency"
    cache.mkdir(parents=True, exist_ok=True)
    documents: Path = args.documents
    with (documents / "oim-truth-volumes.tsv").open() as handle:
        years = {
            r["identifier"]: r["year"] for r in csv.DictReader(handle, delimiter="\t")
        }
    items_tsv = work / "items.tsv"
    if not items_tsv.exists():
        subprocess.run(
            [*AWS, "s3", "cp", f"{BUCKET}/_craft/items.tsv", str(items_tsv)], check=True
        )
    with items_tsv.open() as handle:
        states = {r["item"]: r["state"] for r in csv.DictReader(handle, delimiter="\t")}
    excluded = truth_series()
    pool = [
        (item, export)
        for item, export in sorted(oim_exports(documents).items())
        if item.split("_")[0] not in excluded
        and item in states
        and years.get(item, "").isdigit()
    ]
    random.Random(0).shuffle(pool)
    rng = random.Random(1)
    taken = Counter()
    manifest = []
    for item, export in pool:
        decade = int(years[item]) // 10 * 10
        if taken[decade] >= DECADE_QUOTA.get(decade, 0):
            continue
        mirror = f"{BUCKET}/by-state/{states[item]}/{years[item]}/{item}"
        adjacency = fetch_adjacency(mirror, cache / f"{item}.json")
        if not adjacency:
            continue
        entries = volume_entries(export, adjacency, mirror)
        if not entries:
            continue
        taken[decade] += 1
        for entry in rng.sample(entries, min(PAGES_PER_VOLUME, len(entries))):
            manifest.append({"item": item, "year": years[item], **entry})
    (work / "manifest.json").write_text(json.dumps(manifest))
    print(
        f"{len(manifest)} pages from {sum(taken.values())} volumes; {dict(sorted(taken.items()))}"
    )


# ---------------------------------------------------------------- fetch


def page_name(entry: dict) -> str:
    """A manifest page's file stem: '<item>__<page key>'."""
    return f"{entry['item']}__{entry['key']}"


def cmd_fetch(args: argparse.Namespace) -> None:
    images = args.work / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.work / "manifest.json").read_text())

    def fetch(entry: dict) -> bool:
        target = images / f"{page_name(entry)}.jpg"
        if target.exists() and target.stat().st_size:
            return True
        command = [*AWS, "s3", "cp", entry["image"], str(target), "--quiet"]
        return subprocess.run(command, check=False).returncode == 0

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(fetch, manifest))
    print(f"{sum(results)}/{len(results)} images in {images}")


# ---------------------------------------------------------------- labels


def box_bounds(polygon: list) -> tuple[float, float, float, float]:
    """Axis-aligned bounds (x0, y0, x1, y1) of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def confirmed_reads(entry: dict) -> list[list]:
    """Reads of a predicted neighbour's number near its stretch, as [x0, y0, x1, y1, number]."""
    stretches = {n: LineString(c) for n, c in entry["neighbours"].items()}
    reads = []
    for detection in entry["detections"]:
        number = norm(detection.get("key") or "")
        if number not in stretches:
            continue
        x0, y0, x1, y1 = box_bounds(detection["polygon"])
        if y1 - y0 < MIN_HEIGHT:
            continue
        centre = Point((x0 + x1) / 2, (y0 + y1) / 2)
        if centre.distance(stretches[number]) > NEAR * entry["width"]:
            continue
        reads.append([x0, y0, x1, y1, number])
    return reads


def height_filtered(reads_by_page: dict[str, list]) -> dict[str, list]:
    """Drop reads whose height is off the volume's median (house numbers, merged junk)."""
    heights = [r[3] - r[1] for reads in reads_by_page.values() for r in reads]
    if not heights:
        return reads_by_page
    median = float(np.median(heights))
    low, high = HEIGHT_BAND[0] * median, HEIGHT_BAND[1] * median
    return {
        name: [r for r in reads if low <= r[3] - r[1] <= high]
        for name, reads in reads_by_page.items()
    }


def fill(mask: np.ndarray, geometry, value: int) -> None:
    """Rasterize a shapely (Multi)Polygon into ``mask``."""
    if geometry.is_empty:
        return
    for polygon in getattr(geometry, "geoms", [geometry]):
        if isinstance(polygon, Polygon):
            points = np.round(np.array(polygon.exterior.coords)).astype(np.int32)
            cv2.fillPoly(mask, [points], value)


def page_mask(entry: dict, positives: list[list]) -> np.ndarray:
    """A page's training mask: 1 on positive boxes, IGNORE on unconfirmed outline, else 0.

    A neighbour counts as confirmed when some positive box carries its number;
    the band along an unconfirmed neighbour's stretch, and along outline no OIM
    neighbour borders, is ignored.
    """
    width, height = entry["width"], entry["height"]
    mask = np.zeros((height, width), np.uint8)
    stretches = {n: LineString(c) for n, c in entry["neighbours"].items()}
    confirmed = {norm(str(box[4])) for box in positives}
    outline = Polygon(entry["selector"]).exterior
    bordered = unary_union([s.buffer(NEAR * width) for s in stretches.values()])
    free = outline.difference(bordered) if stretches else outline
    ignore = [
        stretches[n].buffer(IGNORE_BAND * width)
        for n in stretches
        if n not in confirmed
    ]
    if not free.is_empty and free.length > 0.02 * outline.length:
        ignore.append(free.buffer(IGNORE_BAND * width))
    if ignore:
        fill(mask, unary_union(ignore), IGNORE)
    for x0, y0, x1, y1, _ in positives:
        cv2.rectangle(mask, (int(x0), int(y0)), (round(x1), round(y1)), 1, -1)
    return mask


def manifest_positives(manifest: list[dict], adopted: bool = False) -> dict[str, list]:
    """Page name -> positive boxes: height-filtered confirmed reads (+ adopted boxes)."""
    reads: dict[str, list] = {}
    for item in sorted({entry["item"] for entry in manifest}):
        pages = {page_name(e): e for e in manifest if e["item"] == item}
        reads |= height_filtered(
            {name: confirmed_reads(e) for name, e in pages.items()}
        )
    if adopted:
        for entry in manifest:
            reads[page_name(entry)] = reads[page_name(entry)] + entry.get("adopted", [])
    return reads


def cmd_labels(args: argparse.Namespace) -> None:
    masks = args.work / args.masks
    masks.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.work / "manifest.json").read_text())
    positives = manifest_positives(manifest, adopted=args.adopted)
    index = []
    neighbours = confirmed = 0
    for entry in manifest:
        name = page_name(entry)
        if not (args.work / "images" / f"{name}.jpg").exists():
            continue
        boxes = positives[name]
        cv2.imwrite(str(masks / f"{name}.png"), page_mask(entry, boxes))
        index.append({"name": name, "item": entry["item"], "boxes": boxes})
        neighbours += len(entry["neighbours"])
        confirmed += len({norm(str(b[4])) for b in boxes} & set(entry["neighbours"]))
    (masks / "index.json").write_text(json.dumps(index))
    boxes_total = sum(len(e["boxes"]) for e in index)
    print(
        f"{len(index)} pages, {boxes_total} boxes; "
        f"{confirmed}/{neighbours} predicted neighbours confirmed"
    )


# ---------------------------------------------------------------- train


def is_validation(item: str) -> bool:
    """One volume in ten, by hash, is held out to pick nothing but report progress."""
    return int(hashlib.md5(item.encode()).hexdigest(), 16) % 10 == 0


def read_page(work: Path, masks: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(grayscale image, mask) for a page."""
    image = cv2.imread(str(work / "images" / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(masks / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"missing image or mask for {name}")
    return image, mask


def tile_centre(
    shape: tuple[int, int], boxes: list, rng: np.random.Generator, size: int
) -> tuple[float, float]:
    """Where to centre a training tile: half near a positive, a quarter in the outer
    quarter of the page (margins: page numbers, titles -- the hard negatives), a
    quarter anywhere."""
    height, width = shape
    roll = rng.random()
    if roll < 0.5 and boxes:
        x0, y0, x1, y1, _ = boxes[rng.integers(len(boxes))]
        return (
            (x0 + x1) / 2 + rng.uniform(-0.4, 0.4) * size,
            (y0 + y1) / 2 + rng.uniform(-0.4, 0.4) * size,
        )
    if roll < 0.75:
        depth, along = rng.uniform(0, 0.25), rng.uniform(0, 1)
        side = rng.integers(4)
        return [
            (depth * width, along * height),
            ((1 - depth) * width, along * height),
            (along * width, depth * height),
            (along * width, (1 - depth) * height),
        ][side]
    return rng.uniform(0, width), rng.uniform(0, height)


def sample_tile(
    image: np.ndarray, mask: np.ndarray, boxes: list, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A TILE x TILE (image, mask) crop with scale (0.75-1.33x) and photometric jitter."""
    scale = float(np.exp(rng.uniform(math.log(0.75), math.log(1.33))))
    size = round(TILE / scale)
    height, width = image.shape
    cx, cy = tile_centre(image.shape, boxes, rng, size)
    left = int(np.clip(cx - size / 2, 0, max(0, width - size)))
    top = int(np.clip(cy - size / 2, 0, max(0, height - size)))
    crop = image[top : top + size, left : left + size]
    crop_mask = mask[top : top + size, left : left + size]
    short_y, short_x = size - crop.shape[0], size - crop.shape[1]
    if short_y or short_x:
        crop = np.pad(crop, ((0, short_y), (0, short_x)), constant_values=210)
        crop_mask = np.pad(
            crop_mask, ((0, short_y), (0, short_x)), constant_values=IGNORE
        )
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    crop = cv2.resize(crop, (TILE, TILE), interpolation=interpolation)
    crop_mask = cv2.resize(crop_mask, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    jittered = crop.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-25, 25)
    if rng.random() < 0.3:
        jittered = cv2.GaussianBlur(jittered, (0, 0), rng.uniform(0.3, 1.0))
    return np.clip(jittered, 0, 255).astype(np.uint8), crop_mask


def masked_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Positive-weighted BCE plus soft Dice, over the pixels with weight 1 only."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=weights,
        pos_weight=torch.tensor(POS_WEIGHT, device=logits.device),
        reduction="sum",
    ) / weights.sum().clamp(min=1)
    probability = torch.sigmoid(logits) * weights
    intersection = (probability * targets).sum()
    denominator = probability.sum() + (targets * weights).sum()
    return bce + 1 - (2 * intersection + 1) / (denominator + 1)


def validate(model: UNet, pages: list[tuple], device: torch.device | str) -> str:
    """Positive-box recall and stray components per page over held-out (image, mask, boxes)."""
    found = total = stray = 0
    for image, mask, boxes in pages:
        components = number_components(predict_page(model, image, device))
        for x0, y0, x1, y1, _ in boxes:
            total += 1
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            found += any(a <= cx <= c and b <= cy <= d for a, b, c, d in components)
        for a, b, c, d in components:
            region = mask[b:d, a:c]
            if not (region == 1).any() and (region == IGNORE).mean() <= 0.5:
                stray += 1
    model.train()
    return (
        f"val recall {found}/{total} = {found / max(1, total):.1%}, "
        f"stray components/page {stray / max(1, len(pages)):.2f}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    masks = args.work / args.masks
    index = json.loads((masks / "index.json").read_text())
    train = [p for p in index if not is_validation(p["item"])]
    held_out = [p for p in index if is_validation(p["item"])][: args.validation_pages]
    print(f"train {len(train)} pages, validation {len(held_out)}", flush=True)
    device = select_device()
    if args.init:
        model = load_detector(args.init, device)
    else:
        model = build_detector(device)
        with torch.no_grad():  # start at a small positive prior, not at 0.5
            model.head.bias.fill_(math.log(0.01 / 0.99))  # type: ignore[union-attr]
    model.train()
    validation = [
        (*read_page(args.work, masks, p["name"]), p["boxes"]) for p in held_out
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=args.steps, pct_start=0.05
    )
    rng = np.random.default_rng(0)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    losses = []
    started = time.time()
    for step in range(1, args.steps + 1):
        tiles, targets = [], []
        for i in rng.integers(len(train), size=args.batch_size):
            page = train[i]
            if page["name"] not in cache:
                cache[page["name"]] = read_page(args.work, masks, page["name"])
            tile, tile_mask = sample_tile(*cache[page["name"]], page["boxes"], rng)
            tiles.append(normalize_patch(tile))
            targets.append(tile_mask)
        inputs = torch.from_numpy(np.stack(tiles)[:, None]).to(device)
        stacked = np.stack(targets)[:, None]
        positive = torch.from_numpy((stacked == 1).astype(np.float32)).to(device)
        scored = torch.from_numpy((stacked != IGNORE).astype(np.float32)).to(device)
        loss = masked_loss(model(inputs), positive, scored)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        losses.append(float(loss.detach()))
        if step % 500 == 0:
            elapsed = time.time() - started
            print(
                f"step {step}: loss {np.mean(losses[-500:]):.4f} [{elapsed:.0f}s]",
                flush=True,
            )
        if step % 3000 == 0 or step == args.steps:
            print("  " + validate(model, validation, device), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)


# ---------------------------------------------------------------- adopt


def read_component(reader, gray: np.ndarray, component: Component) -> tuple[str, float]:
    """The existing reader's digits-only (text, confidence) for one component."""
    x0, y0, x1, y1 = component
    results = reader.recognize(gray, [[x0, x1, y0, y1]], [], allowlist="0123456789")
    if not results:
        return "", 0.0
    _, text, confidence = results[0]
    return str(text).strip(), float(confidence)


def adoption_heights(boxes: list) -> tuple[float, float]:
    """Plausible component heights for a volume, from its positive boxes.

    Components run a little taller than OCR boxes, hence the extra 30% headroom.
    """
    heights = [b[3] - b[1] for b in boxes]
    if len(heights) < 3:
        return DEFAULT_ADOPT_HEIGHTS
    median = float(np.median(heights))
    return HEIGHT_BAND[0] * median, HEIGHT_BAND[1] * median * 1.3


def candidate_components(
    components: list[Component], positives: list, heights: tuple[float, float]
) -> list[Component]:
    """Components of a plausible height that overlap no existing positive box."""

    def overlaps(c: Component, box: list) -> bool:
        return c[0] < box[2] and box[0] < c[2] and c[1] < box[3] and box[1] < c[3]

    return [
        c
        for c in components
        if heights[0] <= c[3] - c[1] <= heights[1]
        and not any(overlaps(c, box) for box in positives)
    ]


def adopt_boxes(
    entry: dict,
    positives: list,
    candidates: list[Component],
    *,
    read: Callable[[Component], tuple[str, float]] | None = None,
) -> tuple[list[list], Counter]:
    """New positive boxes for the page's unconfirmed neighbours, and what happened to each.

    A neighbour adopts a candidate when exactly one lies within NEAR of its
    stretch and nearer to it than to any other neighbour's stretch. ``read``, the
    reader's (digits, confidence) for a component, vetoes a confident read that
    does not contain the neighbour's number.
    """
    stretches = {n: LineString(c) for n, c in entry["neighbours"].items()}
    confirmed = {norm(str(box[4])) for box in positives}
    outcomes = Counter()
    adopted = []
    for number, line in stretches.items():
        if number in confirmed:
            continue
        nearby = []
        for c in candidates:
            centre = Point((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
            distance = centre.distance(line)
            if distance > NEAR * entry["width"]:
                continue
            if any(
                centre.distance(other) < distance
                for other_number, other in stretches.items()
                if other_number != number
            ):
                continue
            nearby.append(c)
        if len(nearby) != 1:
            outcomes["none" if not nearby else "ambiguous"] += 1
            continue
        component = nearby[0]
        if read is not None:
            text, confidence = read(component)
            if text and number not in text and confidence >= 0.5:
                outcomes["read as another number"] += 1
                continue
        adopted.append([*map(float, component), number])
        outcomes["adopted"] += 1
    return adopted, outcomes


def cmd_adopt(args: argparse.Namespace) -> None:
    import easyocr

    device = select_device()
    model = load_detector(args.weights, device)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    manifest = json.loads((args.work / "manifest.json").read_text())
    positives = manifest_positives(manifest)
    outcomes = Counter()
    for item in sorted({e["item"] for e in manifest}):
        entries = [e for e in manifest if e["item"] == item]
        heights = adoption_heights(
            [b for e in entries for b in positives[page_name(e)]]
        )
        for entry in entries:
            path = args.work / "images" / f"{page_name(entry)}.jpg"
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            boxes = positives[page_name(entry)]
            components = number_components(predict_page(model, gray, device))

            adopted, page_outcomes = adopt_boxes(
                entry,
                boxes,
                candidate_components(components, boxes, heights),
                read=lambda component, gray=gray: read_component(
                    reader, gray, component
                ),
            )
            entry["adopted"] = adopted
            outcomes.update(page_outcomes)
        print(item, dict(outcomes), flush=True)
    (args.work / "manifest.json").write_text(json.dumps(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser(
        "select", help="Choose volumes/pages; write manifest.json"
    )
    select.add_argument("work", type=Path)
    select.add_argument("--documents", type=Path, default=MAPSNAP_DOCUMENTS)
    select.set_defaults(run=cmd_select)
    fetch = commands.add_parser("fetch", help="Download the manifest's page images")
    fetch.add_argument("work", type=Path)
    fetch.add_argument("--workers", type=int, default=12)
    fetch.set_defaults(run=cmd_fetch)
    labels = commands.add_parser("labels", help="Rasterize training masks")
    labels.add_argument("work", type=Path)
    labels.add_argument("--masks", default="masks")
    labels.add_argument("--adopted", action="store_true", help="Include adopted boxes")
    labels.set_defaults(run=cmd_labels)
    train = commands.add_parser("train", help="Train the detector UNet")
    train.add_argument("work", type=Path)
    train.add_argument("--masks", default="masks")
    train.add_argument("--output", type=Path, default=MODEL_PATH)
    train.add_argument("--steps", type=int, default=12000)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument(
        "--init", type=Path, default=None, help="Start from these weights"
    )
    train.add_argument("--validation-pages", type=int, default=120)
    train.set_defaults(run=cmd_train)
    adopt = commands.add_parser("adopt", help="One round of self-training adoption")
    adopt.add_argument("work", type=Path)
    adopt.add_argument("--weights", type=Path, default=MODEL_PATH)
    adopt.set_defaults(run=cmd_adopt)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
