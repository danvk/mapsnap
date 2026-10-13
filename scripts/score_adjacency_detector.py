"""Score the adjacency-number detector against the hand-labelled adjacency truth.

Truth lives in testdata/adjacency_truth/: the 100-page corpus sample (corpus-100.json,
labels in each sheet's 25% mirror image; corpus-100.sources.json ties sheets to
items) and three dev volumes (Hudson vol 9, LA vol 14, Nashville 1a), whose images
and CRAFT boxes are read from data/<volume>/.

    reads      Digits read near each labelled number, CRAFT boxes vs the detector's
               components (at each read size and best-of-three), plus how many labels
               lie inside a box at all. The sample's images, CRAFT boxes and page
               lists are fetched from the mirror into --sample-dir on first use.
    adjacency  The full `mapsnap adjacency` step on the dev volumes with CRAFT boxes
               and with the detector's reads swapped in, scored by claims and edges.
    adoption   How often self-training adoption (train_adjacency_detector adopt)
               picks a real printed reference, on the dev volumes.

    uv run python scripts/score_adjacency_detector.py reads
    uv run python scripts/score_adjacency_detector.py adjacency --out /tmp/adjacency-ab
    uv run python scripts/score_adjacency_detector.py adoption
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch

from mapsnap import page_adjacency
from mapsnap.adjacency_detector import (
    MODEL_PATH,
    READ_SIZES,
    best_reads,
    detection_centre,
    detector_digit_detections,
    load_detector,
    number_components,
    resized_box,
)
from mapsnap.keymap.locate import page_key
from mapsnap.keymap.number_model import select_device
from mapsnap.oim_neighbours import load_pages, predicted_neighbours
from mapsnap.page_adjacency import could_be_claim, digit_detections, horizontal_polygon
from mapsnap.road_model import predict_page
from mapsnap.train_adjacency_detector import (
    adopt_boxes,
    adoption_heights,
    candidate_components,
    confirmed_reads,
    height_filtered,
    norm,
    read_component,
)

REPO = Path(__file__).resolve().parent.parent
TRUTH = REPO / "testdata" / "adjacency_truth"
DEV_VOLUMES = [
    "hudson_co_nj_1950_vol_9",
    "los_angeles_ca_1949_vol_14",
    "nashville_tn_1957_vol1a",
]
RADIUS = 100.0  # px; a read counts for a label when its centre is this close
AWS = ["aws", "--profile", "mapsnap"]


def era(year: int) -> str:
    """The sample's reporting eras."""
    if year < 1900:
        return "1867-1899"
    return "1900-1929" if year < 1930 else "1930-1961"


def matched_labels(labels: list[dict], detections: list[dict]) -> set[int]:
    """Indices of labels read: a detection of the label's number within RADIUS, matched greedily."""
    pairs = []
    for i, label in enumerate(labels):
        text = norm(label["text"])
        for j, detection in enumerate(detections):
            key = detection.get("key") or (
                "0" if detection.get("via") == "zero" else ""
            )
            if text not in (norm(detection.get("text") or ""), norm(key)):
                continue
            x, y = detection_centre(detection)
            distance = ((x - label["x"]) ** 2 + (y - label["y"]) ** 2) ** 0.5
            if distance <= RADIUS:
                pairs.append((distance, i, j))
    used_labels: set[int] = set()
    used_detections: set[int] = set()
    for _, i, j in sorted(pairs):
        if i not in used_labels and j not in used_detections:
            used_labels.add(i)
            used_detections.add(j)
    return used_labels


def located_labels(
    labels: list[dict], boxes: list[list[int]], image_size: tuple[int, int]
) -> set[int]:
    """Indices of labels inside (within 10 px of) a box the adjacency step would read."""
    width, height = image_size
    usable = [b for b in boxes if could_be_claim(horizontal_polygon(b), width, height)]
    return {
        i
        for i, label in enumerate(labels)
        if any(
            b[0] - 10 <= label["x"] <= b[1] + 10
            and b[2] - 10 <= label["y"] <= b[3] + 10
            for b in usable
        )
    }


def craft_boxes(path: Path) -> list[list[int]]:
    """Angle-0 CRAFT boxes [x1, x2, y1, y2] from a boxes.json (empty when missing)."""
    if not path.exists():
        return []
    for entry in json.loads(path.read_text()).get("boxes", []):
        if entry.get("angle") == 0:
            free = [
                [
                    int(min(p[0] for p in quad)),
                    int(max(p[0] for p in quad)),
                    int(min(p[1] for p in quad)),
                    int(max(p[1] for p in quad)),
                ]
                for quad in entry.get("free_list", [])
            ]
            return [
                [int(v) for v in b] for b in entry.get("horizontal_list", [])
            ] + free
    return []


def fetch_sample_page(
    source: dict, name: str, sample_dir: Path
) -> tuple[Path, Path, set[str]]:
    """(image, CRAFT boxes, valid page keys) for a corpus-sample sheet, fetched once."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = sample_dir / f"{name}.jpg"
    boxes = sample_dir / f"{name}.boxes.json"
    keys_path = sample_dir / f"{source['item']}.keys.json"
    mirror = source["mirror"]
    for target, url in (
        (image, mirror),
        (boxes, mirror.removesuffix(".jpg") + ".boxes.json"),
    ):
        if not target.exists():
            subprocess.run([*AWS, "s3", "cp", url, str(target), "--quiet"], check=False)
    if not keys_path.exists():
        listing = subprocess.run(
            [*AWS, "s3", "ls", mirror.rsplit("/", 1)[0] + "/"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        stems = [
            line.split()[-1].removesuffix(".jpg")
            for line in listing.splitlines()
            if line.endswith(".jpg") and ".roadprob" not in line
        ]
        keys_path.write_text(json.dumps(stems))
    stems = json.loads(keys_path.read_text())
    return image, boxes, {k for s in stems if (k := page_key(s)) is not None}


def test_pages(sample_dir: Path) -> list[dict]:
    """Every labelled test page: set, era, image, CRAFT boxes, labels and valid keys."""
    pages = []
    truth = json.loads((TRUTH / "corpus-100.json").read_text())["pages"]
    sources = json.loads((TRUTH / "corpus-100.sources.json").read_text())["pages"]
    for name, entry in truth.items():
        source = sources[name]
        image, boxes, valid = fetch_sample_page(source, name, sample_dir)
        pages.append(
            {
                "set": "sample",
                "era": era(int(source["year"])),
                "name": name,
                "image": image,
                "boxes": boxes,
                "labels": entry.get("labels") or [],
                "valid_keys": valid,
            }
        )
    for volume in DEV_VOLUMES:
        directory = REPO / "data" / volume
        truth = json.loads((TRUTH / f"{volume}.json").read_text())["pages"]
        stems = [p.stem for p in directory.glob("p*.jpg") if "roadprob" not in p.stem]
        valid = {k for s in stems if "__" not in s and (k := page_key(s)) is not None}
        for key, entry in truth.items():
            if (directory / f"{key}.jpg").exists():
                pages.append(
                    {
                        "set": "dev",
                        "era": "",
                        "name": f"{volume}/{key}",
                        "image": directory / f"{key}.jpg",
                        "boxes": directory / f"{key}.boxes.json",
                        "labels": entry.get("labels") or [],
                        "valid_keys": valid,
                    }
                )
    return pages


def cmd_reads(args: argparse.Namespace) -> None:
    import easyocr

    device = select_device()
    model = load_detector(args.weights, device)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    totals: dict[tuple[str, str], Counter] = {}
    for page in test_pages(args.sample_dir):
        gray = cv2.imread(str(page["image"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skipping {page['name']}: no image", file=sys.stderr)
            continue
        height, width = gray.shape
        components = number_components(predict_page(model, gray, device))
        sized = {
            f"detector {factor:+.0%}": [
                resized_box(c, factor, (width, height)) for c in components
            ]
            for factor in READ_SIZES
        }
        arms = {"craft": craft_boxes(page["boxes"]), **sized}
        reads = {
            arm: digit_detections(
                page["image"], reader, page["valid_keys"], (boxes, [])
            )
            for arm, boxes in arms.items()
        }
        reads["detector best3"] = best_reads(components, [reads[a] for a in sized])
        arms["detector best3"] = sized[f"detector {max(READ_SIZES):+.0%}"]
        labels = page["labels"]
        zeros = {i for i, label in enumerate(labels) if norm(label["text"]) == "0"}
        numbers = set(range(len(labels))) - zeros
        groups = [page["set"]] + ([f"sample {page['era']}"] if page["era"] else [])
        for arm, boxes in arms.items():
            located = located_labels(labels, boxes, (width, height))
            read = matched_labels(labels, reads[arm])
            for group in groups:
                t = totals.setdefault((group, arm), Counter())
                t["pages"] += 1
                t["numbers"] += len(numbers)
                t["zeros"] += len(zeros)
                t["located"] += len(located & numbers)
                t["read"] += len(read & numbers)
                t["zeros read"] += len(read & zeros)
                t["unmatched"] += len(reads[arm]) - len(read)
    print(
        f"{'set':22s} {'boxes':16s} {'located':>8s} {'read':>8s} {'zeros':>7s} {'junk/pg':>8s}"
    )
    for (group, arm), t in sorted(totals.items()):
        print(
            f"{group:22s} {arm:16s} {t['located'] / t['numbers']:8.1%} {t['read'] / t['numbers']:8.1%} "
            f"{t['zeros read']:>3d}/{t['zeros']:<3d} {t['unmatched'] / t['pages']:8.2f}"
        )


def score_adjacency(adjacency: dict, truth: dict) -> Counter:
    """Claim and edge counts for one volume's adjacency.json against its truth.

    Panel truth is in the panel image's frame, so panel detections are shifted by
    the panel's bbox. An edge end is correct when the neighbour's number is one of
    the page's labelled numbers.
    """
    t = Counter()
    for key, entry in truth.items():
        labels = [
            label for label in entry.get("labels") or [] if norm(label["text"]) != "0"
        ]
        t["labels"] += len(labels)
        page = adjacency["pages"].get(key)
        if page is None:
            continue
        dx, dy = (page.get("panel_bbox") or [0, 0])[:2]
        claims = [
            {**d, "polygon": [[x - dx, y - dy] for x, y in d["polygon"]]}
            for d in page.get("detections") or []
            if d.get("claim")
        ]
        claimed = matched_labels(labels, claims)
        t["claims"] += len(claims)
        t["labels claimed"] += len(claimed)
        numbers = {norm(label["text"]) for label in labels}
        for kind in ("adjacency", "one_sided"):
            for first, second, *_ in adjacency.get(kind, []):
                if key in (first, second):
                    other = second if first == key else first
                    t[kind] += 1
                    t[f"{kind} correct"] += (
                        norm(other.split("__")[0].removeprefix("p")) in numbers
                    )
    t["mutual edges"] = len(adjacency.get("adjacency", []))
    return t


def cmd_adjacency(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    original = page_adjacency.digit_detections
    device = select_device()
    for arm in ("craft", "detector"):
        page_adjacency.digit_detections = (
            original
            if arm == "craft"
            else detector_digit_detections(load_detector(args.weights, device), device)
        )
        for volume in DEV_VOLUMES:
            output = args.out / f"{volume}.{arm}.json"
            if not output.exists():
                sys.argv = [
                    "adjacency",
                    str(REPO / "data" / volume),
                    "--output",
                    str(output),
                ]
                page_adjacency.main()
    page_adjacency.digit_detections = original
    for arm in ("craft", "detector"):
        t = Counter()
        for volume in DEV_VOLUMES:
            adjacency = json.loads((args.out / f"{volume}.{arm}.json").read_text())
            t.update(
                score_adjacency(
                    adjacency,
                    json.loads((TRUTH / f"{volume}.json").read_text())["pages"],
                )
            )
        print(
            f"{arm:8s} claims {t['claims']} ({t['labels claimed'] / max(1, t['claims']):.1%} correct), "
            f"labels claimed {t['labels claimed']}/{t['labels']} = {t['labels claimed'] / t['labels']:.1%}, "
            f"mutual edges {t['mutual edges']} ({t['adjacency correct']}/{t['adjacency']} ends correct), "
            f"one-sided ends {t['one_sided correct']}/{t['one_sided']}"
        )


def cmd_adoption(args: argparse.Namespace) -> None:
    import easyocr

    device = select_device()
    model = load_detector(args.weights, device)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    verdicts = Counter()
    for volume in DEV_VOLUMES:
        directory = REPO / "data" / volume
        truth = json.loads((TRUTH / f"{volume}.json").read_text())["pages"]
        adjacency = json.loads((directory / "adjacency.json").read_text())["pages"]
        oim_pages = load_pages(directory / "main.iiif.json")
        entries = {}
        for page in oim_pages:
            if "__" in page.key or page.key not in truth or page.key not in adjacency:
                continue
            size = truth[page.key]
            sx, sy = size["width"] / page.width, size["height"] / page.height
            entries[page.key] = {
                "width": size["width"],
                "height": size["height"],
                "detections": adjacency[page.key].get("detections") or [],
                "neighbours": {
                    n: [[x * sx, y * sy] for x, y in line.coords]
                    for n, line in predicted_neighbours(page, oim_pages).items()
                },
            }
        positives = height_filtered({k: confirmed_reads(e) for k, e in entries.items()})
        heights = adoption_heights([b for boxes in positives.values() for b in boxes])
        for key, entry in entries.items():
            gray = cv2.imread(str(directory / f"{key}.jpg"), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            components = number_components(predict_page(model, gray, device))

            adopted, _ = adopt_boxes(
                entry,
                positives[key],
                candidate_components(components, positives[key], heights),
                read=lambda component, gray=gray: read_component(
                    reader, gray, component
                ),
            )
            labels = [
                (label["x"], label["y"], norm(label["text"]))
                for label in truth[key].get("labels") or []
            ]
            for x0, y0, x1, y1, number in adopted:
                near = [
                    t
                    for x, y, t in labels
                    if x0 - 20 <= x <= x1 + 20 and y0 - 20 <= y <= y1 + 20
                ]
                verdicts[
                    "correct"
                    if number in near
                    else ("label, other text" if near else "no label")
                ] += 1
        print(volume, dict(verdicts), flush=True)
    total = sum(verdicts.values())
    print(
        f"adopted {total}: {verdicts['correct'] / max(1, total):.1%} correct; {dict(verdicts)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("command", choices=["reads", "adjacency", "adoption"])
    parser.add_argument("--weights", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=REPO / "data" / "samples" / "adjacency-corpus-100",
        help="Where the corpus sample's images and CRAFT boxes live (fetched if missing)",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/adjacency-detector-ab"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    {"reads": cmd_reads, "adjacency": cmd_adjacency, "adoption": cmd_adoption}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
