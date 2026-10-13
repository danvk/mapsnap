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
    labeler    The corpus sample as two adjacency-labeler volumes beside it:
               <sample>-detector (the detector's reads; pink = a component with
               no valid read) and <sample>-compare (each hand label marked
               ✓ / ✗ read N / ✗ unread / ✗ missed, plus ✗ extra reads).

    uv run python scripts/score_adjacency_detector.py reads
    uv run python scripts/score_adjacency_detector.py adjacency --out /tmp/adjacency-ab
    uv run python scripts/score_adjacency_detector.py adoption
    uv run python scripts/score_adjacency_detector.py labeler
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


def read_text(detection: dict) -> str:
    """The page number a detection resolved to ('0' for an explicit zero)."""
    return detection.get("key") or ("0" if detection.get("via") == "zero" else "")


def label_matches(labels: list[dict], detections: list[dict]) -> dict[int, int]:
    """Label index -> detection index for labels read: a detection of the label's
    number within RADIUS, matched greedily by distance."""
    pairs = []
    for i, label in enumerate(labels):
        text = norm(label["text"])
        for j, detection in enumerate(detections):
            if text not in (
                norm(detection.get("text") or ""),
                norm(read_text(detection)),
            ):
                continue
            x, y = detection_centre(detection)
            distance = ((x - label["x"]) ** 2 + (y - label["y"]) ** 2) ** 0.5
            if distance <= RADIUS:
                pairs.append((distance, i, j))
    matches: dict[int, int] = {}
    for _, i, j in sorted(pairs):
        if i not in matches and j not in matches.values():
            matches[i] = j
    return matches


def matched_labels(labels: list[dict], detections: list[dict]) -> set[int]:
    """Indices of labels read (see label_matches)."""
    return set(label_matches(labels, detections))


def nearest_within(
    point: tuple[float, float], candidates: list[tuple[float, float]], taken: set[int]
) -> int | None:
    """Index of the nearest untaken candidate within RADIUS of point, or None."""
    best = None
    for j, (x, y) in enumerate(candidates):
        distance = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
        if (
            j not in taken
            and distance <= RADIUS
            and (best is None or distance < best[0])
        ):
            best = (distance, j)
    return best[1] if best else None


def comparison_labels(
    labels: list[dict], reads: list[dict], unread: list[tuple[float, float]]
) -> list[dict]:
    """Hand labels annotated with what the detector did, for viewing in the labeler.

    Each hand label becomes '12 ✓' (read), '12 ✗ read 128' (another number read
    within RADIUS), '12 ✗ unread' (a component with no valid read within RADIUS)
    or '12 ✗ missed'. Reads left over are '51 ✗ extra' at their own position;
    leftover unread components are text-less points, which the labeler draws pink.
    """
    matches = label_matches(labels, reads)
    centres = [detection_centre(d) for d in reads]
    used_reads = set(matches.values())
    used_unread: set[int] = set()
    out = []
    for i, label in enumerate(labels):
        text = label["text"].strip()
        point = (label["x"], label["y"])
        if i in matches:
            verdict = f"{text} ✓"
        elif (j := nearest_within(point, centres, used_reads)) is not None:
            used_reads.add(j)
            verdict = f"{text} ✗ read {read_text(reads[j])}"
        elif (j := nearest_within(point, unread, used_unread)) is not None:
            used_unread.add(j)
            verdict = f"{text} ✗ unread"
        else:
            verdict = f"{text} ✗ missed"
        out.append({"x": label["x"], "y": label["y"], "text": verdict})
    for j, (x, y) in enumerate(centres):
        if j not in used_reads:
            out.append({"x": x, "y": y, "text": f"{read_text(reads[j])} ✗ extra"})
    for j, (x, y) in enumerate(unread):
        if j not in used_unread:
            out.append({"x": x, "y": y, "text": ""})
    return out


def verdict(text: str) -> str:
    """The outcome a comparison label records: '✓', '✗ read', '✗ unread', '✗ missed',
    '✗ extra', or 'unread box' for a text-less component."""
    if not text:
        return "unread box"
    outcome = text.split(" ", 1)[1]
    return "✗ read" if outcome.startswith("✗ read") else outcome


def write_volume(directory: Path, sample_dir: Path, pages: dict[str, dict]) -> None:
    """A labeler pseudo-volume: the sample's images (hard links), sources and these labels."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in pages:
        target = directory / f"{name}.jpg"
        if not target.exists():
            target.hardlink_to(sample_dir / f"{name}.jpg")
    sources = sample_dir / "sources.json"
    if sources.exists():
        (directory / "sources.json").write_text(sources.read_text())
    (directory / "adjacency-truth.json").write_text(
        json.dumps({"pages": pages}, indent=1)
    )


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


def test_pages(
    sample_dir: Path, sample_only: bool = False, with_unlabelled: bool = False
) -> list[dict]:
    """Every labelled test page: set, era, image, CRAFT boxes, labels and valid keys.

    ``with_unlabelled`` adds the sample sheets the labeler never recorded (no
    printed references), with no labels.
    """
    pages = []
    truth = json.loads((TRUTH / "corpus-100.json").read_text())["pages"]
    sources = json.loads((TRUTH / "corpus-100.sources.json").read_text())["pages"]
    names = sources if with_unlabelled else truth
    for name in names:
        entry = truth.get(name, {})
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
    for volume in [] if sample_only else DEV_VOLUMES:
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


def cmd_labeler(args: argparse.Namespace) -> None:
    """Write the corpus sample's detector reads, and a comparison, as labeler volumes."""
    import easyocr

    device = select_device()
    model = load_detector(args.weights, device)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    detector_pages: dict[str, dict] = {}
    compare_pages: dict[str, dict] = {}
    for page in test_pages(args.sample_dir, sample_only=True, with_unlabelled=True):
        gray = cv2.imread(str(page["image"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        height, width = gray.shape
        components = number_components(predict_page(model, gray, device))
        reads_by_size = [
            digit_detections(
                page["image"],
                reader,
                page["valid_keys"],
                ([resized_box(c, factor, (width, height)) for c in components], []),
            )
            for factor in READ_SIZES
        ]
        reads, unread = [], []
        for component in components:
            best = best_reads([component], reads_by_size)
            if best:
                reads.append(best[0])
            else:
                x0, y0, x1, y1 = component
                unread.append(((x0 + x1) / 2, (y0 + y1) / 2))
        size = {"width": width, "height": height}
        detector_pages[page["name"]] = size | {
            "labels": [
                {"x": x, "y": y, "text": read_text(d)}
                for d in reads
                for x, y in [detection_centre(d)]
            ]
            + [{"x": x, "y": y, "text": ""} for x, y in unread]
        }
        compare_pages[page["name"]] = size | {
            "labels": comparison_labels(page["labels"], reads, unread)
        }
        print(page["name"], len(reads), "reads,", len(unread), "unread", flush=True)
    base = args.sample_dir.name
    write_volume(
        args.sample_dir.parent / f"{base}-detector", args.sample_dir, detector_pages
    )
    write_volume(
        args.sample_dir.parent / f"{base}-compare", args.sample_dir, compare_pages
    )
    verdicts = Counter(
        verdict(label["text"])
        for page in compare_pages.values()
        for label in page["labels"]
    )
    print(f"wrote {base}-detector and {base}-compare; {dict(verdicts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "command", choices=["reads", "adjacency", "adoption", "labeler"]
    )
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
    commands = {
        "reads": cmd_reads,
        "adjacency": cmd_adjacency,
        "adoption": cmd_adoption,
        "labeler": cmd_labeler,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
