"""Fine-tune EasyOCR's latin recognizer on street-label crops (#265 Phase 2).

The production street path already replaces EasyOCR's decoder with a
vocabulary-trie-constrained CTC beam search over the recognizer's per-timestep
probabilities (ctc_vocab_decode.py), so the recognizer itself is the only
learned piece — and it has never seen Sanborn typography, underline rules,
dashed pipe lines, or squished ordinals. This module fine-tunes the stock
``latin_g2`` CRNN (easyocr.model.vgg_model.Model, 3.8M params) on the corpus's
own inlier crops, half of them corrupted by mapsnap.ocr_augment's
measured-geometry artifacts, producing a state_dict that drops into
``reader.recognizer`` unchanged (``mapsnap ocr --recognizer-weights``).

Training data: the fit-anchored harvest JSONL (records with cls/vol/stem/
text/label/poly/angle). Only rows where the read agreed exactly with the
matched street name (``text == label``) are trusted — every geometric labeling
scheme beyond agreement failed manual review (#265). Fargo and nashville are
excluded end to end: they are the held-out decision volumes.

Subcommands:

    build-dataset  harvest.jsonl --data-dir data --out data/ocr_finetune_cache
    render-review  data/ocr_finetune_cache --out review.html
    train          data/ocr_finetune_cache --out models/street_recognizer.pt
"""

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from mapsnap.ocr_augment import augment_crop

HOLDOUT_VOLUMES = ("fargo_nd_1958", "nashville_tn_1957_vol1a")
VAL_FRACTION = 10  # every 10th pair per volume goes to val
MODEL_HEIGHT = 64  # easyocr 1.7.2 imgH for the latin_g2 recognizer
MAX_FRAGMENTS = 500  # junk-ink pool size for edge-junk augmentation
# Agreement below production's GCP admission floor is untrustworthy: user
# review found 180deg-flipped twins (conf 0.006-0.013) and a "60 ft. wide"
# note force-decoded to OLIVE at conf 0.0000 in the sub-floor tail, while
# genuine reads sit at median 0.984. Costs 1.3% of pairs.
PAIR_CONFIDENCE_FLOOR = 0.15


def load_harvest_pairs(
    jsonl_path: Path, include_volumes: tuple[str, ...] | None = None
) -> tuple[list[dict], list[dict]]:
    """(agreement pairs, junk fragment records) from the harvest JSONL.

    Agreement pairs are INLIER/LOWCONF rows whose read matched the street name
    exactly, at or above PAIR_CONFIDENCE_FLOOR, deduplicated across rotation
    twins (the same box read at 90 and 270 both position-match as inliers; the
    highest-confidence record per (page, label, ~center) wins). Junk records
    are WRONGREAD rows, used only as ink fragments for the edge-junk
    augmentation (their labels are untrusted by design).
    """
    candidates: list[dict] = []
    junk: list[dict] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        volume = record["vol"]
        if include_volumes is not None:
            if volume not in include_volumes:
                continue
        elif volume in HOLDOUT_VOLUMES:
            continue
        if (
            record["cls"] in ("INLIER", "LOWCONF")
            and record["text"] == record["label"]
            and record["conf"] >= PAIR_CONFIDENCE_FLOOR
        ):
            candidates.append(record)
        elif record["cls"] == "WRONGREAD":
            junk.append(record)
    best: dict[tuple, dict] = {}
    for record in candidates:
        xs = [p[0] for p in record["poly"]]
        ys = [p[1] for p in record["poly"]]
        key = (
            record["vol"],
            record["stem"],
            record["label"],
            round(sum(xs) / 4 / 10),
            round(sum(ys) / 4 / 10),
        )
        if key not in best or record["conf"] > best[key]["conf"]:
            best[key] = record
    return list(best.values()), junk


def extract_crop(
    data_dir: Path, record: dict, image_cache: dict[str, Image.Image]
) -> np.ndarray:
    """Grayscale crop for a harvest record, upright in reading orientation.

    ``poly`` is an axis-aligned box in original page coordinates; ``angle`` is
    the recognition pass that read it. Cropping first and rotating the crop by
    the same angle reproduces the recognizer's input (rotations are multiples
    of 90 degrees, so crop-then-rotate equals rotate-then-crop).
    """
    key = f"{record['vol']}/{record['stem']}"
    img = image_cache.get(key)
    if img is None:
        img = Image.open(data_dir / record["vol"] / f"{record['stem']}.jpg").convert(
            "L"
        )
        image_cache[key] = img
    xs = [p[0] for p in record["poly"]]
    ys = [p[1] for p in record["poly"]]
    crop = img.crop((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    if record["angle"]:
        crop = crop.rotate(record["angle"], expand=True)
    return np.array(crop)


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Extract agreement-pair crops into {train,val}/ PNG caches + index files."""
    pairs, junk = load_harvest_pairs(args.harvest)
    print(
        f"{len(pairs)} agreement pairs ({len(junk)} junk candidates) — "
        f"holdout {', '.join(HOLDOUT_VOLUMES)} excluded"
    )
    image_cache: dict[str, Image.Image] = {}
    per_volume: dict[str, int] = {}
    indexes = {"train": [], "val": []}
    for split in indexes:
        (args.out / split).mkdir(parents=True, exist_ok=True)
    for record in pairs:
        n = per_volume.get(record["vol"], 0)
        per_volume[record["vol"]] = n + 1
        split = "val" if n % VAL_FRACTION == VAL_FRACTION - 1 else "train"
        crop = extract_crop(args.data_dir, record, image_cache)
        name = f"{len(indexes[split]):05d}.png"
        Image.fromarray(crop).save(args.out / split / name)
        indexes[split].append(
            {
                "file": name,
                "label": record["label"],
                "vol": record["vol"],
                "stem": record["stem"],
                "conf": record["conf"],
            }
        )
        # Cap the cache; pages are grouped in the harvest so old ones are done.
        if len(image_cache) > 40:
            image_cache.pop(next(iter(image_cache)))
    # Junk fragments: small low-confidence boxes, real neighboring ink.
    rng = np.random.default_rng(0)
    rng.shuffle(junk)  # type: ignore[arg-type]
    fragments_dir = args.out / "fragments"
    fragments_dir.mkdir(exist_ok=True)
    kept = 0
    for record in junk:
        if kept >= MAX_FRAGMENTS:
            break
        xs = [p[0] for p in record["poly"]]
        ys = [p[1] for p in record["poly"]]
        if max(xs) - min(xs) > 80 or max(ys) - min(ys) > 40:
            continue
        crop = extract_crop(args.data_dir, record, image_cache)
        Image.fromarray(crop).save(fragments_dir / f"{kept:04d}.png")
        kept += 1
    for split, index in indexes.items():
        (args.out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in index)
        )
        print(f"{split}: {len(index)} crops")
    print(f"fragments: {kept}")


def load_split(cache_dir: Path, split: str) -> list[tuple[np.ndarray, str, dict]]:
    """(crop, label, meta) triples for a cached split."""
    items = []
    for line in (cache_dir / f"{split}.jsonl").read_text().splitlines():
        meta = json.loads(line)
        crop = np.array(Image.open(cache_dir / split / meta["file"]))
        items.append((crop, meta["label"], meta))
    return items


def load_fragments(cache_dir: Path) -> list[np.ndarray]:
    """The junk-ink fragment pool (may be empty)."""
    fragments_dir = cache_dir / "fragments"
    if not fragments_dir.is_dir():
        return []
    return [np.array(Image.open(p)) for p in sorted(fragments_dir.glob("*.png"))]


def crop_to_data_uri(crop: np.ndarray, scale: int = 3) -> str:
    """Base64 PNG data URI for an HTML review sheet, upscaled for legibility."""
    img = Image.fromarray(crop)
    img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cmd_render_review(args: argparse.Namespace) -> None:
    """HTML sheet: real artifact fixtures vs synthesized corruptions."""
    fixtures_dir = Path(__file__).resolve().parent.parent / "testdata/erase_underlines"
    cases = json.loads((fixtures_dir / "cases.json").read_text())
    rows = [
        "<h1>OCR augmentation review</h1>",
        (
            "<p>Top: the 16 real fixtures (before-erasure crops). Below: "
            "training crops with synthesized corruptions — three variants "
            "each. Judge whether the synthetic artifacts are plausible "
            "stand-ins for the real ones.</p>"
        ),
        "<h2>Real artifacts (testdata/erase_underlines)</h2>",
    ]
    for case in cases:
        crop = np.array(
            Image.open(fixtures_dir / f"{case['name']}.before.png").convert("L")
        )
        rows.append(
            f"<div class='item'><img src='{crop_to_data_uri(crop)}'>"
            f"<span>{case['name']}</span></div>"
        )
    rows.append("<h2>Synthesized (train split sample)</h2>")
    train = load_split(args.cache_dir, "train")
    fragments = load_fragments(args.cache_dir)
    rng = np.random.default_rng(7)
    sample_indices = rng.choice(len(train), size=min(40, len(train)), replace=False)
    for i in sample_indices:
        crop, label, meta = train[int(i)]
        cells = [f"<img src='{crop_to_data_uri(crop)}' title='original'>"]
        for k in range(3):
            aug = augment_crop(
                crop, np.random.default_rng(1000 * int(i) + k), fragments
            )
            cells.append(f"<img src='{crop_to_data_uri(aug)}' title='aug {k}'>")
        rows.append(
            f"<div class='item'>{''.join(cells)}"
            f"<span>{label} ({meta['vol']} {meta['stem']})</span></div>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><style>"
        "body{font-family:system-ui;background:#222;color:#eee}"
        ".item{display:inline-block;margin:8px;padding:6px;background:#333;"
        "border-radius:4px;vertical-align:top}"
        ".item img{display:block;margin-bottom:4px;image-rendering:pixelated}"
        ".item span{font-size:11px;color:#aaa}"
        "h2{margin-top:24px}</style>" + "\n".join(rows)
    )
    args.out.write_text(html)
    print(f"wrote {args.out}")


def greedy_decode(logits: "object", character: list[str]) -> list[str]:
    """Plain CTC greedy decode of a (B, T, C) logits tensor to strings."""
    import torch

    assert isinstance(logits, torch.Tensor)
    indices = logits.argmax(dim=2).cpu().numpy()
    texts = []
    for row in indices:
        chars = []
        previous = 0
        for idx in row:
            if idx != previous and idx != 0:
                chars.append(character[idx])
            previous = idx
        texts.append("".join(chars))
    return texts


def batch_tensor(crops: list[np.ndarray]) -> "object":
    """Stack variable-width crops into one (B, 1, H, W) tensor, EasyOCR-style.

    Aspect-preserving resize to MODEL_HEIGHT then right-pad to the batch's max
    width — the same NormalizePAD convention the production path uses, so
    training and inference see identically prepared pixels.
    """
    from easyocr.recognition import AlignCollate

    images = [Image.fromarray(c, "L") for c in crops]
    max_ratio = max(img.width / img.height for img in images)
    img_w = math.ceil(MODEL_HEIGHT * max(1.0, max_ratio))
    collate = AlignCollate(imgH=MODEL_HEIGHT, imgW=img_w, keep_ratio_with_pad=True)
    return collate(images)


def evaluate(model, items, character, batch_size: int = 32) -> float:
    """Exact-match accuracy of greedy decodes against labels (case-folded)."""
    import torch

    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            image = batch_tensor([crop for crop, _, _ in chunk])
            preds = model(image, None)
            texts = greedy_decode(preds, character)
            for (_, label, _), text in zip(chunk, texts):
                correct += text.upper() == label.upper()
    return correct / max(1, len(items))


def cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune the stock recognizer on clean + augmented agreement crops."""
    import easyocr
    import torch

    torch.manual_seed(args.seed)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    model = reader.recognizer
    # converter.character carries the CTC blank sentinel at index 0.
    character = list(reader.converter.character)
    assert character[0] == "[blank]", character[:3]
    charset = set("".join(character[1:]))

    train_items = load_split(args.cache_dir, "train")
    val_items = load_split(args.cache_dir, "val")
    fragments = load_fragments(args.cache_dir)
    dropped = [label for _, label, _ in train_items if not set(label) <= charset]
    if dropped:
        print(
            f"dropping {len(dropped)} pairs with out-of-charset labels: "
            f"{sorted(set(''.join(dropped)) - charset)}"
        )
        train_items = [t for t in train_items if set(t[1]) <= charset]
        val_items = [t for t in val_items if set(t[1]) <= charset]
    print(f"train {len(train_items)}, val {len(val_items)}, fragments {len(fragments)}")

    # A fixed corrupted copy of val measures artifact robustness each epoch.
    val_corrupted = [
        (augment_crop(crop, np.random.default_rng(9000 + i), fragments), label, meta)
        for i, (crop, label, meta) in enumerate(val_items)
    ]

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng(args.seed * 1000 + epoch)
        order = rng.permutation(len(train_items))
        # Bucket by aspect ratio so batch padding stays modest.
        order = sorted(
            order, key=lambda i: train_items[i][0].shape[1] / train_items[i][0].shape[0]
        )
        batches = [
            order[i : i + args.batch_size]
            for i in range(0, len(order), args.batch_size)
        ]
        rng.shuffle(batches)  # type: ignore[arg-type]
        total_loss = 0.0
        for batch in batches:
            crops, labels = [], []
            for i in batch:
                crop, label, _ = train_items[i]
                if rng.random() < args.augment_fraction:
                    crop = augment_crop(
                        crop, np.random.default_rng(rng.integers(1 << 31)), fragments
                    )
                crops.append(crop)
                labels.append(label)
            image = batch_tensor(crops)
            preds = model(image, None)  # (B, T, C) raw logits
            log_probs = preds.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            targets, target_lengths = reader.converter.encode(labels)
            input_lengths = torch.full(
                (len(crops),), log_probs.shape[0], dtype=torch.int32
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss)
        clean = evaluate(model, val_items, character)
        corrupted = evaluate(model, val_corrupted, character)
        score = (clean + corrupted) / 2
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(1, len(batches)),
                "val_clean": round(clean, 4),
                "val_corrupted": round(corrupted, 4),
            }
        )
        marker = ""
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), args.out)
            marker = "  <- saved"
        print(
            f"epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  "
            f"val clean {clean:.3f}  corrupted {corrupted:.3f}{marker}",
            flush=True,
        )
    (args.out.parent / (args.out.stem + ".history.json")).write_text(
        json.dumps(history, indent=1)
    )
    print(f"best mean val exact-match: {best_score:.3f} -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-dataset", help="Extract crops from the harvest JSONL.")
    p.add_argument("harvest", type=Path, help="ocr-train-harvest JSONL path")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/ocr_finetune_cache"))
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("render-review", help="HTML: real fixtures vs synthetics.")
    p.add_argument("cache_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_render_review)

    p = sub.add_parser("train", help="Fine-tune the recognizer (CPU).")
    p.add_argument("cache_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("models/street_recognizer.pt"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--augment-fraction", type=float, default=0.5)
    p.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
