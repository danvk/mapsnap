#!/usr/bin/env python3
"""Benchmark the fine-tuned street recognizer against stock EasyOCR (#265).

Runs the held-out decision volumes (fargo, nashville) through four offline
eval sets, using the production constrained-CTC decode for both arms so the
only variable is the recognizer weights:

* **E1 clean** — agreement pairs without underline erasure. Production read
  these correctly by construction, so this is a no-regression gate for the
  fine-tuned arm (Gate 1), plus a harness sanity check: the stock arm must
  substantially reproduce the production reads.
* **E2 real artifacts** — agreement pairs whose detection has
  ``underline_removed: true``: real rules/dashes that production only read by
  erasing them. Both arms read the RAW crop, no erasers — measuring whether
  fine-tuning replaces the eraser stack (Gate 2). NOTE: production's
  three-vote accuracy on this set is ~100% *by construction* (the pairs exist
  because production read them), so the honest comparisons are
  fine-tuned-raw vs stock-raw (robustness gained) and fine-tuned-raw vs ~1.0
  (can it match the eraser stack without erasure).
* **E3 synthetic** — E1 crops corrupted by mapsnap.ocr_augment. Fair
  head-to-head (same corruption for both arms), but the corruptions match the
  fine-tune's training augmentation, so a win here is expected and weak
  evidence by itself.
* **E4 upside pool** — WRONGREAD/LOWCONF records: crops production failed to
  turn into a street. Reads that the fine-tuned arm produces at >= 0.15 are
  verified fit-anchored (project through the page's georef; count
  label_inliers) and sampled into an HTML sheet for manual review.

Usage:
  uv run python scripts/ocr_recognizer_bench.py --weights models/street_recognizer.pt
  uv run python scripts/ocr_recognizer_bench.py --limit 50   # stock-only smoke test
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapsnap.ctc_vocab_decode import (
    build_trie,
    generate_vocab_strings,
    prefix_constrained_ctc,
)
from mapsnap.detect_text import NON_STREET_TEXT, SCALE_NOTE_TEXT
from mapsnap.ocr_augment import augment_crop
from mapsnap.road_model import page_world_affine
from mapsnap.streets import build_block_index
from mapsnap.train_street_recognizer import (
    HOLDOUT_VOLUMES,
    batch_tensor,
    extract_crop,
    load_harvest_pairs,
)

DATA = Path(__file__).resolve().parent.parent / "data"
CONFIDENCE_FLOOR = 0.15  # production's --min-confidence
BEAM_WIDTH = 20


def volume_trie(volume: str):
    """(trie root, block_index) built the way production builds its vocabulary."""
    geojson = json.loads((DATA / volume / "centerlines.geojson").read_text())
    block_index = build_block_index(geojson)
    vocab = generate_vocab_strings(set(block_index))
    strings = sorted(set(vocab) | set(NON_STREET_TEXT) | set(SCALE_NOTE_TEXT))
    return build_trie(strings), block_index


def constrained_read(
    model, char_list, trie_root, crop: np.ndarray
) -> tuple[str, float]:
    """One crop through the production decode path: softmax -> trim -> trie beam."""
    import torch
    import torch.nn.functional as F

    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return "", 0.0
    image = batch_tensor([crop])
    with torch.no_grad():
        preds = model(image, None)
        probs = F.softmax(preds, dim=2)[0].numpy()
    effective_t = probs.shape[0]
    for t in range(probs.shape[0] - 1, 0, -1):
        if probs[t, 0] >= 0.9999:
            effective_t = t
        else:
            break
    text, path_prob = prefix_constrained_ctc(
        probs[:effective_t], trie_root, char_list, BEAM_WIDTH
    )
    return text, path_prob ** (1.0 / max(len(text), 1))


def underline_flags(records: list[dict]) -> dict[int, bool]:
    """id(record) -> underline_removed, joined from streets.json by polygon."""
    by_page: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        by_page[(record["vol"], record["stem"])].append(record)
    flags: dict[int, bool] = {}
    for (volume, stem), page_records in by_page.items():
        path = DATA / volume / f"{stem}.streets.json"
        index: dict[tuple, bool] = {}
        if path.exists():
            for det in json.loads(path.read_text())["streets"]:
                key = tuple(tuple(round(c) for c in p) for p in det["polygon"])
                index[key] = bool(det.get("underline_removed"))
        for record in page_records:
            key = tuple(tuple(round(c) for c in p) for p in record["poly"])
            flags[id(record)] = index.get(key, False)
    return flags


def evaluate_set(name, items, arms, char_list, tries) -> dict:
    """Exact-match accuracy per arm on (record, crop) items; returns per-arm stats."""
    stats = {}
    for arm_name, model in arms.items():
        correct = 0
        floor_pass = 0
        for record, crop in items:
            text, conf = constrained_read(
                model, char_list, tries[record["vol"]][0], crop
            )
            ok = text.upper() == record["label"].upper() and conf >= CONFIDENCE_FLOOR
            correct += ok
            floor_pass += conf >= CONFIDENCE_FLOOR
        stats[arm_name] = {
            "n": len(items),
            "exact": round(correct / max(1, len(items)), 4),
            "emitted": round(floor_pass / max(1, len(items)), 4),
        }
        print(
            f"{name:12s} {arm_name:10s} n={len(items):4d} "
            f"exact={stats[arm_name]['exact']:.3f} "
            f"emitted={stats[arm_name]['emitted']:.3f}",
            flush=True,
        )
    return stats


def fit_anchored_verify(candidates: list[dict], block_index, volume: str) -> int:
    """Count candidate new reads verified by the page's existing georef.

    Writes the candidates as a temporary streets.json and runs the production
    feature-prep + inlier predicate (prepare_label_features / label_inliers)
    against the page's fitted affine — the same fit-anchored check the harvest
    itself was built with.
    """
    from mapsnap.georef_from_labels import label_inliers, prepare_label_features

    verified = 0
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_stem[c["stem"]].append(c)
    for stem, group in by_stem.items():
        georef_path = DATA / volume / f"{stem}.georef.json"
        if not georef_path.exists():
            continue
        georef = json.loads(georef_path.read_text())
        detections = []
        for c in group:
            xs = [p[0] for p in c["poly"]]
            ys = [p[1] for p in c["poly"]]
            long_side = max(max(xs) - min(xs), max(ys) - min(ys))
            short_side = min(max(xs) - min(xs), max(ys) - min(ys))
            detections.append(
                {
                    "polygon": c["poly"],
                    "text": c["new_text"],
                    "confidence": c["new_conf"],
                    "angle": c["angle"],
                    "long_side": long_side,
                    "short_side": short_side,
                }
            )
        doc = {
            "width": georef["width"],
            "height": georef["height"],
            "streets": detections,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".streets.json") as tmp:
            json.dump(doc, tmp)
            tmp.flush()
            features = prepare_label_features(
                tmp.name,
                block_index,
                (georef["width"], georef["height"]),
                min_confidence=CONFIDENCE_FLOOR,
                min_long_side=32.0,
                min_short_side=16.0,
                min_aspect_ratio=1.0,
            )
        if not features:
            continue
        inliers, _ = label_inliers(features, block_index, page_world_affine(georef))
        # Attribute verification back to candidates by feature center.
        centers = {
            (round(features[i].center[0]), round(features[i].center[1]))
            for i in inliers
        }
        for c in group:
            xs = [p[0] for p in c["poly"]]
            ys = [p[1] for p in c["poly"]]
            center = (round(sum(xs) / 4), round(sum(ys) / 4))
            if center in centers:
                c["verified"] = True
                verified += 1
    return verified


def render_e4_html(candidates: list[dict], crops: dict[int, np.ndarray], out: Path):
    """100-crop review sheet of the fine-tuned arm's new reads."""
    from mapsnap.train_street_recognizer import crop_to_data_uri

    rng = np.random.default_rng(11)
    sample = list(candidates)
    rng.shuffle(sample)  # type: ignore[arg-type]
    rows = [
        "<h1>E4: new reads from the fine-tuned recognizer</h1>",
        (
            "<p>Boxes where production emitted no street (or junk) and the "
            "fine-tuned arm reads something at &ge;0.15. 'verified' = the "
            "read lands on its street through the page's existing fit.</p>"
        ),
    ]
    for c in sample[:100]:
        crop = crops[c["crop_id"]]
        rows.append(
            "<div class='item'>"
            f"<img src='{crop_to_data_uri(crop)}'>"
            f"<span>{c['vol']} {c['stem']} — was: {c['old_text']!r} "
            f"({c['old_conf']:.2f}, {c['cls']})<br>"
            f"<b>now: {c['new_text']} ({c['new_conf']:.2f})</b> "
            f"{'&#10003; verified' if c.get('verified') else ''}</span></div>"
        )
    out.write_text(
        "<!doctype html><meta charset='utf-8'><style>"
        "body{font-family:system-ui;background:#222;color:#eee}"
        ".item{display:inline-block;margin:8px;padding:6px;background:#333;"
        "border-radius:4px;vertical-align:top;max-width:340px}"
        ".item img{display:block;margin-bottom:4px;image-rendering:pixelated}"
        ".item span{font-size:12px;color:#ccc}</style>" + "\n".join(rows)
    )
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--harvest",
        type=Path,
        default=Path.home() / "Documents/ohm/ocr-train-harvest-2026-08-08.jsonl",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Fine-tuned state_dict; omit for a stock-only smoke run.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap each eval set (smoke testing)."
    )
    parser.add_argument("--e4-html", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write stats JSON.")
    args = parser.parse_args()

    import easyocr
    import torch

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    char_list = reader.converter.character
    arms = {"stock": reader.recognizer}
    if args.weights:
        import copy

        finetuned = copy.deepcopy(reader.recognizer)
        finetuned.load_state_dict(torch.load(args.weights, map_location="cpu"))
        finetuned.eval()
        arms["finetuned"] = finetuned
    reader.recognizer.eval()

    pairs, junk = load_harvest_pairs(args.harvest, include_volumes=HOLDOUT_VOLUMES)
    print(f"holdout: {len(pairs)} agreement pairs, {len(junk)} failed-read records")
    tries = {volume: volume_trie(volume) for volume in HOLDOUT_VOLUMES}

    image_cache: dict[str, object] = {}
    flags = underline_flags(pairs)
    clean_items, artifact_items = [], []
    for record in pairs:
        crop = extract_crop(DATA, record, image_cache)  # type: ignore[arg-type]
        (artifact_items if flags[id(record)] else clean_items).append((record, crop))
    if args.limit:
        clean_items = clean_items[: args.limit]
        artifact_items = artifact_items[: args.limit]
    fragments = []
    for record in junk[:200]:
        xs = [p[0] for p in record["poly"]]
        ys = [p[1] for p in record["poly"]]
        if max(xs) - min(xs) <= 80 and max(ys) - min(ys) <= 40:
            fragments.append(extract_crop(DATA, record, image_cache))  # type: ignore[arg-type]
    corrupted_items = [
        (record, augment_crop(crop, np.random.default_rng(5000 + i), fragments))
        for i, (record, crop) in enumerate(clean_items)
    ]

    stats = {
        "E1_clean": evaluate_set("E1 clean", clean_items, arms, char_list, tries),
        "E2_real_artifact": evaluate_set(
            "E2 artifact", artifact_items, arms, char_list, tries
        ),
        "E3_synthetic": evaluate_set(
            "E3 synthetic", corrupted_items, arms, char_list, tries
        ),
    }

    # E1 harness sanity: the stock arm should reproduce the production reads.
    reproduced = 0
    for record, crop in clean_items:
        text, conf = constrained_read(
            arms["stock"], char_list, tries[record["vol"]][0], crop
        )
        reproduced += text == record["text"]
    stats["harness_sanity"] = {
        "stock_reproduces_production": round(reproduced / max(1, len(clean_items)), 4)
    }
    print(
        f"harness sanity: stock reproduces production on "
        f"{stats['harness_sanity']['stock_reproduces_production']:.1%} of E1"
    )

    # E4: the upside pool, fine-tuned arm only.
    if "finetuned" in arms:
        failed = [r for r in junk if r["vol"] in HOLDOUT_VOLUMES]
        if args.limit:
            failed = failed[: args.limit * 4]
        candidates = []
        crops_by_id: dict[int, np.ndarray] = {}
        for i, record in enumerate(failed):
            crop = extract_crop(DATA, record, image_cache)  # type: ignore[arg-type]
            text, conf = constrained_read(
                arms["finetuned"], char_list, tries[record["vol"]][0], crop
            )
            if conf >= CONFIDENCE_FLOOR and text and text != record["text"]:
                candidate = {
                    "vol": record["vol"],
                    "stem": record["stem"],
                    "poly": record["poly"],
                    "angle": record["angle"],
                    "cls": record["cls"],
                    "old_text": record["text"],
                    "old_conf": record["conf"],
                    "new_text": text,
                    "new_conf": round(conf, 3),
                    "crop_id": i,
                }
                crops_by_id[i] = crop
                candidates.append(candidate)
        verified = 0
        for volume in HOLDOUT_VOLUMES:
            volume_candidates = [c for c in candidates if c["vol"] == volume]
            verified += fit_anchored_verify(volume_candidates, tries[volume][1], volume)
        stats["E4_upside"] = {
            "failed_pool": len(failed),
            "new_reads": len(candidates),
            "fit_verified": verified,
            "precision_lower_bound": round(verified / max(1, len(candidates)), 4),
        }
        print(
            f"E4: {len(candidates)} new reads from {len(failed)} failed boxes; "
            f"{verified} fit-verified "
            f"({stats['E4_upside']['precision_lower_bound']:.1%})"
        )
        if args.e4_html:
            render_e4_html(candidates, crops_by_id, args.e4_html)

    if args.out:
        args.out.write_text(json.dumps(stats, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
