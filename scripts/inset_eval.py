"""Score the inset detector against the hand-labeled key-map truth points.

For every truth-labeled key-map sheet (raw/truth/<stem>.labels.json), run
detect_insets on the sheet the pipeline actually uses (the key-map panel when
the sheet was split) and count the reads its confirmed masks cover, split
into TRUE reads (a truth point inside the read's polygon or within one long
side of its center) and junk. The acceptance bar for any change to the
detector is zero true reads masked.

    uv run python scripts/inset_eval.py [--reread] [--no-gpu]
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

from shapely.geometry import Point, Polygon

from mapsnap.keymap.inset import detect_insets


def sheet_for(volume: str, truth_stem: str) -> str | None:
    """The key-map image the pipeline uses for a truth stem (its panel when split)."""
    keys_path = f"data/{volume}/keymaps.json"
    keys = (
        json.loads(Path(keys_path).read_text())["keys"]
        if os.path.exists(keys_path)
        else []
    )
    stem = next((k for k in keys if k.split("__")[0] == truth_stem), truth_stem)
    image = f"data/{volume}/raw/{stem}.jpg"
    return image if os.path.exists(f"data/{volume}/raw/{stem}.keymap.json") else None


def truth_points(volume: str, truth_stem: str, image: str) -> list[tuple[float, float]]:
    """Truth points in the image's frame (shifted into a panel's crop when split)."""
    doc = json.loads(
        Path(f"data/{volume}/raw/truth/{truth_stem}.labels.json").read_text()
    )
    stem = Path(image).name.split(".")[0]
    ox = oy = 0.0
    if "__" in stem:
        panels = json.loads(Path(f"data/{volume}/{truth_stem}.panels.json").read_text())
        ring = panels["panels"][int(stem.split("__")[1]) - 1]
        scale = doc["width"] / panels["width"]
        ox, oy = min(p[0] for p in ring) * scale, min(p[1] for p in ring) * scale
    return [(t["x"] - ox, t["y"] - oy) for t in doc["labels"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reread", action="store_true", help="Run the CRNN re-read corroboration."
    )
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    crnn = device = None
    if args.reread:
        import torch

        from mapsnap.keymap.crnn_model import build_crnn
        from mapsnap.keymap.number_model import select_device

        device = torch.device("cpu") if args.no_gpu else select_device()
        crnn = build_crnn()
        crnn.load_state_dict(torch.load("models/number_crnn.pt", map_location=device))
        crnn.to(device)

    total_junk = total_true = 0
    print(f"{'sheet':40s} {'masks':>5s} {'junk':>5s} {'TRUE':>5s}  masked reads")
    for truth_path in sorted(glob.glob("data/*/raw/truth/*.labels.json")):
        volume = truth_path.split("/")[1]
        truth_stem = os.path.basename(truth_path).split(".")[0]
        image = sheet_for(volume, truth_stem)
        if image is None:
            continue
        result = detect_insets(image, reread=args.reread, crnn=crnn, device=device)
        reads, _, _ = __import__(
            "mapsnap.keymap.inset", fromlist=["load_reads"]
        ).load_reads(image)
        truth = truth_points(volume, truth_stem, image)
        rings = [Polygon(inset.ring).buffer(0) for inset in result.insets]
        masked, junk, true = [], 0, 0
        for read in reads:
            if not any(ring.contains(Point(read.center)) for ring in rings):
                continue
            poly = Polygon(read.polygon).buffer(0)
            long_side = max(
                math.hypot(
                    read.polygon[1][0] - read.polygon[0][0],
                    read.polygon[1][1] - read.polygon[0][1],
                ),
                math.hypot(
                    read.polygon[2][0] - read.polygon[1][0],
                    read.polygon[2][1] - read.polygon[1][1],
                ),
            )
            is_true = any(
                poly.contains(Point(x, y))
                or math.hypot(x - read.center[0], y - read.center[1]) <= long_side
                for x, y in truth
            )
            true += is_true
            junk += not is_true
            masked.append(read.text + ("*" if is_true else ""))
        total_junk += junk
        total_true += true
        flag = "  <-- TRUE READ MASKED" if true else ""
        print(
            f"{volume + '/' + Path(image).name:40s} {len(rings):5d} {junk:5d} {true:5d}  {' '.join(masked)}{flag}"
        )
    print(
        f"\ntotals: junk masked {total_junk}, TRUE masked {total_true}  (bar: TRUE == 0)"
    )


if __name__ == "__main__":
    main()
