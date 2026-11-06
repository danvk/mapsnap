"""Grade a run's snap candidates against OIM truth, for training snap's accept model (#650).

    grade_snap_candidates.py RUNS_DIR TRUTH_DIR OUT_JSONL

RUNS_DIR/<item>/candidates.jsonl is an item's snap candidates
(``runs/<tag>/artifacts/osm_snap/candidates.jsonl`` from the bucket).
TRUTH_DIR/<item>/ holds its truth: main.iiif.json, oim/, and the run's own
p*.panels.json (score-690's per-item directories have all three).

Writes each candidates record that has truth, with ``rmse_ft`` on every posed
candidate and on the incumbent, plus the record's ``item``. Truth is attached
by snap's own harness (load_page_units, load_panel_units,
attach_missing_truth), so poses are graded exactly as snap grades them when a
volume carries main.iiif.json. That harness reads page images only for their
size, so each target gets a header-only stand-in JPEG of the size its record
gives.
"""

import argparse
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

from mapsnap.page_units import TruthFit, grid_rmse_ft_between, load_page_units
from mapsnap.snap_volume import attach_missing_truth, load_panel_units


def write_stand_in_jpeg(path: Path, width: int, height: int) -> None:
    """A JPEG holding only a start marker and a frame header, enough for jpeg_dimensions."""
    frame = struct.pack(">BHHB", 8, height, width, 1) + bytes([1, 0x11, 0])
    header = b"\xff\xc0" + struct.pack(">H", len(frame) + 2) + frame
    path.write_bytes(b"\xff\xd8" + header + b"\xff\xd9")


def truth_rmse_ft(
    fit: TruthFit, affine: list[list[float]], size: tuple[int, int]
) -> float:
    """A pose's grid RMSE against the page's truth, in feet, to 0.1."""
    width, height = size
    return round(
        grid_rmse_ft_between(fit.affine_local, np.array(affine), width, height), 1
    )


def graded_records(records: list[dict], truth_dir: Path) -> list[dict]:
    """The records with truth, each posed candidate and the incumbent carrying rmse_ft."""
    with tempfile.TemporaryDirectory() as scratch:
        volume = Path(scratch)
        (volume / "main.iiif.json").symlink_to(truth_dir / "main.iiif.json")
        if (truth_dir / "oim").exists():
            (volume / "oim").symlink_to(truth_dir / "oim")
        for panels in truth_dir.glob("p*.panels.json"):
            (volume / panels.name).symlink_to(panels)
        for record in records:
            write_stand_in_jpeg(
                volume / f"{record['target']}.jpg", record["width"], record["height"]
            )
        units = load_page_units(volume) + load_panel_units(volume)
        attach_missing_truth(volume, units)
    truth: dict[str, TruthFit] = {
        unit.stem: unit.truth for unit in units if unit.truth is not None
    }
    graded = []
    for record in records:
        fit = truth.get(record["target"])
        if fit is None:
            continue
        size = (record["width"], record["height"])
        for candidate in record.get("candidates") or []:
            if candidate.get("world_affine"):
                candidate["rmse_ft"] = truth_rmse_ft(
                    fit, candidate["world_affine"], size
                )
        incumbent = record.get("incumbent")
        if incumbent and incumbent.get("world_affine"):
            incumbent["rmse_ft"] = truth_rmse_ft(fit, incumbent["world_affine"], size)
        graded.append(record)
    return graded


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("truth_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    pages = 0
    with args.out.open("w") as out:
        for item_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
            candidates = item_dir / "candidates.jsonl"
            truth_dir = args.truth_dir / item_dir.name
            if not candidates.exists() or not (truth_dir / "main.iiif.json").exists():
                continue
            records = [json.loads(line) for line in candidates.open()]
            records = [r for r in records if r.get("width") and r.get("height")]
            try:
                graded = graded_records(records, truth_dir)
            except (OSError, ValueError, KeyError) as error:
                print(f"{item_dir.name}: {error}", file=sys.stderr)
                continue
            for record in graded:
                out.write(json.dumps({"item": item_dir.name, **record}) + "\n")
            pages += len(graded)
    print(f"{pages} graded pages -> {args.out}")


if __name__ == "__main__":
    main()
