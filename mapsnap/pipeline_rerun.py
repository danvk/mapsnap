"""Re-run the full pipeline on already-downloaded volume(s), reusing cached CRAFT boxes.

For regenerating a release's results tables: every step recomputes its outputs, but nothing
is re-downloaded (images, OSM data, truth JSON are reused as-is) and CRAFT detection — the
most expensive stage — is skipped wherever a ``<stem>.boxes.json`` from a previous run
exists. Freshly split panels and never-OCR'd key maps have no boxes file, so those run full
detection.

Per volume:

  1. verify the volume was set up by a pipeline run (``mapsnap.json`` exists) and has
     ``centerlines.geojson`` — this pipeline never downloads;
  2. ``mapsnap split`` regenerates the split panels (pN__i.jpg + pN.panels.json);
  3. ``mapsnap keymap-detect`` identifies the key map(s), recording them in ``keymaps.json``;
  4. ``mapsnap craft --resume`` refreshes the CRAFT boxes every later step recognizes inside;
  5. ``mapsnap adjacency`` rebuilds the printed-neighbor adjacency graph — before the key-map
     build, so its mutual edges can repair key-map page-number assignments (#213);
  6. ``mapsnap keymap`` rebuilds the key-map sidecars from the full-resolution ``raw/`` scans
     (a detected key map with no raw scan is skipped with a warning — downloading is out of
     scope here), including that assignment repair;
  7. ``mapsnap ocr`` re-recognizes every page (panels supersede their parent),
     auto-discovering the georeferenced key maps; ``--recognizer-weights``
     swaps in fine-tuned weights and forces a full re-read;
  8. ``mapsnap fit --tag <tag>`` georeferences, runs the geometry-first snap
     channel (rescue/arbitrate/refine), builds the IIIF AnnotationPage, and
     compares against truth.

Steps are resumable per volume via ``.pipeline/rerun-<tag>-<step>.done`` stamps (the tag
prefix keeps them distinct from the original pipeline's stamps); ``--force`` redoes them.
A volume that fails is reported and the remaining volumes still run.

    uv run mapsnap rerun data/champaign_ill_1915 --tag 2026-07-17-v1.2
    uv run mapsnap rerun data/*/ --tag 2026-07-17-v1.2   # every dir with a mapsnap.json
"""

import argparse
import sys
from pathlib import Path

from mapsnap.craft import volume_craft_images
from mapsnap.keymap.records import recorded_keymap_keys
from mapsnap.utils import Step, list_pages, run_cmd


def rerun_volume(
    volume: Path, tag: str, force: bool, recognizer_weights: str | None = None
) -> None:
    """Run the six re-run steps for one volume (raises SystemExit on step failure)."""
    if not (volume / "mapsnap.json").exists():
        sys.exit(
            f"{volume} has no mapsnap.json — it was never set up by a pipeline run; "
            "this command re-runs existing volumes and never downloads."
        )
    centerlines = volume / "centerlines.geojson"
    if not centerlines.exists():
        sys.exit(f"{volume} has no centerlines.geojson (re-run does not download OSM).")

    step = Step(volume, force=force)

    with step(f"rerun-{tag}-split"):
        pages = [str(p) for p in sorted(volume.glob("p*.jpg")) if "__" not in p.stem]
        run_cmd(["mapsnap", "split", *pages])

    with step(f"rerun-{tag}-keymap-detect"):
        from mapsnap.keymap.identify import identify_keymaps
        from mapsnap.keymap.records import write_keymaps_record

        write_keymaps_record(volume, identify_keymaps(volume))

    keymap_keys = sorted(recorded_keymap_keys(volume))
    raw_keymaps = []
    for key in keymap_keys:
        raw = volume / "raw" / f"{key}.jpg"
        if raw.exists():
            raw_keymaps.append(str(raw))
        else:
            print(
                f"WARNING: key map {key} has no full-resolution scan at {raw}; "
                "skipping its sidecars (fetch it and re-run with --force to use it).",
                file=sys.stderr,
            )

    # CRAFT once for adjacency, the key-map passes and ocr (#132). --resume
    # keeps existing boxes, which is the whole point of a re-run.
    with step(f"rerun-{tag}-craft"):
        run_cmd(
            ["mapsnap", "craft", "--resume", *volume_craft_images(volume, keymap_keys)]
        )

    # Adjacency before the key-map build so its mutual edges can repair
    # key-map page-number assignments (#213).
    with step(f"rerun-{tag}-adjacency"):
        run_cmd(["mapsnap", "adjacency", str(volume)])

    with step(f"rerun-{tag}-keymap"):
        if raw_keymaps:
            run_cmd(["mapsnap", "keymap", *raw_keymaps])
        else:
            print(f"No usable key map for {volume.name}.", flush=True)

    with step(f"rerun-{tag}-ocr"):
        images = [str(p) for p in list_pages(volume)]
        # Changing the recognizer changes every read, so --resume (which skips
        # any page that already has a .streets.json) must not be passed: the
        # cached reads came from different weights.
        resume = [] if recognizer_weights else ["--resume"]
        weights = (
            ["--recognizer-weights", recognizer_weights] if recognizer_weights else []
        )
        run_cmd(
            [
                "mapsnap",
                "ocr",
                *resume,
                *weights,
                "--centerlines",
                str(centerlines),
                *images,
            ]
        )

    # fit resumes itself (an already-archived run id is skipped), so no stamp.
    run_cmd(["mapsnap", "fit", str(volume), "--tag", tag])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run split/keymap/ocr/adjacency/fit on downloaded volumes, reusing CRAFT boxes."
    )
    parser.add_argument(
        "volumes", nargs="+", type=Path, metavar="DIR", help="Volume directories."
    )
    parser.add_argument(
        "--tag",
        required=True,
        metavar="TAG",
        help="Run tag passed to `mapsnap fit` (e.g. 2026-07-17-v1.2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo steps whose rerun stamps say they already completed.",
    )
    parser.add_argument(
        "--recognizer-weights",
        default=None,
        metavar="PT",
        help=(
            "Fine-tuned recognizer weights to pass to `mapsnap ocr` (#265). "
            "Implies a full re-OCR: cached reads came from other weights."
        ),
    )
    args = parser.parse_args()

    failures: list[str] = []
    for volume in args.volumes:
        print(f"\n=== {volume} ===", flush=True)
        try:
            rerun_volume(volume, args.tag, args.force, args.recognizer_weights)
        except SystemExit as exit_info:
            print(
                f"FAILED {volume}: {exit_info.code}",
                file=sys.stderr,
                flush=True,
            )
            failures.append(str(volume))
    if failures:
        sys.exit(f"{len(failures)} volume(s) failed: {', '.join(failures)}")
    print(f"\nAll {len(args.volumes)} volume(s) completed.", flush=True)


if __name__ == "__main__":
    main()
