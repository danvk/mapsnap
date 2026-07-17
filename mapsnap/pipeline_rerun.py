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
  3. ``mapsnap keymap-detect`` identifies the key map(s); ``mapsnap keymap --reuse-boxes``
     rebuilds their sidecars from the full-resolution ``raw/`` scans (a detected key map
     with no raw scan is skipped with a warning — downloading is out of scope here);
  4. ``mapsnap ocr --reuse-boxes --allow-missing-boxes`` re-recognizes every page (panels
     supersede their parent), auto-discovering the georeferenced key maps;
  5. ``mapsnap adjacency --reuse-boxes`` rebuilds the printed-neighbor adjacency graph;
  6. ``mapsnap fit --tag <tag>`` georeferences, builds the IIIF AnnotationPage, and
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

from mapsnap.utils import Step, list_pages, run_cmd


def rerun_volume(volume: Path, tag: str, force: bool) -> None:
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

    with step(f"rerun-{tag}-keymap"):
        from mapsnap.keymap.identify import identify_keymaps

        keymap_keys = identify_keymaps(volume)
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
        if raw_keymaps:
            run_cmd(["mapsnap", "keymap", "--reuse-boxes", *raw_keymaps])
        else:
            print(f"No usable key map for {volume.name}.", flush=True)

    with step(f"rerun-{tag}-ocr"):
        images = [str(p) for p in list_pages(volume)]
        run_cmd(
            [
                "mapsnap",
                "ocr",
                "--resume",
                "--reuse-boxes",
                "--allow-missing-boxes",
                "--centerlines",
                str(centerlines),
                *images,
            ]
        )

    with step(f"rerun-{tag}-adjacency"):
        run_cmd(["mapsnap", "adjacency", str(volume), "--reuse-boxes"])

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
    args = parser.parse_args()

    failures: list[str] = []
    for volume in args.volumes:
        print(f"\n=== {volume} ===", flush=True)
        try:
            rerun_volume(volume, args.tag, args.force)
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
