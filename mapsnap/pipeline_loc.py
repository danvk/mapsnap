"""Full pipeline for georeferencing a Library of Congress Sanborn volume."""

import argparse
import glob
import sys
from pathlib import Path

from mapsnap.utils import Step, list_pages, run_cmd, write_run_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full LOC pipeline: download images and OSM streets, "
            "run OCR, georeference, and build IIIF. "
            "Expects DIR to exist and contain a manifest.json file."
        )
    )
    parser.add_argument("dir", metavar="DIR", help="Directory containing manifest.json")
    parser.add_argument(
        "relation", metavar="RELATION", help="OSM relation ID for the street network"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run every step even if it already completed. By default a re-run resumes, "
            "skipping steps whose <dir>/.pipeline/<step>.done stamp is present."
        ),
    )
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        sys.exit(f"Directory not found: {dir_path}")
    manifest = dir_path / "manifest.json"
    if not manifest.exists():
        sys.exit(f"manifest.json not found in {dir_path}")

    print(args.dir)
    print(args.relation)
    write_run_record(
        dir_path, "loc", {"manifest": str(manifest), "relation": args.relation}
    )

    step = Step(dir_path, force=args.force)

    with step("download-images"):
        run_cmd(["mapsnap", "download-loc", "--scale", "pct:25", str(manifest)])

    # Detect and write split panels (pN__i.jpg + pN.panels.json) for pages that split.
    page_images = sorted(glob.glob(str(dir_path / "p*.jpg")))
    with step("split"):
        run_cmd(["mapsnap", "split", *page_images])

    with step("download-osm"):
        run_cmd(
            [
                "mapsnap",
                "download-osm",
                args.relation,
                "--output",
                str(dir_path / "streets.osm.json"),
            ]
        )

    with step("osm-to-geojson"):
        run_cmd(
            [
                "mapsnap",
                "osm-to-geojson",
                str(dir_path / "streets.osm.json"),
                "--output",
                str(dir_path / "centerlines.geojson"),
            ]
        )

    # Identify the key map(s) from the 25%-scale pages, download just those at full resolution,
    # and build their sidecars. The subsequent ocr/fit steps then auto-discover raw/*.keymap.json
    # and restrict each page to its key-map neighborhood.
    with step("keymap"):
        from mapsnap.keymap.identify import identify_keymaps

        keymap_keys = identify_keymaps(dir_path)
        if keymap_keys:
            print(f"Key map page(s): {', '.join(keymap_keys)}", flush=True)
            scaled_keymaps = [str(dir_path / f"{key}.jpg") for key in keymap_keys]
            run_cmd(["mapsnap", "download-raw", *scaled_keymaps])
            raw_keymaps = [str(dir_path / "raw" / f"{key}.jpg") for key in keymap_keys]
            run_cmd(["mapsnap", "keymap", *raw_keymaps])
        else:
            print("No key map identified; continuing without one.", flush=True)

    # --resume so an OCR interrupted partway resumes per page on the re-run that follows.
    ocr_images = [str(p) for p in list_pages(dir_path)]
    with step("ocr"):
        run_cmd(
            [
                "mapsnap",
                "ocr",
                "--resume",
                "--centerlines",
                str(dir_path / "centerlines.geojson"),
                *ocr_images,
            ]
        )

    run_cmd(["mapsnap", "fit", str(dir_path), "--tag", "mapsnap"])


if __name__ == "__main__":
    main()
