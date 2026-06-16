"""Full pipeline for georeferencing a Library of Congress Sanborn volume."""

import argparse
import glob
import sys
from pathlib import Path

from mapsnap.utils import list_pages, run_cmd, write_run_record


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
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        sys.exit(f"Directory not found: {dir_path}")
    manifest = dir_path / "manifest.json"
    if not manifest.exists():
        sys.exit(f"manifest.json not found in {dir_path}")

    print(args.dir)
    print(args.relation)

    run_cmd(["mapsnap", "download-loc", "--scale", "pct:25", str(manifest)])

    # Detect and write split panels (pN__i.jpg + pN.panels.json) for pages that split.
    page_images = sorted(glob.glob(str(dir_path / "p*.jpg")))
    run_cmd(["mapsnap", "split", *page_images])

    run_cmd(
        [
            "mapsnap",
            "download-osm",
            args.relation,
            "--output",
            str(dir_path / "streets.osm.json"),
        ]
    )

    run_cmd(
        [
            "mapsnap",
            "osm-to-geojson",
            str(dir_path / "streets.osm.json"),
            "--output",
            str(dir_path / "centerlines.geojson"),
        ]
    )

    ocr_images = [str(p) for p in list_pages(dir_path)]
    run_cmd(
        [
            "mapsnap",
            "ocr",
            "--centerlines",
            str(dir_path / "centerlines.geojson"),
            *ocr_images,
        ]
    )

    run_cmd(["mapsnap", "fit", str(dir_path), "mapsnap"])

    write_run_record(
        dir_path, "loc", {"manifest": str(manifest), "relation": args.relation}
    )


if __name__ == "__main__":
    main()
