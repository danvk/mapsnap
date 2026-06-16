"""Full pipeline for georeferencing an OIM (OldInsuranceMaps.net) Sanborn volume."""

import argparse
import glob
import urllib.request
from pathlib import Path

from mapsnap.utils import list_pages, run_cmd, write_run_record


def download_file(url: str, dest: Path) -> None:
    """Download url to dest, printing the equivalent curl command."""
    print(f"+ curl -o {dest} {url!r}", flush=True)
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full OIM pipeline: download images and OSM streets, "
            "run OCR, georeference, build IIIF, and compare against OIM."
        )
    )
    parser.add_argument(
        "sanborn_slug",
        metavar="SLUG",
        help="OIM Sanborn volume slug, e.g. sanborn05791_053",
    )
    parser.add_argument("dir", metavar="DIR", help="Output directory")
    parser.add_argument(
        "relation", metavar="RELATION", help="OSM relation ID for the street network"
    )
    parser.add_argument(
        "oim_prefix", metavar="OIM_PREFIX", help="OIM URL prefix for image downloads"
    )
    args = parser.parse_args()

    print(args.sanborn_slug)
    print(args.dir)
    print(args.relation)
    print(args.oim_prefix)

    dir_path = Path(args.dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    base_url = f"https://oldinsurancemaps.net/iiif/mosaic/{args.sanborn_slug}"
    download_file(f"{base_url}/main-content/?trim=true", dir_path / "main.iiif.json")
    download_file(f"{base_url}/key-map/?trim=true", dir_path / "key.iiif.json")

    run_cmd(
        [
            "mapsnap",
            "download-oim",
            str(dir_path / "main.iiif.json"),
            "--oim-url-prefix",
            args.oim_prefix,
        ]
    )

    # Downscale the full-resolution raw/ pages to 25% top-level pN.jpg images.
    raw_images = sorted(glob.glob(str(dir_path / "raw" / "*.jpg")))
    run_cmd(["mapsnap", "scale", *raw_images, "--output-dir", str(dir_path)])

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

    # Locate OIM's manual split regions on the canvas (ground truth for compare).
    run_cmd(["mapsnap", "oim-split-truth", str(dir_path / "main.iiif.json")])

    run_cmd(["mapsnap", "fit", str(dir_path), "mapsnap"])

    write_run_record(
        dir_path,
        "oim",
        {
            "sanborn_slug": args.sanborn_slug,
            "relation": args.relation,
            "oim_prefix": args.oim_prefix,
        },
    )


if __name__ == "__main__":
    main()
