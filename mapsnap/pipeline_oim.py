"""Full pipeline for georeferencing an OIM (OldInsuranceMaps.net) Sanborn volume."""

import argparse
import glob
import urllib.request
from pathlib import Path

from mapsnap.utils import (
    image_stem,
    list_pages,
    mark_step_done,
    run_cmd,
    run_step,
    step_done,
    write_run_record,
)


def download_file(url: str, dest: Path) -> None:
    """Download url to dest, printing the equivalent curl command."""
    print(f"+ curl -o {dest} {url!r}", flush=True)
    urllib.request.urlretrieve(url, dest)


def delete_other_raw(dir_path: Path, keymap_keys: list[str]) -> None:
    """Delete every full-resolution ``raw/`` page except the identified key map(s).

    OIM downloads every page at full resolution, but only the key maps are needed at full res
    downstream (to build their sidecars); the rest duplicate the 25% ``pN.jpg`` and waste disk.
    Removes ``raw/*.jpg`` whose stem is not one of ``keymap_keys``; ``--keep_raw`` skips this.
    """
    keep = set(keymap_keys)
    removed = 0
    for image in sorted((dir_path / "raw").glob("*.jpg")):
        if image_stem(str(image)) not in keep:
            image.unlink()
            removed += 1
    print(
        f"Deleted {removed} non-key-map raw image(s); kept {', '.join(sorted(keep))}.",
        flush=True,
    )


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
    parser.add_argument(
        "--keep_raw",
        action="store_true",
        help=(
            "Keep every full-resolution raw/ image. By default only the identified key map(s) "
            "are kept and the other raw pages are deleted (they duplicate the 25%% pN.jpg)."
        ),
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

    print(args.sanborn_slug)
    print(args.dir)
    print(args.relation)
    print(args.oim_prefix)

    dir_path = Path(args.dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    write_run_record(
        dir_path,
        "oim",
        {
            "sanborn_slug": args.sanborn_slug,
            "relation": args.relation,
            "oim_prefix": args.oim_prefix,
        },
    )

    base_url = f"https://oldinsurancemaps.net/iiif/mosaic/{args.sanborn_slug}"
    if args.force or not step_done(dir_path, "manifests"):
        download_file(
            f"{base_url}/main-content/?trim=true", dir_path / "main.iiif.json"
        )
        download_file(f"{base_url}/key-map/?trim=true", dir_path / "key.iiif.json")
        mark_step_done(dir_path, "manifests")
    else:
        print("+ [skip manifests: already completed]", flush=True)

    # Download the full-resolution pages. The key map lives only in key.iiif.json (never in
    # main.iiif.json), so download both into raw/ and let it be treated as just another page —
    # the key-map detector then finds it by content, without the pipeline knowing its origin.
    if args.force or not step_done(dir_path, "download-images"):
        for iiif_name in ("main.iiif.json", "key.iiif.json"):
            run_cmd(
                [
                    "mapsnap",
                    "download-oim",
                    str(dir_path / iiif_name),
                    "--oim-url-prefix",
                    args.oim_prefix,
                ]
            )
        mark_step_done(dir_path, "download-images")
    else:
        print("+ [skip download-images: already completed]", flush=True)

    # Downscale the full-resolution raw/ pages to 25% top-level pN.jpg images.
    raw_images = sorted(glob.glob(str(dir_path / "raw" / "*.jpg")))
    run_step(
        dir_path,
        "scale",
        ["mapsnap", "scale", *raw_images, "--output-dir", str(dir_path)],
        force=args.force,
    )

    # Detect and write split panels (pN__i.jpg + pN.panels.json) for pages that split.
    page_images = sorted(glob.glob(str(dir_path / "p*.jpg")))
    run_step(dir_path, "split", ["mapsnap", "split", *page_images], force=args.force)

    run_step(
        dir_path,
        "download-osm",
        [
            "mapsnap",
            "download-osm",
            args.relation,
            "--output",
            str(dir_path / "streets.osm.json"),
        ],
        force=args.force,
    )

    run_step(
        dir_path,
        "osm-to-geojson",
        [
            "mapsnap",
            "osm-to-geojson",
            str(dir_path / "streets.osm.json"),
            "--output",
            str(dir_path / "centerlines.geojson"),
        ],
        force=args.force,
    )

    # Identify the key map(s) from the 25%-scale pages, keep only those at full resolution (the
    # other raw pages duplicate the 25% pN.jpg — delete them unless --keep_raw), and build their
    # sidecars. The subsequent ocr/fit steps then auto-discover raw/*.keymap.json and restrict
    # each page to its key-map neighborhood.
    if args.force or not step_done(dir_path, "keymap"):
        from mapsnap.keymap.identify import identify_keymaps

        keymap_keys = identify_keymaps(dir_path)
        if keymap_keys:
            print(f"Key map page(s): {', '.join(keymap_keys)}", flush=True)
            if not args.keep_raw:
                delete_other_raw(dir_path, keymap_keys)
            raw_keymaps = [str(dir_path / "raw" / f"{key}.jpg") for key in keymap_keys]
            run_cmd(["mapsnap", "keymap", *raw_keymaps])
        else:
            print("No key map identified; continuing without one.", flush=True)
        mark_step_done(dir_path, "keymap")
    else:
        print("+ [skip keymap: already completed]", flush=True)

    # --resume so an OCR interrupted partway resumes per page on the re-run that follows.
    ocr_images = [str(p) for p in list_pages(dir_path)]
    run_step(
        dir_path,
        "ocr",
        [
            "mapsnap",
            "ocr",
            "--resume",
            "--centerlines",
            str(dir_path / "centerlines.geojson"),
            *ocr_images,
        ],
        force=args.force,
    )

    # Locate OIM's manual split regions on the canvas (ground truth for compare).
    run_step(
        dir_path,
        "oim-split-truth",
        ["mapsnap", "oim-split-truth", str(dir_path / "main.iiif.json")],
        force=args.force,
    )

    run_cmd(["mapsnap", "fit", str(dir_path), "--tag", "mapsnap"])


if __name__ == "__main__":
    main()
