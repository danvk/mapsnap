"""Full pipeline for georeferencing an OIM (OldInsuranceMaps.net) Sanborn volume."""

import argparse
import glob
import urllib.request
from pathlib import Path

from mapsnap.craft import volume_craft_images
from mapsnap.download_osm import BUFFER_M
from mapsnap.keymap.records import recorded_keymap_keys
from mapsnap.utils import (
    Step,
    image_stem,
    list_pages,
    run_cmd,
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
        "--buffer-m",
        type=float,
        default=BUFFER_M,
        metavar="M",
        help=(
            "Download streets this far past the relation boundary "
            "(default: %(default)s). Sheets routinely map ground just outside "
            "the modern administrative line."
        ),
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
            "buffer_m": str(args.buffer_m),
            "oim_prefix": args.oim_prefix,
        },
    )

    step = Step(dir_path, force=args.force)

    base_url = f"https://oldinsurancemaps.net/iiif/mosaic/{args.sanborn_slug}"
    with step("manifests"):
        download_file(
            f"{base_url}/main-content/?trim=true", dir_path / "main.iiif.json"
        )
        download_file(f"{base_url}/key-map/?trim=true", dir_path / "key.iiif.json")

    # Download the full-resolution pages. The key map lives only in key.iiif.json (never in
    # main.iiif.json), so download both into raw/ and let it be treated as just another page —
    # the key-map detector then finds it by content, without the pipeline knowing its origin.
    with step("download-images"):
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

    # Downscale the full-resolution raw/ pages to 25% top-level pN.jpg images.
    raw_images = sorted(glob.glob(str(dir_path / "raw" / "*.jpg")))
    with step("scale"):
        run_cmd(["mapsnap", "scale", *raw_images, "--output-dir", str(dir_path)])

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
                str(dir_path),
                "--buffer-m",
                str(args.buffer_m),
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

    # Identify the key map(s); recorded in keymaps.json so the adjacency scan
    # below can skip them before their sidecars exist.
    with step("keymap-detect"):
        from mapsnap.keymap.identify import identify_keymaps
        from mapsnap.keymap.records import write_keymaps_record

        keymap_keys = identify_keymaps(dir_path)
        write_keymaps_record(dir_path, keymap_keys)
        if keymap_keys:
            print(f"Key map page(s): {', '.join(keymap_keys)}", flush=True)
            # The other raw pages duplicate the 25% pN.jpg — delete them unless
            # --keep_raw. Only the key maps are needed at full resolution.
            if not args.keep_raw:
                delete_other_raw(dir_path, keymap_keys)
        else:
            print("No key map identified; continuing without one.", flush=True)

    # Read the keys back from keymaps.json rather than the step's local: a
    # resumed run skips the step body entirely, so the local would be unbound.
    keymap_keys = sorted(recorded_keymap_keys(dir_path))

    # CRAFT once for everything downstream (#132): adjacency, the key-map
    # passes and ocr all recognize inside these boxes.
    raw_keymaps = [str(dir_path / "raw" / f"{key}.jpg") for key in keymap_keys]
    with step("craft"):
        run_cmd(
            [
                "mapsnap",
                "craft",
                "--resume",
                *volume_craft_images(dir_path, keymap_keys),
            ]
        )

    # Printed adjacent-sheet graph. Runs BEFORE the key-map step so its mutual
    # edges can repair key-map page-number assignments (#213); key-map sheets
    # are skipped via keymaps.json.
    with step("adjacency"):
        run_cmd(["mapsnap", "adjacency", str(dir_path)])

    # Build the key-map sidecars, including the adjacency-informed assignment
    # repair. The subsequent ocr/fit steps then auto-discover raw/*.keymap.json
    # and restrict each page to its key-map neighborhood.
    with step("keymap"):
        if raw_keymaps:
            run_cmd(["mapsnap", "keymap", *raw_keymaps])

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

    # OIM's manual split regions on the canvas (ground truth for compare), read
    # from the boundaries OIM publishes rather than reverse-engineered from
    # crops (#273/#274). The old `oim-split-truth` step template-matched
    # oim/pN__i.jpg files that nothing in this pipeline downloads any more, so
    # it could only warn-and-skip -- a new volume got no split truth at all.
    with step("oim-panels"):
        run_cmd(["mapsnap", "oim-panels", str(dir_path)])

    run_cmd(["mapsnap", "fit", str(dir_path), "--tag", "mapsnap"])


if __name__ == "__main__":
    main()
