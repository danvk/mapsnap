"""Georeference images, build IIIF annotation page, and compare against a reference."""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

from mapsnap.utils import list_pages, run_cmd


def find_centerlines(dir_path: Path) -> Path:
    """Return the centerlines GeoJSON, checking dir then parent dir."""
    for candidate in (
        dir_path / "centerlines.geojson",
        dir_path.parent / "centerlines.geojson",
    ):
        if candidate.exists():
            return candidate
    sys.exit(f"centerlines.geojson not found in {dir_path} or {dir_path.parent}")


def find_input_images(dir_path: Path) -> list[str]:
    """Return the effective page images (split panels supersede their parent page)."""
    images = [str(p) for p in list_pages(dir_path)]
    if not images:
        sys.exit(f"No p*.jpg found in {dir_path}")
    return images


def find_ref_iiif(dir_path: Path) -> Path | None:
    """Return the reference IIIF path, trying main, loc, then any manifest."""
    for name in ("main.iiif.json", "loc.iiif.json"):
        path = dir_path / name
        if path.exists():
            return path
    manifests = sorted(glob.glob(str(dir_path / "*manifest.json")))
    if len(manifests) > 1:
        sys.exit(f"Found multiple manifest.json files in {dir_path}")
    return Path(manifests[0]) if manifests else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Georeference images, build IIIF annotation page, and compare against reference."
    )
    parser.add_argument(
        "dir", metavar="DIR", help="Directory containing images and data files"
    )
    parser.add_argument(
        "tag", metavar="TAG", help="Tag for output files (e.g. 'init' or YYYY-MM-DD)"
    )
    args, georef_extra = parser.parse_known_args()

    dir_path = Path(args.dir)
    centerlines = find_centerlines(dir_path)
    images = find_input_images(dir_path)

    run_cmd(
        ["mapsnap", "georef", *images, "--centerlines", str(centerlines), *georef_extra]
    )

    ref_iiif = find_ref_iiif(dir_path)
    if ref_iiif is None:
        sys.exit(f"No reference IIIF found in {dir_path}")

    output_iiif = str(dir_path / f"{args.tag}.iiif.json")
    # Pass the georef glob as a literal string; make_iiif_georef does its own glob expansion.
    georef_glob = str(dir_path / "*.georef.json")
    run_cmd(
        [
            "mapsnap",
            "iiif",
            str(ref_iiif),
            georef_glob,
            "--centerlines",
            str(centerlines),
            "--output",
            output_iiif,
        ]
    )

    # Compare against OIM, if truth data is available.
    main_iiif = dir_path / "main.iiif.json"
    if main_iiif.exists():
        cmd = ["mapsnap", "compare", str(main_iiif), output_iiif]
        print("+ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        sys.stdout.write(result.stdout)
        (dir_path / f"{args.tag}.txt").write_text(result.stdout)
        if result.returncode != 0:
            sys.exit(result.returncode)
    else:
        print(f"\nNo main.iiif.json in {dir_path}, skipping comparison step.\n")


if __name__ == "__main__":
    main()
