"""Georeference images, build IIIF annotation page, and compare against a reference."""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

from mapsnap import experiments
from mapsnap.utils import default_centerlines, list_pages, run_cmd


def find_centerlines(dir_path: Path) -> Path:
    """Return the centerlines GeoJSON, checking dir then parent dir."""
    centerlines = default_centerlines(dir_path)
    if centerlines is None:
        sys.exit(f"centerlines.geojson not found in {dir_path} or {dir_path.parent}")
    return centerlines


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


def resolve_run_id(
    dir_path: Path,
    tag: str | None,
    flag_tokens: list[str],
    inputs: dict,
    git: dict,
) -> str:
    """Return the run id for this fit: the explicit ``tag`` if given, else the computed id.

    An explicit tag is an ad-hoc named run and is used verbatim. With no tag, the id is
    ``<git-sha8>-<config-hash8>``, which requires a git repository with a clean working tree
    (uncommitted changes to tracked files would make the git-sha provenance a lie); the
    function exits with a message if that requirement isn't met.
    """
    if tag is not None:
        return tag
    if git["sha"] is None:
        sys.exit(f"{dir_path} is not in a git repository; pass --tag to name the run.")
    if not git["clean"]:
        sys.exit(
            "Working tree has uncommitted changes to tracked files. Commit them (even a "
            "throwaway commit) so the run id pins a real revision, or pass --tag."
        )
    return experiments.auto_run_id(
        git["sha"], experiments.compute_config_hash(flag_tokens, inputs)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Georeference images, build IIIF annotation page, and compare against reference."
    )
    parser.add_argument(
        "dir", metavar="DIR", help="Directory containing images and data files"
    )
    parser.add_argument(
        "--tag",
        metavar="TAG",
        default=None,
        help=(
            "Optional tag for output files (e.g. 'init' or YYYY-MM-DD). If omitted, a run id "
            "<git-sha8>-<config-hash8> is computed and the working tree must be clean. (A flag "
            "rather than a positional so passthrough georef flags like --num-workers 2 aren't "
            "mis-parsed.)"
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        metavar="NAME",
        help="Human-readable name recorded alongside the run id in the manifest.",
    )
    args, georef_extra = parser.parse_known_args()

    dir_path = Path(args.dir)
    centerlines = find_centerlines(dir_path)
    images = find_input_images(dir_path)
    ref_iiif = find_ref_iiif(dir_path)
    if ref_iiif is None:
        sys.exit(f"No reference IIIF found in {dir_path}")
    truth = dir_path / "main.iiif.json"

    git = experiments.git_head_info(dir_path)
    inputs = experiments.gather_inputs(
        dir_path, centerlines, truth if truth.exists() else None
    )
    run_id = resolve_run_id(dir_path, args.tag, georef_extra, inputs, git)

    archive_dir = dir_path / experiments.ARTIFACTS_DIRNAME / run_id
    if archive_dir.exists():
        print(f"Run {run_id} already archived at {archive_dir}; skipping computation.")
        return

    run_cmd(
        ["mapsnap", "georef", *images, "--centerlines", str(centerlines), *georef_extra]
    )

    output_iiif = dir_path / f"{run_id}.iiif.json"
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
            str(output_iiif),
        ]
    )

    # Compare against OIM, if truth data is available.
    compare_txt: Path | None = None
    if truth.exists():
        cmd = ["mapsnap", "compare", str(truth), str(output_iiif)]
        print("+ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        sys.stdout.write(result.stdout)
        compare_txt = dir_path / f"{run_id}.txt"
        compare_txt.write_text(result.stdout)
        if result.returncode != 0:
            sys.exit(result.returncode)
    else:
        print(f"\nNo main.iiif.json in {dir_path}, skipping comparison step.\n")

    command = [*sys.argv[0].split(), *sys.argv[1:]]
    archived = experiments.archive_fit_run(
        dir_path,
        run_id,
        georef_extra,
        inputs,
        git,
        command,
        truth if truth.exists() else None,
        output_iiif,
        compare_txt,
        args.label,
    )
    manifest = json.loads((archived / "manifest.json").read_text())
    score = manifest.get("metrics", {}).get("score")
    if score:
        print(
            f"\nScore: {score['net']:.1%} "
            f"(<=25ft {score['good_share']:.1%}, >=200ft {score['disaster_share']:.1%}, "
            f"{score['n_placed']}/{score['n_pages']} pages placed)"
        )
    print(f"\nArchived run {run_id} to {archived}")


if __name__ == "__main__":
    main()
