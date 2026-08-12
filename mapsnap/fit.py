"""Georeference images, build IIIF annotation page, and compare against a reference."""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

from mapsnap import experiments
from mapsnap.utils import default_centerlines, list_pages, run_cmd


def worker_flag(georef_extra: list[str]) -> list[str]:
    """The ``--num-workers N`` pair out of the georef passthrough, or nothing.

    Accepts either spelling argparse does (``--num-workers 4`` or
    ``--num-workers=4``) and normalizes to the two-token form.
    """
    for i, token in enumerate(georef_extra):
        if token == "--num-workers" and i + 1 < len(georef_extra):
            return [token, georef_extra[i + 1]]
        if token.startswith("--num-workers="):
            return ["--num-workers", token.split("=", 1)[1]]
    return []


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


# Every `p<stem>.georef*.json` at the volume root is produced by one of the
# stages below -- georef writes the plain fit and its failure variants, the
# adjacency gate the contradicted ones, snap the -osm ones, street-solve the
# -streets ones. Nothing else writes them.
DERIVED_SIDECAR_GLOB = "p*.georef*.json"


def clear_derived_sidecars(dir_path: Path) -> int:
    """Remove the georef sidecars this run is about to regenerate; return the count.

    Without this a run is not idempotent. `mapsnap iiif` publishes whatever
    sidecars are on disk, and a stage that declines to write one this time
    leaves the *previous* run's file in place to be published instead. Worse,
    snap's `ransac-neighbor` rotation prior reads neighbouring pages' published
    fits, so a single leftover file perturbs its neighbours' searches and
    cascades outward.

    Measured on Grand Rapids: consecutive runs alternated between two fixed
    points, 69.8% and 71.8%, differing by two stale sidecars and 33 of 60
    published pages. Clearing first makes two consecutive runs byte-identical
    (issue #240). Note this is not a fix for nondeterminism -- the pipeline was
    always deterministic given its inputs; the stale files simply *were* part of
    the input.
    """
    removed = 0
    for path in sorted(dir_path.glob(DERIVED_SIDECAR_GLOB)):
        path.unlink()
        removed += 1
    if removed:
        print(f"Cleared {removed} derived georef sidecar(s) from {dir_path}")
    return removed


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
    parser.add_argument(
        "--no-snap",
        action="store_true",
        help=(
            "Skip the geometry channels — OSM snap (rescue/arbitrate/refine) "
            "and the street-constraint solver, whose referee rides on snap's "
            "machinery — so the arbiter weighs RANSAC's poses alone."
        ),
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
    # The config hash must see the snap setting: identical georef flags with
    # snap on vs off produce different outputs and need different run ids
    # (an id collision would silently SKIP the second variant as already
    # archived).
    id_tokens = [*georef_extra, *(["--no-snap"] if args.no_snap else [])]
    run_id = resolve_run_id(dir_path, args.tag, id_tokens, inputs, git)

    archive_dir = dir_path / experiments.ARTIFACTS_DIRNAME / run_id
    # The manifest is written last, so it -- not the directory -- is what says a
    # previous run finished. archive_run creates the directory before copying
    # into it, so an interrupted run leaves an empty one; treating that as done
    # would skip this run's computation for good and leave the tag permanently
    # empty.
    if experiments.is_complete(archive_dir):
        # Skipping is only honest when the archived run would produce the same
        # thing. An auto run id encodes (commit, flags, inputs) so a collision
        # implies a match, but an explicit --tag is just a name: re-using one
        # after changing code or reads silently republishes the OLD run and
        # reports it as this one. That happened -- an A/B arm was "re-run" under
        # a tag the previous experiment had already archived, and its stale
        # numbers were reported as new until the archives were purged by hand.
        stale = experiments.archive_differs(archive_dir, inputs, git, georef_extra)
        if stale:
            sys.exit(
                f"Run {run_id} is already archived at {archive_dir}, but it was "
                f"produced from different {stale}. Refusing to skip and report "
                f"that run as this one: choose a new --tag, or delete the "
                f"archive to recompute."
            )
        print(f"Run {run_id} already archived at {archive_dir}; skipping computation.")
        return

    clear_derived_sidecars(dir_path)

    run_cmd(
        ["mapsnap", "georef", *images, "--centerlines", str(centerlines), *georef_extra]
    )

    # Demote fits that contradict their own printed mutual-adjacency claims
    # (adjacency edges are ~100% precise, so a contradicted, weakly-supported
    # fit is wrong). The demotion leaves partner-stamp re-search hints that
    # snap's rescue picks up, so it runs before snap. No adjacency.json: no-op.
    run_cmd(["mapsnap", "adjacency-gate", str(dir_path)])

    # The geometry-first snap channel: rescue unplaced pages, arbitrate fits
    # OSM contradicts, refine mid-tier fits. Writes pN.georef-snap.json.
    if not args.no_snap:
        # Both passes are per-page and CPU-bound, so one --num-workers governs
        # both; the rest of the georef passthrough is georef-only.
        run_cmd(["mapsnap", "snap", str(dir_path), *worker_flag(georef_extra)])
        # The street-constraint channel: fit key-map-prior pages from their
        # street labels. Writes pN.georef-street.json. Runs after snap because
        # its referee shares machinery with the snap channel; skipped with
        # --no-snap for the same reason.
        run_cmd(["mapsnap", "street-solve", str(dir_path)])

    # The arbiter (#270) weighs every pose the channels produced -- including
    # the ones they rejected -- against each other and against not publishing
    # at all, jointly across the volume, and writes the answer for EVERY page
    # as pN.georef-final.json (poseless when it declines to place the page).
    run_cmd(["mapsnap", "reconcile", str(dir_path), "--publish"])

    output_iiif = dir_path / f"{run_id}.iiif.json"
    # One glob, one channel. Publication used to be first-glob-wins over three
    # channel sidecars, which made stage ORDER the thing that decided what got
    # published and gave every stage a reason to hide its predecessors' files.
    # The arbiter answers for every page instead, so there is nothing to
    # prioritize between (#270 phase 3).
    georef_glob = str(dir_path / "*.georef-final.json")
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=False)
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
