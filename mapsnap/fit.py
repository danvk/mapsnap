"""Georeference images, build IIIF annotation page, and compare against a reference."""

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

from mapsnap import experiments
from mapsnap.utils import list_pages, require_centerlines, run_cmd


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
    """The volume's centerlines (GeoJSON or OSM extract), checking dir then parent."""
    return require_centerlines(dir_path)


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
    if manifests:
        return Path(manifests[0])
    # A mirrored corpus volume has no manifest of its own, but its metadata.json
    # carries every field a canvas needs and points at LoC's image servers (#354).
    metadata = dir_path / "metadata.json"
    return metadata if metadata.exists() else None


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
    # The snap candidate/selection caches are derived state too (#342). Inside
    # a fit they can only ever hit for pages whose records carry no
    # georef_mtime (demoted/failure classes) -- every fitted page's record is
    # invalidated by the georef rewrite above -- and those hits are exactly
    # the cache-temperature flips #342 measured (KC p566__1 at 7.4 vs 282 ft
    # on cache state alone, ~2 net points). Measured across all 18 truth
    # volumes, a fully cold snap stage costs 3 minutes per corpus run (4,958s
    # warm vs 5,113s cold), so the cache buys nothing a fit can keep. The
    # files remain useful within a run and for standalone `mapsnap snap`
    # iteration; a fit just never inherits a previous run's records.
    for cache in (
        *sorted((dir_path / "artifacts" / "osm_snap").glob("candidates.jsonl")),
        *sorted((dir_path / "artifacts" / "osm_snap").glob("selection_*.jsonl")),
        *sorted((dir_path / "artifacts" / "osm_snap").glob("other_edition_prior.json")),
        *sorted((dir_path / "artifacts" / "street_solve").glob("candidates.jsonl")),
    ):
        cache.unlink()
        removed += 1
    if removed:
        print(f"Cleared {removed} derived georef sidecar(s) from {dir_path}")
    return removed


def other_edition_token(annotation: str) -> str:
    """How another edition spells itself in a run id: volume and file name.

    Not the bare name — every volume's is `main.iiif.json` — because an id
    collision silently skips the second run as already archived.
    """
    path = Path(annotation)
    return f"{path.parent.name}/{path.name}" if path.parent.name else path.name


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
        "--run-tag",
        default=None,
        metavar="TAG",
        help=(
            "Name for the run this fit belongs to -- a cut release, say -- "
            "recorded in the manifest and in the published annotation page so "
            "an output can be traced back to the corpus pass that made it. "
            "Defaults to the repo's nearest git tag."
        ),
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
    parser.add_argument(
        "--image-base-url",
        help=(
            "Publish canvases from the page images rather than a reference "
            "manifest; each page's URL is {URL}/{parent_key}.jpg. Needed for a "
            "mirrored volume, which has no annotation page to borrow from."
        ),
    )
    parser.add_argument(
        "--image-source-type",
        default="Image",
        help="IIIF source type for --image-base-url (default: %(default)s).",
    )
    parser.add_argument(
        "--other-edition",
        default=None,
        metavar="FILE",
        help=(
            "A IIIF annotation placing another edition of this atlas. Snap "
            "rescues unplaced pages from its same-numbered sheets."
        ),
    )
    args, georef_extra = parser.parse_known_args()
    if args.other_edition is not None:
        # snap is the only stage that reads the prior, and it runs late: check
        # the file up front rather than after the whole georef pass.
        if args.no_snap:
            parser.error("--other-edition needs the snap stage; drop --no-snap")
        if not Path(args.other_edition).is_file():
            parser.error(f"--other-edition {args.other_edition}: not a file")

    dir_path = Path(args.dir)
    centerlines = find_centerlines(dir_path)
    images = find_input_images(dir_path)
    ref_iiif = find_ref_iiif(dir_path)
    if ref_iiif is None and not args.image_base_url:
        sys.exit(f"No reference IIIF found in {dir_path}")
    truth = dir_path / "main.iiif.json"

    # The repo, not the volume: a corpus worker fits a scratch directory that is
    # not inside a git checkout, and asking there recorded "sha": null on every
    # item -- the one field that says which code produced the run.
    git = experiments.git_head_info(Path(__file__).resolve().parent)
    models = experiments.model_hashes()
    # An explicit tag wins; otherwise the checkout names itself, so a fleet
    # launched at a release records that release without being told.
    run_tag = args.run_tag or git.get("describe")
    inputs = experiments.gather_inputs(
        dir_path, centerlines, truth if truth.exists() else None
    )
    # The config hash must see the snap setting: identical georef flags with
    # snap on vs off produce different outputs and need different run ids
    # (an id collision would silently SKIP the second variant as already
    # archived).
    id_tokens = [
        *georef_extra,
        *(["--no-snap"] if args.no_snap else []),
        *(
            ["--other-edition", other_edition_token(args.other_edition)]
            if args.other_edition
            else []
        ),
    ]
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

    # Wall-clock per stage, printed at the end and recorded in the manifest —
    # the input the #320 incremental-fit work needs, and the first thing anyone
    # asks when a fit is slow ("which stage?") that mtime archaeology used to
    # answer badly.
    stage_seconds: dict[str, float] = {}

    def timed(stage: str, cmd: list[str]) -> None:
        started = time.perf_counter()
        run_cmd(cmd)
        stage_seconds[stage] = round(time.perf_counter() - started, 1)
        print(f"[{stage}: {stage_seconds[stage]:.0f}s]", flush=True)

    timed(
        "georef",
        [
            "mapsnap",
            "georef",
            *images,
            "--centerlines",
            str(centerlines),
            *georef_extra,
        ],
    )

    # Demote fits that contradict their own printed mutual-adjacency claims
    # (adjacency edges are ~100% precise, so a contradicted, weakly-supported
    # fit is wrong). The demotion leaves partner-stamp re-search hints that
    # snap's rescue picks up, so it runs before snap. No adjacency.json: no-op.
    timed("adjacency-gate", ["mapsnap", "adjacency-gate", str(dir_path)])

    # The geometry-first snap channel: rescue unplaced pages, arbitrate fits
    # OSM contradicts, refine mid-tier fits. Writes pN.georef-snap.json.
    if not args.no_snap:
        # Both passes are per-page and CPU-bound, so one --num-workers governs
        # both; the rest of the georef passthrough is georef-only.
        timed(
            "snap",
            [
                "mapsnap",
                "snap",
                str(dir_path),
                *(
                    ["--other-edition", args.other_edition]
                    if args.other_edition
                    else []
                ),
                *worker_flag(georef_extra),
            ],
        )
        # The street-constraint channel: fit key-map-prior pages from their
        # street labels. Writes pN.georef-street.json. Runs after snap because
        # its referee shares machinery with the snap channel; skipped with
        # --no-snap for the same reason.
        timed("street-solve", ["mapsnap", "street-solve", str(dir_path)])

    # The arbiter (#270) weighs every pose the channels produced -- including
    # the ones they rejected -- against each other and against not publishing
    # at all, jointly across the volume, and writes the answer for EVERY page
    # as pN.georef-final.json (poseless when it declines to place the page).
    timed("reconcile", ["mapsnap", "reconcile", str(dir_path), "--publish"])

    output_iiif = dir_path / f"{run_id}.iiif.json"
    # One glob, one channel. Publication used to be first-glob-wins over three
    # channel sidecars, which made stage ORDER the thing that decided what got
    # published and gave every stage a reason to hide its predecessors' files.
    # The arbiter answers for every page instead, so there is nothing to
    # prioritize between (#270 phase 3).
    georef_glob = str(dir_path / "*.georef-final.json")
    # Without a reference manifest the canvases are built from the images on
    # disk, which is how a mirrored corpus volume is published: it has its scans
    # and its metadata, but no annotation page to borrow image services from
    # (#354).
    if args.image_base_url:
        source = [
            georef_glob,
            "--image-base-url",
            args.image_base_url,
            "--image-source-type",
            args.image_source_type,
        ]
    else:
        source = [str(ref_iiif), georef_glob]
    timed(
        "iiif",
        [
            "mapsnap",
            "iiif",
            *source,
            "--centerlines",
            str(centerlines),
            "--output",
            str(output_iiif),
            *(["--run-tag", run_tag] if run_tag else []),
        ],
    )

    # The key map is georeferenced too -- by street GCPs, like any other sheet --
    # and until now that pose was written to raw/<stem>.georef.json and never
    # published. It is a map in its own right, and the one sheet that shows how
    # the volume is laid out. No --centerlines: the block-based clip masks are
    # built for street sheets, and a key map's own footprint lives in its
    # regions.panels.json instead, which is uploaded too -- so a mask can be
    # added later by regenerating this file, without re-running the chain.
    keymap_georefs = sorted((dir_path / "raw").glob("*.georef.json"))
    if keymap_georefs:
        keymap_iiif = dir_path / f"{run_id}.keymap.iiif.json"
        timed(
            "keymap-iiif",
            [
                "mapsnap",
                "iiif",
                *(
                    [str(ref_iiif)]
                    if ref_iiif is not None and not args.image_base_url
                    else []
                ),
                str(dir_path / "raw" / "*.georef.json"),
                *(
                    [
                        "--image-base-url",
                        args.image_base_url,
                        "--image-source-type",
                        args.image_source_type,
                    ]
                    if args.image_base_url
                    else []
                ),
                "--label-note",
                "key map",
                "--output",
                str(keymap_iiif),
                *(["--run-tag", run_tag] if run_tag else []),
            ],
        )
        # A reference annotation page carries canvases only for the sheets it
        # georeferenced, and the key map is not one of them -- so a truth volume
        # produces an empty page here while a mirrored volume, whose canvases
        # come from metadata.json listing every page, produces a real one.
        # Publishing the empty file would claim a key map was published.
        if not json.loads(keymap_iiif.read_text())["items"]:
            keymap_iiif.unlink()
            print(
                f"No key-map annotation written: nothing in {ref_iiif.name if ref_iiif else 'the source'} "
                "provides a canvas for the key-map sheet.",
                file=sys.stderr,
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
        git | {"models": models, "run_tag": run_tag},
        command,
        truth if truth.exists() else None,
        output_iiif,
        compare_txt,
        args.label,
        stage_seconds=stage_seconds,
    )
    total = sum(stage_seconds.values())
    print(
        "Stage times: "
        + "  ".join(
            f"{stage} {seconds:.0f}s" for stage, seconds in stage_seconds.items()
        )
        + f"  (total {total:.0f}s)"
    )
    manifest = json.loads((archived / "manifest.json").read_text())
    # The #300 sheet-equal score lives under metrics.truth (truth_metrics
    # computes it); the old top-level lookup never matched, so fit's own Score
    # line silently never printed and compare's legacy land-weighted footer
    # was the only "Score:" in the log -- the metric confusion behind #330.
    score = manifest.get("metrics", {}).get("truth", {}).get("score")
    if score:
        # fair/poor bands are absent from manifests archived before they existed.
        bands = ""
        if "fair_share" in score:
            bands = (
                f"25-50ft {score['fair_share']:.1%}, "
                f"50-200ft {score['poor_share']:.1%}, "
            )
        print(
            f"\nScore: {score['net']:.1%} "
            f"(<=25ft {score['good_share']:.1%}, {bands}"
            f">=200ft {score['disaster_share']:.1%}, "
            f"{score['n_placed']}/{score['n_pages']} pages placed)"
        )
    print(f"\nArchived run {run_id} to {archived}")


if __name__ == "__main__":
    main()
