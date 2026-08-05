"""Archive an already-produced run into ``data/<volume>/artifacts/<tag>/``.

``mapsnap fit`` archives what it computes, but a run is often assembled by hand:
when only the last stage changed, re-running georef and snap to get a comparable
number is minutes of work for no new information, so the shortcut is to run
``mapsnap iiif`` and ``mapsnap compare`` against the sidecars already on disk.
That leaves ``<volume>/<tag>.iiif.json`` and ``<volume>/<tag>.txt`` at the volume
root with no artifact directory, so the run has no manifest, no provenance, and
no per-page sidecars of its own -- and the volume viewer, which links a run's
pages to the files that produced them, falls back to whatever ran most recently.

This command closes that gap: it takes a tag whose IIIF and compare output
already exist and archives them the same way ``fit`` would, with the same
manifest. It computes nothing, so what it records is what is on disk right now.

    mapsnap archive data/detroit_mich_1929_vol_11 --tag 2026-08-05-kmsnap

Because it archives the *current* sidecars, it is only meaningful straight after
the run that produced them; archiving a tag whose sidecars have since been
overwritten records the wrong files, which ``--check`` reports rather than
guesses about.
"""

import argparse
import sys
from pathlib import Path

from mapsnap import experiments
from mapsnap.fit import find_centerlines, find_ref_iiif


def run_outputs(dir_path: Path, tag: str) -> tuple[Path | None, Path | None]:
    """The ``<tag>.iiif.json`` and ``<tag>.txt`` a run left at the volume root."""
    iiif_path = dir_path / f"{tag}.iiif.json"
    compare_txt = dir_path / f"{tag}.txt"
    return (
        iiif_path if iiif_path.exists() else None,
        compare_txt if compare_txt.exists() else None,
    )


def archived_stems(run_dir: Path) -> set[str]:
    """Page stems whose georef sidecar is already in an archive directory."""
    return {
        path.name.split(".")[0]
        for path in run_dir.glob("p*.georef.json")
        if path.name.startswith("p")
    }


def is_complete(run_dir: Path) -> bool:
    """Whether a run directory holds a finished archive.

    The manifest is written last, so its presence -- not the directory's --
    is what says the copy finished. A directory alone can be the remains of an
    interrupted archive, and treating that as done would silently skip the work
    that would have filled it.
    """
    return (run_dir / "manifest.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Archive an already-produced run's sidecars into "
            "data/<volume>/artifacts/<tag>/, as mapsnap fit does for runs it "
            "computes itself."
        )
    )
    parser.add_argument("dir", metavar="DIR", help="Volume directory")
    parser.add_argument(
        "--tag",
        required=True,
        help=(
            "Run tag, naming <tag>.iiif.json and <tag>.txt at the volume root "
            "and the artifacts/<tag>/ directory to write."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        metavar="NAME",
        help="Human-readable name recorded alongside the run id in the manifest.",
    )
    parser.add_argument(
        "--note",
        default=None,
        metavar="TEXT",
        help=(
            "Recorded in the manifest as how this run was produced. A hand-run "
            "sequence is otherwise unreconstructable: unlike a fit run, the "
            "command line here does not describe what made the sidecars."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would be archived and exit without writing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-archive over a complete existing archive for this tag.",
    )
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        sys.exit(f"No such volume directory: {dir_path}")

    iiif_path, compare_txt = run_outputs(dir_path, args.tag)
    if iiif_path is None:
        sys.exit(
            f"No {args.tag}.iiif.json in {dir_path}. This command archives a run "
            "that has already been produced; it does not compute one."
        )

    run_dir = dir_path / experiments.ARTIFACTS_DIRNAME / args.tag
    sidecars = sorted(dir_path.glob("p*.georef*.json"))
    # --check reports the state rather than acting on it, including the state
    # that would otherwise stop the run: being told "already archived" and
    # nothing else is exactly what a dry run should not do.
    if args.check:
        print(f"Would archive run {args.tag} from {dir_path}:")
        print(f"  iiif    : {iiif_path.name}")
        print(f"  compare : {compare_txt.name if compare_txt else '(none)'}")
        print(
            f"  sidecars: {len(sidecars)} georef, {len(list(dir_path.glob('p*.streets.json')))} streets"
        )
        print(f"  into    : {run_dir}")
        if is_complete(run_dir):
            print("  status  : already archived; archiving needs --force")
        elif run_dir.exists():
            print(
                "  status  : directory exists but has no manifest (interrupted); would complete it"
            )
        else:
            print("  status  : not archived yet")
        return

    if is_complete(run_dir) and not args.force:
        sys.exit(
            f"Run {args.tag} is already archived at {run_dir}; pass --force to replace."
        )

    if not sidecars:
        sys.exit(
            f"No p*.georef*.json in {dir_path}; there is nothing to archive. "
            "Run the fit stages first."
        )

    centerlines = find_centerlines(dir_path)
    if find_ref_iiif(dir_path) is None:
        sys.exit(f"No reference IIIF found in {dir_path}")
    truth = dir_path / "main.iiif.json"
    git = experiments.git_head_info(dir_path)
    inputs = experiments.gather_inputs(
        dir_path, centerlines, truth if truth.exists() else None
    )
    command = [*sys.argv[0].split(), *sys.argv[1:]]

    manifest = experiments.build_manifest(
        dir_path,
        args.tag,
        [],
        inputs,
        git,
        command,
        truth if truth.exists() else None,
        iiif_path,
        args.label,
    )
    # A fit run's command line reproduces it; a hand-assembled one's does not,
    # so record that this was archived after the fact rather than computed here.
    manifest["archived_by"] = "mapsnap archive"
    if args.note:
        manifest["note"] = args.note

    archived = experiments.archive_run(
        dir_path, args.tag, manifest, iiif_path, compare_txt
    )
    stems = archived_stems(archived)
    truth = (manifest.get("metrics") or {}).get("truth") or {}
    placed = len(truth.get("per_page", {}))
    print(f"Archived run {args.tag} to {archived} ({len(stems)} page sidecars)")
    if placed:
        print(f"  {placed} pages have a truth comparison in the manifest.")
    if compare_txt is None:
        print(
            f"No {args.tag}.txt found, so the archive has no compare table.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
