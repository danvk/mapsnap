"""Archive and compare ``mapsnap fit`` experiment runs.

A *run* is one georeferencing pass over a volume. Each run is archived under
``data/<volume>/artifacts/<run-id>/`` so that the inputs, outputs, and metrics of a
``(commit, flags, inputs) → results`` experiment are preserved and diffable. See
EXPERIMENTS.md for the design.

This module provides:
  - run-id computation (``<git-sha8>-<config-hash8>``) and input content hashing,
  - manifest construction (provenance + coverage/truth metrics),
  - archiving of the per-run sidecars, with streets.json deduplicated by content hash,
  - the ``mapsnap experiments diff`` subcommand for comparing two archived runs.

``mapsnap fit`` calls :func:`archive_fit_run` on completion; the CLI entry point
:func:`main` implements ``experiments diff``.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mapsnap import sidecar
from mapsnap.compare_iiif_georef import compare_pages
from mapsnap.score import summarize, volume_page_scores
from mapsnap.utils import default_centerlines, list_pages

ARTIFACTS_DIRNAME = "artifacts"

# Flags that don't change georef output (only parallelism), so they're excluded from the
# config hash — otherwise `-j2` and `-j1` would fragment a shared baseline into two archives.
HASH_EXCLUDED_VALUE_FLAGS = frozenset({"--num-workers"})

# Verdicts georef can record against its own pose (mapsnap.sidecar). These were
# filename suffixes until #270 phase 3; counting them by glob now silently
# reports zero, since every page has exactly one p<stem>.georef.json whatever
# the verdict inside it.
COUNTED_VERDICTS = ("misscale", "outlier", "1gcp", "keymap-outlier", "nofit")


def file_sha256(path: Path) -> str:
    """Return the ``sha256:<hex>`` digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def combined_sha256(paths: list[Path]) -> str:
    """Return a single ``sha256:`` digest over a set of files, order-independent.

    Each file contributes ``<name>\\0<sha256-hex>``; the contributions are sorted by name
    before hashing so the result depends only on the file set's contents and names, not on
    iteration order. Used to fingerprint all of a volume's ``streets.json`` (or ``boxes.json``)
    inputs as one value.
    """
    parts: list[str] = []
    for path in paths:
        digest = file_sha256(path).removeprefix("sha256:")
        parts.append(f"{path.name}\0{digest}")
    h = hashlib.sha256()
    h.update("\n".join(sorted(parts)).encode())
    return f"sha256:{h.hexdigest()}"


def git_head_info(cwd: Path) -> dict:
    """Return ``{sha, branch, subject, clean}`` for the git HEAD containing ``cwd``.

    ``clean`` reflects whether the working tree has uncommitted changes to tracked files
    (``data/`` is gitignored, so generated outputs never count). Returns ``clean=None`` and
    empty fields if ``cwd`` is not inside a git repository.
    """

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "branch": None, "subject": None, "clean": None}
    status = git("status", "--porcelain")
    return {
        "sha": sha,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "subject": git("log", "-1", "--format=%s"),
        "clean": status == "",
    }


def normalize_flags_for_hash(flag_tokens: list[str]) -> list[str]:
    """Drop output-irrelevant flags (e.g. ``--num-workers N``) from a passthrough token list.

    The remaining tokens are returned in their original order; they're folded into the config
    hash so that, e.g., ``--disable-scale-outlier-check`` archives separately from the default.
    """
    result: list[str] = []
    i = 0
    while i < len(flag_tokens):
        token = flag_tokens[i]
        if token in HASH_EXCLUDED_VALUE_FLAGS:
            i += 2  # skip the flag and its value
            continue
        result.append(token)
        i += 1
    return result


def gather_inputs(dir_path: Path, centerlines: Path, truth: Path | None) -> dict:
    """Hash every input that affects a georef run: streets.json, centerlines, truth, boxes.

    Returns the ``inputs`` block recorded in the manifest. ``boxes.json`` is hashed for
    provenance but is not archived (it's CRAFT output, georef-independent, and large).

    Reads are hashed for the EFFECTIVE pages only -- the ones `mapsnap georef`
    is actually given. A split page's parent keeps its old `p12.streets.json`
    after the split, but `list_pages` hands georef `p12__1`/`p12__2` instead, so
    folding the parent in claimed a file as an input that cannot change any
    result. Eight of the eighteen truth volumes carry such a leftover, and the
    stale hash both muddied provenance and made the file look live enough that
    an audit of recognizer coverage mis-reported 15 pages as needing re-OCR.
    """
    effective = {path.stem for path in list_pages(dir_path)}
    if not effective:
        # No page images to go by (a bare sidecar directory). Hashing nothing
        # would make every such directory hash alike, so fall back to the glob.
        effective = {
            path.name[: -len(".streets.json")]
            for path in dir_path.glob("p*.streets.json")
        } | {
            path.name[: -len(".boxes.json")] for path in dir_path.glob("p*.boxes.json")
        }
    streets = sorted(
        path
        for path in dir_path.glob("p*.streets.json")
        if path.name[: -len(".streets.json")] in effective
    )
    boxes = sorted(
        path
        for path in dir_path.glob("p*.boxes.json")
        if path.name[: -len(".boxes.json")] in effective
    )
    inputs: dict = {
        "centerlines_sha": file_sha256(centerlines),
        "streets": {"count": len(streets), "combined_sha": combined_sha256(streets)},
        "boxes_sha": combined_sha256(boxes) if boxes else None,
    }
    if truth is not None and truth.exists():
        inputs["truth"] = truth.name
        inputs["truth_sha"] = file_sha256(truth)
    return inputs


def compute_config_hash(flag_tokens: list[str], inputs: dict) -> str:
    """Return the 8-char config hash over the effective flags and input content hashes.

    Folding input hashes in means a re-run after re-OCR (new streets.json) gets a fresh id
    rather than silently reusing a stale archive.
    """
    payload = {
        "flags": normalize_flags_for_hash(flag_tokens),
        "inputs": inputs,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:8]


def auto_run_id(git_sha: str, config_hash: str) -> str:
    """Compose the automatic run id ``<git-sha8>-<config-hash8>`` from a full git sha."""
    return f"{git_sha[:8]}-{config_hash}"


def coverage_counts(dir_path: Path) -> dict:
    """Count pages and georef outcomes from the live sidecar files in ``dir_path``.

    ``georeferenced`` counts pages whose RANSAC sidecar holds a pose georef
    STANDS BEHIND, and each verdict is counted under its own name. Both read
    the recorded status rather than the filename: after #270 phase 3 every page
    has a ``p<stem>.georef.json`` whatever georef concluded about it, so
    counting files reported every page as georeferenced and every verdict as
    zero. ``pages`` is the number of effective input page images.
    """
    counts: dict[str, int] = {
        "pages": len(list_pages(dir_path)),
        "georeferenced": 0,
    }
    for verdict in COUNTED_VERDICTS:
        counts[verdict] = 0
    for path in dir_path.glob("p*.georef.json"):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if sidecar.internally_valid(doc):
            counts["georeferenced"] += 1
        else:
            verdict = sidecar.status(doc)
            if verdict in counts:
                counts[verdict] += 1
    return counts


def truth_metrics(truth_path: Path, iiif_path: Path) -> dict:
    """Compute aggregate and per-page error of a generated IIIF against truth.

    Error is each page's RMSE in feet (the headline metric of ``mapsnap compare``). Returns
    ``compared``/``missing`` counts, mean/median RMSE, a ``per_page`` map, and — when the
    volume has centerlines for land weighting — the project success ``score``.
    """
    rows, missing = compare_pages(truth_path, iiif_path)
    rmses = sorted(r["rmse_ft"] for r in rows)
    n = len(rmses)
    if n:
        mean_rmse = sum(rmses) / n
        median_rmse = (
            rmses[n // 2] if n % 2 else (rmses[n // 2 - 1] + rmses[n // 2]) / 2
        )
    else:
        mean_rmse = median_rmse = 0.0
    metrics = {
        "compared": n,
        "missing": len(missing),
        "mean_rmse_ft": round(mean_rmse, 1),
        "median_rmse_ft": round(median_rmse, 1),
        "per_page": {r["page_key"]: r["rmse_ft"] for r in rows},
    }
    # The project success metric (see mapsnap.score): land-weighted good share
    # minus disaster share. Needs centerlines for the land weights, which a
    # truth-bearing volume normally has.
    if default_centerlines(iiif_path.parent) is not None:
        score = summarize(volume_page_scores(iiif_path))
        metrics["score"] = {
            "net": round(score.net_score, 4),
            "good_share": round(score.good_share, 4),
            "fair_share": round(score.fair_share, 4),
            "poor_share": round(score.poor_share, 4),
            "disaster_share": round(score.disaster_share, 4),
            "n_pages": score.n_pages,
            "n_placed": score.n_placed,
        }
    return metrics


def build_manifest(
    dir_path: Path,
    run_id: str,
    flag_tokens: list[str],
    inputs: dict,
    git: dict,
    command: list[str],
    truth_path: Path | None,
    iiif_path: Path | None,
    label: str | None = None,
) -> dict:
    """Assemble the manifest dict: run identity, provenance, and metrics.

    The ``truth`` block under ``metrics`` is included only when both a truth file and a
    generated IIIF exist; volumes without truth report coverage-only metrics.
    """
    metrics: dict = coverage_counts(dir_path)
    if truth_path is not None and truth_path.exists() and iiif_path is not None:
        metrics["truth"] = truth_metrics(truth_path, iiif_path)
    manifest: dict = {
        "run_id": run_id,
        "git": git,
        "command": command,
        "flags": flag_tokens,
        "inputs": inputs,
        "metrics": metrics,
    }
    if label is not None:
        manifest["label"] = label
    return manifest


def find_dedup_streets_dir(
    artifacts_dir: Path, run_id: str, combined_sha: str
) -> Path | None:
    """Return an existing archive whose streets combined_sha matches, for symlink dedup.

    Scans sibling archives under ``artifacts_dir`` (excluding ``run_id`` itself) and returns
    the first whose manifest records the same ``streets.combined_sha``, or None.
    """
    if not artifacts_dir.is_dir():
        return None
    for entry in sorted(artifacts_dir.iterdir()):
        if entry.name == run_id or not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        prior_sha = manifest.get("inputs", {}).get("streets", {}).get("combined_sha")
        if prior_sha == combined_sha:
            return entry
    return None


def archive_streets(dir_path: Path, run_dir: Path, dedup_source: Path | None) -> None:
    """Archive each ``p*.streets.json`` into ``run_dir``, symlinking when a dedup source exists.

    When ``dedup_source`` (a prior archive with identical streets content) is given, each
    streets file is symlinked to that archive's copy (resolving through any existing symlink
    so chains don't form). Otherwise the files are copied.
    """
    for streets in sorted(dir_path.glob("p*.streets.json")):
        dest = run_dir / streets.name
        if dedup_source is not None:
            source = dedup_source / streets.name
            if source.exists():
                dest.symlink_to(os.path.realpath(source))
                continue
        shutil.copy2(streets, dest)


def is_complete(run_dir: Path) -> bool:
    """Whether a run directory holds a finished archive.

    :func:`archive_run` creates the directory before copying into it, so an
    interrupted archive leaves one behind with nothing (or half) in it. The
    manifest is written last, which makes its presence -- not the directory's --
    the honest completion marker.

    Lives here rather than in ``mapsnap.archive`` because ``mapsnap.fit`` needs
    it too, and ``mapsnap.archive`` imports from ``mapsnap.fit``.
    """
    return (run_dir / "manifest.json").exists()


def archive_differs(
    run_dir: Path, inputs: dict, git: dict, flag_tokens: list[str]
) -> str | None:
    """What an existing archive was built from that this run is not, or None.

    Guards the "already archived, skipping" path. With an auto run id the id
    itself encodes (commit, flags, inputs), so a collision means a genuine
    match and this always returns None. With an explicit ``--tag`` it does not:
    the tag is just a name, and re-using one after changing code or reads makes
    `fit` republish the archived run and present it as the new one.

    Returns a short human phrase naming the FIRST mismatch, for the error
    message. A manifest that cannot be read counts as differing -- an archive
    we cannot verify is not one we should silently trust.
    """
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, ValueError):
        return "an unreadable manifest"
    if manifest.get("inputs") != inputs:
        return "inputs (reads, centerlines or truth)"
    archived_sha = (manifest.get("git") or {}).get("sha")
    if archived_sha and git.get("sha") and archived_sha != git["sha"]:
        return f"a different commit ({archived_sha[:8]})"
    if (manifest.get("flags") or []) != list(flag_tokens):
        return "different flags"
    return None


def archive_run(
    dir_path: Path,
    run_id: str,
    manifest: dict,
    iiif_path: Path | None,
    compare_txt: Path | None,
) -> Path:
    """Copy a run's sidecars and manifest into ``data/<volume>/artifacts/<run-id>/``.

    Archives the georef JSON (canonical and variants), per-page ``.txt`` logs, the IIIF and
    compare files, and ``manifest.json``. ``streets.json`` is deduplicated against any prior
    archive with matching content (see :func:`archive_streets`). Returns the archive path.
    """
    artifacts_dir = dir_path / ARTIFACTS_DIRNAME
    run_dir = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    combined_sha = manifest["inputs"]["streets"]["combined_sha"]
    dedup_source = find_dedup_streets_dir(artifacts_dir, run_id, combined_sha)
    archive_streets(dir_path, run_dir, dedup_source)

    for georef in sorted(dir_path.glob("p*.georef*.json")):
        shutil.copy2(georef, run_dir / georef.name)
    for log in sorted(dir_path.glob("p*.txt")):
        shutil.copy2(log, run_dir / log.name)
    # Channel candidate stores, so a later reconcile/analysis pass can see the
    # poses this run actually weighed (#270). Without these the baseline is
    # unrecoverable once any post-run experiment re-matches: the files live at
    # fixed paths and are overwritten in place.
    # Globbed, not enumerated: clear_derived_sidecars deletes every
    # candidates/selection file at the next fit's start, so anything the
    # archive misses is destroyed rather than merely stale -- and the
    # selection variants have grown before (selection_volume, selection_argmax
    # were never in the old hardcoded list).
    for channel in ("osm_snap", "street_solve"):
        channel_dir = artifacts_dir / channel
        for source in (
            *sorted(channel_dir.glob("candidates.jsonl")),
            *sorted(channel_dir.glob("selection_*.jsonl")),
        ):
            target_dir = run_dir / channel
            target_dir.mkdir(exist_ok=True)
            shutil.copy2(source, target_dir / source.name)
    if iiif_path is not None and iiif_path.exists():
        shutil.copy2(iiif_path, run_dir / iiif_path.name)
    if compare_txt is not None and compare_txt.exists():
        shutil.copy2(compare_txt, run_dir / compare_txt.name)

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def archive_fit_run(
    dir_path: Path,
    run_id: str,
    flag_tokens: list[str],
    inputs: dict,
    git: dict,
    command: list[str],
    truth_path: Path | None,
    iiif_path: Path | None,
    compare_txt: Path | None,
    label: str | None = None,
    stage_seconds: dict[str, float] | None = None,
) -> Path:
    """Build the manifest for a completed ``mapsnap fit`` run and archive it.

    Convenience wrapper combining :func:`build_manifest` and :func:`archive_run`, called by
    ``mapsnap fit`` once georef/IIIF/compare have produced their sidecars. Returns the archive
    directory.
    """
    manifest = build_manifest(
        dir_path,
        run_id,
        flag_tokens,
        inputs,
        git,
        command,
        truth_path,
        iiif_path,
        label,
    )
    if stage_seconds:
        manifest["stage_seconds"] = stage_seconds
    return archive_run(dir_path, run_id, manifest, iiif_path, compare_txt)


# ---------------------------------------------------------------------------
# experiments diff
# ---------------------------------------------------------------------------


def load_manifest(volume_dir: Path, run_id: str) -> dict | None:
    """Load ``data/<volume>/artifacts/<run-id>/manifest.json``, or None if absent."""
    path = volume_dir / ARTIFACTS_DIRNAME / run_id / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def volume_dirs_with_run(data_root: Path, run_id: str) -> list[Path]:
    """Return volume dirs under ``data_root`` that have an archive for ``run_id``."""
    return sorted(
        p.parent.parent
        for p in data_root.glob(f"*/{ARTIFACTS_DIRNAME}/{run_id}")
        if p.is_dir()
    )


def page_status(manifest: dict, page_key: str) -> str:
    """Classify a page as ``ok``/``missing`` for the coverage diff.

    Uses the per-page truth comparison when available: a page with a truth RMSE is ``ok``,
    one absent from ``per_page`` is ``missing``. Falls back to ``ok`` when there's no truth.
    """
    truth = manifest.get("metrics", {}).get("truth")
    if truth is None:
        return "ok"
    return "ok" if page_key in truth.get("per_page", {}) else "missing"


def diff_volume(man_a: dict, man_b: dict) -> dict:
    """Compute the coverage and per-page error deltas between two manifests of one volume.

    Returns a dict with ``coverage`` (per-status count deltas), ``regressions`` and
    ``improvements`` (per-page RMSE changes, each ``(page, before, after, delta)``), and the
    aggregate ``mean``/``median`` RMSE before/after. Pages newly placed or newly failed are
    surfaced under ``newly_placed`` / ``newly_failed``.
    """
    cov_a = man_a.get("metrics", {})
    cov_b = man_b.get("metrics", {})
    coverage = {
        key: cov_b.get(key, 0) - cov_a.get(key, 0)
        for key in ("georeferenced", *COUNTED_VERDICTS)
    }

    truth_a = (cov_a.get("truth") or {}).get("per_page", {})
    truth_b = (cov_b.get("truth") or {}).get("per_page", {})
    newly_placed = sorted(set(truth_b) - set(truth_a))
    newly_failed = sorted(set(truth_a) - set(truth_b))

    changes: list[tuple[str, float, float, float]] = []
    for page in sorted(set(truth_a) & set(truth_b)):
        before, after = truth_a[page], truth_b[page]
        if before != after:
            changes.append((page, before, after, after - before))
    regressions = sorted(
        (c for c in changes if c[3] > 0), key=lambda c: c[3], reverse=True
    )
    improvements = sorted((c for c in changes if c[3] < 0), key=lambda c: c[3])

    return {
        "coverage": coverage,
        "newly_placed": newly_placed,
        "newly_failed": newly_failed,
        "regressions": regressions,
        "improvements": improvements,
        "mean_a": (cov_a.get("truth") or {}).get("mean_rmse_ft"),
        "mean_b": (cov_b.get("truth") or {}).get("mean_rmse_ft"),
        "median_a": (cov_a.get("truth") or {}).get("median_rmse_ft"),
        "median_b": (cov_b.get("truth") or {}).get("median_rmse_ft"),
    }


def format_volume_diff(volume_name: str, diff: dict) -> str:
    """Render a single volume's :func:`diff_volume` result as a human-readable block."""
    lines = [f"## {volume_name}"]

    cov = diff["coverage"]
    cov_parts = [f"{key} {val:+d}" for key, val in cov.items() if val]
    lines.append("coverage: " + (", ".join(cov_parts) if cov_parts else "no change"))

    if diff["newly_placed"]:
        lines.append("newly placed: " + ", ".join(diff["newly_placed"]))
    if diff["newly_failed"]:
        lines.append("newly failed: " + ", ".join(diff["newly_failed"]))

    for label, rows in (
        ("regressions", diff["regressions"]),
        ("improvements", diff["improvements"]),
    ):
        for page, before, after, delta in rows:
            lines.append(
                f"  {label[:-1]:<11} {page:<10} {before:>8.1f} → {after:>8.1f} ft "
                f"({delta:+.1f})"
            )

    mean_a, mean_b = diff["mean_a"], diff["mean_b"]
    if mean_a is not None and mean_b is not None:
        lines.append(
            f"mean RMSE {mean_a:.1f} → {mean_b:.1f} ft ({mean_b - mean_a:+.1f}); "
            f"median {diff['median_a']:.1f} → {diff['median_b']:.1f} ft"
        )
    lines.append(
        f"roll-up: Δ placed {cov['georeferenced']:+d}, "
        f"{len(diff['regressions'])} regressions, "
        f"{len(diff['improvements'])} improvements"
    )
    return "\n".join(lines)


def run_diff(run_id_a: str, run_id_b: str, volumes: list[str], data_root: Path) -> int:
    """Implement ``mapsnap experiments diff``: print per-volume deltas between two runs.

    With no explicit ``volumes``, scans every volume under ``data_root`` that has both run
    archives. Returns a process exit code (1 if neither run is found anywhere).
    """
    if volumes:
        volume_dirs = [data_root / v if "/" not in v else Path(v) for v in volumes]
    else:
        dirs_a = set(volume_dirs_with_run(data_root, run_id_a))
        dirs_b = set(volume_dirs_with_run(data_root, run_id_b))
        volume_dirs = sorted(dirs_a & dirs_b)

    if not volume_dirs:
        print(
            f"No volume has archives for both {run_id_a} and {run_id_b} under {data_root}.",
            file=sys.stderr,
        )
        return 1

    found_any = False
    for volume_dir in volume_dirs:
        man_a = load_manifest(volume_dir, run_id_a)
        man_b = load_manifest(volume_dir, run_id_b)
        if man_a is None or man_b is None:
            missing = run_id_a if man_a is None else run_id_b
            print(f"{volume_dir.name}: missing archive for {missing}, skipping.")
            continue
        found_any = True
        print(format_volume_diff(volume_dir.name, diff_volume(man_a, man_b)))
        print()
    return 0 if found_any else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare archived `mapsnap fit` experiment runs."
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    diff_parser = sub.add_parser("diff", help="Compare two archived runs by id.")
    diff_parser.add_argument("run_id_a", metavar="RUN_ID_A")
    diff_parser.add_argument("run_id_b", metavar="RUN_ID_B")
    diff_parser.add_argument(
        "volumes",
        nargs="*",
        metavar="VOLUME",
        help="Volume name(s) under the data root; default scans all volumes.",
    )
    diff_parser.add_argument(
        "--data-root",
        default="data",
        metavar="DIR",
        help="Root directory containing volume subdirectories (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.subcommand == "diff":
        sys.exit(
            run_diff(args.run_id_a, args.run_id_b, args.volumes, Path(args.data_root))
        )


if __name__ == "__main__":
    main()
