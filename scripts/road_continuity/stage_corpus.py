#!/usr/bin/env python
"""Assemble corpus-run inputs for the truth volumes, so #481 can be A/B'd on them.

    scripts/road_continuity/stage_corpus.py --run-tag corpus-v1 --out work/rc

The truth volumes under ``data/`` were fitted from the old IIIF renders at JPEG
quality 75. The corpus run reads the mirror's q94 copies, which yield ~34% more
CRAFT boxes and so different reads, different fits and different P(road) --
measuring road-continuity against the q75 baseline would answer a question
nobody is asking. This builds a volume-shaped directory per truth volume out of
the corpus run's own outputs, which `mapsnap road-continuity` can then be
pointed at directly.

What comes from where, and why:

* ``p*.streets.json`` and ``p*.georef*.json`` from ``runs/<tag>/`` -- the run's
  reads and its published poses. These are what make it a corpus A/B at all.
* ``p*.roadprob.jpg`` from the ITEM ROOT, not the run directory. `loc-craft`
  writes P(road) once per item, where every run shares it, and `loc-fit`
  excludes ``*.jpg`` from its upload for exactly that reason. It is inferred
  from the q94 scan, so it is the q94 P(road).
* ``main.iiif.json``, ``oim/`` and ``centerlines.geojson`` from the local truth
  volume. Truth is truth whatever rendered the scan. The centerlines are the
  one deliberate deviation: the corpus run reads a county-wide OSM extract and
  this copies the volume's own, which covers the same ground in the same
  vintage and saves 20 MB an item. It feeds label canonicalisation, not the
  search.

Reads S3 as the ``mapsnap`` profile, not the default one: the default is a root
login session that dies every 10-20 minutes and fails mid-sync as an empty
listing rather than an error. Each item is still checked for having yielded
files, since a run can simply not have reached it yet.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING = Path.home() / "Downloads/loc-sanborn-maps.mapping.tsv"
# Copied from the truth volume rather than fetched: see the module docstring.
LOCAL_INPUTS = ("main.iiif.json", "centerlines.geojson")


def run_aws(args: list[str], profile: str) -> str:
    """An aws CLI call under ``profile``, or exit naming what failed."""
    result = subprocess.run(
        [*args, "--profile", profile], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if "expired" in message or "credentials" in message.lower():
            sys.exit(f"AWS session is not usable: {message}\nRun `aws login` first.")
        sys.exit(f"{' '.join(args[:4])}... failed: {message}")
    return result.stdout


def item_of(volume: Path) -> str | None:
    """The volume's LoC item id.

    `run-oim` volumes record it as ``params.sanborn_slug``; `run-loc` ones --
    half the truth set -- record only an OSM relation, so the id has to come
    out of the manifest's own LoC URLs. Both forms are needed: without the
    fallback, ten of the twenty truth volumes look absent from the mirror.
    """
    record = volume / "mapsnap.json"
    if record.exists():
        params = json.loads(record.read_text()).get("params") or {}
        slug = params.get("sanborn_slug")
        if slug:
            return slug
    manifest = volume / "manifest.json"
    if manifest.exists():
        found = sorted(set(re.findall(r"sanborn\d+_\d+", manifest.read_text())))
        # More than one means the manifest mixes volumes, which no truth volume
        # does; naming it is better than silently staging the wrong item.
        if len(found) == 1:
            return found[0]
        if found:
            print(f"  {volume.name}: ambiguous item ids {found}", file=sys.stderr)
    return None


def mirror_prefixes(mapping: Path) -> dict[str, tuple[str, str]]:
    """Item -> (state, year), the mirror's own layout for it."""
    prefixes: dict[str, tuple[str, str]] = {}
    with open(mapping, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            prefixes.setdefault(row["item"], (row["state"], row["year"]))
    return prefixes


def stage_volume(
    volume: Path,
    item: str,
    state: str,
    year: str,
    *,
    bucket: str,
    run_tag: str,
    out: Path,
    profile: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Pull one volume's corpus inputs; return (sidecar files, roadprob files)."""
    target = out / volume.name
    target.mkdir(parents=True, exist_ok=True)
    item_prefix = f"{bucket}/by-state/{state}/{year}/{item}"

    sidecars = [
        "aws", "s3", "sync", f"{item_prefix}/runs/{run_tag}/", str(target),
        "--exclude", "*",
        "--include", "p*.streets.json",
        "--include", "p*.georef.json",
        "--include", "p*.georef-final.json",
        "--include", "keymaps.json",
        "--only-show-errors",
    ]  # fmt: skip
    rasters = [
        "aws", "s3", "sync", f"{item_prefix}/", str(target),
        "--exclude", "*",
        "--include", "p*.roadprob.jpg",
        "--only-show-errors",
    ]  # fmt: skip
    if dry_run:
        print(f"  would run: {' '.join(sidecars[:5])} ...")
        return (0, 0)
    run_aws(sidecars, profile)
    run_aws(rasters, profile)

    for name in LOCAL_INPUTS:
        source = volume / name
        if source.exists():
            shutil.copy2(source, target / name)
    if (volume / "oim").is_dir() and not (target / "oim").exists():
        shutil.copytree(volume / "oim", target / "oim")
    return (
        len(list(target.glob("p*.streets.json"))),
        len(list(target.glob("p*.roadprob.jpg"))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-tag", default="corpus-v1")
    parser.add_argument("--bucket", default="s3://mapsnap-sanborn")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, default=REPO / "work/rc-corpus")
    parser.add_argument(
        "--volumes",
        help="Comma-separated volume directory names (default: every truth volume)",
    )
    parser.add_argument(
        "--profile",
        default="mapsnap",
        help=(
            "AWS profile (default: %(default)s). The default profile is a root "
            "login session that expires every 10-20 minutes, which shows up "
            "mid-sync as an empty listing rather than an error; the mapsnap "
            "profile is a long-lived IAM user."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prefixes = mirror_prefixes(args.mapping)
    names = (
        args.volumes.split(",")
        if args.volumes
        # A truth volume is one with a main.iiif.json to score against.
        else sorted(p.parent.name for p in args.data.glob("*/main.iiif.json"))
    )

    print(f"{len(names)} volume(s) -> {args.out}", file=sys.stderr)
    missing: list[str] = []
    empty: list[str] = []
    for name in names:
        volume = args.data / name
        item = item_of(volume)
        if item is None or item not in prefixes:
            missing.append(f"{name} ({item or 'no item id'})")
            continue
        state, year = prefixes[item]
        n_sidecars, n_rasters = stage_volume(
            volume,
            item,
            state,
            year,
            bucket=args.bucket.rstrip("/"),
            run_tag=args.run_tag,
            out=args.out,
            profile=args.profile,
            dry_run=args.dry_run,
        )
        print(f"  {name:30s} {item:20s} {n_sidecars:4d} reads, {n_rasters:4d} P(road)")
        if not args.dry_run and (n_sidecars == 0 or n_rasters == 0):
            empty.append(name)

    if missing:
        print(f"\nnot in the mirror ({len(missing)}): {', '.join(missing)}")
    if empty:
        print(
            f"\nEMPTY ({len(empty)}): {', '.join(empty)}"
            "\nAn item with no files usually means the run has not reached it yet,"
            "\nor the session expired mid-sync. Check one by hand before trusting"
            "\nthe A/B."
        )


if __name__ == "__main__":
    main()
