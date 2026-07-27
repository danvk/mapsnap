"""Download full-resolution images from a Library of Congress IIIF Presentation manifest.

Reads a local LOC IIIF v2 manifest JSON file, extracts a page key from each
canvas @id (e.g. "p451" from "...04720_04_1951-0451"), constructs a
full-resolution IIIF image URL, and downloads it.

A manifest can concatenate several LOC volumes, and each of those carries its
own key map at page 0 — Grand Rapids' manifest holds volumes 07 and 08, whose
"...-0000" canvases both reduce to "p0". Their page numbers are volume-prefixed
(701-729, 809-846) so only page 0 ever collides, but the collision used to be
silent: the first canvas wrote p0.jpg and the second was skipped as "already
done", leaving the volume with half its key maps. Colliding keys are now
suffixed in manifest order (p0a, p0b) so every canvas is kept. Keys that do not
collide are untouched, so existing volumes keep their filenames.

Usage:
    python download_loc_iiif.py <iiif_file> [--pages p0a,p0b] [--dry-run]
"""

import argparse
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from mapsnap.download_oim_iiif import download_with_retry
from mapsnap.utils import jpeg_dimensions, source_id_to_page_key


@dataclass
class CanvasTarget:
    """One manifest canvas and the file it will be downloaded to."""

    canvas: dict
    output_dir: Path
    #: The key this canvas's @id implies on its own ("p0").
    base_key: str
    #: The key actually used, after disambiguating collisions ("p0a").
    page_key: str

    @property
    def label(self) -> str:
        return self.canvas.get("label", "unknown")

    @property
    def path(self) -> Path:
        return self.output_dir / f"{self.page_key}.jpg"

    def image_url(self, scale: str) -> str:
        return f"{self.canvas.get('@id', '')}/full/{scale}/0/default.jpg"


def canvas_to_page_key(canvas_id: str, label: str) -> str:
    """Extract a short page key from a LOC IIIF canvas @id and label.

    Delegates to source_id_to_page_key for numeric page IDs (e.g. p451).
    Falls back to the last colon-separated segment for non-numeric ones
    (e.g. "covr", "titl", "ind1").
    """
    last_segment = canvas_id.split(":")[-1]
    if re.search(r"-\d", last_segment) or re.match(
        r"sb\d", last_segment, re.IGNORECASE
    ):
        return source_id_to_page_key(canvas_id, label)
    return last_segment


def disambiguate_keys(targets: list[CanvasTarget]) -> list[tuple[str, list[str]]]:
    """Suffix page keys that collide within an output directory, in manifest order.

    Two canvases reducing to the same key would otherwise write the same file,
    and the second would be skipped as already downloaded. Colliding keys become
    "<key>a", "<key>b", ... in the order the canvases appear; a suffix already
    claimed by another canvas's own key is skipped, so a manifest holding both a
    real "p0a" and two "p0"s cannot be made to collide again.

    Mutates ``targets`` and returns the renames as (base_key, [new keys]) so the
    caller can report them.
    """
    by_dir: dict[Path, list[CanvasTarget]] = defaultdict(list)
    for target in targets:
        by_dir[target.output_dir].append(target)

    renames: list[tuple[str, list[str]]] = []
    for group in by_dir.values():
        counts = Counter(target.base_key for target in group)
        taken = set(counts)
        for base_key, count in counts.items():
            if count < 2:
                continue
            colliding = [t for t in group if t.base_key == base_key]
            suffixes = (s for s in string.ascii_lowercase if base_key + s not in taken)
            new_keys = []
            for target in colliding:
                suffix = next(suffixes, None)
                if suffix is None:
                    raise ValueError(f"more than 26 canvases collide on {base_key!r}")
                target.page_key = base_key + suffix
                taken.add(target.page_key)
                new_keys.append(target.page_key)
            renames.append((base_key, new_keys))
    return renames


def select_pages(targets: list[CanvasTarget], wanted: set[str]) -> list[CanvasTarget]:
    """Keep only the targets a --pages selection names.

    A selection matches either the final page key ("p0a") or the base key it was
    disambiguated from ("p0", which selects every one of its variants), so a
    subset can be named without knowing in advance which keys collided.
    """
    return [t for t in targets if t.page_key in wanted or t.base_key in wanted]


def load_targets(iiif_path: Path) -> list[CanvasTarget]:
    """Every canvas of one manifest, keyed but not yet disambiguated."""
    data: dict = json.loads(iiif_path.read_text())
    sequences: list[dict] = data.get("sequences", [])
    if not sequences:
        print(f"No sequences found in {iiif_path}.", file=sys.stderr)
        sys.exit(1)
    canvases: list[dict] = sequences[0].get("canvases", [])
    print(f"Found {len(canvases)} canvases in {iiif_path.name}.", file=sys.stderr)
    targets = []
    for canvas in canvases:
        canvas_id: str = canvas.get("@id", "")
        key = canvas_to_page_key(canvas_id, canvas.get("label", "unknown"))
        assert key != "iiif", f"Could not extract valid page key from {canvas_id}"
        targets.append(
            CanvasTarget(canvas, iiif_path.parent, base_key=key, page_key=key)
        )
    return targets


def process_canvas(target: CanvasTarget, scale: str, dry_run: bool) -> Path:
    """Download the full-resolution image for one canvas.

    Returns a Path to the output file, or raises on failure.
    """
    image_path = target.path
    if image_path.exists():
        print(f"  Already done: {image_path.name}", file=sys.stderr)
        return image_path

    image_url = target.image_url(scale)
    print(
        f"  {target.label} ({target.page_key}) {image_path} {image_url}",
        file=sys.stderr,
    )
    if dry_run:
        return image_path

    print(f"    Downloading → {image_path.name} ...", file=sys.stderr)
    download_with_retry(image_url, image_path, initial_delay=15.0)
    # unconditional delay to avoid making too many requests if they all succeed.
    time.sleep(15.0)
    dl_width, dl_height = jpeg_dimensions(image_path)
    print(f"    Downloaded: {dl_width}×{dl_height}", file=sys.stderr)
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download LOC images from a local IIIF Presentation manifest. "
            "Saves one <page_key>.jpg per canvas alongside the IIIF file."
        )
    )
    parser.add_argument(
        "iiif_files",
        nargs="+",
        metavar="FILE",
        help="Local LOC IIIF Presentation manifest JSON file(s)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="full",
        help='Scale at which to download imagery. Default is "full", but can also be '
        'set to "pct:25", "pct:50", etc.',
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        metavar="KEYS",
        help=(
            "Comma-separated page keys to download (e.g. 'p0a,p0b'), instead of "
            "every canvas. A key that collided may be named either by its final "
            "key ('p0a') or by the base key it came from ('p0', which selects "
            "all its variants). Use --dry-run to list the keys."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    targets: list[CanvasTarget] = []
    for iiif_file in args.iiif_files:
        targets.extend(load_targets(Path(iiif_file)))

    for base_key, new_keys in disambiguate_keys(targets):
        print(
            f"Collision: {len(new_keys)} canvases map to {base_key!r} "
            f"-> {', '.join(new_keys)}",
            file=sys.stderr,
        )

    if args.pages:
        wanted = {p.strip() for p in args.pages.split(",") if p.strip()}
        selected = select_pages(targets, wanted)
        unmatched = (
            wanted - {t.page_key for t in selected} - {t.base_key for t in selected}
        )
        if unmatched:
            print(f"No canvas matches: {', '.join(sorted(unmatched))}", file=sys.stderr)
        print(f"Selected {len(selected)} of {len(targets)} canvases.", file=sys.stderr)
        targets = selected

    num_total = 0
    out_paths = Counter[Path]()
    for i, target in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] ", file=sys.stderr, end="")
        out_paths[process_canvas(target, scale=args.scale, dry_run=args.dry_run)] += 1
        num_total += 1

    print(
        f"\nDone: {num_total} canvases {'would be ' if args.dry_run else ''}processed.",
        file=sys.stderr,
    )
    if len(out_paths) < num_total:
        print(f"Unique paths: {len(out_paths)}; there are collisions.", file=sys.stderr)
        print(
            [*((path, count) for path, count in out_paths.most_common() if count > 1)],
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
