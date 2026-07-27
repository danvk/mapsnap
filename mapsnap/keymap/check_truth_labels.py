"""Sanity-check hand-labelled key-map truth against a volume's page images.

Key-map page numbers are typed by hand from a scan, and the awkward ones —
small raised letter suffixes, ornate fonts, numbers sitting on dark blocks —
are easy to mistype or mis-read. Every label should name a real sheet in the
volume, every sheet should be labelled once, and the volume's own page images
are the ground truth for both. This reports the three ways that can break:

  UNKNOWN     a label naming no page image ("35A?", "3S" for "35")
  MISSING     a page image no label names (a number skipped on the key map)
  DUPLICATED  a label used twice for a sheet that is not split into panels

A split sheet is exempt from the duplicate check: the key map draws one region
per panel, so a legitimately split page carries one label per panel.

Usage:
    uv run python -m mapsnap.keymap.check_truth_labels data/<volume>
    uv run python -m mapsnap.keymap.check_truth_labels <...>/truth/p0.labels.json

Exits non-zero when anything is reported, so it can gate a labelling session.
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# A page image stem: 'p' then digits, optionally a letter suffix, optionally a
# '__N' split panel. Front matter ('pcover') and other names are not pages.
PAGE_STEM = re.compile(r"^p(\d+[A-Za-z]?)(?:__(\d+))?$", re.IGNORECASE)


@dataclass
class VolumePages:
    """The sheet labels a volume's page images imply."""

    # Sheet label (upper-case, e.g. "35B") -> its whole-page image stem.
    stems: dict[str, str] = field(default_factory=dict)
    # Sheet label -> how many split panels it has (0 when not split).
    panels: Counter = field(default_factory=Counter)
    # Sheet labels belonging to key-map sheets, which never label themselves.
    keymaps: set[str] = field(default_factory=set)
    # Image stems that are not page images at all (front matter, etc.).
    skipped: list[str] = field(default_factory=list)


@dataclass
class Report:
    """What a single truth file got wrong, relative to the volume's images."""

    labels_path: Path
    volume: Path
    n_labels: int
    n_sheets: int
    unknown: list[tuple[str, str | None]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    duplicated: list[tuple[str, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unknown or self.missing or self.duplicated)


def sheet_label(stem: str) -> tuple[str, int | None] | None:
    """('35B', panel_index) for a page-image stem, or None if it is not a page.

    ``panel_index`` is None for a whole page and the 1-based panel number for a
    split ('p35B__2' -> ('35B', 2)).
    """
    match = PAGE_STEM.match(stem)
    if not match:
        return None
    panel = int(match.group(2)) if match.group(2) else None
    return match.group(1).upper(), panel


def normalize_label(text: str) -> str:
    """A truth label as it would be written on a page image ('  35b ' -> '35B')."""
    return text.strip().upper()


def volume_pages(volume: Path) -> VolumePages:
    """Index a volume's page images by the sheet label each implies."""
    pages = VolumePages()
    for path in sorted(volume.glob("p*.jpg")):
        parsed = sheet_label(path.stem)
        if parsed is None:
            pages.skipped.append(path.stem)
            continue
        label, panel = parsed
        if panel is None:
            pages.stems[label] = path.stem
        else:
            pages.panels[label] += 1
    # A key map does not print its own number, so it is never "missing".
    for path in (volume / "raw").glob("*.keymap.json"):
        parsed = sheet_label(path.name.split(".")[0])
        if parsed is not None:
            pages.keymaps.add(parsed[0])
    return pages


def suggest(label: str, known: set[str]) -> str | None:
    """A real sheet label this one was probably meant to be, or None.

    Only the mechanical slips are suggested — stray punctuation such as the
    "35A?" a labeller leaves to mark their own uncertainty — rather than
    guessing at fuzzy matches, which would invite trusting a bad suggestion.
    """
    stripped = re.sub(r"[^0-9A-Z]", "", label)
    return stripped if stripped != label and stripped in known else None


def check_truth_file(labels_path: Path, volume: Path) -> Report:
    """Compare one <stem>.labels.json against its volume's page images."""
    doc = json.loads(labels_path.read_text())
    texts = [
        normalize_label(entry.get("text", ""))
        for entry in doc.get("labels", [])
        if normalize_label(entry.get("text", ""))
    ]
    pages = volume_pages(volume)
    known = set(pages.stems)
    counts = Counter(texts)

    report = Report(
        labels_path=labels_path,
        volume=volume,
        n_labels=len(texts),
        n_sheets=len(known - pages.keymaps),
    )
    for label in sorted(set(texts) - known):
        report.unknown.append((label, suggest(label, known)))
    report.missing = sorted(known - set(texts) - pages.keymaps)
    for label, count in sorted(counts.items()):
        # A split sheet is drawn once per panel, so repeats are expected there.
        if count > 1 and label in known and not pages.panels[label]:
            report.duplicated.append((label, count))
    return report


def format_report(report: Report) -> str:
    """Render one report as human-readable lines."""
    lines = [
        f"{report.volume.name} / {report.labels_path.stem.replace('.labels', '')}",
        f"  {report.n_labels} labels vs {report.n_sheets} page images",
    ]
    if report.ok:
        lines.append("  OK — every sheet labelled exactly once")
        return "\n".join(lines)
    if report.unknown:
        lines.append(f"  UNKNOWN ({len(report.unknown)}) — label names no page image:")
        for label, hint in report.unknown:
            suffix = f"   (did you mean {hint}?)" if hint else ""
            lines.append(f"    {label!r}{suffix}")
    if report.missing:
        lines.append(
            f"  MISSING ({len(report.missing)}) — page image has no label: "
            + ", ".join(report.missing)
        )
    if report.duplicated:
        lines.append(
            f"  DUPLICATED ({len(report.duplicated)}) — repeated, but not a split sheet:"
        )
        for label, count in report.duplicated:
            lines.append(f"    {label} x{count}")
    return "\n".join(lines)


def truth_files(target: Path) -> list[tuple[Path, Path]]:
    """(labels file, volume) pairs for a volume directory or a labels file."""
    if target.is_dir():
        return [
            (path, target) for path in sorted(target.glob("raw/truth/*.labels.json"))
        ]
    # <volume>/raw/truth/<stem>.labels.json
    return [(target, target.parent.parent.parent)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check hand-labelled key-map truth against a volume's page images."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        metavar="VOLUME_OR_LABELS",
        help="Volume directory (checks every raw/truth/*.labels.json) or a labels file.",
    )
    args = parser.parse_args()

    pairs: list[tuple[Path, Path]] = []
    for target in args.targets:
        found = truth_files(target)
        if not found:
            print(f"{target}: no raw/truth/*.labels.json found", file=sys.stderr)
        pairs.extend(found)

    problems = 0
    for labels_path, volume in pairs:
        report = check_truth_file(labels_path, volume)
        print(format_report(report))
        problems += len(report.unknown) + len(report.missing) + len(report.duplicated)
    if problems:
        print(f"\n{problems} suspicious label(s) across {len(pairs)} file(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
