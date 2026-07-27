"""Sanity-check hand-labelled key-map truth against a volume's page images.

Key-map page numbers are typed by hand from a scan, and the awkward ones —
small raised letter suffixes, ornate fonts, numbers sitting on dark blocks —
are easy to mistype or mis-read. Every label should name a real sheet in the
volume, every sheet should be labelled once, and the volume's own page images
are the ground truth for both. This reports the three ways that can break:

  UNKNOWN     a label naming no page image ("35A?", "3S" for "35")
  MISSING     a page image NO truth file labels
  DUPLICATED  a label used twice for a sheet that is not split into panels

UNKNOWN and DUPLICATED are per-file, since each is a slip made on one sheet.
MISSING is per-volume: a volume's key maps divide the city between them (Los
Angeles has pa and pb, Brooklyn a SW and a NE half), so a sheet drawn on one
key map is not missing just because the other does not show it. It is only
reported when no truth file in the volume names it.

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
class FileReport:
    """What one truth file got wrong on its own key map."""

    labels_path: Path
    n_labels: int
    unknown: list[tuple[str, str | None]] = field(default_factory=list)
    duplicated: list[tuple[str, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unknown or self.duplicated)


@dataclass
class VolumeReport:
    """A volume's truth files, plus the sheets none of them label."""

    volume: Path
    files: list[FileReport] = field(default_factory=list)
    n_sheets: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def n_problems(self) -> int:
        return len(self.missing) + sum(
            len(f.unknown) + len(f.duplicated) for f in self.files
        )

    @property
    def ok(self) -> bool:
        return self.n_problems == 0


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


def read_labels(labels_path: Path) -> list[str]:
    """The non-empty label texts of one truth file, normalized."""
    doc = json.loads(labels_path.read_text())
    texts = (normalize_label(e.get("text", "")) for e in doc.get("labels", []))
    return [text for text in texts if text]


def check_volume(volume: Path, labels_paths: list[Path]) -> VolumeReport:
    """Check every truth file of one volume against its page images."""
    pages = volume_pages(volume)
    known = set(pages.stems)
    report = VolumeReport(volume=volume, n_sheets=len(known - pages.keymaps))

    labelled: set[str] = set()
    for labels_path in labels_paths:
        texts = read_labels(labels_path)
        labelled.update(texts)
        counts = Counter(texts)
        file_report = FileReport(labels_path=labels_path, n_labels=len(texts))
        for label in sorted(set(texts) - known):
            file_report.unknown.append((label, suggest(label, known)))
        for label, count in sorted(counts.items()):
            # A split sheet is drawn once per panel, so repeats are expected.
            if count > 1 and label in known and not pages.panels[label]:
                file_report.duplicated.append((label, count))
        report.files.append(file_report)

    # Volume-wide: a sheet drawn on one key map is not missing from the volume
    # just because another key map does not show it.
    report.missing = sorted(known - labelled - pages.keymaps)
    return report


def format_report(report: VolumeReport) -> str:
    """Render one volume's report as human-readable lines."""
    n_labels = sum(f.n_labels for f in report.files)
    lines = [
        report.volume.name,
        (
            f"  {len(report.files)} truth file(s), {n_labels} labels "
            f"vs {report.n_sheets} page images"
        ),
    ]
    for file_report in report.files:
        stem = file_report.labels_path.name.replace(".labels.json", "")
        lines.append(f"  {stem} ({file_report.n_labels} labels)")
        if file_report.ok:
            lines.append("    no bad labels")
        for label, hint in file_report.unknown:
            suffix = f"   (did you mean {hint}?)" if hint else ""
            lines.append(f"    UNKNOWN {label!r} — names no page image{suffix}")
        for label, count in file_report.duplicated:
            lines.append(
                f"    DUPLICATED {label} x{count} — repeated, but not a split sheet"
            )
    if report.missing:
        lines.append(
            f"  MISSING ({len(report.missing)}) — labelled by no truth file: "
            + ", ".join(report.missing)
        )
    elif report.ok:
        lines.append("  OK — every sheet labelled, no bad labels")
    return "\n".join(lines)


def volume_of(target: Path) -> Path:
    """The volume a target names, whether it is the directory or a labels file."""
    # <volume>/raw/truth/<stem>.labels.json
    return target if target.is_dir() else target.parent.parent.parent


def truth_files(volume: Path) -> list[Path]:
    """Every truth file of a volume, sorted.

    Always all of them, even when the caller named just one: the MISSING check
    is only meaningful across a volume's whole set of key maps.
    """
    return sorted(volume.glob("raw/truth/*.labels.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check hand-labelled key-map truth against a volume's page images."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        metavar="VOLUME_OR_LABELS",
        help=(
            "Volume directory, or a labels file (its whole volume is checked "
            "either way, since a volume's key maps share the sheets between them)."
        ),
    )
    args = parser.parse_args()

    volumes: list[Path] = []
    for target in args.targets:
        volume = volume_of(target)
        if volume not in volumes:
            volumes.append(volume)

    problems = 0
    for volume in volumes:
        labels_paths = truth_files(volume)
        if not labels_paths:
            print(f"{volume}: no raw/truth/*.labels.json found", file=sys.stderr)
            continue
        report = check_volume(volume, labels_paths)
        print(format_report(report))
        problems += report.n_problems
    if problems:
        print(f"\n{problems} suspicious label(s) across {len(volumes)} volume(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
