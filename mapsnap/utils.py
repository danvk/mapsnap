"""Shared utilities for mapsnap scripts."""

import json
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    """Print and run a subprocess command, exiting with its return code on failure."""
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def write_run_record(dir_path: Path, source: str, params: dict[str, str]) -> None:
    """Record the pipeline invocation in dir_path/mapsnap.json for reproducibility.

    source is the pipeline kind ("loc" or "oim"); params holds the volume-specific
    arguments (slug/manifest, relation, etc.). The command line and a UTC timestamp are
    added automatically.
    """
    # cli.py rewrites argv[0] to "mapsnap <subcommand>"; split it back into tokens.
    record = {
        "source": source,
        "command": [*sys.argv[0].split(), *sys.argv[1:]],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": params,
    }
    (dir_path / "mapsnap.json").write_text(json.dumps(record, indent=2))


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) from a JPEG file by scanning SOF markers."""
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG: {path}")
        while True:
            marker = f.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                break
            seg_type = marker[1]
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if seg_type in (0xC0, 0xC1, 0xC2, 0xC3):  # SOF0–SOF3
                data = f.read(length - 2)
                height = struct.unpack(">H", data[1:3])[0]
                width = struct.unpack(">H", data[3:5])[0]
                return width, height
            f.seek(length - 2, 1)
    raise ValueError(f"No SOF marker found in {path}")


def source_id_to_page_key(source_id: str, label: str) -> str:
    """Extract a short page key like 'p425' from a LOC IIIF image service URL.

    Leading zeros in the page number are stripped and a 'p' prefix is added:
      "https://...1950-0006N/info.json", ""           → "p6N"
      "https://...1951-0425/info.json",  ""           → "p425"
      "https://...service:...:sb001250", ""           → "p125"
      "https://...service:...:sb00154s", ""           → "p154s"
    Labels ending with "[N]" produce a split suffix:
      "https://...1950-0156/info.json",  "... [2]"   → "p156__2"

    The sb-format encodes the page number as 5 zero-padded digits followed by
    one character: '0' means no suffix, a letter is used directly as a suffix.
    """
    split_suffix = ""
    if label.endswith("]"):
        m = re.search(r"\[(\d+)\]$", label)
        assert m
        split_suffix = f"__{m.group(1)}"

    # Sanborn sb-format: service:...:sb{5-digit page}{suffix char}
    m = re.search(r":sb(\d{5})([a-z0-9])$", source_id, re.IGNORECASE)
    if m:
        page_num = int(m.group(1))
        suffix_char = m.group(2).lower()
        suffix = "" if suffix_char == "0" else suffix_char
        return f"p{page_num}{suffix}" + split_suffix

    # Standard LOC format: -NNNN[letter] optionally followed by /info.json
    m = re.search(r"-(\d+)([a-zA-Z]?)(?:/info\.json)?$", source_id)
    if m:
        page_key = f"p{int(m.group(1))}{m.group(2)}"
    else:
        # Fall back to the suffix after the last hyphen in the last colon-segment
        # (e.g. "...01790_01N_1950-covr" → "covr", "...01790_01N_1950-ind1" → "ind1").
        last_segment = source_id.removesuffix("/info.json").split(":")[-1]
        page_key = last_segment.rsplit("-", 1)[-1]
    return page_key + split_suffix


def label_to_page_key(label: str) -> str | None:
    """Extract the page key from an OIM IIIF annotation label.

    The page key is the last pipe-separated segment's page identifier, normalized
    to lowercase with bracket variants collapsed to double-underscores:
      "New Orleans, La. | 1951 | Vol. 5 p428 [2]" → "p428__2"
      "New Orleans, La. | 1896 | Vol. 2 p156"     → "p156"
    Returns None if no page identifier is found.
    """
    last_part = label.rsplit("|", 1)[-1].strip()
    m = re.search(r"\b(p\d+[a-z]?)(?:\s*\[(\d+)\])?$", last_part, re.IGNORECASE)
    if m is None:
        return None
    page = m.group(1).lower()
    variant = m.group(2)
    return f"{page}__{variant}" if variant else page


def image_stem(image_path: str) -> str:
    """Return filename stem with ALL extensions stripped.

    Unlike Path.stem (which only strips the last extension), this strips
    everything from the first '.' so multi-extension filenames like
    ``p50n.2048px.jpg`` become ``p50n`` rather than ``p50n.2048px``.
    """
    return Path(image_path).name.split(".")[0]


def default_centerlines(dir_path: Path) -> Path | None:
    """Return the ``centerlines.geojson`` next to the inputs, or None if absent.

    Checks ``dir_path`` then its parent (so it is found whether commands are run on a
    volume directory or on split panels in a subdirectory). Returns None rather than
    exiting so callers can decide whether the file is required.
    """
    for candidate in (
        dir_path / "centerlines.geojson",
        dir_path.parent / "centerlines.geojson",
    ):
        if candidate.exists():
            return candidate
    return None


def list_pages(dir_path: Path) -> list[Path]:
    """Return the effective page images in dir_path, splits superseding their parent.

    Globs top-level ``p*.jpg`` (ignoring the ``raw/`` and ``oim/`` subdirectories). A
    whole-page ``pN.jpg`` is dropped when any of its panels ``pN__*.jpg`` is present, so
    callers operate on the split panels instead. Returns paths sorted by name.
    """
    images = sorted(dir_path.glob("p*.jpg"))
    split_parents = {p.stem.split("__")[0] for p in images if "__" in p.stem}
    return [
        p
        for p in images
        if "__" in p.stem or p.stem.split("__")[0] not in split_parents
    ]
