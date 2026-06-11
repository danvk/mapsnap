"""Shared utilities for mapsnap scripts."""

import re
import struct
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    """Print and run a subprocess command, exiting with its return code on failure."""
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


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
      "https://...1950-0006N/info.json", ""     → "p6N"
      "https://...1951-0425/info.json",  ""     → "p425"
    Labels ending with "[N]" produce a split suffix:
      "https://...1950-0156/info.json", "... [2]" → "p156__2"
    """
    split_suffix = ""
    if label.endswith("]"):
        m = re.search(r"\[(\d+)\]$", label)
        assert m
        split_suffix = f"__{m.group(1)}"
    m = re.search(r"-(\d+)([a-zA-Z]?)(?:/info\.json)?$", source_id)
    if m:
        page_key = f"p{int(m.group(1))}{m.group(2)}"
    else:
        page_key = source_id.split("/")[-2] or source_id
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
