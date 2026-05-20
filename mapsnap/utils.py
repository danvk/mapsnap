"""Shared utilities for mapsnap scripts."""

from pathlib import Path


def image_stem(image_path: str) -> str:
    """Return filename stem with ALL extensions stripped.

    Unlike Path.stem (which only strips the last extension), this strips
    everything from the first '.' so multi-extension filenames like
    ``p50n.2048px.jpg`` become ``p50n`` rather than ``p50n.2048px``.
    """
    return Path(image_path).name.split(".")[0]
