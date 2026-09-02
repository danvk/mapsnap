"""The per-sheet decision log of the key-map pipeline: ``raw/<stem>.keymap.txt``.

Every stage that decides something about a key-map sheet -- whether its split
stands, whether it is a key map, what its page numbers read as, which
assignments were repaired, what cartouche words it carries, and (soon) which
regions are volume-index insets -- appends a section here, the way the RANSAC
georeferencer's ``<stem>.txt`` collects a page's decisions and the snap
channel appends its own section to it. One section per stage: a rerun
replaces the stage's previous section rather than duplicating it, and the
section header carries the time the decision was made.

The log sits beside the sheet's other key-map sidecars under ``raw/``,
whichever image (scaled ``<volume>/<stem>.jpg`` or full-resolution
``<volume>/raw/<stem>.jpg``) the stage was handed.
"""

from datetime import UTC, datetime
from pathlib import Path

from mapsnap.utils import image_stem

LOG_SUFFIX = ".keymap.txt"


def keymap_log_path(image_path: str | Path) -> Path:
    """``<volume>/raw/<stem>.keymap.txt`` for a key-map sheet's scaled or raw image."""
    image_path = Path(image_path)
    parent = image_path.parent
    raw_dir = parent if parent.name == "raw" else parent / "raw"
    return raw_dir / (image_stem(str(image_path)) + LOG_SUFFIX)


def section_markers(stage: str) -> tuple[str, str]:
    """(begin, end) markers for a stage's section; the begin marker is a prefix
    because the header also carries a timestamp."""
    return f"==== mapsnap {stage} ", f"==== end mapsnap {stage} ===="


def append_keymap_log(image_path: str | Path, stage: str, lines: list[str]) -> Path:
    """Write ``stage``'s section for the sheet, replacing any earlier one; return the path.

    ``lines`` are the decision, one per line, as the stage would print them.
    """
    path = keymap_log_path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    begin, end = section_markers(stage)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    section = "\n".join([f"{begin}({stamp}) ====", *lines, end]) + "\n"
    text = path.read_text() if path.exists() else ""
    if begin in text:
        # Replace the stage's earlier section where it stands, so the log keeps
        # its stage order across reruns.
        head, _, rest = text.partition(begin)
        _, _, tail = rest.partition(end)
        text = head + section + tail.lstrip("\n")
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += section
    path.write_text(text)
    return path


def read_section(image_path: str | Path, stage: str) -> list[str] | None:
    """The lines of ``stage``'s section (without its markers), or None if absent."""
    path = keymap_log_path(image_path)
    if not path.exists():
        return None
    begin, end = section_markers(stage)
    text = path.read_text()
    if begin not in text:
        return None
    _, _, rest = text.partition(begin)
    body, _, _ = rest.partition(end)
    lines = body.split("\n")
    return [line for line in lines[1:] if line != ""]  # drop the header remainder
