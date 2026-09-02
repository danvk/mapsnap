"""Tests for the key-map cartouche pass."""

import json
from pathlib import Path

from PIL import Image

from mapsnap.keymap import cartouche
from mapsnap.keymap.cartouche import (
    CARTOUCHE_KIND,
    CARTOUCHE_SPECIFIC,
    CARTOUCHE_VOCAB,
    cartouche_log_lines,
    cartouche_reads,
    cartouche_sidecar_path,
    is_specific,
    write_cartouche_sidecar,
)


def test_every_vocabulary_word_has_a_kind():
    assert set(CARTOUCHE_VOCAB) == set(CARTOUCHE_KIND)
    assert CARTOUCHE_SPECIFIC <= set(CARTOUCHE_VOCAB)


def test_only_inset_titles_are_specific():
    # MAP alone reads the sheet's own "KEY MAP" title; GRAPHIC never appears elsewhere.
    assert (
        is_specific("GRAPHIC") and is_specific("VOLUMES") and is_specific("GRAPHIC MAP")
    )
    assert (
        not is_specific("MAP") and not is_specific("GENERAL") and not is_specific("KEY")
    )


def test_cartouche_sidecar_path_sits_beside_the_image():
    assert cartouche_sidecar_path("data/vol/raw/p0.jpg") == Path(
        "data/vol/raw/p0.cartouche.json"
    )
    assert (
        cartouche_sidecar_path(Path("data/vol/raw/p0__1.jpg")).name
        == "p0__1.cartouche.json"
    )


def test_cartouche_reads_keeps_vocabulary_hits_above_the_floor(monkeypatch):
    def fake_detect_text(image_path, vocab_strings, min_size=15, reader=None, **kwargs):
        assert vocab_strings == CARTOUCHE_VOCAB
        return [
            {
                "text": "VOLUMES",
                "confidence": 0.99,
                "angle": 0,
                "dir_pix": 0.0,
                "long_side": 70,
                "short_side": 20,
                "polygon": [[10, 10], [80, 10], [80, 30], [10, 30]],
            },
            {
                "text": "GENERAL",
                "confidence": 0.77,
                "angle": 0,
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            },
            {
                "text": "KEY",
                "confidence": 0.12,
                "angle": 0,
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            },
            {
                "text": "MAIN",
                "confidence": 0.95,
                "angle": 0,
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            },
        ]

    monkeypatch.setattr(cartouche, "detect_text", fake_detect_text)
    reads = cartouche_reads("any.jpg")
    assert [(r["text"], r["kind"], r["specific"]) for r in reads] == [
        ("VOLUMES", "volumes", True),
        ("GENERAL", "volumes", False),
    ]
    # detect_text's own fields ride along, so the file stays a streets.json.
    assert reads[0]["dir_pix"] == 0.0 and reads[0]["long_side"] == 70
    assert reads[0]["polygon"] == [
        [10.0, 10.0],
        [80.0, 10.0],
        [80.0, 30.0],
        [10.0, 30.0],
    ]
    # The floor is a parameter: lower it and the weak KEY read comes back last.
    assert [r["text"] for r in cartouche_reads("any.jpg", min_confidence=0.1)] == [
        "VOLUMES",
        "GENERAL",
        "KEY",
    ]


def test_write_cartouche_sidecar_is_a_streets_json(tmp_path: Path):
    """The debugger classifies a dropped file by content: an object with a
    `streets` list whose entries carry `confidence` loads like any page's
    reads, so the sidecar uses exactly that shape (as <stem>.keymap.json does)."""
    image = tmp_path / "p0.jpg"
    Image.new("RGB", (40, 30)).save(image)
    reads = [
        {
            "text": "KEY",
            "kind": "legend",
            "specific": False,
            "confidence": 0.9,
            "angle": 0,
            "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
        }
    ]
    path = write_cartouche_sidecar(image, reads)
    assert path == tmp_path / "p0.cartouche.json"
    data = json.loads(path.read_text())
    assert (data["width"], data["height"]) == (40, 30)
    assert data["streets"] == reads
    assert "timestamp" in data and isinstance(data["command"], list)
    assert "reads" not in data


def test_cartouche_log_lines_mark_specific_words():
    reads = [
        {
            "text": "GRAPHIC",
            "kind": "volumes",
            "confidence": 0.99,
            "polygon": [[0, 0], [10, 0], [10, 4], [0, 4]],
        },
        {
            "text": "MAP",
            "kind": "volumes",
            "confidence": 0.95,
            "polygon": [[20, 20], [30, 20], [30, 24], [20, 24]],
        },
    ]
    assert cartouche_log_lines(reads) == [
        "2 cartouche read(s):",
        "  GRAPHIC @0.99 (volumes, specific) at (5, 2)",
        "  MAP @0.95 (volumes, weak) at (25, 22)",
    ]
    assert cartouche_log_lines([]) == ["no cartouche words read"]
