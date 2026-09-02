"""Tests for the key-map cartouche pass."""

import json
from pathlib import Path

from PIL import Image

from mapsnap.keymap import cartouche
from mapsnap.keymap.cartouche import (
    CARTOUCHE_KIND,
    CARTOUCHE_SPECIFIC,
    CARTOUCHE_VOCAB,
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
    assert [(r["text"], r["kind"]) for r in reads] == [
        ("VOLUMES", "volumes"),
        ("GENERAL", "volumes"),
    ]
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


def test_write_cartouche_sidecar_records_image_size(tmp_path: Path):
    image = tmp_path / "p0.jpg"
    Image.new("RGB", (40, 30)).save(image)
    reads = [
        {
            "text": "KEY",
            "kind": "legend",
            "confidence": 0.9,
            "angle": 0,
            "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]],
        }
    ]
    path = write_cartouche_sidecar(image, reads)
    assert path == tmp_path / "p0.cartouche.json"
    data = json.loads(path.read_text())
    assert (data["image"], data["width"], data["height"]) == ("p0.jpg", 40, 30)
    assert data["reads"] == reads
