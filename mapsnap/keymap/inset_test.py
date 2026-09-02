"""Tests for the volume-index inset detector."""

import json
from pathlib import Path

from shapely.geometry import Point

from mapsnap.keymap.inset import (
    Inset,
    annotate_keymap_reads,
    cluster_reads,
    detect_insets,
    inset_rings,
    inset_sidecar_path,
    is_small,
    log_lines,
    median_spacing,
    write_inset_sidecar,
)


def _record(text: str, x: float, y: float, size: float = 30.0) -> dict:
    return {
        "text": text,
        "confidence": 0.95,
        "angle": 0,
        "polygon": [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
    }


def _sheet(
    tmp_path: Path, reads: list[dict], cartouche: list[dict] | None = None
) -> Path:
    raw = tmp_path / "vol" / "raw"
    raw.mkdir(parents=True)
    image = raw / "p0.jpg"
    image.touch()
    (raw / "p0.keymap.json").write_text(
        json.dumps({"width": 4000, "height": 5000, "streets": reads})
    )
    if cartouche is not None:
        (raw / "p0.cartouche.json").write_text(
            json.dumps({"width": 4000, "height": 5000, "streets": cartouche})
        )
    return image


def _grid(
    n_cols: int, n_rows: int, x0: float, y0: float, step: float, start: int = 300
):
    """A main key map: page numbers on a regular grid."""
    return [
        _record(str(start + r * n_cols + c), x0 + c * step, y0 + r * step)
        for r in range(n_rows)
        for c in range(n_cols)
    ]


def test_is_small_is_a_volume_index_range():
    assert is_small("1") and is_small("15")
    assert (
        not is_small("0")
        and not is_small("16")
        and not is_small("4A")
        and not is_small("311")
    )


def test_cluster_reads_single_linkage_largest_first():
    centers = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (500.0, 500.0), (505.0, 500.0)]
    assert cluster_reads(centers, 15) == [[0, 1, 2], [3, 4]]
    assert median_spacing(centers) == 10.0


def test_richmond_shaped_inset_is_found_with_its_snapped_read(tmp_path: Path):
    """An isolated cluster reading `2 3 4 311` (the inset's "1" snapped to a valid
    page) is an inset: small majority, far from the grid, GENERAL INDEX cartouche."""
    main = _grid(6, 6, 1500, 800, 300)  # 36 pages, 300 px apart
    inset = [
        _record("2", 300, 4300),
        _record("3", 450, 4200),
        _record("4", 250, 4450),
        _record("311", 400, 4500),
    ]
    cartouche = [
        {
            "text": "VOLUMES",
            "confidence": 0.9,
            "polygon": [[300, 3950], [500, 3950], [500, 4000], [300, 4000]],
        },
        {
            "text": "KEY",
            "confidence": 1.0,
            "polygon": [[3000, 100], [3200, 100], [3200, 150], [3000, 150]],
        },
    ]
    image = _sheet(tmp_path, main + inset, cartouche)
    result = detect_insets(image)
    assert [c.texts for c in result.candidates] == [["2", "3", "4", "311"]]
    assert result.candidates[0].cartouche == ["VOLUMES"]
    assert len(result.insets) == 1 and result.insets[0].confirmed
    assert not result.separate_keymaps
    path = write_inset_sidecar(image, result)
    assert (
        path
        == inset_sidecar_path(image)
        == tmp_path / "vol" / "raw" / "p0.inset.panels.json"
    )
    data = json.loads(path.read_text())
    assert (data["width"], data["height"], len(data["panels"])) == (4000, 5000, 1)
    assert data["labels"] == ["volumes inset: 2, 3, 4, 311 (cartouche VOLUMES)"]
    # The mask covers the inset's reads and none of the key map's.
    (ring,) = inset_rings(image)
    assert all(ring.contains(Point(r["polygon"][0])) for r in inset)
    assert not any(ring.contains(Point(r["polygon"][0])) for r in main)


def test_cartouche_title_counts_within_a_few_spacings_of_the_cluster(tmp_path: Path):
    """The title is printed above or beside the inset map, not among its numerals."""
    main = _grid(6, 6, 1500, 800, 300)  # spacing 300
    inset = [
        _record("2", 300, 4300),
        _record("3", 450, 4200),
        _record("4", 250, 4450),
        _record("1", 400, 4500),
    ]

    def title_at(y: float) -> list[dict]:
        return [
            {
                "text": "GRAPHIC",
                "confidence": 0.9,
                "polygon": [[300, y], [500, y], [500, y + 40], [300, y + 40]],
            }
        ]

    near = _sheet(tmp_path / "near", inset + main, title_at(3500))  # ~2 spacings above
    (tmp_path / "near").mkdir(exist_ok=True)
    assert detect_insets(near).candidates[0].cartouche == ["GRAPHIC"]
    far = _sheet(tmp_path / "far", inset + main, title_at(2300))  # ~6 spacings above
    assert detect_insets(far).candidates[0].cartouche == []


def test_unconfirmed_cluster_is_reported_but_not_masked(tmp_path: Path):
    main = _grid(6, 6, 1500, 800, 300)
    junk = [_record("5", 300, 4300), _record("7", 450, 4400)]
    image = _sheet(tmp_path, main + junk)  # no cartouche sidecar, no re-read
    result = detect_insets(image)
    assert len(result.candidates) == 1 and result.insets == []
    assert "unconfirmed (not masked)" in "\n".join(log_lines(result))
    assert (
        inset_rings(image) == []
        or write_inset_sidecar(image, result)
        and inset_rings(image) == []
    )


def test_ellenville_shaped_sheet_is_separate_keymaps_not_an_inset(tmp_path: Path):
    """Two real key maps of small page numbers, neither dominating: nothing is masked."""
    village = [
        _record(str(n), 500 + (n % 4) * 300, 500 + (n // 4) * 300, size=40)
        for n in range(2, 14)
    ]
    napanoch = [
        _record(str(n), 500 + (n % 3) * 300, 3500 + (n // 3) * 300, size=40)
        for n in range(11, 16)
    ]
    image = _sheet(tmp_path, village + napanoch)
    result = detect_insets(image)
    assert result.candidates == [] and result.insets == []
    assert result.separate_keymaps
    assert "separate key maps" in "\n".join(log_lines(result))


def test_a_lone_far_read_is_never_an_inset(tmp_path: Path):
    main = _grid(6, 6, 1500, 800, 300)
    image = _sheet(tmp_path, main + [_record("9", 200, 4600)])
    assert detect_insets(image).candidates == []


def test_inset_label_reports_the_reread():
    inset = Inset(
        indices=[0, 1],
        texts=["3", "4"],
        n_small=2,
        ring=[[0, 0], [1, 0], [1, 1]],
        reread=(2, 2),
    )
    assert inset.confirmed
    assert inset.label() == "volumes inset: 3, 4 (re-read small 2/2)"
    unconfirmed = Inset(
        indices=[0, 1],
        texts=["3", "4"],
        n_small=2,
        ring=[[0, 0], [1, 0], [1, 1]],
        reread=(0, 2),
    )
    assert not unconfirmed.confirmed and unconfirmed.label().endswith(
        "(re-read small 0/2)"
    )


def test_annotate_keymap_reads_flags_masked_reads_and_clears_stale_flags(
    tmp_path: Path,
):
    main = _grid(6, 6, 1500, 800, 300)
    inset = [
        _record("2", 300, 4300),
        _record("3", 450, 4200),
        _record("4", 250, 4450),
        _record("311", 400, 4500),
    ]
    cartouche = [
        {
            "text": "VOLUMES",
            "confidence": 0.9,
            "polygon": [[300, 3950], [500, 3950], [500, 4000], [300, 4000]],
        }
    ]
    image = _sheet(tmp_path, main + inset, cartouche)
    result = detect_insets(image)
    assert annotate_keymap_reads(image, result) == 4
    doc = json.loads((image.parent / "p0.keymap.json").read_text())
    flagged = sorted(r["text"] for r in doc["streets"] if r.get("inset"))
    assert flagged == ["2", "3", "311", "4"]
    assert all(
        "inset" not in r
        for r in doc["streets"]
        if r["text"].isdigit() and int(r["text"]) >= 300 and r["text"] != "311"
    )
    # The detector re-judges from ALL reads, so a run that confirms nothing clears the flags.
    (image.parent / "p0.cartouche.json").unlink()
    result = detect_insets(image)
    assert result.insets == [] and annotate_keymap_reads(image, result) == 0
    doc = json.loads((image.parent / "p0.keymap.json").read_text())
    assert not any(r.get("inset") for r in doc["streets"])
