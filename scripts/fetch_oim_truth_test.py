"""Tests for fetch_oim_truth.py."""

import json
from pathlib import Path

from fetch_oim_truth import (
    complete_sanborn_volumes,
    export_canvas_sizes,
    has_key_map,
    ring_origin_repairs,
    split_keys,
)

HEADER = "identifier\tyear\ttitle\tdocument_ct\tregion_ct\tcompletion_pct\n"


def test_complete_sanborn_volumes_keeps_finished_sanborn_rows(tmp_path: Path):
    tsv = tmp_path / "volumes.tsv"
    tsv.write_text(
        HEADER
        + "sanborn03264_001\t1895\tAbbeville\t2\t3\t100\n"
        + "sanborn01778_005\t1909\tChampaign\t19\t31\t48\n"
        + "S7BFS0\t1937\tFHA Block Data Map\t2\t2\t100\n"
    )
    assert [row["identifier"] for row in complete_sanborn_volumes(tsv)] == [
        "sanborn03264_001"
    ]


def test_has_key_map_needs_a_key_map_layer_set_with_layers():
    with_key = '"LAYERSETS": [{"id": "main-content", "layers": [1]}, {"id": "key-map", "layers": [{"id": 5}]}]'
    empty_key = '"LAYERSETS": [{"id": "key-map", "layers": []}]'
    assert has_key_map(f"<script>{with_key}</script>")
    assert not has_key_map(f"<script>{empty_key}</script>")
    assert not has_key_map("<html>no layer sets</html>")


def item(label: str, width: int = 6000, height: int = 7000) -> dict:
    """An export item with a label and a source size."""
    return {"label": label, "target": {"source": {"width": width, "height": height}}}


def test_split_keys_and_canvas_sizes_come_from_the_labels():
    export = {
        "items": [
            item("Fargo, N.D. | 1958 p7 [2]", 6660, 7700),
            item("Fargo, N.D. | 1958 p8"),
            item("Fargo, N.D. | 1958 p20 [1]"),
            item("Fargo, N.D. | 1958 p20 [3]"),
        ]
    }
    assert split_keys(export) == ["p20", "p7"]
    # A split item's source is its parent's full image.
    assert export_canvas_sizes(export)["p7"] == [6660, 7700]
    assert set(export_canvas_sizes(export)) == {"p7", "p8", "p20"}


def split_item(
    label: str, points: list[tuple[int, int]], gcps: list[tuple[int, int]]
) -> dict:
    """A split annotation whose selector is in its crop's frame and GCPs in the parent's."""
    rendered = " ".join(f"{x},{y}" for x, y in points)
    return {
        "label": label,
        "target": {
            "source": {"width": 6000, "height": 7000},
            "selector": {
                "type": "SvgSelector",
                "value": f'<svg><polygon points="{rendered}" /></svg>',
            },
        },
        "body": {
            "features": [
                {
                    "properties": {"resourceCoords": list(gcp)},
                    "geometry": {"coordinates": [-96.8, 46.9]},
                }
                for gcp in gcps
            ]
        },
    }


def test_ring_origin_repairs_shift_a_flagged_selector_to_its_region(tmp_path: Path):
    # Region 2 is the right half of the sheet; the volunteer's mask covers only
    # its lower part, so aligning bounding boxes would not recover the crop offset.
    (tmp_path / "p10.panels.json").write_text(
        json.dumps(
            {
                "panels": [
                    [[0, 0], [3000, 0], [3000, 7000], [0, 7000], [0, 0]],
                    [[3000, 0], [6000, 0], [6000, 7000], [3000, 7000], [3000, 0]],
                ]
            }
        )
    )
    mask = [(100, 4000), (2900, 4000), (2900, 6900), (100, 6900)]
    broken = split_item(
        "Fargo, N.D. | 1958 p10 [2]", mask, [(3500, 4500), (5500, 6500)]
    )
    unflagged = split_item("Fargo, N.D. | 1958 p10 [1]", mask, [(3500, 4500)])
    doc = {"items": [broken, unflagged]}
    log = ring_origin_repairs(doc, tmp_path, {"p10 [2]"})
    assert log == [
        "p10 [2]: shifted by region origin (3000, 0); gcps inside 0% -> 100%"
    ]
    assert "3100.0,4000.0" in broken["target"]["selector"]["value"]
    assert "3100" not in unflagged["target"]["selector"]["value"]


def test_ring_origin_repairs_keep_a_shift_that_does_not_help(tmp_path: Path):
    (tmp_path / "p10.panels.json").write_text(
        json.dumps(
            {
                "panels": [
                    [[0, 0], [10, 0], [10, 10]],
                    [[3000, 0], [6000, 0], [6000, 7000]],
                ]
            }
        )
    )
    # The GCPs already sit inside the unshifted selector.
    fine = split_item(
        "Fargo, N.D. | 1958 p10 [2]", [(0, 0), (500, 0), (500, 500)], [(100, 100)]
    )
    before = fine["target"]["selector"]["value"]
    assert ring_origin_repairs({"items": [fine]}, tmp_path, {"p10 [2]"}) == []
    assert fine["target"]["selector"]["value"] == before
