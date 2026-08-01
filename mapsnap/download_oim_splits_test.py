import json

from mapsnap.download_oim_splits import slug_candidates, split_pages


def test_split_pages_groups_by_parent(tmp_path):
    doc = {
        "items": [
            {"label": "Columbus, Ohio | 1951 | Vol. 3 p248 [1]"},
            {"label": "Columbus, Ohio | 1951 | Vol. 3 p248 [2]"},
            {"label": "Columbus, Ohio | 1951 | Vol. 3 p201 [1]"},
            {"label": "Columbus, Ohio | 1951 | Vol. 3 p200"},
        ]
    }
    path = tmp_path / "main.iiif.json"
    path.write_text(json.dumps(doc))
    assert split_pages(path) == {"p201": [1], "p248": [1, 2]}


def test_slug_candidates_expand_state_codes():
    cands = slug_candidates("columbus_oh_1951_vol_3")
    assert cands[0] == "columbus_oh_1951_vol_3"
    assert "columbus_ohio_1951_vol_3" in cands
    # No expansion available -> just the name itself.
    assert slug_candidates("champaign_ill_1915") == ["champaign_ill_1915"]
