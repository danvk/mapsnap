"""Tests for sample_pages.py."""

from pathlib import Path

from sample_pages import Sheet, draw, read_sheets, rejection, source_record

MAPPING = (
    "item\tstate\tyear\tcity\tseq\tstem\tpage_key\tsource\tbytes\tstorage_dir\n"
    "sanborn03297_002\tlouisiana\t1909\tcovington\t1\t03297_1909-0001\tp1\tjp2\t1\tgmd/x\n"
    "sanborn03297_002\tlouisiana\t1909\tcovington\t2\t03297_1909-0002\tp2\tjp2\t1\tgmd/x\n"
    "sanborn00001_001\talabama\t1907\tabbeville\t1\t00001_1907-0001\tp1\tjp2\t1\tgmd/y\n"
)
SHEET = Sheet("sanborn03297_002", "louisiana", "1909", "covington", "p2")


def test_read_sheets_keeps_one_row_per_sheet_and_filters_items(tmp_path: Path):
    path = tmp_path / "mapping.tsv"
    path.write_text(MAPPING)
    assert len(read_sheets(path)) == 3
    sheets = read_sheets(path, {"sanborn03297_002"})
    assert [s.page for s in sheets] == ["p1", "p2"]
    assert sheets[0].location.prefix == "by-state/louisiana/1909/sanborn03297_002"


def test_draw_is_a_seeded_permutation():
    sheets = [Sheet("i", "s", "y", "c", f"p{n}") for n in range(50)]
    assert draw(sheets, 7) == draw(sheets, 7)
    assert draw(sheets, 7) != draw(sheets, 8)
    assert sorted(draw(sheets, 7), key=lambda s: int(s.page[1:])) == sheets


def test_rejection_keeps_whole_mirrored_sheets_of_finished_items():
    finished = ["p2.jpg", "runs/corpus-v1/mapsnap.iiif.json"]
    assert rejection(SHEET, finished, "corpus-v1", []) is None
    assert (
        rejection(SHEET, ["p2.jpg"], "corpus-v1", [])
        == "corpus-v1 did not finish the item"
    )
    assert rejection(SHEET, finished[1:], "corpus-v1", []) == "no mirror image"
    split = [*finished, "runs/corpus-v1/p2.panels.json"]
    assert rejection(SHEET, split, "corpus-v1", []) == "split by corpus-v1"
    assert rejection(SHEET, finished, "corpus-v1", ["p2"]) == "key map"


def test_source_record_ties_the_page_back_to_loc_and_the_mirror():
    metadata = {
        "loc_url": "https://www.loc.gov/item/sanborn03297_002/",
        "storage_dir": "gmd/gmd401m/g4014m/g4014cm/g032971909",
        "sheets": [
            {"key": "p2", "stem": "03297_1909-0002", "width": 1613, "height": 1913}
        ],
    }
    record = source_record(SHEET, metadata, "s3://mapsnap-sanborn")
    assert record["item"] == "sanborn03297_002" and record["page"] == "p2"
    assert record["loc_iiif"].endswith(
        "service:gmd:gmd401m:g4014m:g4014cm:g032971909:03297_1909-0002"
    )
    assert record["mirror"] == (
        "s3://mapsnap-sanborn/by-state/louisiana/1909/sanborn03297_002/p2.jpg"
    )
    assert (record["width"], record["height"]) == (1613, 1913)
    # A sheet the metadata doesn't describe still gets the item's links.
    bare = source_record(SHEET, {}, "s3://mapsnap-sanborn")
    assert bare["loc_url"] == "https://www.loc.gov/item/sanborn03297_002/"
    assert bare["loc_iiif"] is None
