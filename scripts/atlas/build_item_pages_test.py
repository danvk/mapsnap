import json
from pathlib import Path

from build_item_pages import (
    INDEX_COLUMNS,
    Fit,
    RunInfo,
    cached_read,
    catalogue_fields,
    fits_by_sheet,
    index_row,
    item_record,
    loc_resource,
    loc_source_url,
    manifest,
    page_notes,
    panel_shapes,
    prefetch,
    read_catalogue,
    read_fits,
    read_item_list,
    read_mirror_metadata,
    read_pages,
    read_run_panels,
    run_tag,
    service_id,
    split_sheets,
    thumbnail_size,
)
from shapely.geometry import Point, Polygon

STORAGE = "gmd/gmd397m/g3974m/g3974am/g3974am_g000021926"
SERVICE = "service:gmd:gmd397m:g3974m:g3974am:g3974am_g000021926"
CDN = f"https://cdn.chronoscope.io/mapsnap/{SERVICE}"


def sheet(seq: int, key: str, source: str = "torrent-jp2") -> dict:
    """A mirror metadata.json sheet entry."""
    return {
        "seq": seq,
        "stem": f"00002_1926-{seq:04d}",
        "key": key,
        "source": source,
        "storage_dir": STORAGE,
        "width": 1613,
        "height": 1913,
    }


MIRROR = {
    "item": "sanborn00002_001",
    "year": "1926",
    "sheets": [sheet(2, "p2"), sheet(1, "p1")],
}
CATALOGUE_RECORD = {
    "Id": "http://www.loc.gov/item/sanborn00002_001/",
    "Title": "Sanborn Fire Insurance Map from Alabama City, Etowah County, Alabama.",
    "Date": "1926-01",
    "City_text": ["Alabama City"],
    "County_text": ["Etowah County"],
    "State_text": ["Alabama"],
    "Notes": ["Jan 1926.", "9 sheet(s)."],
    "Repository": "Library of Congress",
    "IIIF_manifest": "http://www.loc.gov/item/sanborn00002_001/manifest.json",
    "Location": [{"Coordinates": [34.02, -86.04]}],
}
ROW = {
    "item": "sanborn00002_001",
    "city": "Alabama City",
    "state": "AL",
    "year": "1926",
    "date": "1926-01",
    "volume": "",
    "title": "Sanborn Fire Insurance Map from Alabama City, Etowah County, Alabama.",
    "sheets": "2",
    "scans": "3",
    "images": "3",
    "placed": "2",
    "published": "1",
    "sheets_placed": "1",
    "status": "published",
    "main": "sanborn00002_001.main.iiif.json",
    "keymap": "sanborn00002_001.keymap.iiif.json",
}


def annotation_page(labels: list[str], metadata: list[dict] | None = None) -> dict:
    """A published AnnotationPage whose items carry these labels."""
    return {
        "type": "AnnotationPage",
        "metadata": metadata or [],
        "items": [
            {
                "id": f"{CDN}:{label.split()[-1]}/georef",
                "type": "Annotation",
                "label": label,
            }
            for label in labels
        ],
    }


def run_pages() -> dict[str, dict]:
    """A run's pages for an item with a key map (p1) and a split p2 whose second panel was placed."""
    main = annotation_page(
        ["Alabama City, Alabama | 1926 | sanborn00002_001 p2 [2]"],
        [
            {"label": "run", "value": "corpus-v1"},
            {
                "label": "unplaced pages",
                "value": "p2__1 (tier 1, 33.99,-86.02 +/-179m)",
            },
            {
                "label": "withheld",
                "value": "p1 (the key map, published in the key-map file)",
            },
        ],
    )
    keymap = annotation_page(["Alabama City, Alabama | 1926 | sanborn00002_001 p1"])
    return {"main": main, "keymap": keymap}


def write_run(run_dir: Path) -> None:
    """A run dump holding run_pages()."""
    iiif = run_dir / "iiif/chronoscope"
    iiif.mkdir(parents=True)
    for column, page in run_pages().items():
        (iiif / ROW[column]).write_text(json.dumps(page))


class FakeReader:
    """An S3Reader over a dict of objects, counting reads."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.reads: list[str] = []

    def connect(self) -> None:
        pass

    def read(self, s3_url: str) -> bytes | None:
        self.reads.append(s3_url)
        return self.objects.get(s3_url)


def test_service_id():
    assert service_id(STORAGE, "00002_1926-0001") == f"{SERVICE}:00002_1926-0001"


def test_thumbnail_size_rounds_up():
    assert thumbnail_size(1613, 1913) == (404, 479)
    assert thumbnail_size(1600, 2000) == (400, 500)


def test_loc_resource():
    assert loc_resource(STORAGE) == "g3974am.g000021926"
    assert loc_resource("gmd/gmd401m/g4014m/g4014cm/g032971909") == "g4014cm.g032971909"


def test_loc_source_url():
    base = "https://tile.loc.gov/storage-services"
    assert (
        loc_source_url(sheet(1, "p1"))
        == f"{base}/service/{STORAGE}/00002_1926-0001.jp2"
    )
    tif = sheet(1, "p0a", source="torrent-master-tif")
    assert loc_source_url(tif) == f"{base}/master/{STORAGE}/00002_1926-0001.tif"


def test_page_notes():
    page = annotation_page(
        [],
        [
            {
                "label": "withheld",
                "value": "p3 (a skeleton sheet; its full-color sheet is published), "
                "p4__2 (12.4 km from the catalogue location)",
            }
        ],
    )
    assert page_notes(page, "withheld") == [
        ("p3", "a skeleton sheet; its full-color sheet is published"),
        ("p4__2", "12.4 km from the catalogue location"),
    ]
    assert page_notes(page, "unplaced pages") == []


def test_read_pages(tmp_path: Path):
    write_run(tmp_path)
    assert read_pages(tmp_path, ROW) == run_pages()
    assert list(read_pages(tmp_path, {**ROW, "keymap": ""})) == ["main"]


def test_run_tag():
    assert run_tag(run_pages()) == "corpus-v1"
    assert run_tag({}) == ""


def test_read_fits():
    fits = read_fits(run_pages())
    # The key map's "withheld" entry is dropped: the key-map file places it.
    assert sorted((fit.image, fit.status, fit.file) for fit in fits) == [
        ("p1", "key map", "keymap"),
        ("p2__1", "unplaced", ""),
        ("p2__2", "placed", "main"),
    ]
    placed = next(fit for fit in fits if fit.status == "placed")
    # The annotation's id, which Allmaps takes as the map id.
    assert placed.map == run_pages()["main"]["items"][0]["id"]


def test_read_fits_keeps_a_withheld_page():
    page = annotation_page(
        [], [{"label": "withheld", "value": "p5 (8.0 km from the catalogue location)"}]
    )
    fits = read_fits({"main": page})
    assert fits == [Fit("p5", "withheld", "8.0 km from the catalogue location")]


def test_fit_as_json_leaves_out_empty_fields():
    assert Fit("p1", "unplaced").as_json() == {"image": "p1", "status": "unplaced"}
    assert Fit("p1", "placed", map="m", file="main").as_json() == {
        "image": "p1",
        "status": "placed",
        "map": "m",
        "file": "main",
    }


def test_panel_shapes_label_inside_each_panel():
    # An L-shaped panel, whose centroid falls outside it, and the square it wraps.
    ell = [[0, 0], [100, 0], [100, 20], [20, 20], [20, 100], [0, 100], [0, 0]]
    square = [[20, 20], [100, 20], [100, 100], [20, 100], [20, 20]]
    shapes = panel_shapes({"panels": [ell, square]})
    assert [shape["points"] for shape in shapes] == [ell, square]
    for ring, shape in zip([ell, square], shapes, strict=True):
        assert Polygon(ring).contains(Point(shape["label"]))


def test_cached_read(tmp_path: Path):
    reader = FakeReader({"s3://b/a.json": b"{}"})
    assert cached_read(reader, "s3://b/a.json", tmp_path / "a.json") == b"{}"
    assert cached_read(reader, "s3://b/a.json", tmp_path / "a.json") == b"{}"
    # A missing object is remembered, so it isn't asked for again either.
    assert cached_read(reader, "s3://b/none.json", tmp_path / "none.json") is None
    assert cached_read(reader, "s3://b/none.json", tmp_path / "none.json") is None
    assert reader.reads == ["s3://b/a.json", "s3://b/none.json"]


def test_prefetch_fills_only_missing_copies(tmp_path: Path):
    reader = FakeReader({"s3://b/a.json": b"a", "s3://b/b.json": b"b"})
    (tmp_path / "a.json").write_bytes(b"cached")
    sources = [
        ("s3://b/a.json", tmp_path / "a.json"),
        ("s3://b/b.json", tmp_path / "b.json"),
        ("s3://b/none.json", tmp_path / "none.json"),
    ]
    prefetch(reader, sources)
    assert sorted(reader.reads) == ["s3://b/b.json", "s3://b/none.json"]
    assert (tmp_path / "b.json").read_bytes() == b"b"
    assert (tmp_path / "none.json").read_bytes() == b""
    assert not list(tmp_path.glob("*.partial"))


def test_split_sheets():
    fits = [Fit("p2__2", "placed"), Fit("p2__1", "unplaced"), Fit("p1", "key map")]
    assert split_sheets(fits) == ["p2"]


def test_read_run_panels(tmp_path: Path):
    prefix = "s3://mapsnap-sanborn/by-state/alabama/1926/sanborn00002_001"
    square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    reader = FakeReader(
        {
            f"{prefix}/runs/corpus-v1/p2.panels.json": json.dumps(
                {"panels": [square, square]}
            ).encode()
        }
    )
    panels = read_run_panels(
        reader, prefix, "corpus-v1", read_fits(run_pages()), tmp_path
    )
    # Only p2 is split; p1, the key map, isn't fetched.
    assert list(panels) == ["p2"]
    assert len(panels["p2"]) == 2
    assert reader.reads == [f"{prefix}/runs/corpus-v1/p2.panels.json"]


def test_read_mirror_metadata(tmp_path: Path):
    prefix = "s3://mapsnap-sanborn/by-state/alabama/1926/sanborn00002_001"
    reader = FakeReader({f"{prefix}/metadata.json": json.dumps(MIRROR).encode()})
    assert read_mirror_metadata(reader, prefix, tmp_path) == MIRROR
    assert (tmp_path / "sanborn00002_001.json").exists()
    assert read_mirror_metadata(reader, f"{prefix}x", tmp_path) is None


def test_fits_by_sheet():
    grouped = fits_by_sheet(
        [Fit("p2__2", "placed"), Fit("p2__1", "unplaced"), Fit("p1", "key map")]
    )
    assert [fit.image for fit in grouped["p2"]] == ["p2__1", "p2__2"]
    assert [fit.image for fit in grouped["p1"]] == ["p1"]


def test_catalogue_fields():
    fields = catalogue_fields(CATALOGUE_RECORD)
    assert fields["city"] == "Alabama City"
    assert fields["coordinates"] == {"lat": 34.02, "lon": -86.04}
    assert catalogue_fields({})["coordinates"] is None


def test_read_catalogue(tmp_path: Path):
    path = tmp_path / "metadata.jsonl"
    other = {**CATALOGUE_RECORD, "Id": "http://www.loc.gov/item/sanborn00009_001/"}
    path.write_text(json.dumps(CATALOGUE_RECORD) + "\n" + json.dumps(other) + "\n")
    assert list(read_catalogue(path, {"sanborn00002_001"})) == ["sanborn00002_001"]


def test_item_record(tmp_path: Path):
    square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    run = RunInfo("v1.3", ROW, read_fits(run_pages()))
    run.panels = {"p2": panel_shapes({"panels": [square, square]})}
    record = item_record(
        "sanborn00002_001", catalogue_fields(CATALOGUE_RECORD), MIRROR, run
    )
    assert record["name"] == "Alabama City, Etowah County, Alabama"
    assert [s["key"] for s in record["sheets"]] == ["p1", "p2"]  # LoC's order
    first = record["sheets"][0]
    assert first["image"] == f"{CDN}:00002_1926-0001"
    assert (
        first["thumbnail"]["url"] == f"{CDN}:00002_1926-0001/full/404,479/0/default.jpg"
    )
    assert first["loc_page"] == "https://www.loc.gov/resource/g3974am.g000021926/?sp=1"
    assert first["fit"] == [
        {
            "image": "p1",
            "status": "key map",
            "map": f"{CDN}:p1/georef",
            "file": "keymap",
        }
    ]
    assert "panels" not in first
    assert len(record["sheets"][1]["panels"]) == 2
    assert [fit["status"] for fit in record["sheets"][1]["fit"]] == [
        "unplaced",
        "placed",
    ]
    assert record["run"]["sheets_placed"] == 1
    assert record["run"]["main"]["chronoscope"] == (
        "https://data.mapsnap.org/runs/v1.3/iiif/chronoscope/sanborn00002_001.main.iiif.json"
    )
    assert record["run"]["main"]["allmaps_chronoscope"].startswith(
        "https://viewer.allmaps.org/?url="
    )


def test_item_record_without_catalogue_or_run():
    record = item_record(
        "2010593241", {}, {"year": "1867", "sheets": [sheet(1, "p1")]}, None
    )
    assert record["name"] == "2010593241"
    assert record["year"] == "1867"
    assert record["sheets"][0]["fit"] == []
    assert "run" not in record


def test_manifest():
    record = item_record(
        "sanborn00002_001", catalogue_fields(CATALOGUE_RECORD), MIRROR, None
    )
    doc = manifest(record)
    assert doc["id"] == "https://data.mapsnap.org/manifests/sanborn00002_001.json"
    assert len(doc["items"]) == 2
    canvas = doc["items"][0]
    body = canvas["items"][0]["items"][0]["body"]
    assert (canvas["width"], canvas["height"]) == (1613, 1913)
    assert body["service"][0] == {
        "id": f"{CDN}:00002_1926-0001",
        "type": "ImageService3",
        "profile": "level0",
    }
    assert doc["homepage"][0]["id"] == "https://mapsnap.org/item?id=sanborn00002_001"
    assert {entry["label"]["en"][0] for entry in doc["metadata"]} == {
        "Date",
        "Location",
        "Notes",
        "Repository",
    }


def test_index_row():
    run = RunInfo("v1.3", ROW, read_fits(run_pages()))
    record = item_record(
        "sanborn00002_001", catalogue_fields(CATALOGUE_RECORD), MIRROR, run
    )
    row = dict(zip(INDEX_COLUMNS, index_row(record), strict=True))
    assert row == {
        "id": "sanborn00002_001",
        "city": "Alabama City",
        "state": "AL",
        "year": "1926",
        "date": "1926-01",
        "volume": "",
        "sheets": 2,
        "status": "published",
    }


def test_read_item_list(tmp_path: Path):
    path = tmp_path / "items.txt"
    path.write_text("# pilot\nsanborn00002_001\n\nsanborn00009_001 extra words\n")
    assert read_item_list(path) == ["sanborn00002_001", "sanborn00009_001"]
