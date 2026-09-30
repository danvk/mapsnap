import zipfile

from build_run_site import (
    Site,
    Totals,
    allmaps_url,
    display_title,
    percent,
    read_items,
    readme,
    run_index,
    state_page,
    totals_by_state,
    volume_sort_key,
    write_zip,
)

HEADER = [
    "item",
    "city",
    "state",
    "year",
    "date",
    "volume",
    "title",
    "sheets",
    "images",
    "placed",
    "published",
    "sheets_placed",
    "dropped_over_6km",
    "dropped_over_5km_from_location",
    "status",
    "main",
    "keymap",
]
SITE = Site("v1.3", "https://data.mapsnap.org/runs/v1.3")


def row(item: str, **values: str) -> dict[str, str]:
    """An items.tsv row for a published Brooklyn volume, with any cells replaced."""
    cells = {
        "item": item,
        "city": "Brooklyn",
        "state": "NY",
        "year": "1939",
        "date": "1939",
        "volume": "",
        "title": "Sanborn Fire Insurance Map from Brooklyn, Kings County, New York.",
        "sheets": "10",
        "images": "12",
        "placed": "9",
        "published": "9",
        "sheets_placed": "8",
        "dropped_over_6km": "0",
        "dropped_over_5km_from_location": "0",
        "status": "published",
        "main": f"{item}.main.iiif.json",
        "keymap": "",
    }
    return cells | values


def test_read_items_reads_a_tsv(tmp_path):
    path = tmp_path / "items.tsv"
    path.write_text("\t".join(HEADER) + "\n" + "\t".join(row("x").values()) + "\n")
    assert read_items(path) == [row("x")]


def test_display_title_drops_the_boilerplate_and_adds_the_volume():
    assert display_title(row("x")) == "Brooklyn, Kings County, New York"
    assert (
        display_title(row("x", volume="2"))
        == "Brooklyn, Kings County, New York, vol. 2"
    )
    assert display_title(row("x", title="")) == "Brooklyn"


def test_allmaps_url_encodes_the_file_url():
    assert allmaps_url("https://data.mapsnap.org/a b.json") == (
        "https://viewer.allmaps.org/?url=https%3A%2F%2Fdata.mapsnap.org%2Fa%20b.json"
    )


def test_percent_rounds_and_handles_nothing():
    assert percent(1, 3) == "33%"
    assert percent(0, 0) == "–"


def test_totals_count_sheets_of_every_volume_but_placed_of_published_ones():
    totals = Totals()
    totals.add(row("a", keymap="a.keymap.iiif.json"))
    totals.add(row("b", status="no page placed", sheets_placed="0", main=""))
    assert totals.volumes == 2
    assert totals.published == 1
    assert totals.sheets == 20
    assert totals.sheets_placed == 8
    assert totals.keymaps == 1
    assert totals.towns == {("NY", "Brooklyn")}
    assert totals.years == {1939}


def test_totals_by_state_skips_rows_without_a_state():
    totals = totals_by_state([row("a"), row("b", state="NJ"), row("c", state="")])
    assert sorted(totals) == ["NJ", "NY"]


def test_volume_sort_key_orders_volumes_numerically():
    rows = [row("c", volume="10"), row("a", volume="2"), row("b", volume="")]
    assert [r["item"] for r in sorted(rows, key=volume_sort_key)] == ["b", "a", "c"]


def test_site_links_every_source_and_allmaps():
    links = SITE.file_links("x.main.iiif.json")
    assert (
        'href="https://data.mapsnap.org/runs/v1.3/iiif/loc/x.main.iiif.json"' in links
    )
    assert (
        'href="https://data.mapsnap.org/runs/v1.3/iiif/chronoscope/x.main.iiif.json"'
        in links
    )
    assert links.count("viewer.allmaps.org") == 2
    assert SITE.file_links("") == '<span class="none">—</span>'


def test_run_index_links_states_downloads_and_the_example(tmp_path):
    rows = [
        row("sanborn06116_046", volume="2"),
        row("b", state="NJ", status="withheld: key map under 2.5 km", main=""),
    ]
    html = run_index(rows, SITE, tmp_path)
    assert 'href="/runs/v1.3/states/ny"' in html
    assert 'href="/runs/v1.3/states/nj"' in html
    assert "https://data.mapsnap.org/runs/v1.3/mapsnap-v1.3-chronoscope.zip" in html
    assert "https://data.mapsnap.org/runs/v1.3/items.tsv" in html
    assert (
        "viewer.allmaps.org/?url=https%3A%2F%2Fdata.mapsnap.org%2Fruns%2Fv1.3%2Fiiif%2F"
        "chronoscope%2Fsanborn06116_046.main.iiif.json"
    ) in html
    assert 'href="mailto:danvdk+mapsnap@gmail.com"' in html
    assert 'href="https://chronoscope.io/"' in html
    assert 'href="https://iiif.io/api/extension/georef/"' in html
    assert 'href="https://github.com/danvk/mapsnap/issues/541"' in html
    assert 'href="https://oldinsurancemaps.net/"' in html


def test_state_page_lists_every_volume_with_its_files():
    rows = [
        row("a", keymap="a.keymap.iiif.json"),
        row("b", status="no page placed", main="", sheets_placed="0"),
    ]
    html = state_page("NY", rows, SITE)
    assert "<h1>New York</h1>" in html
    assert 'href="https://www.loc.gov/item/a/"' in html
    assert "iiif/chronoscope/a.keymap.iiif.json" in html
    assert '<tr data-item="b" class="withheld">' in html
    assert "no page placed" in html
    # No id column: the id is only in the row's data, for the filter.
    assert "<th>ID</th>" not in html and "<code>a</code>" not in html
    assert "row.dataset.item" in html
    assert "1 of 2 volumes published" in html
    assert 'href="https://github.com/danvk/mapsnap/issues/541"' in html
    assert 'href="https://oldinsurancemaps.net/"' in html


def test_write_zip_holds_one_sources_files_with_the_tsv_and_readme(tmp_path):
    (tmp_path / "iiif/loc").mkdir(parents=True)
    (tmp_path / "iiif/chronoscope").mkdir(parents=True)
    (tmp_path / "iiif/loc/a.main.iiif.json").write_text("{}")
    (tmp_path / "iiif/chronoscope/a.main.iiif.json").write_text("{}")
    (tmp_path / "items.tsv").write_text("item\n")
    path = write_zip(tmp_path, SITE, "loc")
    assert path == tmp_path / "mapsnap-v1.3-loc.zip"
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == [
            "mapsnap-v1.3-loc/README.txt",
            "mapsnap-v1.3-loc/iiif/loc/a.main.iiif.json",
            "mapsnap-v1.3-loc/items.tsv",
        ]


def test_readme_states_the_license():
    text = readme(SITE)
    assert "Open Database License" in text
    assert "https://data.mapsnap.org/runs/v1.3" in text


def test_a_key_map_only_volume_counts_as_published_and_links_its_key_map():
    only = row(
        "k",
        status="published: key map only",
        main="",
        keymap="k.keymap.iiif.json",
        sheets_placed="0",
    )
    totals = Totals()
    totals.add(only)
    assert totals.published == 1
    html = state_page("NY", [only], SITE)
    assert '<tr class="withheld">' not in html
    assert "iiif/loc/k.keymap.iiif.json" in html
