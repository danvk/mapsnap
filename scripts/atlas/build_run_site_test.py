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
    volumes_json,
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


def test_site_links_view_and_json_on_the_default_source():
    links = SITE.file_links("x.main.iiif.json")
    assert (
        'href="https://data.mapsnap.org/runs/v1.3/iiif/chronoscope/x.main.iiif.json"'
        in links
    )
    assert "iiif/loc/" not in links
    assert links.count("viewer.allmaps.org") == 1
    assert ">View</a>" in links and ">JSON</a>" in links
    # The toggle rebuilds both links from the file's name.
    assert links.count('data-name="x.main.iiif.json"') == 2
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
    assert 'href="/faq"' in html and "OSMUS Slack" in html
    assert "try the loc.gov link" in html
    assert 'href="https://chronoscope.io/"' in html
    assert 'href="https://iiif.io/api/extension/georef/"' in html
    assert 'href="https://github.com/danvk/mapsnap/issues/541"' in html
    assert 'href="https://oldinsurancemaps.net/"' in html
    # The item viewer has nothing to do with the run, so the run page doesn't describe it.
    assert "Browse a volume" not in html
    # Background sections come after the states, before the items.tsv reference.
    order = [
        html.index(f"<h2>{h}</h2>")
        for h in (
            "Find a volume",
            "Download",
            "States",
            "The files",
            "Limitations",
            "OldInsuranceMaps.net",
            "items.tsv",
        )
    ]
    assert order == sorted(order)


def test_state_page_lists_every_volume_with_its_files():
    rows = [
        row("a", keymap="a.keymap.iiif.json"),
        row("b", status="no page placed", main="", sheets_placed="0"),
    ]
    html = state_page("NY", rows, SITE)
    assert "<h1>New York</h1>" in html
    # Titles open the item viewer, not loc.gov.
    assert 'href="/item?id=a"' in html
    assert "loc.gov/item/a/" not in html
    assert "iiif/chronoscope/a.keymap.iiif.json" in html
    assert '<tr data-item="b" class="withheld">' in html
    assert "no page placed" in html
    # No id column: the id is only in the row's data, for the filter.
    assert "<th>ID</th>" not in html and "<code>a</code>" not in html
    assert "row.dataset.item" in html
    assert "switch Images to loc.gov" in html
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
    assert "iiif/chronoscope/k.keymap.iiif.json" in html


def test_state_page_has_one_sticky_source_toggle():
    html = state_page("NY", [row("a", keymap="a.keymap.iiif.json")], SITE)
    assert '<div class="controls">' in html
    assert 'value="chronoscope" checked' in html
    assert 'value="loc">' in html
    # No per-row source labels: one toggle switches every link.
    assert '<span class="src">' not in html
    assert 'const IIIF = "https://data.mapsnap.org/runs/v1.3/iiif/"' in html
    assert "localStorage" in html


def test_volumes_json_is_one_compact_row_per_volume():
    rows = [
        row("b", state="NJ", status="no page placed", main="", sheets_placed="0"),
        row("a", keymap="a.keymap.iiif.json"),
    ]
    volumes = volumes_json(rows)
    # Sorted by state, then as the state pages sort.
    assert [v[0] for v in volumes] == ["b", "a"]
    a = volumes[1]
    assert (a[0], a[4], a[7], a[8], a[9]) == ("a", "NY", "published", 1, 1)
    assert isinstance(a[5], int) and isinstance(a[6], int)
    assert volumes[0][7:] == ["no page placed", 0, 0]


def test_run_index_searches_every_volume(tmp_path):
    html = run_index([row("sanborn06116_046", volume="2")], SITE, tmp_path)
    assert "<h2>Find a volume</h2>" in html
    assert '<div class="controls">' in html
    assert 'fetch("/runs/v1.3/volumes.json")' in html
    assert '"NY": "New York"' in html
    assert "fileHref(name, view)" in html
    # Result rows get the copy button too.
    assert "copyButton(item)" in html and "navigator.clipboard.writeText" in html


def test_state_page_titles_have_a_copy_id_button():
    html = state_page("NY", [row("sanborn06116_046")], SITE)
    assert 'data-copy="sanborn06116_046"' in html
    assert 'title="Copy the volume ID (sanborn06116_046)"' in html
    assert "navigator.clipboard.writeText" in html
    # The checkmark shown after a copy is in the script, not the page's rows.
    assert html.count("M20 6 9 17l-5-5") == 1
