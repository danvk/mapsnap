"""Build a published run's web pages, and its zip files, from its items.tsv.

    uv run python scripts/atlas/build_run_site.py \\
        --run-dir ~/Documents/mapsnap/data.mapsnap.org/runs/v1.3 \\
        --site-dir ~/Documents/mapsnap/mapsnap.org \\
        --version v1.3 --data-url https://data.mapsnap.org/runs/v1.3 --zip

``--run-dir`` is what publish_run.py wrote (items.tsv and iiif/), which is served
from ``--data-url``. This writes, under ``--site-dir``:

- ``runs/<version>/index.html``: the run's summary, downloads, an example, and a
  table of states;
- ``runs/<version>/states/<xx>.html``: one page per state, keyed by postal code,
  listing every volume with links to its IIIF files and to view them in Allmaps.

The pages use the site's shared ``/style.css``. With ``--zip``, it also writes one
zip per image source into ``--run-dir`` (``mapsnap-<version>-<source>.zip``), each
holding that source's IIIF files, items.tsv and README.txt.
"""

import argparse
import csv
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from urllib.parse import quote

from gazetteer import STATE_CODES

SOURCES = {"loc": "loc.gov", "chronoscope": "Chronoscope"}
ALLMAPS_VIEWER = "https://viewer.allmaps.org/?url="
LOC_ITEM = "https://www.loc.gov/item/"
REPO = "https://github.com/danvk/mapsnap"
# Where readers report a volume that is misplaced or missing pages.
PROBLEMS_URL = "https://github.com/danvk/mapsnap/issues/541"
GEOREF_SPEC = "https://iiif.io/api/extension/georef/"
CHRONOSCOPE = '<a href="https://chronoscope.io/">Chronoscope</a>'
OLD_INSURANCE_MAPS = '<a href="https://oldinsurancemaps.net/">OldInsuranceMaps.net</a>'
STATE_NAMES = {code: name for name, code in STATE_CODES.items()}
TITLE_PREFIX = "Sanborn Fire Insurance Map from "
# The example volume on the run page: Brooklyn, 1939, vol. 2.
EXAMPLE_ITEM = "sanborn05791_054"

LICENSE_TEXT = (
    "The georeferencing annotations are © OpenStreetMap contributors and mapsnap, "
    "available under the Open Database License 1.0 "
    "(https://opendatacommons.org/licenses/odbl/1-0/). Their control points are "
    "OpenStreetMap positions. Individual contents are under the Database Contents "
    "License (https://opendatacommons.org/licenses/dbcl/1-0/). The sheet images are "
    "the Library of Congress's Sanborn Maps Collection "
    "(https://www.loc.gov/collections/sanborn-maps/)."
)

LICENSE_HTML = (
    "The georeferencing annotations are © "
    '<a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> '
    "and mapsnap, available under the "
    '<a href="https://opendatacommons.org/licenses/odbl/1-0/">Open Database License 1.0</a>: '
    "their control points are OpenStreetMap positions. Individual contents are under the "
    '<a href="https://opendatacommons.org/licenses/dbcl/1-0/">Database Contents License</a>. '
    "The sheet images are the Library of Congress's "
    '<a href="https://www.loc.gov/collections/sanborn-maps/">Sanborn Maps Collection</a>.'
)

# items.tsv's columns, as the run page and README.txt document them.
COLUMNS = [
    (
        "item",
        "The Library of Congress item id; its page is https://www.loc.gov/item/<item>/.",
    ),
    (
        "city, state, year, date",
        "The town, its postal code, and the volume's year and catalogue date.",
    ),
    ("volume", "The volume number, where the catalogue gives one."),
    ("title", "The catalogue title."),
    ("sheets", "Scanned sheets in the volume."),
    ("images", "Images the run split the sheets into (split sheets become several)."),
    ("placed", "Images the run georeferenced."),
    ("published", "Images published: those placed, less the ones dropped below."),
    ("sheets_placed", "Sheets with at least one published image."),
    (
        "dropped_over_6km",
        "Images dropped because their sheet would span over 6 km of ground.",
    ),
    (
        "dropped_over_5km_from_location",
        (
            "Images dropped because, in a volume of 10 sheets or fewer, they sit over "
            "5 km from the catalogue's location for the town."
        ),
    ),
    ("status", "published, no page placed, or withheld and why."),
    (
        "main, keymap",
        "The volume's IIIF file names, under iiif/loc/ and iiif/chronoscope/; blank if none.",
    ),
]


def read_items(path: Path) -> list[dict[str, str]]:
    """The rows of a run's items.tsv."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_int(value: str) -> int:
    """A TSV cell as an integer, blank as 0."""
    return int(value) if value.strip() else 0


def display_title(row: dict[str, str]) -> str:
    """A volume's title without the catalogue's boilerplate, with its volume number."""
    title = row["title"].removeprefix(TITLE_PREFIX).rstrip(".") or row["city"]
    return f"{title}, vol. {row['volume']}" if row["volume"] else title


def allmaps_url(file_url: str) -> str:
    """The Allmaps viewer, opened on one IIIF file."""
    return ALLMAPS_VIEWER + quote(file_url, safe="")


def number(value: int) -> str:
    """An integer with thousands separators."""
    return f"{value:,}"


def percent(part: int, whole: int) -> str:
    """part/whole as a whole-number percentage."""
    return f"{round(100 * part / whole)}%" if whole else "–"


@dataclass
class Totals:
    """Counts over a set of items.tsv rows."""

    volumes: int = 0
    published: int = 0
    sheets: int = 0
    sheets_placed: int = 0
    images_published: int = 0
    keymaps: int = 0
    towns: set[tuple[str, str]] = field(default_factory=set)
    years: set[int] = field(default_factory=set)

    def add(self, row: dict[str, str]) -> None:
        """Count one volume."""
        self.volumes += 1
        self.sheets += as_int(row["sheets"])
        if row["status"] == "published":
            self.published += 1
            self.sheets_placed += as_int(row["sheets_placed"])
            self.images_published += as_int(row["published"])
            self.keymaps += bool(row["keymap"])
            self.towns.add((row["state"], row["city"]))
            if row["year"].isdigit():
                self.years.add(int(row["year"]))


def totals_by_state(rows: list[dict[str, str]]) -> dict[str, Totals]:
    """Postal code -> that state's totals; rows without a state are left out."""
    states: dict[str, Totals] = defaultdict(Totals)
    for row in rows:
        if row["state"]:
            states[row["state"]].add(row)
    return dict(states)


def volume_sort_key(row: dict[str, str]) -> tuple:
    """Town, then year, then volume number, then item id."""
    volume = row["volume"]
    digits = "".join(ch for ch in volume if ch.isdigit())
    return (row["city"], row["year"], int(digits) if digits else 0, volume, row["item"])


@dataclass
class Site:
    """Where a run's files are served, and how its pages are named."""

    version: str
    data_url: str

    def page_path(self) -> str:
        """The site path of the run's own page; state pages sit under it."""
        return f"/runs/{self.version}/"

    def file_url(self, source: str, name: str) -> str:
        """The URL of one published IIIF file."""
        return f"{self.data_url.rstrip('/')}/iiif/{source}/{name}"

    def file_links(self, name: str) -> str:
        """Download and Allmaps links for one IIIF file, from each image source."""
        if not name:
            return '<span class="none">—</span>'
        parts = []
        for source, label in SOURCES.items():
            url = self.file_url(source, name)
            parts.append(
                f'<span class="src">{label}: <a href="{escape(url)}">JSON</a> · '
                f'<a href="{escape(allmaps_url(url))}">Allmaps</a></span>'
            )
        return "".join(parts)


def page(title: str, body: str) -> str:
    """A complete page around ``body``, with the site's header and footer."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<link rel="stylesheet" href="/style.css">
</head>
<body>
<header class="site"><a href="/">mapsnap</a></header>
<main>
{body}
</main>
<footer class="site">Georeferencing annotations © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a> and mapsnap, under the <a href="https://opendatacommons.org/licenses/odbl/1-0/">ODbL</a>. Sheet images: <a href="https://www.loc.gov/collections/sanborn-maps/">Library of Congress, Sanborn Maps Collection</a>.</footer>
</body>
</html>
"""


def zip_name(version: str, source: str) -> str:
    """The file name of one image source's zip."""
    return f"mapsnap-{version}-{source}.zip"


def file_size(path: Path) -> str:
    """A file's size in MB, or a dash if it is missing."""
    return f"{path.stat().st_size / 1e6:,.0f} MB" if path.exists() else "–"


def run_index(rows: list[dict[str, str]], site: Site, run_dir: Path) -> str:
    """The run page: summary, downloads, example, file layout, states, license."""
    totals = Totals()
    for row in rows:
        totals.add(row)
    years = f"{min(totals.years)}–{max(totals.years)}" if totals.years else "–"

    stats = [
        (
            number(totals.published),
            f"volumes published, of {number(totals.volumes)} in the collection",
        ),
        (
            number(totals.sheets_placed),
            f"sheets placed, of {number(totals.sheets)} ({percent(totals.sheets_placed, totals.sheets)})",
        ),
        (
            number(totals.images_published),
            "georeferenced images (a split sheet is several)",
        ),
        (number(totals.keymaps), "key maps"),
        (number(len(totals.towns)), "towns"),
        (years, "years"),
    ]
    stat_html = "\n".join(
        f'<div class="stat"><b>{value}</b><span>{escape(label)}</span></div>'
        for value, label in stats
    )

    data = site.data_url.rstrip("/")
    downloads = "\n".join(
        f'<tr><td><a href="{data}/{zip_name(site.version, source)}">{zip_name(site.version, source)}</a></td>'
        f"<td>Every IIIF file, pointing at {label}, with items.tsv and README.txt</td>"
        f'<td class="num">{file_size(run_dir / zip_name(site.version, source))}</td></tr>'
        for source, label in SOURCES.items()
    )
    downloads += (
        f'\n<tr><td><a href="{data}/items.tsv">items.tsv</a></td>'
        "<td>One row per volume, published or not</td>"
        f'<td class="num">{file_size(run_dir / "items.tsv")}</td></tr>'
    )

    example = next((row for row in rows if row["item"] == EXAMPLE_ITEM), None)
    example_html = ""
    if example and example["main"]:
        url = site.file_url("chronoscope", example["main"])
        example_html = f"""<h2>Try one</h2>
<p><a href="{escape(allmaps_url(url))}">{escape(display_title(example))}, {escape(example["year"])}</a>
opens in the Allmaps viewer: {example["sheets_placed"]} georeferenced sheets over today's map, drawn from {CHRONOSCOPE}'s copies of the scans. The file behind it is
<a href="{escape(url)}"><code>{escape(url)}</code></a>.</p>"""

    state_rows = []
    for code, state in sorted(
        totals_by_state(rows).items(),
        key=lambda item: STATE_NAMES.get(item[0], item[0]),
    ):
        name = STATE_NAMES.get(code, code)
        state_rows.append(
            f'<tr><td><a href="{site.page_path()}states/{code.lower()}">{escape(name)}</a></td>'
            f'<td class="num">{number(state.published)}</td><td class="num">{number(state.volumes)}</td>'
            f'<td class="num">{number(state.sheets_placed)}</td><td class="num">{number(state.sheets)}</td>'
            f'<td class="num">{percent(state.sheets_placed, state.sheets)}</td></tr>'
        )

    column_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{escape(text)}</td></tr>"
        for name, text in COLUMNS
    )

    body = f"""<h1>Run {escape(site.version)}</h1>
<p class="lede">Georeferencing for the Library of Congress's <a href="https://www.loc.gov/collections/sanborn-maps/">Sanborn Maps Collection</a>, made automatically by <a href="{REPO}">mapsnap</a>: each placed sheet is a <a href="{GEOREF_SPEC}">IIIF Georeference Annotation</a> you can open in <a href="https://allmaps.org/">Allmaps</a> or any viewer that reads them.</p>
<div class="stats">
{stat_html}
</div>

{example_html}

<h2>Download</h2>
<table class="downloads">
{downloads}
</table>

<h2>The files</h2>
<p>Each published volume has a IIIF AnnotationPage of its main content, <code>&lt;item&gt;.main.iiif.json</code>, and of its key map when one was placed, <code>&lt;item&gt;.keymap.iiif.json</code>. Every file comes in two versions that differ only in where the scans are fetched from:</p>
<ul>
<li><code>{data}/iiif/loc/</code> points at the Library of Congress's own image servers, the original scans.</li>
<li><code>{data}/iiif/chronoscope/</code> points at {CHRONOSCOPE}'s copies of the same scans at a quarter of their size, which load much faster and are kinder to loc.gov.</li>
</ul>
<p>Each file's <code>id</code> is its own URL. Every image is a georeference annotation: control points, a transformation, and a clip outline for the part of the sheet that is map.</p>

<h2>Limitations</h2>
<p>mapsnap can't place every page, and not every page it places is accurate: a sheet can land on the wrong block, or in the wrong town. If you find a volume that's wrong, please report it on <a href="{PROBLEMS_URL}">GitHub</a>.</p>

<h2>OldInsuranceMaps.net</h2>
<p>mapsnap was built with data from {OLD_INSURANCE_MAPS}, where volunteers have georeferenced Sanborn maps by hand; their work is what mapsnap was developed and measured against. If a volume you want isn't listed here, or is missing pages, look for it on OldInsuranceMaps.net, or georeference it by hand there.</p>

<h2>States</h2>
<table class="states">
<thead><tr><th>State</th><th class="num">Volumes published</th><th class="num">of</th><th class="num">Sheets placed</th><th class="num">of</th><th class="num">%</th></tr></thead>
<tbody>
{chr(10).join(state_rows)}
</tbody>
</table>

<h2>items.tsv</h2>
<table class="columns">
{column_rows}
</table>

<h2>License</h2>
<p>{LICENSE_HTML}</p>
"""
    return page(f"mapsnap run {site.version}", body)


def state_page(code: str, rows: list[dict[str, str]], site: Site) -> str:
    """One state's page: every volume, with links to its IIIF files."""
    name = STATE_NAMES.get(code, code)
    totals = Totals()
    for row in rows:
        totals.add(row)
    body_rows = []
    for row in sorted(rows, key=volume_sort_key):
        published = row["status"] == "published"
        if published:
            files = f'<td class="files">{site.file_links(row["main"])}</td><td class="files">{site.file_links(row["keymap"])}</td>'
        else:
            files = f'<td class="files status" colspan="2">{escape(row["status"])}</td>'
        body_rows.append(
            f"<tr{'' if published else ' class="withheld"'}>"
            f'<td><a href="{LOC_ITEM}{escape(row["item"])}/"><code>{escape(row["item"])}</code></a></td>'
            f"<td>{escape(display_title(row))}</td><td>{escape(row['year'])}</td><td>{escape(row['city'])}</td>"
            f'<td class="num">{escape(row["sheets"])}</td><td class="num">{escape(row["sheets_placed"])}</td>'
            f"{files}</tr>"
        )
    body = f"""<p class="crumbs"><a href="{site.page_path()}">Run {escape(site.version)}</a> › {escape(name)}</p>
<h1>{escape(name)}</h1>
<p class="lede">{number(totals.published)} of {number(totals.volumes)} volumes published, with {number(totals.sheets_placed)} of {number(totals.sheets)} sheets placed. <b>JSON</b> is the IIIF file; <b>Allmaps</b> opens it over today's map. loc.gov files draw the Library of Congress's scans, Chronoscope files its faster copies of them.</p>
<p>Not every page is placed, or placed accurately: please <a href="{PROBLEMS_URL}">report problems</a>. Missing a volume, or pages of one? Look for it on {OLD_INSURANCE_MAPS}, or georeference it by hand there.</p>
<p><input type="search" id="filter" placeholder="Filter by town, year or id" aria-label="Filter volumes"> <span id="count"></span></p>
<div class="table-scroll">
<table class="volumes">
<thead><tr><th>ID</th><th>Title</th><th>Year</th><th>Location</th><th class="num">Sheets</th><th class="num">Placed</th><th>Main content</th><th>Key map</th></tr></thead>
<tbody>
{chr(10).join(body_rows)}
</tbody>
</table>
</div>
<script>
const input = document.getElementById("filter");
const count = document.getElementById("count");
const rows = [...document.querySelectorAll("table.volumes tbody tr")];
function update() {{
  const terms = input.value.toLowerCase().split(/\\s+/).filter(Boolean);
  let shown = 0;
  for (const row of rows) {{
    const text = row.textContent.toLowerCase();
    const visible = terms.every((term) => text.includes(term));
    row.hidden = !visible;
    shown += visible;
  }}
  count.textContent = terms.length ? `${{shown}} of ${{rows.length}} volumes` : "";
}}
input.addEventListener("input", update);
</script>
"""
    return page(f"{name} — mapsnap run {site.version}", body)


def readme(site: Site) -> str:
    """README.txt for the zips: what the files are, and their license."""
    columns = "\n".join(f"  {name}: {text}" for name, text in COLUMNS)
    return f"""mapsnap run {site.version}

Georeferencing for the Library of Congress's Sanborn Maps Collection, made
automatically by mapsnap ({REPO}). Each published volume has a IIIF
AnnotationPage of its main content (iiif/<source>/<item>.main.iiif.json) and,
when one was placed, of its key map (<item>.keymap.iiif.json). Files under
iiif/loc/ point at loc.gov's scans; files under iiif/chronoscope/ point at
Chronoscope's quarter-size copies of them. Each file's id is its URL under
{site.data_url}.

items.tsv has one row per volume, published or not:
{columns}

{LICENSE_TEXT}
"""


def write_zip(run_dir: Path, site: Site, source: str) -> Path:
    """One image source's IIIF files, items.tsv and README.txt, as a zip in ``run_dir``."""
    path = run_dir / zip_name(site.version, source)
    root = f"mapsnap-{site.version}-{source}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(f"{root}/README.txt", readme(site))
        archive.write(run_dir / "items.tsv", f"{root}/items.tsv")
        for file in sorted((run_dir / "iiif" / source).glob("*.json")):
            archive.write(file, f"{root}/iiif/{source}/{file.name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="publish_run.py's output"
    )
    parser.add_argument(
        "--site-dir", type=Path, required=True, help="The website's root directory"
    )
    parser.add_argument(
        "--version", required=True, help="The run's name in URLs, e.g. v1.3"
    )
    parser.add_argument(
        "--data-url", required=True, help="The URL --run-dir is served at"
    )
    parser.add_argument("--zip", action="store_true", help="Also (re)build the zips")
    args = parser.parse_args()

    site = Site(args.version, args.data_url)
    rows = read_items(args.run_dir / "items.tsv")
    if args.zip:
        for source in SOURCES:
            print(f"wrote {write_zip(args.run_dir, site, source)}")
        (args.run_dir / "README.txt").write_text(readme(site))

    out = args.site_dir / "runs" / args.version
    (out / "states").mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(run_index(rows, site, args.run_dir))
    by_state: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["state"]:
            by_state[row["state"]].append(row)
    for code, state_rows in by_state.items():
        (out / "states" / f"{code.lower()}.html").write_text(
            state_page(code, state_rows, site)
        )
    print(f"wrote {out / 'index.html'} and {len(by_state)} state pages")


if __name__ == "__main__":
    main()
