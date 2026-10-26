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
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from urllib.parse import quote

from gazetteer import STATE_CODES

SOURCES = {"loc": "loc.gov", "chronoscope": "Chronoscope"}
# The image source a state page's links start on; its toggle switches them all,
# and the viewer's choice is remembered in the browser.
DEFAULT_SOURCE = "chronoscope"
# mapsnap.org's item viewer: a volume's catalog record and every scan, whether or
# not mapsnap placed any of it.
ITEM_PAGE = "/item?id="
# How many of the run page's search matches are shown at a time.
RESULTS_PAGE = 200
# The copy-ID button's icons (Lucide's clipboard and check), drawn in currentColor.
COPY_ICON = (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="8" y="2" width="8" height="4" rx="1"/>'
    '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>'
)
CHECK_ICON = (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
    'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M20 6 9 17l-5-5"/></svg>'
)
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
# The example volume on the run page: New York, 1923 (lower Manhattan).
EXAMPLE_ITEM = "sanborn06116_046"
CONTACT_EMAIL = "danvdk+mapsnap@gmail.com"
FAQ_PATH = "/faq"
OSMUS_SLACK = "https://slack.openstreetmap.us/"
CONTACT_HTML = (
    f'<p class="contact">Report issues on <a href="{REPO}/issues">GitHub</a>. '
    f'See the <a href="{FAQ_PATH}">FAQ</a>. For other questions, '
    f'<a href="mailto:{CONTACT_EMAIL}">contact Dan</a> or join the #mapsnap channel on '
    f'<a href="{OSMUS_SLACK}">OSMUS Slack</a>.</p>'
)
# Allmaps sizes its GPU textures by the largest tile an image declares, and the
# Chronoscope copies declare the whole image as one tile, so a big volume can
# exhaust a phone's memory where loc.gov's 512 px tiles don't.
MOBILE_NOTE = (
    "If a Chronoscope Allmaps link doesn't load on mobile, try the loc.gov link "
    "or a desktop browser."
)
# The state pages' version: their links follow the Images toggle.
STATE_MOBILE_NOTE = (
    "If a volume doesn't load on your phone, switch Images to loc.gov, or try a "
    "desktop browser."
)

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
        "The town, its postal code, and the volume's year and catalog date.",
    ),
    ("volume", "The volume number, where the catalog gives one."),
    ("title", "The catalog title."),
    (
        "sheets",
        (
            "Map sheets in the volume. A skeleton sheet (pNs) and its full-color "
            "sheet (pN) count once, since at most one of them is published; title "
            "and index pages are left out."
        ),
    ),
    ("scans", "Every image the Library of Congress catalogs for the volume."),
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
            "5 km from the catalog's location for the town."
        ),
    ),
    (
        "dropped_keymap",
        (
            "Images dropped because they are the volume's key map, which is "
            "published in its own file."
        ),
    ),
    (
        "status",
        "published, published: key map only, no page placed, or withheld and why.",
    ),
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
    """A volume's title without the catalog's boilerplate, with its volume number."""
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
        if row["status"].startswith("published"):
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
        """View (Allmaps) and JSON links for one IIIF file, on DEFAULT_SOURCE.

        Both carry the file's name, which is the same under every source, so the
        state page's Images toggle can point them at another source's copy.
        """
        if not name:
            return '<span class="none">—</span>'
        url = self.file_url(DEFAULT_SOURCE, name)
        return (
            f'<a class="file view" data-name="{escape(name)}" '
            f'href="{escape(allmaps_url(url))}">View</a> · '
            f'<a class="file json" data-name="{escape(name)}" href="{escape(url)}">JSON</a>'
        )


def controls_html(placeholder: str) -> str:
    """The sticky bar of a filter box and the Images toggle, shared by run and state pages."""
    radios = "\n".join(
        f'<label><input type="radio" name="source" value="{source}"'
        f"{' checked' if source == DEFAULT_SOURCE else ''}> {label}</label>"
        for source, label in sorted(
            SOURCES.items(), key=lambda kv: kv[0] != DEFAULT_SOURCE
        )
    )
    return f"""<div class="controls">
<input type="search" id="filter" placeholder="{escape(placeholder)}" aria-label="Filter volumes">
<span class="source" role="radiogroup" aria-label="Images">Images:
{radios}
</span>
<span id="count"></span>
</div>"""


def source_script(site: "Site") -> str:
    """Script for the Images toggle: points every a.file link at the chosen source.

    Links are looked up when the source changes, so rows a page adds later
    (the run page's search results) follow the toggle too; they build their
    own hrefs with fileHref().
    """
    return f"""const IIIF = "{site.data_url.rstrip("/")}/iiif/";
const ALLMAPS = "{ALLMAPS_VIEWER}";
const DEFAULT_SOURCE = "{DEFAULT_SOURCE}";
let source = DEFAULT_SOURCE;
// The URL of one IIIF file on the current source, or its Allmaps view.
function fileHref(name, view) {{
  const url = IIIF + source + "/" + name;
  return view ? ALLMAPS + encodeURIComponent(url) : url;
}}
function setSource(value) {{
  source = value;
  for (const link of document.querySelectorAll("a.file")) {{
    link.href = fileHref(link.dataset.name, link.classList.contains("view"));
  }}
}}
for (const radio of document.querySelectorAll('input[name="source"]')) {{
  radio.addEventListener("change", () => {{
    setSource(radio.value);
    try {{ localStorage.setItem("mapsnap-source", radio.value); }} catch (error) {{}}
  }});
}}
let saved = null;
try {{ saved = localStorage.getItem("mapsnap-source"); }} catch (error) {{}}
if (saved && saved !== DEFAULT_SOURCE) {{
  const radio = document.querySelector(`input[name="source"][value="${{saved}}"]`);
  if (radio) {{
    radio.checked = true;
    setSource(saved);
  }}
}}"""


def find_script(site: "Site") -> str:
    """Script for the run page's search: fetch volumes.json once, filter it, show matches.

    Every term must appear in the volume's id, title, year, town, state (code or
    name) or status. Matches are shown RESULTS_PAGE at a time.
    """
    state_names = json.dumps(STATE_NAMES, sort_keys=True)
    return f"""const STATE_NAMES = {state_names};
const RESULTS_PAGE = {RESULTS_PAGE};
const input = document.getElementById("filter");
const count = document.getElementById("count");
const table = document.getElementById("results");
const body = table.querySelector("tbody");
const more = document.getElementById("more");
let volumes = null;
let matches = [];
let shown = 0;
// The run's volumes, fetched on first use, each with the text its terms match.
async function load() {{
  if (!volumes) {{
    const response = await fetch("{site.page_path()}volumes.json");
    volumes = (await response.json()).map((v) => ({{
      v,
      text: [v[0], v[1], v[2], v[3], v[4], STATE_NAMES[v[4]] || "", v[7]].join(" ").toLowerCase(),
    }}));
  }}
  return volumes;
}}
function cell(text, className) {{
  const td = document.createElement("td");
  if (className) td.className = className;
  td.textContent = text;
  return td;
}}
// A View · JSON pair for one of the volume's files, or a dash.
function fileCell(item, kind, has) {{
  const td = document.createElement("td");
  td.className = "files";
  if (!has) {{
    td.innerHTML = '<span class="none">—</span>';
    return td;
  }}
  const name = `${{item}}.${{kind}}.iiif.json`;
  for (const [label, view] of [["View", true], ["JSON", false]]) {{
    const link = document.createElement("a");
    link.className = view ? "file view" : "file json";
    link.dataset.name = name;
    link.href = fileHref(name, view);
    link.textContent = label;
    td.append(link);
    if (view) td.append(" · ");
  }}
  return td;
}}
function row(v) {{
  const [item, title, year, city, state, sheets, placed, status, main, keymap] = v;
  const published = status.startsWith("published");
  const tr = document.createElement("tr");
  if (!published) tr.className = "withheld";
  const titleCell = document.createElement("td");
  const titleLink = document.createElement("a");
  titleLink.href = "{ITEM_PAGE}" + encodeURIComponent(item);
  titleLink.textContent = title;
  titleCell.append(titleLink, copyButton(item));
  const stateCell = document.createElement("td");
  const stateLink = document.createElement("a");
  stateLink.href = "{site.page_path()}states/" + state.toLowerCase();
  stateLink.textContent = state;
  stateCell.append(stateLink);
  tr.append(titleCell, cell(year), cell(city), stateCell, cell(sheets, "num"), cell(placed, "num"));
  if (published) {{
    tr.append(fileCell(item, "main", main), fileCell(item, "keymap", keymap));
  }} else {{
    const td = cell(status, "files status");
    td.colSpan = 2;
    tr.append(td);
  }}
  return tr;
}}
function showMore() {{
  const next = matches.slice(shown, shown + RESULTS_PAGE);
  body.append(...next.map((m) => row(m.v)));
  shown += next.length;
  more.hidden = shown >= matches.length;
  more.textContent = `Show ${{Math.min(RESULTS_PAGE, matches.length - shown)}} more`;
}}
async function update() {{
  const terms = input.value.toLowerCase().split(/\\s+/).filter(Boolean);
  body.replaceChildren();
  shown = 0;
  if (!terms.length) {{
    table.hidden = true;
    more.hidden = true;
    count.textContent = "";
    return;
  }}
  let all;
  try {{
    all = await load();
  }} catch (error) {{
    count.textContent = "Couldn't load the list of volumes.";
    return;
  }}
  // A newer keystroke may have finished first.
  if (input.value.toLowerCase().split(/\\s+/).filter(Boolean).join(" ") !== terms.join(" ")) return;
  matches = all.filter((m) => terms.every((term) => m.text.includes(term)));
  table.hidden = !matches.length;
  count.textContent = `${{matches.length.toLocaleString()}} of ${{all.length.toLocaleString()}} volumes`;
  showMore();
}}
let timer = null;
input.addEventListener("input", () => {{
  clearTimeout(timer);
  timer = setTimeout(update, 150);
}});
more.addEventListener("click", showMore);"""


def copy_button(item: str) -> str:
    """A button beside a volume's title that copies its id; shown when its row is hovered."""
    return (
        f'<button type="button" class="copy" data-copy="{escape(item)}" '
        f'title="Copy the volume ID ({escape(item)})" '
        f'aria-label="Copy the volume ID {escape(item)}">{COPY_ICON}</button>'
    )


def copy_script() -> str:
    """Script for every copy button on the page, including rows added later.

    Copies the id, then shows a checkmark for a moment.
    """
    return f"""const COPY_ICON = {json.dumps(COPY_ICON)};
const CHECK_ICON = {json.dumps(CHECK_ICON)};
// A copy button for rows the page builds itself.
function copyButton(item) {{
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy";
  button.dataset.copy = item;
  button.title = `Copy the volume ID (${{item}})`;
  button.setAttribute("aria-label", `Copy the volume ID ${{item}}`);
  button.innerHTML = COPY_ICON;
  return button;
}}
document.addEventListener("click", async (event) => {{
  const button = event.target.closest("button.copy");
  if (!button) return;
  const item = button.dataset.copy;
  try {{
    await navigator.clipboard.writeText(item);
  }} catch (error) {{
    button.title = "Your browser didn't allow copying";
    return;
  }}
  button.innerHTML = CHECK_ICON;
  button.classList.add("copied");
  button.title = `Copied ${{item}}`;
  clearTimeout(button.copiedTimer);
  button.copiedTimer = setTimeout(() => {{
    button.innerHTML = COPY_ICON;
    button.classList.remove("copied");
    button.title = `Copy the volume ID (${{item}})`;
  }}, 1500);
}});"""


def volumes_json(rows: list[dict[str, str]]) -> list[list]:
    """Every volume as [item, title, year, town, state, sheets, placed, status, main, key map].

    The run page's search reads this: the last two are 1 when the volume has that
    IIIF file, whose name follows from the item (<item>.main.iiif.json).
    """
    return [
        [
            row["item"],
            display_title(row),
            row["year"],
            row["city"],
            row["state"],
            as_int(row["sheets"]),
            as_int(row["sheets_placed"]),
            row["status"],
            int(bool(row["main"])),
            int(bool(row["keymap"])),
        ]
        for row in sorted(rows, key=lambda r: (r["state"],) + volume_sort_key(r))
    ]


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
        (number(len(totals.towns)), "cities and towns"),
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

<h2>Find a volume</h2>
<p>Search every volume in the run by town, state, year or Library of Congress id. A title opens the volume's page; <b>View</b> opens its placed sheets over today's map in Allmaps, and <b>JSON</b> is the IIIF file behind them.</p>
{controls_html("Find a volume by town, state, year or id")}
<div class="table-scroll">
<table class="volumes" id="results" hidden>
<thead><tr><th>Title</th><th>Year</th><th>Location</th><th>State</th><th class="num">Sheets</th><th class="num">Placed</th><th>Main content</th><th>Key map</th></tr></thead>
<tbody></tbody>
</table>
</div>
<p><button type="button" id="more" hidden>Show more</button></p>
<script>
{source_script(site)}
{copy_script()}
{find_script(site)}
</script>

<h2>Download</h2>
<table class="downloads">
{downloads}
</table>

<h2>States</h2>
<table class="states">
<thead><tr><th>State</th><th class="num">Volumes published</th><th class="num">of</th><th class="num">Sheets placed</th><th class="num">of</th><th class="num">%</th></tr></thead>
<tbody>
{chr(10).join(state_rows)}
</tbody>
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
<p>{MOBILE_NOTE}</p>

<h2>OldInsuranceMaps.net</h2>
<p>mapsnap was built with data from {OLD_INSURANCE_MAPS}, where volunteers have georeferenced Sanborn maps by hand; their work is what mapsnap was developed and measured against. If a volume you want isn't listed here, or is missing pages, look for it on OldInsuranceMaps.net, or georeference it by hand there.</p>

<h2>items.tsv</h2>
<table class="columns">
{column_rows}
</table>

<h2>License</h2>
<p>{LICENSE_HTML}</p>

{CONTACT_HTML}
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
        published = row["status"].startswith("published")
        if published:
            files = f'<td class="files">{site.file_links(row["main"])}</td><td class="files">{site.file_links(row["keymap"])}</td>'
        else:
            files = f'<td class="files status" colspan="2">{escape(row["status"])}</td>'
        # The item id isn't shown, but the filter still finds it, and the title
        # links to the volume's page in mapsnap.org's item viewer.
        body_rows.append(
            f'<tr data-item="{escape(row["item"])}"{"" if published else ' class="withheld"'}>'
            f'<td><a href="{ITEM_PAGE}{escape(row["item"])}">{escape(display_title(row))}</a>'
            f"{copy_button(row['item'])}</td>"
            f"<td>{escape(row['year'])}</td><td>{escape(row['city'])}</td>"
            f'<td class="num">{escape(row["sheets"])}</td><td class="num">{escape(row["sheets_placed"])}</td>'
            f"{files}</tr>"
        )
    body = f"""<p class="crumbs"><a href="{site.page_path()}">Run {escape(site.version)}</a> › {escape(name)}</p>
<h1>{escape(name)}</h1>
<p class="lede">{number(totals.published)} of {number(totals.volumes)} volumes published, with {number(totals.sheets_placed)} of {number(totals.sheets)} sheets placed. A title opens the volume's page, with its catalog record and every scan. <b>View</b> opens its placed sheets over today's map in Allmaps; <b>JSON</b> is the IIIF file behind them. <b>Images</b> chooses whose scans they draw: {CHRONOSCOPE}'s faster copies, or the Library of Congress's originals.</p>
<p>Not every page is placed, or placed accurately: please <a href="{PROBLEMS_URL}">report problems</a>. Missing a volume, or pages of one? Look for it on {OLD_INSURANCE_MAPS}, or georeference it by hand there. {STATE_MOBILE_NOTE}</p>
{controls_html("Filter by town, year or id")}
<div class="table-scroll">
<table class="volumes">
<thead><tr><th>Title</th><th>Year</th><th>Location</th><th class="num">Sheets</th><th class="num">Placed</th><th>Main content</th><th>Key map</th></tr></thead>
<tbody>
{chr(10).join(body_rows)}
</tbody>
</table>
</div>
<script>
{source_script(site)}
{copy_script()}
const input = document.getElementById("filter");
const count = document.getElementById("count");
const rows = [...document.querySelectorAll("table.volumes tbody tr")];
function update() {{
  const terms = input.value.toLowerCase().split(/\\s+/).filter(Boolean);
  let shown = 0;
  for (const row of rows) {{
    const text = (row.dataset.item + " " + row.textContent).toLowerCase();
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
    (out / "volumes.json").write_text(
        json.dumps(volumes_json(rows), separators=(",", ":"))
    )
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
