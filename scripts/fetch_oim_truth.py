#!/usr/bin/env python
"""Assemble OIM truth for every 100%-complete Sanborn volume, politely.

For each volume in the OIM truth listing whose completion is 100%, writes to
``<out>/<identifier>/``:

- ``main.export.iiif.json``: OIM's main-content export, as downloaded;
- ``main.iiif.json``: the same, with split pages' SvgSelectors repaired
  (``fix-truth-splits``, OIM#402), which is the file to grade against;
- ``key.iiif.json``: the key-map export, when the volume has a key-map layer set;
- ``oim/pN.panels.json`` and ``oim/pN.cutlines.json``: OIM's own region
  boundaries and cutlines for every split page (``oim-panels``);
- ``volume.json``: what was fetched, the split repairs, and any problems.

No images are downloaded. The API region boundaries let ``fix-truth-splits``
align each split selector without crop images, and a document page that omits
its canvas size falls back to the export's image dimensions.

Every request goes through one throttle (``--delay`` seconds between requests,
exponential backoff on 429/5xx), and a finished volume is skipped on rerun.

    uv run python scripts/fetch_oim_truth.py ~/Documents/mapsnap/oim-truth-volumes.tsv \
        ~/Documents/mapsnap/oim --delay 1.5
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from mapsnap import oim_panels
from mapsnap.compare_iiif_georef import label_split_index, parse_svg_polygon
from mapsnap.fix_truth_splits import (
    fix_annotation_page,
    gcp_containment,
    shifted_selector,
)
from mapsnap.oim_panels import (
    OIM_BASE,
    document_regions,
    embedded_json,
    title_page_key,
    write_page_files,
)
from mapsnap.utils import label_to_page_key

USER_AGENT = "mapsnap/0.1 (truth corpus fetch)"


class Throttle:
    """At most one request per ``delay`` seconds, with backoff on server trouble."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.last = 0.0
        self.requests = 0

    def get(self, url: str, max_attempts: int = 5) -> tuple[int, bytes]:
        """(HTTP status, body) for a GET; retries 429 and 5xx, returns the last status."""
        backoff = max(self.delay, 5.0)
        status, body = 0, b""
        for attempt in range(1, max_attempts + 1):
            wait = self.last + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.requests += 1
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                status, body = error.code, b""
                if error.code != 429 and error.code < 500:
                    return status, body
            except (urllib.error.URLError, TimeoutError) as error:
                status, body = 0, str(error).encode()
            if attempt < max_attempts:
                print(
                    f"    {url}: {status or body.decode()}; retrying in {backoff:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 300.0)
        return status, body

    def text(self, url: str) -> str:
        """A page's text, or raise if it could not be fetched."""
        status, body = self.get(url)
        if status != 200:
            raise OSError(f"{url}: HTTP {status}")
        return body.decode("utf-8", errors="replace")


def complete_sanborn_volumes(tsv: Path) -> list[dict[str, str]]:
    """Rows of the listing at 100% completion whose identifier is a Sanborn volume."""
    with tsv.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        row
        for row in rows
        if row["completion_pct"] == "100" and row["identifier"].startswith("sanborn")
    ]


def has_key_map(map_html: str) -> bool:
    """Whether the map page lists a key-map layer set with any layers in it."""
    layersets = embedded_json(map_html, "LAYERSETS") or []
    return any(
        isinstance(layerset, dict)
        and layerset.get("id") == "key-map"
        and layerset.get("layers")
        for layerset in layersets
    )


def export_canvas_sizes(doc: dict) -> dict[str, list[int]]:
    """Parent page key -> [width, height] of its image, from the export's items.

    A split item's source is its parent's full image, so its size is the canvas's.
    """
    sizes: dict[str, list[int]] = {}
    for item in doc.get("items", []):
        key = label_to_page_key(str(item.get("label", "")))
        source = item.get("target", {}).get("source", {})
        if key and source.get("width") and source.get("height"):
            sizes[key.split("__")[0]] = [int(source["width"]), int(source["height"])]
    return sizes


def split_keys(doc: dict) -> list[str]:
    """Parent keys of the pages this export shows as split."""
    keys = set()
    for item in doc.get("items", []):
        key = label_to_page_key(str(item.get("label", "")))
        if key and "__" in key:
            keys.add(key.split("__")[0])
        elif key and label_split_index(item) is not None:
            keys.add(key)
    return sorted(keys)


def ring_origin_repairs(doc: dict, oim_dir: Path, flagged: set[str]) -> list[str]:
    """Shift flagged split selectors by their region's top-left corner; log what moved.

    ``fix-truth-splits`` aligns a selector's bounding box with its region ring's,
    which is exact only when the volunteer's mask traces the whole region. OIM
    crops each split image to its region's bounding box, so the ring's top-left
    corner is the crop offset whatever the mask covers. As with the first
    repair, a shift is kept only when more of the item's own GCPs fall inside.
    ``flagged`` holds labels like "p10 [2]" that the first repair left broken.
    """
    log = []
    for item in doc.get("items", []):
        key = label_to_page_key(str(item.get("label", "")))
        index = label_split_index(item)
        if not key or index is None:
            continue
        parent = key.split("__")[0]
        label = f"{parent} [{index}]"
        selector = item["target"].get("selector") or {}
        panels = oim_dir / f"{parent}.panels.json"
        if (
            label not in flagged
            or selector.get("type") != "SvgSelector"
            or not panels.exists()
        ):
            continue
        rings = json.loads(panels.read_text())["panels"]
        if index > len(rings):
            continue
        ring = rings[index - 1]
        offset = (max(0.0, min(x for x, _ in ring)), max(0.0, min(y for _, y in ring)))
        before = gcp_containment(item, parse_svg_polygon(selector["value"]))
        value, points = shifted_selector(selector["value"], offset)
        after = gcp_containment(item, points)
        if after > before:
            selector["value"] = value
            log.append(
                f"{label}: shifted by region origin ({offset[0]:.0f}, {offset[1]:.0f}); "
                f"gcps inside {before:.0%} -> {after:.0%}"
            )
    return log


def repair_selectors(export: dict, oim_dir: Path) -> tuple[dict, dict]:
    """(repaired annotation page, record fields) from an export and its oim/ panels."""
    repaired = json.loads(json.dumps(export))
    first = fix_annotation_page(repaired, oim_dir)
    flagged = {
        line.split(":")[0]
        for line in first
        if "still broken" in line or "LOOKS BROKEN" in line
    }
    second = ring_origin_repairs(repaired, oim_dir, flagged)
    fixed = {line.split(":")[0] for line in second}
    return repaired, {
        "split_repairs": first,
        "region_origin_repairs": second,
        # Still below half their GCPs: compare grades splits by the panel rings,
        # but anything that projects the selector inherits these.
        "broken_selectors": [
            line for line in first if line.split(":")[0] in flagged - fixed
        ],
    }


def fetch_volume(row: dict[str, str], out: Path, throttle: Throttle) -> dict:
    """Fetch one volume's truth into out/<identifier>/ and return its record."""
    slug = row["identifier"]
    volume = out / slug
    volume.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "identifier": slug,
        "title": row["title"],
        "year": row["year"],
        "document_ct": int(row["document_ct"]),
        "region_ct": int(row["region_ct"]),
        "problems": [],
    }

    main_url = f"{OIM_BASE}/iiif/mosaic/{slug}/main-content/?trim=true"
    main_text = throttle.text(main_url)
    (volume / "main.export.iiif.json").write_text(main_text)
    export = json.loads(main_text)
    record["main_items"] = len(export.get("items", []))

    map_html = throttle.text(f"{OIM_BASE}/map/{slug}")
    documents = embedded_json(map_html, "documents") or []
    record["key_map"] = has_key_map(map_html)
    if record["key_map"]:
        status, body = throttle.get(f"{OIM_BASE}/iiif/mosaic/{slug}/key-map/?trim=true")
        if status == 200:
            (volume / "key.iiif.json").write_bytes(body)
            record["key_items"] = len(json.loads(body).get("items", []))
        else:
            record["problems"].append(f"key-map export: HTTP {status}")

    # The split pages' region boundaries and cutlines, one document page each.
    by_key: dict[str, list[int]] = {}
    for document in documents:
        key = title_page_key(str(document.get("title", "")))
        if key:
            by_key.setdefault(key, []).append(document["id"])
    sizes = export_canvas_sizes(export)
    split = split_keys(export)
    record["split_pages"] = split
    written = []
    for page_key in split:
        doc_ids = by_key.get(page_key)
        if not doc_ids:
            record["problems"].append(f"{page_key}: no OIM document")
            continue
        best: tuple[list, list, list | None] = ([], [], None)
        for doc_id in doc_ids:
            candidate = document_regions(doc_id)
            if len(candidate[0]) > len(best[0]):
                best = candidate
            if len(best[0]) >= 2:
                break
        regions, cutlines, canvas = best
        canvas = canvas or sizes.get(page_key)
        if canvas is None:
            record["problems"].append(f"{page_key}: no canvas size")
            continue
        if write_page_files(volume, page_key, regions, cutlines, canvas):
            written.append(page_key)
        else:
            record["problems"].append(f"{page_key}: fewer than 2 region boundaries")
    record["panels_written"] = len(written)

    repaired, fields = repair_selectors(export, volume / "oim")
    record.update(fields)
    (volume / "main.iiif.json").write_text(json.dumps(repaired))
    record["fetched"] = datetime.now(UTC).isoformat(timespec="seconds")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble OIM truth for every 100%-complete Sanborn volume."
    )
    parser.add_argument("tsv", type=Path, help="The OIM truth volume listing")
    parser.add_argument(
        "out", type=Path, help="Output directory, one subdirectory per volume"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5, help="Seconds between requests"
    )
    parser.add_argument(
        "--limit", type=int, help="Only the first N volumes (smallest first)"
    )
    parser.add_argument("--only", nargs="*", help="Only these identifiers")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Offline: re-derive every fetched volume's main.iiif.json and repair "
        "fields from its main.export.iiif.json and oim/ panels.",
    )
    args = parser.parse_args()

    if args.rebuild:
        for record_path in sorted(args.out.glob("*/volume.json")):
            volume = record_path.parent
            export = json.loads((volume / "main.export.iiif.json").read_text())
            repaired, fields = repair_selectors(export, volume / "oim")
            (volume / "main.iiif.json").write_text(json.dumps(repaired))
            record = json.loads(record_path.read_text())
            record.update(fields)
            record_path.write_text(json.dumps(record, indent=1))
        return

    rows = complete_sanborn_volumes(args.tsv)
    if args.only:
        rows = [row for row in rows if row["identifier"] in set(args.only)]
    rows.sort(key=lambda row: int(row["document_ct"]))
    if args.limit:
        rows = rows[: args.limit]
    throttle = Throttle(args.delay)
    # oim-panels' document fetches go through the same throttle.
    oim_panels.fetch = throttle.text
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        record_path = args.out / row["identifier"] / "volume.json"
        if record_path.exists():
            continue
        try:
            record = fetch_volume(row, args.out, throttle)
        except (OSError, ValueError) as error:
            print(
                f"[{index}/{len(rows)}] {row['identifier']}: FAILED {error}", flush=True
            )
            continue
        record_path.write_text(json.dumps(record, indent=1))
        repairs = sum(1 for line in record["split_repairs"] if "shifted" in line) + len(
            record["region_origin_repairs"]
        )
        print(
            f"[{index}/{len(rows)}] {row['identifier']} {row['title']}: "
            f"{record['main_items']} items, key map {'yes' if record['key_map'] else 'no'}, "
            f"{record['panels_written']}/{len(record['split_pages'])} split pages, "
            f"{repairs} selectors repaired, {len(record['broken_selectors'])} unrepaired, "
            f"{len(record['problems'])} problems "
            f"({throttle.requests} requests, {(time.monotonic() - started) / 60:.0f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
