"""Download the OIM split images a volume's truth needs, straight from main.iiif.json.

Scoring truth splits properly needs ``oim/pN.panels.json``, whose inputs are
OIM's manually-cut split crops (``oim/pN__i.jpg``) and the full-resolution
parent scans (``raw/pN.jpg``). ``mapsnap download-oim`` fetches those for whole
OIM volumes but needs a caller-supplied URL prefix; this command derives
everything itself for the split pages only:

- which pages split, and into how many panels, from main.iiif.json labels;
- the image URL prefix by probing OIM's S3 bucket with candidate volume slugs
  (the data directory's name, plus state-name expansions: LOC-style dirs say
  ``columbus_oh`` where OIM's files say ``columbus_ohio``);
- full pages live under ``uploaded/documents/``, split crops under
  ``uploaded/regions/`` — the same layout download-oim's 404-swap expects.

Then run ``mapsnap oim-split-truth`` to build the panels files:

    mapsnap download-oim-splits data/columbus_oh_1951_vol_3
    mapsnap oim-split-truth data/columbus_oh_1951_vol_3/main.iiif.json
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from mapsnap.compare_iiif_georef import label_split_index
from mapsnap.download_oim_iiif import download_oim_image
from mapsnap.utils import label_to_page_key

OIM_BUCKET = "https://s3.us-central-1.wasabisys.com/oldinsurancemaps/uploaded"

STATE_NAMES = {
    "al": "alabama",
    "ca": "calif",
    "co": "colorado",
    "fl": "florida",
    "ga": "georgia",
    "il": "ill",
    "in": "indiana",
    "ky": "kentucky",
    "la": "la",
    "ma": "mass",
    "md": "maryland",
    "mi": "mich",
    "mn": "minnesota",
    "mo": "mo",
    "nc": "north_carolina",
    "nj": "nj",
    "ny": "ny",
    "oh": "ohio",
    "pa": "pa",
    "tn": "tn",
    "tx": "texas",
    "va": "virginia",
    "wa": "washington",
    "wi": "wisconsin",
}
"""Postal code -> the spelling OIM's file slugs tend to use. Only consulted
when the directory-name slug itself 404s, and always verified by a probe."""


def split_pages(iiif_path: Path) -> dict[str, list[int]]:
    """parent page key -> sorted split indices, from the truth labels."""
    groups: dict[str, set[int]] = {}
    for item in json.loads(iiif_path.read_text()).get("items", []):
        index = label_split_index(item)
        if index is None:
            continue
        key = label_to_page_key(item.get("label", ""))
        if key is None:
            continue
        groups.setdefault(key.split("__")[0], set()).add(index)
    return {parent: sorted(indices) for parent, indices in sorted(groups.items())}


def slug_candidates(volume_name: str) -> list[str]:
    """Volume slugs to probe, most likely first."""
    candidates = [volume_name]
    parts = volume_name.split("_")
    for i, part in enumerate(parts):
        expansion = STATE_NAMES.get(part)
        if expansion and expansion != part:
            candidates.append("_".join(parts[:i] + [expansion] + parts[i + 1 :]))
    return candidates


def url_exists(url: str) -> bool:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "mapsnap/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def discover_prefix(volume: Path, probe_page: str) -> str | None:
    """The documents/ URL prefix whose probe page actually exists, or None."""
    for slug in slug_candidates(volume.name):
        prefix = f"{OIM_BUCKET}/documents/{slug}_"
        if url_exists(f"{prefix}{probe_page}.jpg"):
            return prefix
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "volume", type=Path, help="Volume directory (with main.iiif.json)"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="OIM documents/ URL prefix override (default: discovered by probing).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    iiif_path = args.volume / "main.iiif.json"
    if not iiif_path.exists():
        sys.exit(f"no {iiif_path}")
    groups = split_pages(iiif_path)
    if not groups:
        print("no split pages in the truth; nothing to download")
        return
    total = sum(len(v) for v in groups.values())
    print(
        f"{len(groups)} split page(s), {total} panels: "
        + ", ".join(f"{k}[{len(v)}]" for k, v in groups.items())
    )

    probe = next(iter(groups))
    prefix = args.prefix or discover_prefix(args.volume, probe)
    if prefix is None:
        sys.exit(
            f"could not discover the OIM URL prefix (probed slugs: "
            f"{slug_candidates(args.volume.name)}); pass --prefix"
        )
    print(f"prefix: {prefix}")

    for parent, indices in groups.items():
        targets = [(f"{prefix}{parent}.jpg", args.volume / "raw" / f"{parent}.jpg")]
        for index in indices:
            targets.append(
                (
                    f"{prefix}{parent}__{index}.jpg",
                    args.volume / "oim" / f"{parent}__{index}.jpg",
                )
            )
        for url, dest in targets:
            if dest.exists():
                print(f"  have {dest.name}")
                continue
            print(f"  {url} -> {dest}")
            if args.dry_run:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            download_oim_image(url, dest)


if __name__ == "__main__":
    main()
