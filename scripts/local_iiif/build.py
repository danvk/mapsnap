#!/usr/bin/env python
"""Build the static IIIF tree #501 proposes, from whatever the S3 cache holds.

    scripts/local_iiif/build.py [--root DIR] [--quality 75] [--item ID ...]

For every item in ~/.cache/mapsnap/s3 that has both a published annotation and
its scans, writes a halving pyramid of each page at the given JPEG quality and
an IIIF Image API **level 0** service around it:

    <root>/<item>/<page>/info.json
    <root>/<item>/<page>/full/<w>,<h>/0/default.jpg     one per level
    <root>/<item>/<page>/full/max/0/default.jpg         the largest, by its other name

No tiling: a level-0 service whose `sizes` are whole images is the simpler half
of the proposal, and enough to see whether serving the corpus ourselves works
at all. Tiles are the same files cut up, and can be added without moving
anything.

The annotation is rewritten beside the original in the cache as
`mapsnap.local.iiif.json`: its image service points at this tree, and its GCPs
and clipping selector are rescaled out of the LoC full-resolution frame the
annotation is written in (6,372 x 7,548 for Fountain City) into the mirrored
scan's (1,593 x 1,887). That rescale is the same one the debugger does in
memory; doing it on disk is what makes the result a plain file any IIIF viewer
can open.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

DEFAULT_CACHE = Path.home() / ".cache/mapsnap/s3"
DEFAULT_ROOT = Path.home() / ".cache/mapsnap/local-iiif"
# How far down to halve. The 25% scan is level 0; four levels reach ~3% of the
# original scan, which is a thumbnail of a sheet.
LEVELS = 4
# A tileset the size of the whole image makes every zoom level a 1x1 grid --
# one file per level, no grid of tiles -- which is what a viewer needs to
# accept the service at all: Allmaps' parser refuses an image declaring neither
# `tiles` nor support for arbitrary regions ("Image does not support tiles or
# custom regions and sizes").
SCALE_FACTORS = [1 << level for level in range(LEVELS)]


def page_key(service_id: str, label: str, annotation_id: str) -> str | None:
    """The page this annotation is of, from whichever field carries the number."""
    for text in (annotation_id, service_id, label):
        if not text:
            continue
        # "...-0019__2/georef" or "... p19 [2]" -> p19__2
        match = re.search(r"-(\d+)([A-Za-z]*)(__\d+)?(?:/georef)?$", text)
        if match:
            number = str(int(match.group(1)))
            return f"p{number}{match.group(2)}{match.group(3) or ''}"
        match = re.search(r"\bp(\d+[A-Za-z]*)(?:\s*\[(\d+)\])?", text)
        if match:
            return f"p{match.group(1)}" + (
                f"__{match.group(2)}" if match.group(2) else ""
            )
    return None


def tile_size(dimension: int, factor: int) -> int:
    """The size a client asks for, for a single tile covering the whole image.

    The Image API's tile region calculation, which the viewer follows exactly:

        tileWidth = floor((imageWidth - regionX + scaleFactor - 1) / scaleFactor)

    With one tile the region starts at 0, so this is ceil(dimension / factor).
    It is NOT round: 1593 at factor 8 is 200, not 199, and a pixel of
    disagreement is a 404 -- which is how the first build of this failed.

    https://iiif.io/api/image/3.0/implementation/#3-tile-region-parameter-calculation
    """
    return max(1, -(-dimension // factor))


def pyramid_sizes(width: int, height: int) -> list[tuple[int, int]]:
    """(width, height) at each scale factor, largest first, stopping at 1 px."""
    sizes: list[tuple[int, int]] = []
    for factor in SCALE_FACTORS:
        w, h = tile_size(width, factor), tile_size(height, factor)
        if sizes and (w, h) == sizes[-1]:
            continue
        sizes.append((w, h))
        if w == 1 or h == 1:
            break
    return sizes


def link_to(link: Path, target: Path) -> None:
    """Point `link` at `target`, replacing whatever was there."""
    link.unlink(missing_ok=True)
    link.symlink_to(target)


def write_page(image_path: Path, out_dir: Path, service_id: str, quality: int) -> dict:
    """Write one page's pyramid and info.json; return the info document."""
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        width, height = source.size
        sizes = pyramid_sizes(width, height)
        for index, (w, h) in enumerate(sizes):
            level = (
                source
                if index == 0
                else source.resize((w, h), Image.Resampling.LANCZOS)
            )
            target = out_dir / "full" / f"{w},{h}" / "0"
            target.mkdir(parents=True, exist_ok=True)
            level.save(target / "default.jpg", "JPEG", quality=quality, optimize=True)
            # The same bytes under the name a tiled client asks for: it sends an
            # explicit region rather than `full`, even when that region is the
            # whole image. Symlinked, so each level is stored once.
            region = out_dir / f"0,0,{width},{height}" / f"{w},{h}" / "0"
            region.mkdir(parents=True, exist_ok=True)
            link_to(
                region / "default.jpg",
                Path("../../..") / "full" / f"{w},{h}" / "0" / "default.jpg",
            )
    # `full/max` is what a viewer asks for when it wants the whole image and has
    # not read `sizes` yet. Symlinked, not copied: it is the same bytes as the
    # largest level, and duplicating them would double the store. An object
    # store has no symlinks, so hosting this for real means either paying for
    # the copy or declaring `maxWidth` so clients ask for the size by name.
    max_dir = out_dir / "full" / "max" / "0"
    max_dir.mkdir(parents=True, exist_ok=True)
    link_to(
        max_dir / "default.jpg",
        Path("..") / ".." / f"{sizes[0][0]},{sizes[0][1]}" / "0" / "default.jpg",
    )

    info = {
        "@context": "http://iiif.io/api/image/3/context.json",
        "id": service_id,
        "type": "ImageService3",
        "protocol": "http://iiif.io/api/image",
        "profile": "level0",
        "width": width,
        "height": height,
        "sizes": [{"width": w, "height": h} for w, h in sizes],
        "tiles": [
            {
                "width": width,
                "height": height,
                "scaleFactors": SCALE_FACTORS[: len(sizes)],
            }
        ],
        "extraFormats": ["jpg"],
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def rescale_selector(
    svg: str, scale_x: float, scale_y: float, width: int, height: int
) -> str:
    """The clipping polygon in the scan's pixel frame."""

    def points(match: re.Match[str]) -> str:
        pairs = []
        for pair in match.group(1).split():
            x, _, y = pair.partition(",")
            nx = min(max(float(x) * scale_x, 0.0), float(width))
            ny = min(max(float(y) * scale_y, 0.0), float(height))
            pairs.append(f"{round(nx, 1)},{round(ny, 1)}")
        return 'points="' + " ".join(pairs) + '"'

    return re.sub(r'points="([^"]*)"', points, svg)


def build_item(
    item_dir: Path, annotation: Path, root: Path, base_url: str, quality: int
) -> dict:
    """Write one item's pyramid and its rewritten annotation. Returns a summary."""
    page = json.loads(annotation.read_text())
    item = item_dir.name
    written: dict[str, dict] = {}
    kept, skipped = [], []
    for entry in page.get("items", []):
        target = entry.get("target") or {}
        source = target.get("source") or {}
        key = page_key(
            str(source.get("id", "")),
            str(entry.get("label", "")),
            str(entry.get("id", "")),
        )
        # A split panel is georeferenced against its parent sheet, so the scan
        # to serve is the parent's and the panel keeps its own identity.
        parent = re.sub(r"__\d+$", "", key) if key else None
        scan = item_dir / f"{parent}.jpg" if parent else None
        if not parent or not scan or not scan.exists() or not source.get("width"):
            skipped.append(key or str(entry.get("label")))
            continue
        if parent not in written:
            written[parent] = write_page(
                scan, root / item / parent, f"{base_url}/{item}/{parent}", quality
            )
        info = written[parent]
        scale_x = info["width"] / source["width"]
        scale_y = info["height"] / source["height"]
        target["source"] = {
            "id": info["id"],
            "type": "ImageService3",
            "width": info["width"],
            "height": info["height"],
        }
        for feature in (entry.get("body") or {}).get("features", []):
            coords = (feature.get("properties") or {}).get("resourceCoords")
            if coords and len(coords) >= 2:
                feature["properties"]["resourceCoords"] = [
                    round(coords[0] * scale_x, 1),
                    round(coords[1] * scale_y, 1),
                ]
        selector = target.get("selector") or {}
        if selector.get("type") == "SvgSelector" and selector.get("value"):
            selector["value"] = rescale_selector(
                selector["value"], scale_x, scale_y, info["width"], info["height"]
            )
        kept.append(entry)
    page["items"] = kept
    out = annotation.with_name("mapsnap.local.iiif.json")
    body = json.dumps(page, indent=2) + "\n"
    out.write_text(body)
    # And under the served tree, so a viewer can be pointed at a URL rather than
    # a file: the annotation is the entry point, and Allmaps takes one by ?url=.
    served = root / "annotations" / f"{item}.json"
    served.parent.mkdir(parents=True, exist_ok=True)
    served.write_text(body)
    return {
        "item": item,
        "pages": len(written),
        "annotations": len(kept),
        "skipped": skipped,
        "out": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="Where the static tree goes."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8183/iiif",
        help="Public URL of --root's iiif tree.",
    )
    parser.add_argument(
        "--quality", type=int, default=75, help="JPEG quality (default: %(default)s)."
    )
    parser.add_argument(
        "--item", action="append", help="Only this item id; repeatable."
    )
    args = parser.parse_args()

    wanted = set(args.item or [])
    annotations = sorted(args.cache.rglob("mapsnap.iiif.json"))
    built, pages = [], 0
    for annotation in annotations:
        item_dir = annotation.parent.parent.parent
        if wanted and item_dir.name not in wanted:
            continue
        if not any(item_dir.glob("*.jpg")):
            continue
        summary = build_item(
            item_dir, annotation, args.root, args.base_url.rstrip("/"), args.quality
        )
        built.append(summary)
        pages += summary["pages"]
        print(
            f"  {summary['item']:20} {summary['pages']:4} pages, {summary['annotations']:4} annotations"
            + (f", {len(summary['skipped'])} skipped" if summary["skipped"] else ""),
            file=sys.stderr,
        )
    # Symlinks are not storage: counting full/max would double every level 0.
    total = (
        sum(f.stat().st_size for f in args.root.rglob("*.jpg") if not f.is_symlink())
        if args.root.exists()
        else 0
    )
    source = sum(
        f.stat().st_size
        for a in built
        for f in (a["out"].parent.parent.parent).glob("*.jpg")
    )
    print(f"\n{len(built)} items, {pages} pages -> {args.root}", file=sys.stderr)
    if source:
        print(
            f"  pyramid {total / 1e6:,.1f} MB from {source / 1e6:,.1f} MB of q95 "
            f"scans ({total / source:.0%}), quality {args.quality}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
