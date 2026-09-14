"""Cut one OSM extract per county from a single national dump (#408).

The corpus needs street geometry for every county that has a Sanborn volume:
2,714 of them, covering 432,410 sheets. Downloading each from an OSM service
would be thousands of requests; cutting them locally from one dump is a single
pass per batch.

Counties come from ``mapsnap loc-counties``' ``counties.tsv`` and their
boundaries from Natural Earth, keyed on FIPS. Each boundary is buffered before
cutting -- a town on a county line needs the streets on the other side of it,
and the boundary itself is only as accurate as Natural Earth's generalization.

Two things this learned the hard way, both worth keeping:

  * the dump must have **renumbered** ids (``osmium renumber``). osmium sizes a
    node-id bitmap per extract from the highest id it sees, so raw OSM ids cost
    about 1.4 GB per polygon: a 100-polygon batch needed 141 GB, was killed, and
    wrote truncated, silently-valid output files.
  * memory still scales with batch size even renumbered (100 polygons measured
    at 3.7 GB), so batches are capped rather than run as one pass.

Outputs are named ``<FIPS>.osm.pbf`` and a county already on disk is skipped, so
an interrupted run resumes by being run again.

    mapsnap osm-counties --counties counties.tsv --natural-earth ne.geojson \\
        --pbf us-named-highways.renumbered.osm.pbf --out-dir /data/osm-by-county
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# osmium holds a node-id bitmap per extract, so memory grows with the batch.
# 100 polygons measured at 3.7 GB against the renumbered dump.
DEFAULT_BATCH = 200
DEFAULT_BUFFER_KM = 3.0
# Metres per degree, near enough for a buffer whose purpose is slack.
METRES_PER_DEGREE_LAT = 110_570.0
METRES_PER_DEGREE_LON_EQUATOR = 111_320.0


@dataclass(frozen=True)
class County:
    """One county to cut, and what it is worth cutting for."""

    fips: str
    state: str
    name: str
    sheets: int

    @property
    def filename(self) -> str:
        return f"{self.fips}.osm.pbf"


def read_counties(path: Path) -> list[County]:
    """The counties to extract, from ``loc-counties``' counties.tsv."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [
            name
            for name in ("fips", "state", "ne_name", "sheets")
            if name not in (reader.fieldnames or [])
        ]
        if missing:
            sys.exit(f"{path}: no {missing[0]!r} column")
        return [
            County(row["fips"], row["state"], row["ne_name"], int(row["sheets"] or 0))
            for row in reader
            if row["fips"]
        ]


def natural_earth_geojson(source: Path) -> Path:
    """A GeoJSON of the Natural Earth counties, converting a shapefile if needed."""
    if source.is_dir():
        shapefiles = sorted(source.glob("*.shp"))
        if not shapefiles:
            sys.exit(f"{source}: no .shp inside")
        source = shapefiles[0]
    if source.suffix.lower() in (".geojson", ".json"):
        return source
    out = source.with_suffix(".geojson")
    if not out.exists():
        subprocess.run(["ogr2ogr", "-f", "GeoJSON", str(out), str(source)], check=True)
    return out


def load_boundaries(geojson: Path) -> dict[str, dict]:
    """Natural Earth geometries keyed by FIPS (``US06037``)."""
    features = json.loads(geojson.read_text())["features"]
    return {
        feature["properties"]["FIPS"]: feature["geometry"]
        for feature in features
        if feature.get("properties", {}).get("FIPS")
    }


def buffered_rings(geometry: dict, buffer_km: float) -> list[list[list[list[float]]]]:
    """The geometry grown by ``buffer_km``, as a list of polygons of rings.

    Buffering happens in a local metre-ish frame scaled from the geometry's own
    latitude, because a degree of longitude is only 111 km at the equator and
    85 km at 40 degrees north; buffering in raw degrees would stretch the north
    of the country. Interior rings are dropped: a hole in a county boundary is
    not somewhere streets should be excluded from.
    """
    from shapely.affinity import scale
    from shapely.geometry import MultiPolygon, Polygon, shape

    shapely_geometry = shape(geometry)
    latitude = shapely_geometry.centroid.y
    per_lon = METRES_PER_DEGREE_LON_EQUATOR * math.cos(math.radians(latitude))
    per_lat = METRES_PER_DEGREE_LAT
    metric = scale(shapely_geometry, xfact=per_lon, yfact=per_lat, origin=(0, 0))
    grown = metric.buffer(buffer_km * 1000.0, quad_segs=4)
    back = scale(grown, xfact=1 / per_lon, yfact=1 / per_lat, origin=(0, 0))
    polygons = back.geoms if isinstance(back, MultiPolygon) else [back]
    return [
        [[[round(x, 6), round(y, 6)] for x, y in polygon.exterior.coords]]
        for polygon in polygons
        if isinstance(polygon, Polygon)
    ]


def extract_entry(county: County, rings: list[list[list[list[float]]]]) -> dict:
    """One osmium extract-config entry for a county."""
    entry: dict = {"output": county.filename, "output_format": "pbf"}
    if len(rings) == 1:
        entry["polygon"] = rings[0]
    else:
        entry["multipolygon"] = rings
    return entry


def pending(counties: list[County], out_dir: Path) -> list[County]:
    """Counties with no extract on disk yet, so a killed run resumes."""
    return [c for c in counties if not (out_dir / c.filename).exists()]


def run_batch(
    entries: list[dict], pbf: Path, out_dir: Path, work_dir: Path
) -> subprocess.CompletedProcess:
    """Cut one batch of counties in a single pass over the dump."""
    config = work_dir / "extracts.json"
    config.write_text(json.dumps({"directory": str(out_dir), "extracts": entries}))
    return subprocess.run(
        [
            "osmium",
            "extract",
            "--config",
            str(config),
            "--strategy",
            "simple",
            "--overwrite",
            str(pbf),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def format_duration(seconds: float) -> str:
    """``h:mm`` for a duration."""
    minutes = int(seconds // 60)
    return f"{minutes // 60}:{minutes % 60:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut one OSM extract per Sanborn county from a national dump."
    )
    parser.add_argument("--counties", type=Path, required=True, help="counties.tsv")
    parser.add_argument(
        "--natural-earth",
        type=Path,
        required=True,
        help="Natural Earth admin-2 counties: a .geojson, a .shp, or its directory.",
    )
    parser.add_argument(
        "--pbf", type=Path, required=True, help="The national dump, RENUMBERED."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help="Counties per osmium pass (default: %(default)s; memory grows with it).",
    )
    parser.add_argument(
        "--buffer-km",
        type=float,
        default=DEFAULT_BUFFER_KM,
        help="Grow each boundary by this much (default: %(default)s).",
    )
    parser.add_argument("--limit", type=int, help="Stop after this many counties.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be cut, and write the first batch's config, without running osmium.",
    )
    args = parser.parse_args()

    counties = read_counties(args.counties)
    boundaries = load_boundaries(natural_earth_geojson(args.natural_earth))
    unknown = [c for c in counties if c.fips not in boundaries]
    if unknown:
        print(
            f"{len(unknown)} counties are not in Natural Earth, skipping: "
            f"{', '.join(c.fips for c in unknown[:5])}",
            file=sys.stderr,
        )
    counties = [c for c in counties if c.fips in boundaries]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = pending(counties, args.out_dir)
    if args.limit:
        todo = todo[: args.limit]
    done_already = len(counties) - len(pending(counties, args.out_dir))
    print(
        f"{len(counties):,} counties wanted, {done_already:,} already cut, "
        f"{len(todo):,} to do in batches of {args.batch_size}",
        file=sys.stderr,
        flush=True,
    )
    if not todo:
        return

    started = time.perf_counter()
    cut = failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for index in range(0, len(todo), args.batch_size):
            batch = todo[index : index + args.batch_size]
            entries = [
                extract_entry(c, buffered_rings(boundaries[c.fips], args.buffer_km))
                for c in batch
            ]
            if args.dry_run:
                config = work_dir / "extracts.json"
                config.write_text(
                    json.dumps({"directory": str(args.out_dir), "extracts": entries})
                )
                print(
                    f"batch {index // args.batch_size + 1}: {len(batch)} counties, "
                    f"config {config.stat().st_size / 1024:.0f} KB, "
                    f"first {batch[0].fips} ({batch[0].name}, {batch[0].state})"
                )
                return
            result = run_batch(entries, args.pbf, args.out_dir, work_dir)
            written = sum(1 for c in batch if (args.out_dir / c.filename).exists())
            cut += written
            if result.returncode != 0 or written < len(batch):
                failed += len(batch) - written
                print(
                    f"  batch {index // args.batch_size + 1}: osmium exit "
                    f"{result.returncode}, {written}/{len(batch)} written: "
                    f"{result.stderr.strip()[:300]}",
                    file=sys.stderr,
                    flush=True,
                )
            elapsed = time.perf_counter() - started
            rate = cut / elapsed if elapsed else 0.0
            print(
                f"batch {index // args.batch_size + 1}: {written}/{len(batch)} cut "
                f"| {cut}/{len(todo)} total, {rate * 3600:.0f} counties/h, "
                f"eta {format_duration((len(todo) - cut) / rate if rate else 0)}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    size = sum(
        (args.out_dir / c.filename).stat().st_size
        for c in counties
        if (args.out_dir / c.filename).exists()
    )
    print(
        f"cut {cut} counties ({failed} failed) in {format_duration(elapsed)}; "
        f"{size / 2**30:.1f} GB in {args.out_dir}",
        file=sys.stderr,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
