---
name: local-volume
description: "Bring a corpus item (sanbornNNNNN_NNN) down from the mirror into data/ so the volume viewer can show a run, or so it can be scored or re-run locally: pull_item.py, the county centerlines, OIM truth, panel crops, and the compare .txt the viewer reads. Use when asked to pull, inspect, view, debug, or score a corpus volume locally."
---

# A corpus volume on the laptop

## 1. Pull the item

```bash
uv run python scripts/pull_item.py sanborn05791_054 --run corpus-v2 --name brooklyn_ny_1939_vol_2
```

- **The stable half:** without `--run`, it syncs the item's whole prefix (page JPEGs, P(road) maps, CRAFT boxes, `metadata.json`, `raw/`) into `data/<city>_<state>_<year>/`.
- **One run only:** `--run TAG` syncs only `runs/<TAG>/` into `data/<vol>/runs/<TAG>/`. That's how to add a run to a volume that's already local: a full sync would overwrite its page images.
- **Item lookup:** the item → prefix mapping comes from `~/Documents/mapsnap/loc-sanborn-volumes.tsv` (`--tsv`).
- **Credentials:** it runs `aws` with `AWS_PROFILE=mapsnap` unless you set another; re-run `aws login` if listings come back empty.

## 2. Centerlines (the county extract)

Scoring, `compare` and re-running stages need `centerlines.osm.pbf` (or `.geojson`) in the volume root.
1. **Find the item's county FIPS code.** `s3://mapsnap-sanborn/_craft/city-and-hand-items.tsv` (independent cities plus hand-labelled counties) wins over `s3://mapsnap-sanborn/_craft/items.tsv`, the same precedence loc-fit uses. Both have `item` and `fips` columns.
2. **Fetch the extract:** `s3://mapsnap-sanborn/osm-by-county/<FIPS>.osm.pbf`. Copies of many extracts are in `~/Documents/mapsnap/oim-score/counties/`.

```bash
aws s3 cp s3://mapsnap-sanborn/_craft/city-and-hand-items.tsv /tmp/ --profile mapsnap
aws s3 cp s3://mapsnap-sanborn/_craft/items.tsv /tmp/ --profile mapsnap
fips=$(awk -F'\t' -v i=ITEM 'FNR==1 {for (c=1;c<=NF;c++) if ($c=="fips") f=c; next} $1==i {print $f}' /tmp/items.tsv /tmp/city-and-hand-items.tsv | tail -1)
aws s3 cp s3://mapsnap-sanborn/osm-by-county/$fips.osm.pbf data/<vol>/centerlines.osm.pbf --profile mapsnap
```

## 3. Truth (if the item has OIM truth)

- **Where it lives:** `~/Documents/mapsnap/oim/<item>/`, holding `main.iiif.json` (the file to grade against, with split selectors repaired) and `oim/` (OIM's panel polygons). `scripts/fetch_oim_truth.py` builds these.
- **What to link:** link (or copy) both into the volume root as `data/<vol>/main.iiif.json` and `data/<vol>/oim/`.
- **Scoring needs `oim/`:** without it, scoring split-panel truth silently counts every panel as unplaced.

## 4. What the viewer and scoring need beside a run's IIIF

- **Panel cuts:** `runs/<tag>/pN.panels.json`, the run's own split cuts. Without them, its split panels can't be matched to truth.
- **The compare text:** the viewer's "color by RMSE" reads a `.txt` sibling of the IIIF:

  ```bash
  uv run mapsnap compare data/<vol>/main.iiif.json data/<vol>/runs/<tag>/mapsnap.iiif.json \
      > data/<vol>/runs/<tag>/mapsnap.iiif.txt
  ```

- **Scoring:** `uv run mapsnap score data/<vol>/runs/<tag>/mapsnap.iiif.json`. Scores are against the volume root's truth, `oim/` and centerlines (`score.volume_root`); `--csv FILE` also writes per-page rows.

## 5. Re-running stages locally

- **Panel crops (`pN__k.jpg`) aren't uploaded.** Cut them before running region, snap or anything else that reads panels:

  ```python
  import json
  from pathlib import Path
  from shapely.geometry import Polygon
  from mapsnap.split import write_panels

  for panels in Path("data/<vol>").glob("p*.panels.json"):
      base = panels.name.removesuffix(".panels.json")
      doc = json.loads(panels.read_text())
      write_panels(
          panels.parent / f"{base}.jpg", [Polygon(r) for r in doc["panels"]], base
      )
  ```

- **Region maps** (`artifacts/region/`) aren't uploaded either: `mapsnap region data/<vol>`.
- **Run sidecars:** stages read them from the directory they're given, so copy or sync `runs/<tag>/*` up to the volume root for a re-run, or work in a scratch copy outside `data/`. Experiments must not create new top-level directories under `data/`.
- **One heavy job at a time** on the laptop (see the `background-jobs` skill).
