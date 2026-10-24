---
name: truth-ab
description: "Run an A/B of a pipeline change (snap, reconcile, region, georef, OCR, splitting) on OIM-truth volumes and score it with the sheet-equal metric. Use when asked to A/B, sweep, or measure a change's effect on the score across volumes, or to compare two runs of the same volumes."
---

# A/B on truth volumes

The goal is cross-volume evidence that a change generalizes, available as early as possible, from a run that can't silently test the same code twice.

## Design rules

- **One build plus a runtime flag.** Put the change behind a CLI flag that defaults to today's behavior (booleans default false). Thread it through `fit` if needed, and run both arms from one venv. Never use two worktrees, venvs or PATH tricks: `fit` spawns stages by bare name, and editable installs make the working tree the running code. Don't edit `mapsnap/*.py` while a run is going.
- **Include a canary:** a volume where the change has a known, non-zero effect. A broken A/B looks exactly like a null result, and only the canary tells them apart.
- **Check that the base arm reproduces the published run** on the first volume (same score as the published IIIF) before trusting anything else.
- **Order volumes smallest first** (ascending truth pages), so signal arrives while there's still time to redirect.
- **Re-run only the stages the change affects.** Reuse upstream outputs from the corpus run: snap's `artifacts/osm_snap/candidates.jsonl` is an incremental cache, and the georef sidecars come with the run. Judge snap changes on `georef-final` (after reconcile), not on snap's own sidecars.
- **Skip volumes where the change can't fire**, e.g. region overlap is off below `REGION_MIN_SHEETS` (6), and say so in the write-up.
- **Experiments live outside `data/`.** Anything under `data/` with truth and centerlines gets scored as a volume. Use `~/Documents/mapsnap/<experiment>/` or the scratchpad.
- **Run one volume at a time** (see the `background-jobs` skill for launching and waiting).

## Inputs

- **Truth items:** `~/Documents/mapsnap/score-690/corpus-v2/<item>/` (690 OIM-truth items). Each has links to `main.iiif.json`, `oim/` and the county's `centerlines.osm.pbf`, plus the run's `p*.panels.json`. Published per-page scores are in `score-690/pages-corpus-v2.tsv`.
- **Run inputs:** from the bucket. The stable half (page JPEGs, P(road) maps, `metadata.json`, `keymaps.json`) is the item prefix minus `runs/*`. Then add `runs/<tag>/` on top, in the same directory. Skip `raw/*.jpg` (full-resolution key-map scans) and `*.boxes.json` unless a stage needs them.
- **Panel crops (`pN__k.jpg`) aren't uploaded.** Cut them from `pN.panels.json` with `mapsnap.split.write_panels(image, [Polygon(ring) for ring in doc["panels"]], base)`.
- **Disk:** stream each volume (download, run both arms, score, delete). Check `df -h ~` before starting.

## Template

The #643 harness in `~/Documents/mapsnap/ab643/` is the template to copy:
- `run_volume.py`: assemble one volume, predict region maps once, run each arm's reconcile, build a mask-free IIIF per arm;
- `run_all.sh`: the smallest-first loop, skipping finished volumes;
- `score.py`: both arms, cached per volume;
- `pages_diff.py`: every page whose outcome changed.

For a mask-free IIIF: `mapsnap iiif <dir>/metadata.json '<dir>/*.georef-final.json' --no-clip-masks --output <arm>/<item>/mapsnap.iiif.json`. Put links to the truth (`main.iiif.json`, `oim/`, `centerlines.osm.pbf`) and copies of the run's `p*.panels.json` beside each arm's IIIF, so it scores as a volume.

## Scoring

- **Use `mapsnap.score.volume_page_scores(iiif)` and `summarize(pages)`.** They are what `mapsnap score` uses, and `net_score` = good share − disaster share. Report:
  - land-weighted net (`summarize(pages)`);
  - flat net (`summarize([dataclasses.replace(p, land_m2=p.area_m2) for p in pages])`);
  - both volume-equal and sheet-weighted means over volumes.
- **Never weight pages by `land_km2`** from the compare txt (the old absolute-area metric). Per-page weights are `sheet_portion × land_fraction`.
- **Scoring needs** `oim/` and centerlines in the volume root, and `pN.panels.json` beside the IIIF. Without them, split panels silently score as unplaced.
- **Hand-built IIIFs in a real volume directory** also need `mapsnap compare main.iiif.json <tag>.iiif.json > <tag>.txt` beside them for the viewer.

## Reporting

- Give the score deltas, then how many volumes moved (up and down), then the flipped pages grouped by what matters for the change (e.g. by panel size).
- Name the biggest movers and the canary's outcome.
- Separate snap-level wins from what the published run (after reconcile) would show: reconcile often recovers what snap leaves.
- Record the results on the issue, with figures on the `issue-images` branch (see that skill).
