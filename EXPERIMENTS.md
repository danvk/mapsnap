# Running Experiments

In order to measure the impact of a change and avoid unexpected regressions, it's important to run experiments across a diverse test set.

The mapsnap pipeline has three main components:

1. CRAFT text detection (produces boxes.json)
2. Text OCR (produces streets.json)
3. Georeferencing (produces georef.json)

The first step is the slowest but only ever needs to be run once. The second step is faster (~3s/page on MPS) and the third is faster still. Some changes affect the OCR step and require it to be re-run, but most only affect the georeferencing step.

## The current state

Once CRAFT and OCR have completed for a volume, you can run the georeferencing step, produce a IIIF file and compare against any truth data by running, for example:

    uv run mapsnap fit data/brooklyn_ny_1939_vol_1 --tag 2026-06-24

This updates all the georef.json files under data/brooklyn_ny_1939_vol_1, produces data/brooklyn_ny_1939_vol_1/2026-06-24.iiif.json and data/brooklyn_ny_1939_vol_1/2026-06-24.txt, which compares the IIIF file to the truth data. It can be run with `--num-workers 2` to speed things up.

This is convenient but has a few drawbacks:

1. The tagging system (`2026-06-24`) is ad-hoc and doesn't cleanly map to any particular version of the code. This makes finding a good baseline harder than necessary.
2. Any existing georef.json files are clobbered, which makes comparing what changed across two runs difficult.

## The proposal

Every `mapsnap fit` run is archived under `data/<volume>/artifacts/<run-id>/`. The per-page JSON, IIIF, and compare files are still written to their current sidecar locations (current behavior); on completion `mapsnap fit` copies them, plus a manifest, into the artifact directory.

### Run identity

The tag parameter becomes optional. When it is omitted, the run id is computed automatically as:

    <git-sha8>-<config-hash8>

- **`git-sha8`** — the short SHA of `HEAD`. `mapsnap fit` requires the working tree to be clean (no uncommitted changes to tracked files; `data/` is gitignored so generated outputs don't count). To run an experiment you commit the change first — even a throwaway commit you intend to squash or roll back. This deliberately leans on git for code provenance rather than re-implementing diff tracking. (Caveat: if you later discard the commit, keep it reachable on a branch/tag if you want the run to stay reproducible, since the manifest references the SHA.)

- **`config-hash8`** — a short hash over everything else that affects this step's output:
  - the **effective flags** (e.g. `--scale-outlier-threshold`, `--seed`, `--num-workers`), and
  - the **content hashes of the inputs**: every `p*.streets.json`, `centerlines.geojson`, and the truth IIIF.

  Folding flags in means `--scale-outlier-threshold 0` gets its own archive instead of colliding with the default run. Folding input hashes in means a re-run after re-OCR (new streets.json) gets a fresh id instead of silently reusing a stale archive — so the early-out below is safe.

If `data/<volume>/artifacts/<run-id>/` already exists, `mapsnap fit` skips the computation and prints the path to the existing archive. (It does not touch the live sidecar files; restoring an archived run to the live locations isn't needed yet.) Shared baselines — `main` vs `branchA` and `branchB` — therefore reuse the same baseline archive.

An explicit tag (`mapsnap fit <volume> --tag <tag>`) still overrides the computed id, for ad-hoc named runs. It's a flag rather than a positional so that passthrough georef flags (e.g. `--num-workers 2`) can't be mis-parsed as the tag.

### What gets archived

Copied into `artifacts/<run-id>/` on completion:

- `p*.streets.json`, `p*.georef.json` (including the `-misscale` / `-1gcp` / `-outlier` variants)
- the IIIF file and the compare `.txt`
- the per-page `.txt` logs (the GCP / inlier / RANSAC detail — needed to understand *why* a page changed)
- `manifest.json` (provenance + metrics; below)

`streets.json` is the largest of these (~90% of a run, see [Disk usage](#disk-usage)) and is OCR output, unchanged across the georef-only experiments that are most of them. So it is **deduplicated**: when this run's `streets` `combined_sha` matches that of an existing archive for the same volume, the new archive symlinks each `p*.streets.json` to that prior copy instead of re-copying it. The manifest still records the full `combined_sha`, so the archive stays self-describing; only the bytes are shared.

`boxes.json` is **not** archived at all — it is CRAFT output, georef-independent, and large; only its hash is recorded in the manifest.

### manifest.json

One file per run, combining provenance and metrics so a run is reproducible *and* diffable:

    {
      "run_id": "1a2b3c4d-9f8e7d6c",
      "label": "van-brunt-ocr-fix",
      "git": { "sha": "1a2b3c4d...", "branch": "ocr-fix", "subject": "two-word street OCR", "clean": true },
      "command": ["mapsnap", "fit", "data/brooklyn_ny_1939_vol_1"],
      "flags": { "scale_outlier_threshold": 0.25, "seed": 0, "num_workers": 2 },
      "created": "2026-06-26T12:00:00Z",
      "inputs": {
        "centerlines_sha": "sha256:...",
        "truth": "main.iiif.json", "truth_sha": "sha256:...",
        "streets": { "count": 63, "combined_sha": "sha256:..." },
        "boxes_sha": "sha256:..."
      },
      "metrics": {
        "pages": 63,
        "georeferenced": 57, "misscale": 2, "outlier": 0, "deferred_unconfirmed": 1,
        "truth": {
          "compared": 55, "mean_error_m": 41.3, "median_error_m": 22.0,
          "per_page": { "p33": 12.4, "p44": 62.1 }
        }
      }
    }

The `truth` block under `metrics` is omitted for volumes with no truth data; those report coverage-only metrics (georeferenced / misscale / outlier / deferred counts). The optional `--label` gives a run a human-readable name alongside the hash.

Record `seed` (and pin it) so an A/B diff reflects the code change rather than RANSAC RNG.

### Comparing runs

The archive is only as useful as the diff, so add a first-class:

    uv run mapsnap experiments diff <run-id-a> <run-id-b> [volume ...]

It reads the two `manifest.json` files (and georef corners where needed) and reports:

- **coverage delta** — pages newly georeferenced / newly failed / changed status (e.g. `misscale → ok`);
- **error delta vs truth** — per page and mean/median, sorted so regressions surface first;
- a **one-line roll-up** — Δ pages placed, Δ mean error, # regressions, # improvements.

With no `volume` argument it scans every volume's `artifacts/<id>/`, so even with the volume-oriented layout you get a cross-volume roll-up. (Both ids must exist; run `fit` on each baseline first.)

### Disk usage

Measured on `brooklyn_ny_1939_vol_1` (63 pages):

| artifact          | size per run |
| ----------------- | ------------ |
| `p*.streets.json` | ~6.5 MB      |
| `p*.georef.json`  | ~0.44 MB     |
| `p*.txt` logs     | ~0.09 MB     |
| IIIF + compare    | ~0.15 MB     |
| `manifest.json`   | ~0.05 MB     |
| **total**         | **~7.2 MB**  |

`streets.json` is ~90% of the total. It is OCR output and is unchanged across georef-only experiments — which are most of them — so copying it on every run would be the dominant cost. At ~7 MB/run that adds up across a diverse set: e.g. 20 volumes × 10 experiments ≈ **1.4 GB**.

The streets.json deduplication described under [What gets archived](#what-gets-archived) avoids this: when a run's `streets` `combined_sha` matches an existing archive's, the new archive symlinks to that archive's `streets.json` instead of copying. That drops a georef-only run to **~0.7 MB** (≈140 MB for the example above) while keeping the manifest self-describing.

## What's good about this

- Creates a clear archive of **(commit, flags, inputs) → results**, not just commit → results.
- Early-out lets experiments share a baseline (`main` vs `branchA`/`branchB`), and keying on input hashes keeps that early-out from serving stale results.
- Preserves artifacts for debugging and comparison, and the built-in `experiments diff` turns the archive into an actual regression check.

## Downsides

- It's still up to the user to decide when they need to rerun `mapsnap ocr` rather than just `mapsnap fit`. (Partial help: streets.json records the OCR command/timestamp, so a "streets.json is stale relative to the current OCR code" warning can be derived.)
- Archiving an experiment requires committing the code change first — accepted, since we lean on git rather than reinventing diff tracking.
- The streets.json symlinks make an archive non-self-contained: deleting the archive that holds the real bytes leaves the symlinks in other archives dangling. (Acceptable for a local experiment cache; a future `experiments gc` could re-materialize before pruning.)
