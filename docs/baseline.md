# Running a corpus baseline

How to produce a full tagged run across the scored volumes, and the mistakes
that have cost real time. Companion to `docs/fit-pipeline.md`, which explains
what happens *inside* a single page's fit.

A baseline is the acceptance test for anything that changes admission,
recognition, or fitting: per-volume scores against the previous tagged run,
with enough hygiene that a difference means the code changed rather than the
caches did.

## The volume set

The scored corpus is every volume with all three of `mapsnap.json`,
`centerlines.geojson` and `main.iiif.json` (truth) — 20 as of 2026-08-28.
Volumes with the first two but no truth can be run, but they contribute
nothing to a score comparison.

```sh
for d in data/*/; do
  [ -f "$d/mapsnap.json" ] && [ -f "$d/centerlines.geojson" ] &&
  [ -f "$d/main.iiif.json" ] && echo "$(basename $d) $(ls $d/p*.jpg | wc -l)"
done
```

## Pre-flight (do this before the lanes start)

Three things the pipeline does not do for itself. All are cheap; skipping any
of them silently corrupts the comparison.

1. **Re-split every parent page.** Splits change when `mapsnap/split.py`
   changes, and volumes that have not been re-split since carry stale panels.
   `rerun` does run `split`, but doing it up front tells you *which* volumes
   moved before you interpret their scores:

   ```sh
   uv run mapsnap split $(ls data/$VOL/p*.jpg | grep -v '__')
   ```

   Two-panel sheets number the panel holding the bottom-left corner first
   (#379, matching OIM), so a volume split before that change renumbers about
   half of its two-panel pages here — expect `p12__1` and `p12__2` to swap
   meaning on those, in every artifact that names them.

2. **Let `split` drop the reads of panels that changed, then `ocr --resume`.**
   `split` compares the new panel rings with the previous `panels.json` and
   deletes every derived sidecar (`boxes.json` / `streets.json` / `txt` /
   `georef*.json` / `contradiction.json`) of each panel whose ring moved, was
   renumbered, or no longer exists — it prints which. Panels whose ring did not
   change keep their reads and fits.

   That matters because **`ocr --resume` skips on `.streets.json` existence
   alone** (plus the recognizer weights): before this, a panel whose pixels
   changed kept its stale reads. `craft --resume` compares mtimes and always
   re-detected on its own. So after re-splitting, `ocr --resume` re-reads
   exactly the panels that need it and nothing else.

3. **Delete `p*.contradiction.json` (#258).** `fit`'s `clear_derived_sidecars`
   globs only `p*.georef*.json`, so adjacency-gate demotions survive into the
   next run and keep steering snap's search centers. The 2026-08-10 pre-flight
   found 17 of these across 8 volumes, several written two runs earlier.

   ```sh
   rm -f data/*/p*.contradiction.json
   ```

(Since #349, `fit` deletes `artifacts/osm_snap/` and `artifacts/street_solve/`
candidates and selection files itself, so the old #275 stale-candidates
pre-flight is no longer needed — every fit is cold-cache by construction.)

## Two lanes, small volumes first

**Lane A owns OCR and is serial.** Two concurrent EasyOCR jobs bog the machine
down; never run two. **Lane B fits** each volume as lane A releases it, which
is safe to overlap because a fit only reads its own volume. A fit does not
monopolise the GPU the way OCR does, so **the fit lane can run two at a
time** — useful when it falls behind, and the right shape for a parameter
sweep, where every job is a fit.

Single-lane runs cost roughly double: on 2026-08-10, washington_dc took 29 min
and kansas_city 69 min end to end, and the fit half of each was pure blocking
time for the next volume's reads.

```sh
# lane A
for VOL in $VOLUMES; do
  uv run mapsnap rerun data/$VOL --tag $TAG --recognizer-weights $W --no-fit
  touch $READY/$VOL
done

# lane B
for marker in $READY/*; do
  VOL=$(basename $marker)
  uv run mapsnap fit data/$VOL --tag $TAG
  uv run mapsnap score data/$VOL/$TAG.iiif.json   # report as they land
done
```

**Order the volumes smallest first.** Cross-volume signal is what tells you
whether a change generalizes, and a champaign (41 pages) plus a brooklyn (65)
plus a nashville (74) in the time one washington_dc (159) takes is three
independent readings instead of one. Save the big volumes for the tail, when
you already know roughly what you are looking at.

**A marker file must mean the step succeeded, not that it ran.** Gate the
release on the exit code. A lane A whose `rerun` failed but still marked its
volume ready leaves the fit lane fitting whatever `streets.json` was on disk
from an earlier run — which produces *plausible* wrong numbers rather than
obviously wrong ones. This cost most of a night on 2026-08-10.

Score each volume as it lands rather than only at the end — a change that goes
badly wrong is visible in the first two or three volumes, and there is no point
burning the night on the other fourteen.

## Changing the recognizer

`--recognizer-weights PT` implies a full re-OCR: it drops `--resume` from the
OCR step, because cached `.streets.json` reads came from different weights and
resuming would produce a corpus that silently mixes two recognizers.

That makes the OCR lane much slower than a normal re-run (which reuses reads
and only re-recognizes new panels), so budget for it.

**Resumed reads are only valid when the key maps did not change either.** OCR
vocabularies are keymap-restricted where a usable key map exists, so a baseline
that regenerates the keymap chain (the normal case — `rerun` always rebuilds
it) must re-OCR from scratch: delete `p*.streets.json` up front. The 2026-08-28
run initially resumed reads and had to be restarted.

## Fixing bugs found along the way

You can fix small issues that arise while running the pipeline. Commit these
changes on the same branch as the baseline update and include them in the PR.
Do not commit directly to `main` or push commits directly onto `origin/main`.
This avoids breaking CI (see #360).

## Reading the results

```sh
uv run mapsnap score data/*/$TAG.iiif.json                  # the table
uv run mapsnap experiments diff $PREVIOUS_TAG $TAG          # per-page
```

Interpretation rules learned the hard way:

- **Compare against a run scored under the same truth.** Truth changes
  (`oim-panels`, split-truth rebuilds) move scores on their own: the #274
  rebuild moved grand_rapids 68.3 → 70.3 with no pipeline change at all. When
  in doubt, re-score the *old* IIIF under today's truth and compare that.
- **`experiments diff` compares archived `.txt` files**, which were written
  under whatever truth was current then — so it straddles truth revisions
  silently. It is the right tool for attribution *within* a truth regime and
  the wrong one across it.
- **A "regression" may be an accounting change.** Split-panel RMSEs are graded
  over each panel's own truth region, so a region rebuild changes the number
  without the fit moving a millimetre. Check whether the georef sidecars are
  byte-identical before blaming the pipeline.
- **One volume is not a result.** Changes sign-flip by volume routinely (the
  1-GCP ablation: Detroit +2.5, Brooklyn −4.0). Weight by land, and read the
  aggregate.
- **When several changes land in one run**, note which volumes each could have
  touched *before* looking at the scores, so attribution is a check rather than
  a story fitted after the fact.

## Recording the run

Write `data/<TAG>-rerun.md` with the per-volume table (score, ≤25 ft share,
disaster share, placed/total), the aggregate, wall-clock and lane timings, and
a short note per volume that moved more than ~1 point. Past run reports are the
only durable record of why a number changed.
