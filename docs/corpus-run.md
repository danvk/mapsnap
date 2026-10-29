# Running a corpus pass on AWS Batch and publishing it

How to take the pipeline from a merged commit to a published run on
data.mapsnap.org, and what went wrong the last time we did it. This is the
end-to-end view. The mechanics of each Batch script are in
`scripts/batch/README.md`, and `docs/baseline.md` covers the 20-volume
baseline that should come before any corpus pass.

Everything here was learned on corpus-v2 (2026-10-16 to 2026-10-22), which
became the v2.0 publish.

## The shape of a pass

| Step | corpus-v2 | Took |
|---|---|---|
| Canary | `d5d2593d` | — |
| Batch 1: a random 1,000 items | `74af3f0f`, 1,000/1,000, $24.69 | — |
| The rest: 34,159 items, 8 per child | `b78a856e`, 4,228/4,270 children, ~$300 | Sat 10:26 PT → Tue 01:21 ET (~60 h) |
| Mop-up on regular machines | `c69a5f65`, 32 items | 2 h |
| Mop-up on 16 GB | `e89ebb7b` (Atlantic City), `2720965d` (two giant key maps) | 2 h |
| Correction pass (`corpus-v2.1`) | `3601ec45`, 2,500 small volumes with `--locations`, ~$8 | 1.5 h |
| Scoring against OIM truth | 690 volumes | ~20 min |
| Confidence re-score | 379,001 annotations | ~1 h, mostly downloading |
| Publish build | `publish_run` (90 s), `build_run_site` with zips (40 s), `build_item_pages` | see step 6 below |
| Upload to R2 | `aws s3 sync --profile r2` | — |

The correction pass shouldn't have been needed; see the next section.

## Before submitting anything

### Check the job definition's command line against `loc-fit --help`

**Why this matters:** corpus-v1 and corpus-v2 both ran without `--locations`.
`loc-fit` had supported it since #531: a volume of ≤ 10 pages with a catalogue
coordinate reads only the streets within 5 km of town. But the job definition's
command never passed the flag, so every small volume was read against its whole
county. `publish_run.py` then dropped 2,069 pages placed more than 5 km from
town and withheld 590 volumes. Fixing that took a 2,500-volume correction pass
(`corpus-v2.1`) and an afternoon of analysis (#641).

**The rule:** when a pipeline flag should be on in production, add it to
`setup.sh`'s command and re-register the definitions in the same PR. Before
every pass, read the live definition and compare its flags with
`loc-fit --help`:

```sh
aws batch describe-job-definitions --job-definition-name mapsnap-loc-fit --status ACTIVE \
  --query 'reverse(sort_by(jobDefinitions,&revision))[0].{rev:revision,image:containerProperties.image,cmd:join(` `,containerProperties.command),params:parameters,attempts:retryStrategy.attempts,timeout:timeout.attemptDurationSeconds}'
docker run --rm --platform linux/amd64 mapsnap:<sha> loc-fit --help
```

Look at every `loc-fit` option that isn't in the command and decide on purpose
whether it should be there. Then check that the image parses the exact command
line:

```sh
docker run --rm --platform linux/amd64 mapsnap:<sha> loc-fit --check-args <the definition's arguments>
```

As of #641 the command passes `--counties items.tsv city-and-hand-items.tsv`
and `--locations item-locations.tsv`, with 10 attempts of 12 hours each.

### Re-register with the script, and keep its defaults

`setup.sh --job-definitions-only` registers whatever its defaults say. Before
#641 those were 5 attempts of 3 hours, while the live definitions had 10 of
12, so a plain re-register would have quietly cut both. The defaults now match
what the corpus needs. Never copy a revision from `describe-job-definitions`:
it lists revisions in no particular order.

### Build the image from `main`, and check it

```sh
git checkout --detach origin/main
docker buildx build --platform linux/amd64 --build-arg GIT_SHA=$(git rev-parse HEAD) \
  -t mapsnap:$(git rev-parse --short HEAD) --load .
docker run --rm --platform linux/amd64 --entrypoint /bin/sh mapsnap:<sha> -c 'env | grep MAPSNAP_GIT_SHA'
```

One corpus-v2 build came out with `MAPSNAP_GIT_SHA=unknown`, because the
build argument hadn't reached the build. A rebuild with the full SHA written
out fixed it. Check the variable before pushing, because every published file
records it.

Then:
1. Push with `scripts/batch/push-image.sh`. It only moves the source layer
   after the first push.
2. Point the definitions at the image:
   `IMAGE=…/mapsnap:<sha> scripts/batch/setup.sh --job-definitions-only`.

In corpus-v2 both steps ran under the admin `aws login` session, which expires
every 10–20 minutes.

### Check inputs the pipeline can't fix for itself

- **County:** every item needs one in `_craft/items.tsv` or
  `_craft/city-and-hand-items.tsv`. Without one, `loc-fit` exits 4 and Batch
  never retries it. In corpus-v2 that was 36 children:
  - the NJ coast atlases, which cover multi-town shorelines;
  - Alaska volumes, whose boroughs weren't cut;
  - Mexico, Cuba and multi-state warehouse sets;
  - a few one-offs.

  The fixable ones are hand-labelled in `_craft/hand-items.tsv`, with new
  extracts under `osm-by-county/`:
  - Alaska boroughs, cut with `mapsnap osm-counties` from the renumbered
    named-highways file on fivetera;
  - Junction City, an `osmium merge` named `US05139;US22111;US22027`, which
    follows the multi-county precedent.

  Count the items with no mapping before a pass, not after.
- **Scans:** some catalogue entries have none. Toledo 1868 (`2016586562`) is a
  "ghost" row in the mirror mapping and exits 3. That's expected, but don't
  keep resubmitting it.
- **Run tag:** use letters, digits, hyphens and dots. Batch job names can't
  contain dots, and `submit.sh` now maps them to hyphens (`fit-corpus-v2-1`).

## When to run

**Run big passes overnight and on weekends, Pacific time.** Spot reclaims
follow the on-demand work day in `us-west-2`. Instance terminations during
corpus-v2:

| Window (UTC) | Terminations |
|---|---|
| Sat 17:00 – Sun 23:00 (30 h) | 25 |
| Mon 00:00 – 14:00 | 30 |
| **Mon 15:00 – 22:00** (08:00–15:00 PT) | **87** |
| Mon 23:00 onward | 4 |

On Monday afternoon the fleet fell from 128 to 113 running children, and two
children were reclaimed on all five of their attempts. A child resumes from its
checkpoints, so extra attempts are cheap; that's why the definitions now allow
10.

A Friday-evening start puts most of a 2–3 day pass on the weekend.

**Don't change the compute environment while a job runs.** We changed its
instance types during the canary, and Batch replaced the running instances
about 30 minutes later (`jobExecutionTimeoutMinutes`). The children recovered
through retries, but every one of them restarted.

## Running a pass

1. **Canary** (a handful of items, including one with a key map), then a first
   batch. Make the batch a random 1,000 (`sample-items.py`) **plus every OIM
   truth volume**, and check cost, failures and peak memory before the rest.
2. **Retrain p(good) on that batch before running the rest.** The confidence
   model is trained against OIM truth on a run's own provenance, and scores
   drift as the pipeline changes: corpus-v2 ran with the corpus-v1 model and
   was 13 points under-confident. Retraining afterwards meant downloading and
   rewriting every published IIIF (see *Confidence* below). Instead:
   - score the batch's truth volumes;
   - run `scripts/train_confidence.py` over them;
   - merge the new `models/confidence.json`, then build and push the image the
     rest of the corpus runs on.

   Only the first batch then needs `rescore_confidence.py`, a few minutes for
   1,000 volumes.
3. **Plan and submit the rest under the same run tag:**
   ```sh
   scripts/batch/plan-items.py rest.txt --per-job 8 --out planned.txt
   PER_JOB=8 scripts/batch/submit.sh corpus-v2 planned.txt
   ```
   An item already published under the tag is skipped, so passes can overlap
   and resubmissions are safe.
4. **Sample the fleet alongside it:**
   `scripts/batch/snapshot-instances.sh <job> instances.tsv &`. Terminated
   instances leave `describe-instances` within the hour, so the cost can't be
   reconstructed afterwards.
5. **Watch it** with `scripts/batch/status.sh <job> --watch`, or a loop that
   reports only new failures and progress milestones. That saves reading the
   console for 60 hours.

### Don't trust the early estimates

Time and cost projected from the first few hundred finished children run high.
In corpus-v2 they were still far too high after 500 items. That's a side effect
of the planner's order, which is deliberate:

- **The heaviest children run first.** `balance_items` places items
  largest-first into the lightest child, and ties go to the lowest index. So
  the first children hold the biggest volumes. In corpus-v2, planned work fell
  from about 149 sheet-equivalents a child at the start of the array to 104 at
  the end. Batch starts children roughly in index order, so the first to finish
  are the heaviest.
- **Each child runs its biggest item first,** because items keep the planner's
  order.

This is worth keeping. A long child that starts late sets the end of the whole
run, and a big item reclaimed near the end has to start over while the rest of
the fleet sits idle. Starting the large volumes early shortens that tail (#625
has the straggler numbers). The cost is only in the estimates.

**Estimate from work done, not children done.**
- **Time:** sum the planned sheet-equivalents of the finished children
  (`sheets + 3` per item, as `balance_items` weighs them), divide by elapsed
  time, and apply that rate to what's left.
- **Cost:** price the fleet snapshots, dollars an hour multiplied by the hours
  left.

Expect the rate to rise as the run goes on. corpus-v2's estimated finish moved
from Tuesday to Monday between the 500-item and 2,300-item marks. Separately,
`collect-run.py`'s corpus projection is per child, which overstates a chunked
run's cost about 4× (#624, a real bug).

### Reading failures

| What you see | Meaning | What to do |
|---|---|---|
| exit 4 | No county (or other missing input) | Fix the mapping, then mop up |
| exit 3 | Unprocessable: no scans, or not in the mirror | Nothing |
| `Host EC2 … terminated` on every attempt | Spot reclaims, not a bug | `missing-items.py` finds what wasn't published; mop up |
| exit 1 whose log ends in `died with signal 9 … at a peak of 7,8xx MB` | Out of memory | Find the stage; mop up on 16 GB; file the cause |
| exit 1 with a traceback | A bug | Read it; it'll fail all attempts the same way |
| exit 137 | Container killed, also for memory, or terminated by us | Same as OOM |

- **Lost children aren't lost items.** A failed child of 8 items has usually
  published 7. Work from `missing-items.py` (what S3 doesn't have), not from
  failed children.
- **Logs:** each attempt's log is the CloudWatch stream named in `describe-jobs`
  (`container.logStreamName`; `attempts[]` for earlier ones). The stages print `[stage: Ns, peak N MB]`.
- **Stop a child that will fail every attempt:** `aws batch terminate-job`. It
  takes a minute or two to stop, so wait for FAILED before resubmitting the same
  item. An item repeating the same OOM uses its attempts at about 12 minutes
  each.

### Known memory problems

| Item | Stage | Peak | Status |
|---|---|---|---|
| Brooklyn 1908 (163 MP key map) | key-map cartouche | 7.9 GB | fixed by #637 |
| Boston 1867 (146 MP key map) | key-map OCR | 7.9 GB | fixed by #637 |
| both of the above | key-map `page_regions` | 8–9 GB | #640 |
| Atlantic City 1921 | region masker in `iiif` | 11.4 GB | #638, open; needs 16 GB |
| sanborn05408_004 (167 pages) | region masker in `iiif` | 11.6 GB | needed 16 GB in the v2.0 rerun |

- **Key maps:** a giant key map (> 100 MP; a typical one is about 50 MP) is the
  common thread in the first three. Full-resolution float64 copies of the
  image cost 24 bytes a pixel.
- **The masker:** Atlantic City isn't explained by grid size. Brooklyn 1909's
  grid is 2.6× bigger and fit in 7.5 GB.

### Mop-up

- **Same run tag, `PER_JOB=1`:** use `mapsnap-loc-fit` for items lost to
  reclaims or fixed by a code change, and `mapsnap-loc-fit-large` for memory.
  An attempt resumes from the checkpoints the earlier attempts uploaded
  (`runs/<tag>/checkpoints/`), so it goes straight back to the stage that
  failed.
- **Arrays need at least 2 children.** Pair a lone item with any
  already-published one, which exits within seconds.
- **Overriding a parameter at submit:** use
  `--parameters cityCounties=…,locations=…` in `aws batch submit-job`.

### Check the masks before publishing

corpus-v2 published 127 volumes whose annotations were all whole sheets, and
about 180 more with whole groups of pages unmasked (#647). Nothing failed, and
nothing in the job logs showed it.

- **Each item's manifest counts them:** `metrics.masks` has `annotations` and
  `whole_sheet`. Flag any volume where `whole_sheet` is close to
  `annotations`. A handful of whole-sheet pages is normal; a whole volume is
  not.
- **`runs/<tag>/fit.log`** holds `fit`'s full output for each item, including
  the masker's `region masks: …` lines.
- **The cause in corpus-v2 was letter case** (#647, fixed by #657). The iiif
  step looked for `p100l.jpg` where the scan is `p100L.jpg`. That works on the
  Mac, whose filesystem ignores case, and finds nothing on Batch's Linux. Since
  #648 an unreadable region map fails the item instead of publishing whole
  sheets.

### Re-running only the last stages

When a fix only changes reconcile or the iiif step, an item doesn't need its
chain re-run. This is how v2.0 was republished in 2026-10 (#644, #647): 961
items, about 4 minutes each.

1. **Seed a new run tag:** `scripts/batch/seed-run.sh <new-tag> <items.txt>
   [<source-tag>]`. Each line of the list may name its own source tag, which
   matters when the published set mixes runs (v2.0 is corpus-v2 plus
   corpus-v2.1). It's a server-side copy of each item's run without its done
   marker, key-map IIIF, manifest and `fit.log`. loc-fit's checkpoints and
   `fit --resume` then skip everything up to the region stage.
   - **Seed everything before submitting anything** under the new tag, and
     never seed an item that has already run under it. `aws s3 sync` re-copies
     any file whose size differs, so it puts the old run's sidecars back over
     the new run's. That happened to NY 1923's pilot: its published IIIF is
     right, but its corpus-v2.2 sidecars are corpus-v2's. The script now
     refuses items with a done marker.
   - **`--copy-props none`** is needed: `mapsnap-mirror` can't read object
     tags, which a plain S3-to-S3 copy does.
2. **Build, push and re-register** as above, then **pilot two items**
   (`PER_JOB=1`; an array needs two). Check each item's `fit.log`: only
   split, craft, region, reconcile and iiif should run.
3. **Submit** under the new tag (`PER_JOB=8`).
4. **The job definitions pass `--locations` by default.** In a rerun, items of
   ten pages or fewer have their streets clipped to 5 km of the catalog
   location, even if they originally ran without it.
5. **Collect and check:** each item's `runs/<tag>/mapsnap.iiif.json` and the
   manifests' `metrics.masks`, then publish as below, copying the new tag's
   files over the published set.

### Cost

- **Collect promptly:** run `collect-run.py` within 24 hours. Batch drops the
  attempt history after that.
- **Its corpus projection is per child, which is wrong for chunked runs**
  (#624). Price from the fleet snapshots instead.
- **corpus-v2 came to about $300** for 6,833 job-hours. 13.8% of children were
  interrupted at least once, and about 8% of the work was redone.

## Scoring against OIM truth

The truth set is the 690 OIM volumes that are done, selected on 2026-10-17:
- ≥ 90% complete with a multimask, or ≥ 95% complete;
- or done apart from skeleton sheets and small splits;

and in the mirror. The selection and fetch scripts are in
`~/Documents/mapsnap/oim-list/`. Truth lives in
`~/Documents/mapsnap/oim/<item>/` (`main.iiif.json`, plus `oim/` when there
are split sheets).

To score a run, build one directory per volume holding:
- the run's `mapsnap.iiif.json` and `p*.panels.json`;
- links to the truth (`main.iiif.json`, `oim/`);
- the county's `centerlines.osm.pbf`.

Then call `mapsnap.score.volume_page_scores` on each one. The working copy of
this is `~/Documents/mapsnap/score-690/` (`setup.sh` and `compare.py`). Report
both means:
- **volume-equal** (the headline): 71.6 for corpus-v2, 67.3 for corpus-v1;
- **sheet-weighted:** 76.9 and 73.7.

Some truth volumes are suspect. Rochester 1950 Vol. 1 holds Vol. 1s's fits
(ohmg-dev/OldInsuranceMaps#434), and the confident-but-wrong list from #633
finds more.

## Confidence (p(good))

Retrain the model on the first batch, before the rest of the corpus runs (see
*Running a pass*); the steps below are for when that didn't happen.

The model is trained on a run's own provenance against OIM truth
(`scripts/train_confidence.py`). Retrain it after any change to placement,
because an old model's scores drift with the pipeline. Scored with the
corpus-v1 model, corpus-v2's poses were 13 points under-confident.

To update a published run without re-fitting it:
1. Fetch each item's `mapsnap.iiif.json`, key-map IIIF, `p*.provenance.json`
   and `p*.panels.json`. That's 35,101 items, 4.9 GB and 774k files. Use
   `get_object` with 48 threads (about 13 items a second); `download_file`
   is 3× slower on small files.
2. Run `scripts/rescore_confidence.py <item dirs…>`, which writes
   `mapsnap.confidence-<model>.iiif.json` beside each input.
3. Publish only one main IIIF per volume. Copy the re-scored file into the flat
   `publish_run` input as `<item>.iiif.json`; the model name lives inside it,
   as `confidence_model`.

A run made after the model was retrained needs no re-score: its annotations
already say `confidence_model: corpus-v2`.

## Publishing to data.mapsnap.org

The bucket is the R2 bucket `mapsnap`, reached with `aws s3 … --profile r2`.
The local mirror is `~/Documents/mapsnap/data.mapsnap.org/` (`runs/`,
`items/`, `manifests/`).

1. **Assemble the inputs.** `publish_run.py` wants flat directories:
   `iiif/<item>.iiif.json` and `keymap-iiif/<item>.iiif.json`. For a run
   filled from two tags (v2.0 is `corpus-v2` plus 2,500 `corpus-v2.1`
   volumes), copy the later tag over the earlier. Remove a stale key-map file
   when the later tag has none.
2. **Build the run:**
   ```sh
   PYTHONPATH=.:scripts/atlas python scripts/atlas/publish_run.py \
     --iiif-dir …/iiif --keymap-dir …/keymap-iiif \
     --small-keymaps <empty file> --atlas-dir ~/github/mapsnap/app/public/atlas \
     --out ~/Documents/mapsnap/data.mapsnap.org/runs/v2.0 \
     --base-url https://data.mapsnap.org/runs/v2.0
   ```
   - **Start from an empty `--out`.** A volume that drops out otherwise leaves
     its old files behind.
   - **`--small-keymaps`:** pass an empty file. The pipeline has ignored
     key maps under 2.5 km on its own since #525, and the old list was
     corpus-v1's.
   - **`--atlas-dir`:** `app/public/atlas` is generated and isn't in a clean
     checkout.
3. **Read the filter counts** it prints and the `dropped_*` columns of
   `items.tsv`:
   - A rule the pipeline already enforces should drop nothing. For v2.0, the
     "sheet too large", skeleton and key-map rules dropped 0 pages.
   - The proximity rule dropped 2,069 pages in the first build and 5 after
     `corpus-v2.1`.
   - The last five are a boundary mismatch: the clip keeps whole streets that
     cross 5 km, while the filter judges page centers (#641).
4. **Check the run tags.** Every published file carries a page-level
   `metadata` entry `run` (`corpus-v2` or `corpus-v2.1`). Count them against
   the list of volumes you meant to replace.
5. **Build the README, the zips and the site pages:**
   `build_run_site.py --run-dir … --site-dir ~/Documents/mapsnap/mapsnap.org --version v2.0 --data-url https://data.mapsnap.org/runs/v2.0 --zip`.
   The site is a local git repo with no remote; Dan deploys it.
6. **Rebuild the item pages:** first
   `PYTHONPATH=.:scripts/atlas uv run python scripts/atlas/unmirrored_scans.py`,
   then
   `AWS_PROFILE=mapsnap build_item_pages.py --run-dir … --version v2.0 --out ~/Documents/mapsnap/data.mapsnap.org --run-iiif-dir …/iiif`.
   The first fetches, at 2 a second, the size of each scan the mirror skips
   (covers, title pages, indexes, #467's map sheets) from the CDN, into
   `~/Documents/mapsnap/cdn-info/`. The cache makes later runs instant; the
   first took about 74 minutes for 8,864 scans. Item pages show those scans
   too, and the 44 items the mirror skipped entirely get pages of their own.
   They're baked per run, and the script reads panels from the run tag recorded
   in each annotation. `--run-iiif-dir` is step 2's `--iiif-dir`. A volume the
   run placed nothing of has no published file, so its unplaced images and
   their panels come from the run's own annotation instead. Without it, the
   records of 4,638 v2.0 volumes list no images at all.
7. **Upload:** `aws s3 sync … s3://mapsnap/… --profile r2` for
   `runs/<version>/`, `items/` and `manifests/`. The edge serves
   `max-age=14400` whatever the upload says.
8. **Purge the edge cache** once every sync has finished, or the old files are
   served for up to 4 hours. In the Cloudflare dashboard: the mapsnap.org zone,
   then Caching → Configuration → Purge Everything. (Purging by prefix needs an
   Enterprise plan.) Browsers keep their own copies for the same 4 hours.
9. **Rebuild and deploy the atlas** (mapsnap.org/atlas):
   ```sh
   uv run python scripts/atlas/build_places.py
   uv run python scripts/atlas/build_footprints.py \
     --run-dir ~/Documents/mapsnap/data.mapsnap.org/runs/<version>
   aws s3 sync app/public/atlas s3://mapsnap/atlas/<version>/ --profile r2
   cd app && VITE_ATLAS_DATA_URL=https://data.mapsnap.org/atlas/<version> \
     npm run build:atlas
   cp -R dist-atlas/atlas.html dist-atlas/atlas-assets ~/Documents/mapsnap/mapsnap.org/
   ```
   `app/public/atlas` is generated and gitignored. The page reads it from
   data.mapsnap.org and reads the run's annotations and item records from
   there too, so deploy it after steps 6–8. Then deploy the site.

Back up the local `runs/<version>/`, `items/` and `manifests/` before
rebuilding them in place. The v2.0 republish kept its backup in
`~/Documents/mapsnap/backups/`.

## Working on the Mac

- **zsh doesn't word-split `$var`,** and `path`, `status` and `keymaps` are
  read-only names. Put loops in `bash` scripts.
- **Non-interactive shells may not have Homebrew on `PATH`:**
  `aws: command not found`. Export `/opt/homebrew/bin`.
- **`pkill -f pattern` also kills the shell that runs it,** if the pattern
  appears in its command line.
- **The admin `aws login` session expires every 10–20 minutes.** An empty
  `aws s3 ls` means the session expired, not an empty prefix. Use
  `AWS_PROFILE=mapsnap` for everything that allows it.
- **The amd64 image can't run torch under Docker's x86 emulation on Apple
  Silicon** (illegal instruction). For a local repro, build the arm64 variant
  (`scripts/batch/README.md`).
- **macOS compresses memory,** so a local peak RSS understates Linux. Compare
  NumPy/PIL peaks with `tracemalloc`. That's how #637 and #640 were measured.
- **One heavy local job at a time.** Even two OCR workers bog down the laptop.

## Open threads after v2.0

v2.0 was republished on 2026-10-24 with #644, #645 and #657: 343,403 sheets
placed, up 966. Still open:

- **#643 / #653:** content regions that are too large (NY 1923 p26 and its
  inset).
- **#646:** snap skips panel-only volumes (Rosendale).
- **#650 / #651:** a learned accept score for snap. The replace model holds up
  on held-out volumes; the rescue model doesn't yet. Reconcile needs
  recalibrating with it.
- **#652:** panel bars and priors ignore panel size.
- **#658:** the scorer finds `panels.json` by page key (case, zero padding).
- **#659:** skeletons published instead of an agreeing full-color twin (88
  sheets).

- **#638:** Atlantic City's masker memory.
- **#614:** guarding rasters sized by geography.
- **#624:** `collect-run.py`'s per-child projection.
- **#625:** planner balance.
- **#626:** key-map stage checkpoints.
- **#627:** annexed-town counties.
- **#628:** the cutline splitter misses cuts interrupted by insets.
- **#630:** the cartouche pass reads every word box.
- **#634:** low-priority cleanup.
- **`page_regions` is very sensitive to tiny input changes** (#640): ±1 in the
  resized pixels changed regions on 39 of 41 key maps.
- **The publish proximity rule's boundary** (#641).
