# How a page gets its fit — and every way it can be rejected

Stage structure as of #270 phase 3; the per-gate examples are from the
`2026-08-07` run (main @ `31146e1`) and still name the right pages, though a
rejection that used to rename a file now records a `status` inside it. Grep
the page's `<stem>.txt` sidecar or the run's fit log for the quoted messages.

`mapsnap fit DIR --tag T` runs six stages in a fixed order. Each stage
communicates with the next entirely through per-page sidecar files:

```
clear_derived_sidecars          (#247: delete p*.georef*.json so nothing stale survives)
        │
        ▼
1. mapsnap georef               RANSAC pose           → p*.georef.json
        │
        ▼
2. mapsnap adjacency-gate       contradiction verdict + re-search hints
        │
        ▼
3. mapsnap snap                 rescue / arbitrate / refine → p*.georef-snap.json
        │
        ▼
4. mapsnap street-solve         street-constraint pose     → p*.georef-street.json
        │
        ▼
5. mapsnap reconcile --publish  weighs every pose jointly  → p*.georef-final.json
        │
        ▼
6. mapsnap iiif                 publishes p*.georef-final.json, and only that
```

**One sidecar per channel, one verdict inside it** (#270 phase 3). A channel
writes its pose to its own file and records what it concluded in a `status`
field; `status: fitted` (or no status at all) means the channel stands behind
the pose, and anything else — `misscale`, `outlier`, `keymap-outlier`, `1gcp`,
`nofit`, `contradicted` — is a pose the channel produced and declined. The
pose stays readable either way, which is the point: the arbiter weighs
declined poses too, and the fargo p55 class is precisely a declined pose that
was right. A channel that reached more than one pose (georef's key-map retry)
keeps the loser under `rejected` in the same file.

A page is **published** iff its `georef-final.json` carries corners. The
arbiter writes one for *every* page, so declining to place a page is a
recorded decision (`corners: null`) rather than the absence of a file.

This replaced a scheme where the *filename* carried the judgement — a page
had up to nine kinds of `p*.georef*.json` and publication was first-glob-wins
over them. That made stage ORDER the thing that decided what got published,
gave each stage a reason to hide its predecessors' files, and let a reader
that didn't know the whole variant vocabulary drop pages silently.

---

## Stage 1: RANSAC georef (`georef_from_labels.py`)

### 1a. Label admission (per detection)

`prepare_label_features` turns `<stem>.streets.json` into matchable features:

1. Drop `ignore`-flagged reads (scale notes, pipe labels).
2. **Hint promotion** (`promote_avenue_letters`): a letter/word beside a type
   hint (`ST`, `AV`) is promoted and **bypasses the size floors entirely**.
   *Example: fargo p64 — five of the six streets its fit uses (`2 ND`, `7TH`,
   `8TH`, `12TH`, `14TH`) are under the 20 px floor and enter only via the
   `ST`/`AV` hints beside them.*
3. **Multiword assembly** (`assemble_multiword_streets`): collinear adjacent
   words whose concatenation names a street merge (`VAN` + `BRUNT`); one
   partner may be weak if the other is confident.
4. Dedup, then the **admission gate** (`passes_admission_gate`): confidence ≥
   0.15; short/long size floors (auto-calibrated per volume from the p25 of
   confident reads, e.g. 20/40 px on fargo) with **confidence relaxation**
   down to 0.7× for high-confidence reads; aspect ratio; number-only,
   bare-letter, and direction-word rejections; labels on coloured building
   fill dropped (`drop_labels_on_fill`).

Note the fragility: a detection admitted only via relaxation sits on a
confidence cliff. *Example: champaign p21__1's `W. HILL` (0.229 → 0.431 across
the #263 change) joined the fit and moved the pose ~0.9°.*

### 1b. Vocabulary tiers (key-map prior)

For a volume with usable key maps (`raw/*.keymap.json`), each page matches
against progressively wider street sets until something fits:

| tier | vocabulary | extra gates |
|---|---|---|
| 1. neighborhood | streets within the calibrated radius of the page's key-map location | — |
| 1b. relaxed retry | same, size floors × 0.6 | fitted scale must be plausible vs the region prior |
| 2. rectangle | all streets in the key-map bounding rectangle | scale prior 0.1–6.0×; **key-map-outlier check** (below) |
| 3. no locator / unplaced page | rectangle, else whole county | — |

**Key-map-outlier rejection + retry (#259):** a rectangle-tier fit placed
beyond the key-map radius, when the location is anchored and confident
in-radius streets exist, is recorded `status: keymap-outlier`, then the
neighborhood is retried at relaxed floors; the retry publishes only if it has
at least as many inliers as the rejected fit.

*Examples (fargo p58__2.txt, both messages):* `rejecting broadened fit: placed
3569 m from the key-map location (7.4x the 484 m radius)` then `accepting
in-radius retry: 6 inliers >= the rejected fit's 3` — published at 21.7 ft.
*Counter-example: fargo p55*, whose key-map location itself is ~2.9 km wrong:
the (correct!) RANSAC fit is rejected as a key-map outlier and the retry finds
nothing — the page leaves stage 1 unplaced. (Snap later makes this worse; see
stage 3.)

### 1c. GCPs and RANSAC

Admitted labels are extrapolated along their text direction; label pairs whose
streets intersect near the page **and** in OSM become intersection GCPs
(deduped). Then `ransac_hybrid`:

- **< 1 GCP** → `status: nofit` (a neighborhood-only sidecar so the debugger can show
  the key-map expectation). *Example: fargo p59__2.*
- **exactly 1 distinct intersection** → **deferred** for median-scale
  processing (see 1e). Log: `Only 1 distinct intersection: 10TH x 14TH;
  deferring for median-scale processing.`
- **≥ 2 GCPs**: every seed pair is solved as a 4-parameter similarity and
  scored by per-label inliers (point-to-polyline distance + direction), plus a
  rotation-histogram prior. A pair is vetoed when one of its own seed streets
  is a **rotation outlier** under the model it defines. Highest score wins;
  score ties are razor-thin on gridded cities (fargo p58__2's two quadrant
  hypotheses differed by 0.9%).

### 1d. Per-volume passes (after all pages fit)

These run once per `mapsnap georef` invocation, over all fitted pages:

**Reference scale + scale-outlier check.** The volume reference is the median
fitted scale (`Reference scale: 1.4538 px/ft (66 images)` — fargo). Each
page's ratio to it must land in `SAME_SCALE_BAND` (0.75–1.25),
`HALF_SCALE_BAND` (0.4–0.6) or `DOUBLE_SCALE_BAND` (1.8–2.2), **except** that
two escape hatches are consulted *before* the rename — a flagged page is
kept, never un-rejected later:

1. **Printed scale note** (`SCALE 100 FT TO ONE INCH` read off the sheet): a
   `keep` verdict overrides the bands (requires an over-determined fit); a
   mismatch against the note drops a fit the bands would have kept.
   *Drop example: washington_dc p227 — `15.4838 px/ft is 12.31x the page's
   PRINTED scale note` → p227.georef.json gets `status: misscale`.*
2. **Neighbor corroboration** (`scale_corroborated_by_neighbors`): if the
   page's scale is within the SAME band of the median of ≥ 2 *adjacent* fitted
   pages, the off-rung scale is treated as a real local drawing scale.
   *Example: fargo p59__1 — `Keeping scale outlier … 0.9294 px/ft (0.64x
   reference) corroborated by 6 adjacent page(s)`.* This is also the
   pipeline's current biggest known scale hole: p59__1's kept fit is
   cross-matched to truth panel p59__2 at a 22% scale error and scores as a
   239 ft disaster. Six neighbors agreeing does not make the neighborhood
   right.

   Ordering note: the corroboration check runs *inside* the scale pass, before
   any rename — so there is no path by which a rejected page later "recovers"
   within stage 1. What can resurrect a rejected page is stage 3.

   *Plain band rejection example: champaign p1 — `0.3726 px/ft vs reference
   1.4504 (0.26×)` → p1.georef.json gets `status: misscale`.*

**Deferred (1-intersection) processing.** Each deferred page is fitted at
every scale candidate (volume reference, key-map region rung, adjacent-page
rungs) × rotation candidates (from its own two labels, validated against ≥ 2
neighbors within 1.5× page dimensions); the page's own labels arbitrate.
Confirmed fits (≥ 2 agreeing neighbors) publish as `georef.json`; unconfirmed
ones are marked `status: 1gcp` and stay unplaced (snap may rescue them).
*Example: chicago p1n — `Deferred: 2 / 3 inlier labels`, unconfirmed →
`status: 1gcp` on p1n.georef.json.*

**Location outlier.** With ≥ 5 fitted pages, any page whose center is more
than `--min-distance-for-outlier-km` from every other page is marked
`status: outlier`. *Examples: grand_rapids p819 (`1.6 km from closest
map`), miami p91__3 (1.9 km).*

### Stage-1 outcome summary

| sidecar | meaning | 08-07 example |
|---|---|---|
| `georef.json` | published RANSAC fit | fargo p64 (11.1 ft) |
| `misscale` | scale off the volume rungs / printed note | champaign p1; dc p227 |
| `1gcp` | deferred fit, unconfirmed by neighbors | chicago p1n |
| `nofit` | key-map-placed page, no usable fit | fargo p59__2 |
| `keymap-outlier` | rectangle fit far from anchored key-map location | fargo p55 |
| `outlier` | center km from every other page | grand_rapids p819 |

---

## Stage 2: adjacency gate (`adjacency_gate.py`)

Pages print their neighbors' sheet numbers at the shared boundary; mutual
claims are ~100% precise. Both sides' printed claims are mapped through their
own fits; if the stamps land more than `GATE_STAMP_M` (100 m, widened by the
coarser sheet's scale relative to the volume median — #254) apart, the edge is
a contradiction and arbitration names a suspect:

1. **multi-contradiction** — a page contradicting ≥ 2 neighbors is the liar.
   *Example: fargo `Demoted p32: contradicts p22, p33, p36
   [multi-contradiction; gcps=2]`.*
2. **corroboration** — a page with other compatible mutual edges is vouched
   for; one with none is the suspect. *Example: fargo `Demoted p16:
   contradicts p15 [uncorroborated; gcps=2]`.*
3. both sides vouched for → **the edge is junk**; demote nobody.
4. tie-break: **fewer effective GCPs**. *Example: kansas_city `Demoted p551:
   contradicts p545 [fewer-gcps; gcps=1]`.*

A named suspect is demoted only with a **hard signal**: effective GCPs ≤ 2, or
rotation/scale deviation from the *volume median* beyond 15° / 0.2 log (known
flaw on multi-family volumes, #256). Demotion renames every channel sidecar to
`status: contradicted` on each channel sidecar and writes `p*.contradiction.json` with the partners'
stamp positions — which stage 3 reads as extra search centers. (These hint
files currently outlive the run; #258.)

---

## Stage 3: snap (`osm_snap_experiment.py`)

One pass per page matches its road-probability map against rasterized OSM
geometry over a rotation ladder and the volume's scale rungs, producing scored
candidates (`select_score`, `margin`, chamfer `verification`, `ncc_fine`, name
alignment). What happens next depends on the page's state:

**Unplaced pages** (`nofit`, `misscale`, `1gcp`, `outlier`, `none` — including
gate-demoted pages) get **rescue**: the top candidate publishes as
`georef-snap.json` if `select_score ≥ 1.25` and `margin ≥ 0.25`.
*Example: fargo p31 (prev=none) rescued at score 2.19 → 6.5 ft.*

- **Stamp-corroborated rescue**: for a contradiction-demoted page whose
  candidate lands its printed claim back on the hinting neighbor's stamp, the
  bar drops to `0.7`. *Example: kansas_city p551 — demoted in stage 2, rescued
  at select 0.74 (under the normal 1.25 bar) → 29.9 ft.*
- **Failure mode to know**: rescue searches from the key-map prior. When that
  prior is wrong, a plausible-scoring wrong pose can publish. *Example: fargo
  p55 — correct RANSAC fit killed in stage 1 by the wrong key-map location,
  then snap "rescued" it near that same wrong location (the standing #248
  trust problem).*

**Fitted pages** face two head-to-heads against the top candidate:

- **Challenge** (replace a *disagreeing* incumbent): candidate must clear
  score ≥ 1.5 with margin ≥ 0.25, disagree by ≥ 100 ft, win both verification
  and name alignment, **and** the incumbent must be geometrically indefensible
  (verification ≤ 0.1). *Example: columbia p15 (1 challenge in the volume).*
- **Refine** (adopt an *agreeing* challenger): when the two agree on the lock,
  the chamfer-locked pose is the more precise estimator; adopted when its
  verification beats the incumbent's by ≥ 0.05. *Example: fargo p1,
  prev=fitted → 7.5 ft. Volume counts: chicago `0 challenges, 27 refinements`,
  brooklyn `0 challenges, 18 refinements, 1 rung flips accepted`.*

Snap consumes rungs from the family-scale estimator over *published* fits, so
any change to which pages survive stage 1 perturbs snap's refinement on pages
no gate touched (observed: nashville p4, 26.2 → 402.3 ft under the ±12%
scale-band experiment).

**Debugger records (#325 phase 2).** Each `candidates.jsonl` record also
carries `truth_pose` — the truth affine scored with the ladder's own evidence
and soft bonuses (`rank_pose`), present whenever truth exists, so the snap
explorer can say whether the search never reached the right pose (truth
outscores every candidate) or the evidence itself prefers a wrong one (a
far-off candidate outscores truth) — and `decision`, the per-page bars
(need/got/verdict) for the path the page's state selects, with the rules that
cannot be judged from one record (panel sheet agreement, the volume energy
committee, printed-note rung ratios) listed as skipped. The selection files
remain the authority on what published; `decision` explains the per-page part.

---

## Stage 4: street-solve (`street_solve_run.py`)

Every page with a key-map location prior gets a pose solved from its street
labels as position+angle constraints against named OSM polylines — no
intersections needed. The channel alone is a coin flip, so a pose is adopted
**only where an independent referee** (`osm_snap.evaluate_pose`: road-skeleton
chamfer + name alignment, derived from neither channel) prefers it over the
currently-published pose by a clear margin. Adopted poses write
`georef-street.json`.

*08-07 adoptions (13 pages corpus-wide): new_orleans_1896 p125/p164/p181,
kansas_city p453/p470/p548, grand_rapids p703/p711, philadelphia p237/p238,
los_angeles p1499j, nashville p66, washington_dc p166.*

---

## Stage 5: arbitration (`reconcile.py`)

Every pose the channels produced — including the ones they declined, the top
snap candidates, and the street-solve pose — is weighed against every other
and against not placing the page at all, jointly across the volume (#270).
The winner is written to `p*.georef-final.json`; a page the arbiter declines
gets one with `corners: null`.

This is the stage that fixes the pathology stage 5 used to have: fargo p55
ended a run holding both `keymap-outlier` (the correct pose, declined) and a
snap pose (wrong, published) and *nothing ever compared them*, because
publication was glob precedence and precedence does not look inside files.

## Stage 6: publication

`mapsnap iiif` expands one glob, `*.georef-final.json`. There is no
precedence left to reason about: exactly one sidecar answers for each page.

- A poseless `georef-final.json` leaves the page unplaced *and* claims its
  page key, so nothing else can supply a pose for it.
- Truth comparison then cross-matches split panels: a truth panel with no fit
  of its own is scored against another panel's fit — a row whose `page_truth`
  and `page_gen` columns differ in compare output (#267; older archived tables
  wrote a `(t)` marker with the generated key in trailing parens instead). This
  is how an *unfitted* page (fargo p59__2) can still contribute a disaster: the
  fit being graded belongs to p59__1.

---

## The full gate inventory

Reject paths, in the order a page can hit them:

| # | gate | stage | sidecar / action | 08-07 example |
|---|---|---|---|---|
| 1 | detection admission (conf/size/aspect/words/fill) | 1a | label dropped | fargo p58__2's 14 px cross-streets (strict tier) |
| 2 | seed rotation outlier | 1c | candidate pair vetoed | (debug-only message; see `ransac_hybrid` tests) |
| 3 | < 2 GCPs | 1c | defer or `nofit` | fargo p59__2 |
| 4 | relaxed-fit region-scale check | 1b | relaxed fit discarded | — |
| 5 | broadened-fit scale prior (0.1–6.0×) | 1b | fit removed | — |
| 6 | key-map outlier (+ retry) | 1b | `keymap-outlier`, maybe retried | fargo p55 (stays), p58__2 (retry wins) |
| 7 | printed-scale-note mismatch | 1d | `misscale` | dc p227 (12.31×) |
| 8 | scale bands vs reference | 1d | `misscale` unless note/neighbors keep it | champaign p1 (0.26×); fargo p59__1 **kept** |
| 9 | deferred fit unconfirmed | 1d | `1gcp` | chicago p1n |
| 10 | location outlier | 1d | `outlier` | grand_rapids p819 |
| 11 | adjacency contradiction + hard signal | 2 | `contradicted` + hints | fargo p32, kansas_city p551 |
| 12 | snap rescue bars (1.25 / 0.7 / margin) | 3 | candidate not published | fargo p60__1 in pre-#263 runs |
| 13 | snap challenge bars | 3 | incumbent kept | (default outcome for most fitted pages) |
| 14 | snap refine margin (0.05) | 3 | incumbent kept | — |
| 15 | street-solve referee margin | 4 | streets pose discarded | (most solved pages) |

Resurrection paths — the only ways a rejected page returns:

| from | via | example |
|---|---|---|
| any stage-1 rejection or gate demotion | snap rescue (1.25, or 0.7 with stamp corroboration) | kansas_city p551 |
| flagged scale outlier (before rename) | printed note keep / neighbor corroboration | fargo p59__1 |
| key-map-outlier rejection | neighborhood relaxed retry | fargo p58__2 |
| any published pose | street-solve referee adoption (replacement, not resurrection) | nashville p66 |

Note what is *not* on the resurrection list: nothing ever un-renames a
demoted sidecar's `status` back to `fitted`. All
recovery flows through a different channel's sidecar outranking or standing in
for the lost one.
