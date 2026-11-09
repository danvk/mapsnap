# Models

Trained weights for the key-map page-number pipeline (`mapsnap.keymap`).

| File | What it is | Trainer | Device |
|---|---|---|---|
| `number_detector.pt` | CNN **localizer** — a MobileNetV3-small patch classifier that finds page-number centers | `mapsnap.keymap.train_number_detector` | GPU (MPS/CUDA) |
| `number_crnn.pt` | CRNN **recognizer** — reads the page key (digits + optional letter suffix) from a crop around each center | `mapsnap.keymap.train_crnn` | CPU |

Both are consumed by `python -m mapsnap.keymap.detect_numbers_crnn` (and the localizer alone by `python -m mapsnap.keymap.detect_numbers_cnn --debug`).

## Training data

Both models train on every hand-labeled key map under `data/**/raw/truth/*.labels.json` — the full-resolution sheet `<volume>/raw/<stem>.jpg` paired with the point labels the labeler tool writes (`app/`, keymap.html). Discovery is `keymap_patches.labelled_keymaps`, so nested volumes are found; mixed scan DPIs are normalised by `keymap_patches.working_scale` (a plain 1.0 or 0.25 when that lands near the 1950 px working long side, else scaled to it).

One key map is held out via `--val-image` (a `<volume>/<stem>` key; bare stems are ambiguous — ten volumes have a `p0`) for validation and best-checkpoint selection. `--exclude` removes whole volumes from training entirely, for leave-one-volume-out evaluation without letting the held-out volume drive checkpoint selection.

The current weights (2026-07-27) were trained on 25 key maps / ~1,610 labels (detector: 6,440 patches; CRNN: 2,412 strips) with `--val-image hudson_co_nj_1950_vol_9/p0`; CRNN val exact-match 0.986, detector val AP ≈ 0.99. Leave-one-volume-out spot checks: held-out asheville reads 18% → 49% exact vs the previous weights; localizer recall on held-out asheville 60% → 91%.

## Retraining

Run both from the repo root. Each writes its `.pt` to `models/` (override with `--out`).

### 1. Localizer → `number_detector.pt`

```sh
uv run python -m mapsnap.keymap.train_number_detector --val-image hudson_co_nj_1950_vol_9/p0
```

Fine-tunes a pretrained MobileNetV3-small on positive (label-centered) vs negative (sampled-away) patches; saves the best weights by validation average precision. Defaults: `--epochs 20`, `--batch-size 64`, `--lr 3e-4`, `--data-dir data`. Runs on the GPU (`select_device()` prefers MPS, then CUDA, then CPU).

### 2. Recognizer → `number_crnn.pt`

```sh
uv run python -m mapsnap.keymap.train_crnn --val-image hudson_co_nj_1950_vol_9/p0 --synthetic 12000
```

Crops a fixed strip around each labeled number (plus empty-target "no-number" negatives so the model learns to reject the localizer's false positives) and trains the CRNN with CTC, saving the best weights by exact-match accuracy. Defaults: `--epochs 40`, `--batch-size 64`, `--lr 1e-3`, `--negative-ratio 0.5`, `--seed 0`, `--data-dir data`.

The charset is digits **plus letters** (#316), so suffixed page keys (columbia's underlined `53E`, los_angeles's quote-marked `1499K`, nashville's inline `9A`) are readable. Letters are rare in the hand labels, so training leans on synthesis:

- **Pass `--synthetic 12000`.** Adds strips from `mapsnap.keymap.synth_strips`: matched fonts over real background tiles, the three suffix-adornment conventions, partial neighbor numbers at strip edges. With them 40 epochs reaches ~0.95 val exact-match; without them letters are undertrained. (The old digits-only advice to run 250 epochs is superseded — the synthetic pool is what fixed the underfit.)
- **The generator's realism is load-bearing.** Two trained-in failures came from subtle render/scan mismatches: strips rendered at final scale phase-locked the conv features (reads alternated blank/correct with a one-timestep period on real sheets), and strips without the pipeline's 160x60 -> 160x48 vertical squeeze taught the wrong glyph proportions. `synth_strips` renders at 4x and downsamples through a continuous-jitter crop for this reason; keep it that way.
- **It runs on CPU by design** — `nn.CTCLoss` is not reliable on MPS, and the model/data are small enough that CPU training takes ~45 minutes with the synthetic pool.

Off-center robustness is handled at **inference**, not training: `detect_numbers_crnn` reads every candidate at three vertical offsets and keeps the best read (`read_best_offset`). Widening training translation to cover the localizer's offset range was tried and traded letter accuracy for offset robustness in every combination.

## Verifying a retrain

Regenerate detections on the held-out page and score against its labels:

```sh
uv run python -m mapsnap.keymap.detect_numbers_crnn --pages 1-112 data/chicago_il_1950_vol_1/raw/p0.jpg
uv run python -m mapsnap.keymap.score_keymap_labels \
    data/chicago_il_1950_vol_1/raw/p0.keymap.json data/chicago_il_1950_vol_1/raw/truth/p0.labels.json
```

## Note on these binaries

`data/` is gitignored, so these weights live here (outside it) to ship with the repo and keep the pipeline runnable on a fresh clone. They are regenerable from the truth data with the commands above; if the history bloat becomes a concern, move them to Git LFS or a release asset.

## keymap_road_unet.pt

P(road) for key-map sheets (issue #211): a colour UNet (base=24, 3-channel)
trained by `mapsnap.keymap.train_road_prob` on OSM centerlines rendered
through each sheet's own georef -- per-sheet pixel stroke widths, soft labels,
loss masked to the mapped extent. Holdout (detroit, miami, grand_rapids,
brooklyn_1939_1 -- whole volumes): buffered F1 0.865 at tolerance 40px,
completeness 0.73-0.93 with the fill-background stratum >= paper everywhere.
Run `python -m mapsnap.keymap.road_prob predict` to write
`raw/<stem>.roadprob.png` (masked to the mapped extent) plus an overlay render.

## street_recognizer.pt

Fine-tuned street-label **recognizer** (#265): EasyOCR's `latin_g2` CRNN
(`easyocr.model.vgg_model.Model`, 3.8M params) fine-tuned on 3,916 inlier
agreement crops from 15 volumes (conf >= 0.15, rotation-twin-deduped), half of
each batch corrupted by `mapsnap.ocr_augment`'s measured-geometry artifacts
(underline rules fused to glyph bottoms, dashed pipe lines at glyph height,
capped junk fragments, resolution squeeze, photometric jitter). Holdout =
fargo + nashville, excluded end to end.

Consumed by `mapsnap ocr --recognizer-weights models/street_recognizer.pt` —
a plain state_dict swapped into `reader.recognizer`; detection, vocabulary
trie, constrained CTC decode, and the three-vote underline arbitration are
all unchanged.

Held-out benchmarks (2026-08-09, `scripts/ocr_recognizer_bench.py`): clean
crops 0.958 vs stock 0.932 exact; real underline/dash crops read RAW (no
erasers) 0.871 vs stock 0.655. End-to-end A/B: fargo 52.4 -> 63.9,
nashville 65.7 -> 64.8 (the regression decomposes to snap-refine #277 and
yellow-fill #278, not read quality). Training curve:
`street_recognizer.history.json` (best epoch 18).

Retrain (CPU, ~2.5 h; `nn.CTCLoss` is unreliable on MPS):

```sh
uv run python -m mapsnap.train_street_recognizer build-dataset \
    ~/Documents/ohm/ocr-train-harvest-2026-08-08.jsonl --out data/ocr_finetune_cache
uv run python -m mapsnap.train_street_recognizer render-review \
    data/ocr_finetune_cache --out review.html   # eyeball before training
uv run python -m mapsnap.train_street_recognizer train data/ocr_finetune_cache
```

## region_unet.pt

The page **content-region** model (#226, PR #355): a whole-page UNet (base 24,
GroupNorm) that predicts, per pixel of a letterboxed 512² page, whether the
position holds content exclusive to that page rather than margin, title block,
or a strip duplicated from a neighbour. Labels are the OIM multimask
selectors in each truth volume's `main.iiif.json`, rasterized.

Consumed by `mapsnap region data/<vol>` (writes `artifacts/region/<stem>.png`),
which feeds the volume viewer's Page / Region / P(road) toggle, the
reconciler's `--region-overlap` factor (#352) and the clip masks
(`mapsnap iiif --masks region`, the default; #544). `mapsnap iiif` predicts any
maps that are missing. Maps written before #530 retrained this model are stale:
refresh them with `mapsnap region data/<vol> --force`.

**Current weights (2026-09-08)** were trained on 18 truth volumes (1,518
pages), Hudson held out: held-out IoU **0.856** (mean; median 0.891). Two
things differ from the #355 training set. Columbus 1951 vol 3 is excluded:
its OIM truth has no multimask, so its selectors are whole-sheet rectangles
that teach the model to paint margins as content. New Orleans 1896 vol 2 is
included with its truth re-exported after OIM's multimask was saved; the
#355 weights had learned its stale whole-sheet selectors and predicted 97% of
each sheet as content there (IoU 0.744 against the real mask; 0.911 now).
Measured at published poses, that pollution showed as a good-page median
region overlap of 20–23% on those two volumes against 2–6% elsewhere; after
this retrain both sit at 2–4%.

Retrain (about 2 h on MPS at 10–11 min per epoch):

```
uv run python -m mapsnap.region_model train $(ls -d data/*/ | grep -v 'columbus_oh_1951_vol_3\|hudson_co_nj_1950_vol_9') --val data/hudson_co_nj_1950_vol_9
```

Verify with the corpus overlap measurement in #352 (good-page median overlap
per volume) and the held-out IoU the trainer prints; #355's ablations
(longer training, base 32) still hold and are not worth repeating.

## keymap_region_unet.pt

The **key-map page-region** model (`mapsnap.keymap.region_unet`): a UNet (base
16, BatchNorm, 2.0M parameters) that labels every pixel of a key-map sheet
background (paper, cartouche, inset), block **interior**, or block **boundary**
(the heavy line or color change between two pages). The blocks come out of a
watershed on P(boundary) seeded from the detected page numbers, as a planar
partition; a block no detected number sits in becomes a region labeled `?`. It
is the alternative to `page_regions`' color rules, which lose blocks whenever
color stops meaning "same block" (two neighbors in one color, one block in two
colors, pale fills) and move regions on 39 of 41 key maps for a one-pixel change
in the input (#640). The sheet is worked at the scale that puts the median
spacing between page numbers at 128 px, tiled at 512 px, and predicted with
flip averaging. `MAPSNAP_KEYMAP_REGIONS=unet` makes `mapsnap keymap` use it;
`MAPSNAP_KEYMAP_REGION_WEIGHTS` loads other weights. About 10 s per key map on
an M2 (4 s of it the watershed).

Labels need no hand tracing: every OIM-georeferenced page's footprint, projected
through the key map's own georeference, is a block. Three corrections make
footprints usable (`mapsnap.keymap.train_region_unet`): each footprint is shrunk
to a core and a marker watershed on an edge-strength map (ink darkness and
color change) snaps its boundary to the drawn line within a band of 15% of the
spacing (the key-map georef's error is 15–50 px); where `page_regions` agrees
with a footprint at IoU ≥ 0.8 its polygon is taken as the exact block; and
blocks nobody labeled (unclipped OIM footprints, the inset, detected numbers
with no footprint, paper bubbles) are *ignored*, not taught as paper. Training
data lives outside the repo in `~/Documents/mapsnap/keymap-region-training`:
`prep` builds it from the volumes under `data/`, and a streaming pull of the
corpus-v2 key maps for the other OIM-truth items adds the rest (the sheets are
deleted after their example is built; disk, not bandwidth, is the limit).

**Current weights (2026-11-09)** are the average of epochs 18–24 of a 24-epoch
run on 549 key maps (the 59 with truth under `data/` plus 490 corpus-v2 key
maps of other OIM-truth volumes; Hudson, Chicago 1950 vol 1, Miami 1950 and
San Antonio 1952 held out), about 5.5 h on an M2. On the four held-out maps,
mean IoU against the projected OIM footprints is **0.430** for the model
against **0.415** for `page_regions`, matched by page number; matched by shape
alone (any region to any footprint, so the page-number reads drop out)
**0.544** against **0.498**. A first model trained on the 59 `data/` maps
alone scored 0.422 / 0.537 on the same four, and 0.427 / 0.524 against 0.406 /
0.475 on 60 corpus maps it had never seen (ahead on 35 of the 60 by page
number and 44 by shape, 2,241 blocks at IoU ≥ 0.5 against 2,082 of 4,016); the
extra 490 maps bought about 0.01, so data is not what limits this model.
Over the 63 key maps with truth under `data/` (59 of them in the training
set, so this flatters the model) it is 0.465 against 0.433 (shape 0.559
against 0.498), ahead on 39, with 2,537 blocks at IoU ≥ 0.5 against 2,222 of
4,171; the wins are large (Richmond 1952 +0.31, Asheville +0.28, Columbia
+0.21) and the losses small (≤ 0.06). Both methods are capped near 0.75 by the truth itself: OIM
clips pages to the street centerline, the drawn block stops at the curb, and
both methods' regions are about three quarters of their footprint's area. A
one-pixel change in the input (the #640 perturbation) leaves the model's
regions at a self-consistency of 0.95 (Chicago), 0.96 (Hudson), 0.93 (San
Antonio) and 0.89 (Miami) by mean per-page IoU; `page_regions` on the same
Hudson sheet scores 0.81, with 14 of 89 pages moving by more than half their
area.

Retrain:

```
uv run python -m mapsnap.keymap.train_region_unet prep        # labels from data/
uv run python -m mapsnap.keymap.train_region_unet train --checkpoint-dir CKPT --epochs 24
uv run python -m mapsnap.keymap.train_region_unet average models/keymap_region_unet.pt CKPT/epoch_{18..24}.pt
uv run python -m mapsnap.keymap.train_region_unet eval --stability
```

Averaging the late epochs' weights gives a model as good as the best single
epoch on the held-out maps without choosing it by those same maps; the
trainer also keeps the best epoch by held-out IoU as its output. `eval`
prints, per held-out map, the model and `page_regions` side by side
(mean IoU and blocks at IoU ≥ 0.5, the shape-only IoU, self-overlap, unseeded
regions, seeds the paper test dropped) and the stability check. The decoder's
thresholds (`DecodeParams`) were swept on the held-out maps and the result did
not move; the model's probabilities are confident enough that they do not
matter.

## cutline_unet.pt

The splitter's **cutline** model (#83): a whole-page UNet (base 16, GroupNorm,
2.0M parameters) that draws, per pixel of the page letterboxed to 768², the
dividing lines that cut a sheet into panels. `mapsnap split` uses it by default
(`mapsnap.cutline_model`); `MAPSNAP_SPLITTER=classical` falls back to the
classical line detector, and `MAPSNAP_CUTLINE_WEIGHTS` loads other weights.
About 1 s per page on one CPU thread.

Labels are the cutlines OldInsuranceMaps volunteers traced, rasterized as
3 px strokes; pages volunteers marked unsplit (OIM's `prepared` flag with one
region) are empty masks. The benchmark lives outside the repo in
`~/Documents/mapsnap/cutline-training`: 2,000 split pages whose cuts follow
printed ink (weighted toward pages with many panels) and 2,000 unsplit pages
matched to them by volume size, from complete and unfinished volumes, San
Francisco excluded. One volume in five is the test fold, which training never
sees; one train-fold volume in eight picks the best epoch.

**Current weights (2026-10-05)** trained 15 epochs on an M2 (~5 h), best epoch
11 by held-out Dice (0.742). On the test fold (760 pages), through the
production splitter: right panel count on **87.4%** of split pages (tuned
classical: 76.3%), unsplit pages left whole **99.2%** (98.5%). They were
trained with an extra page-level "split?" head, since dropped because it changed
nothing; `mapsnap.train_cutline_unet` trains the plain UNet.

Retrain:

```
uv run python -m mapsnap.train_cutline_unet prep ~/Documents/mapsnap/cutline-training
uv run python -m mapsnap.train_cutline_unet train ~/Documents/mapsnap/cutline-training
```

Verify with the benchmark, through the production path:

```
uv run python scripts/score_splits_oim.py --manifest ~/Documents/mapsnap/cutline-training/manifest.tsv
MAPSNAP_SPLITTER=classical uv run python scripts/score_splits_oim.py --manifest ...   # the comparison
```

## confidence.json

Not weights for a network: the coefficients of a logistic model that estimates,
for every pose reconcile publishes, the probability that it lies within 25 ft of
the truth (`mapsnap.confidence`, #579). Reconcile writes it as `confidence` in
each placed page's `pN.provenance.json` and `georef-final.json`. The IIIF carries
it as `confidence` metadata, plus `confidence_provisional` for panels.

Each column has a fill value for a missing input, a mean and scale to
standardize by, a coefficient, and a missing-input coefficient where an input
can be absent. `mapsnap.confidence.p_good` applies them without any ML
dependency.

Trained on corpus-v1 in the 400 OIM-truth volumes (23,371 placed poses), with
every pose labelled by its RMSE against OIM truth:

    uv run python scripts/train_confidence.py <volume dirs> \
        --iiif corpus-v1.iiif.json --version corpus-v1 --out models/confidence.json

Each volume directory holds the run's `pN.provenance.json` and `pN.panels.json`,
its published annotation page, and the OIM truth (`main.iiif.json`, `oim/`).
Grouped 5-fold AUC (whole volumes held out): 0.886 overall, 0.888 for pages,
0.771 for panels. **Retrain after each corpus run:** any change to reconcile or
snap can shift what the inputs mean, and panel scores need a current-code run
before they stop being provisional.

## Adjacency-number detector — `adjacency_unet.pt`

A UNet (`mapsnap.road_model.UNet`, base 16, grayscale) that marks the printed
adjacent-sheet numbers on the mirror's 25% page images, so `mapsnap adjacency`
can read numbers CRAFT never boxes (big thin numerals, the hollow outline font).
Inference helpers are in `mapsnap.adjacency_detector`; it is not wired into the
pipeline yet.

Trained by `mapsnap.train_adjacency_detector` with no hand labels: OIM's
georeferencing predicts each sheet's neighbours and the stretch of outline
each borders (`mapsnap.oim_neighbours`), and corpus-v1's digit reads that
match a predicted neighbour near its stretch are the positive boxes.
`adjacency_unet.pages.json.gz` is the training manifest: 1,889 pages from 206
OIM volumes (1867–1961, no city series that has adjacency truth), 4,281 positive
boxes, plus 1,257 boxes adopted by one round of self-training (`adopted`; the
shipped weights do not use them). 12,000 steps take about five hours on an M2.

    gunzip -c models/adjacency_unet.pages.json.gz > WORK/manifest.json
    uv run python -m mapsnap.train_adjacency_detector fetch WORK
    uv run python -m mapsnap.train_adjacency_detector labels WORK
    uv run python -m mapsnap.train_adjacency_detector train WORK

Scored against `testdata/adjacency_truth/` by `scripts/score_adjacency_detector.py`.
