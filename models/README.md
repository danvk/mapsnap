# Models

Trained weights for the key-map page-number pipeline (`mapsnap.keymap`).

| File | What it is | Trainer | Device |
|---|---|---|---|
| `number_detector.pt` | CNN **localizer** — a MobileNetV3-small patch classifier that finds page-number centers | `mapsnap.keymap.train_number_detector` | GPU (MPS/CUDA) |
| `number_crnn.pt` | CRNN **recognizer** — reads the digit string from a crop around each center | `mapsnap.keymap.train_crnn` | CPU |

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
uv run python -m mapsnap.keymap.train_crnn --val-image hudson_co_nj_1950_vol_9/p0 --epochs 250
```

Crops a fixed strip around each labeled number (plus empty-target "no-number" negatives so the model learns to reject the localizer's false positives) and trains the CRNN with CTC, saving the best weights by exact-match accuracy. Defaults: `--epochs 40`, `--batch-size 64`, `--lr 1e-3`, `--negative-ratio 0.5`, `--seed 0`, `--data-dir data`.

Two things to know:

- **Pass `--epochs 250`.** The 40-epoch default badly underfits (val exact-match ~0.16); ~250 epochs reaches ~0.95+.
- **It runs on CPU by design** — `nn.CTCLoss` is not reliable on MPS, and the model/data are small enough that CPU training takes only a couple of minutes.

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
