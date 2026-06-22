# Models

Trained weights for the key-map page-number pipeline (`mapsnap.keymap`).

| File | What it is | Trainer | Device |
|---|---|---|---|
| `number_detector.pt` | CNN **localizer** — a MobileNetV3-small patch classifier that finds page-number centers | `mapsnap.keymap.train_number_detector` | GPU (MPS/CUDA) |
| `number_crnn.pt` | CRNN **recognizer** — reads the digit string from a crop around each center | `mapsnap.keymap.train_crnn` | CPU |

Both are consumed by `python -m mapsnap.keymap.detect_numbers_crnn` (and the localizer alone by `python -m mapsnap.keymap.detect_numbers_cnn --debug`).

## Training data

Both models train on the hand-labeled key maps in `data/keymaps/`: each `<stem>.jpg` paired with a `<stem>.labels.json` of point labels (`{image, width, height, labels: [{x, y, text}]}`) produced by the labeler tool (`app/`, `npm run keymap`). Mixed scan resolutions are handled automatically — `keymap_patches.working_scale` brings full scans (max side ≥ 4000 px) and already-25% scans to a common working size, so just point the trainers at the directory.

One image is held out for validation via `--val-image` (default `chicago-p0b`) so the reported metric reflects generalization to an unseen scan.

## Retraining

Run both from the repo root. Each writes its `.pt` to `models/` (override with `--out`).

### 1. Localizer → `number_detector.pt`

```sh
uv run python -m mapsnap.keymap.train_number_detector --val-image chicago-p0b
```

Fine-tunes a pretrained MobileNetV3-small on positive (label-centered) vs negative (sampled-away) patches; saves the best weights by validation average precision. Defaults: `--epochs 20`, `--batch-size 64`, `--lr 3e-4`, `--data-dir data/keymaps`. Runs on the GPU (`select_device()` prefers MPS, then CUDA, then CPU).

### 2. Recognizer → `number_crnn.pt`

```sh
uv run python -m mapsnap.keymap.train_crnn --val-image chicago-p0b --epochs 250
```

Crops a fixed strip around each labeled number (plus empty-target "no-number" negatives so the model learns to reject the localizer's false positives) and trains the CRNN with CTC, saving the best weights by exact-match accuracy. Defaults: `--epochs 40`, `--batch-size 64`, `--lr 1e-3`, `--negative-ratio 0.5`, `--seed 0`, `--data-dir data/keymaps`.

Two things to know:

- **Pass `--epochs 250`.** The 40-epoch default badly underfits (val exact-match ~0.16); ~250 epochs reaches ~0.95+.
- **It runs on CPU by design** — `nn.CTCLoss` is not reliable on MPS, and the model/data are small enough that CPU training takes only a couple of minutes.

## Verifying a retrain

Regenerate detections on the held-out page and score against its labels:

```sh
uv run python -m mapsnap.keymap.detect_numbers_crnn --pages 1-112 data/keymaps/chicago-p0b.jpg
uv run python -m mapsnap.keymap.score_keymap_labels \
    data/keymaps/chicago-p0b.keymap.json data/keymaps/chicago-p0b.labels.json
```

## Note on these binaries

`data/` is gitignored, so these weights live here (outside it) to ship with the repo and keep the pipeline runnable on a fresh clone. They are regenerable from `data/keymaps/` with the commands above; if the history bloat becomes a concern, move them to Git LFS or a release asset.
