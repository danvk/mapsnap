# A local stand-in for the hosted corpus (#501)

Builds the static IIIF tree [#501](https://github.com/danvk/mapsnap/issues/501) proposes, out of whatever the debugger's S3 cache already holds, and serves it. Enough to see whether hosting the corpus ourselves works before paying for a bucket.

```bash
scripts/local_iiif/build.py            # ~/.cache/mapsnap/s3 -> ~/.cache/mapsnap/local-iiif
scripts/local_iiif/serve.py            # http://localhost:8183/iiif/
```

`build.py --item sanborn09554_005` does one volume; `--quality`, `--root` and `--base-url` are the other knobs. Re-running is idempotent.

## What it writes

```
<root>/<item>/<page>/info.json
<root>/<item>/<page>/full/<w>,<h>/0/default.jpg     one real file per level
<root>/<item>/<page>/full/max/0/default.jpg         symlink to the largest
<root>/<item>/<page>/0,0,<W>,<H>/<w>,<h>/0/default.jpg   symlinks, for tiled clients
<root>/annotations/<item>.json                      the annotation, servable by URL
```

and, beside each original in the S3 cache, `mapsnap.local.iiif.json` — the same annotation with its image service pointed at this tree and its GCPs and clipping selector rescaled out of the LoC full-resolution frame into the mirrored scan's. That rescale is what the debugger does in memory; doing it on disk is what makes the result a plain file any viewer can open.

## No tiling, but a tileset

Each page is a halving pyramid of whole images — four levels, no grid. The service still *declares* a tileset the size of the whole image, so every zoom level is a 1×1 grid of one file. Without it Allmaps' parser refuses the image outright:

> Image does not support tiles or custom regions and sizes.

A level-0 service must therefore declare either `tiles` or support for arbitrary regions, and only the first is true here. Two consequences worth knowing if you change the level maths:

- A tiled client asks for an explicit region (`0,0,1593,1887/797,944/0/default.jpg`) even when that region is the whole image, so those paths exist as symlinks to the `full/` ones.
- The size is `ceil(dimension / scaleFactor)`, per the Image API's [tile region calculation](https://iiif.io/api/image/3.0/implementation/#3-tile-region-parameter-calculation) — **not** round. 1593 at factor 8 is 200, not 199. The first build of this used round and 404'd on the smallest level of every page.

Take the URLs from `image.getTileImageRequest`, which is what the renderer calls. Reimplementing the arithmetic is how the rounding bug above survived its own validation: the check agreed with the builder and both disagreed with the viewer.

## Measured on the 20 cached volumes

| | |
|---|---|
| pages | 483 |
| pyramid | 254.0 MB, from 467.6 MB of q95 scans (54%) |
| per page | 0.526 MB (#501 estimated 0.48) |
| corpus projection | **0.23 TB** over 441,179 sheets (#501 estimated 0.21) |
| objects | 4 files a page → 1.8 M for the corpus, against 9.5 M for the tiled plan |
| build | 53 s for 483 pages, single process |

Validated with `app/validate-local-iiif.mjs`, which parses every output with the libraries the viewer uses (`@allmaps/annotation`, `@allmaps/iiif-parser`) and fetches every URL `getTileImageRequest` produces, plus every declared size and `full/max`:

```
$ cd app && node validate-local-iiif.mjs ~/.cache/mapsnap/local-iiif/annotations/*.json
  4473 URLs resolve, 0 missing
```

## Viewing it

The annotations are served, so a viewer can take a URL:

```
https://viewer.allmaps.org/?url=http://localhost:8183/iiif/annotations/sanborn09554_005.json
```

The mapsnap debugger's `?iiif=s3://…` path will **not** show this tree: it rewrites `target.source` to its own `/s3-iiif` service, which overrides the local one. Pointing the debugger here would mean teaching it to leave an already-local annotation alone.

## What this leaves out

Symlinks stand in for duplicate bytes. An object store has none, so hosting this for real means either paying for the copy (which would put level 0 twice in the bill) or declaring `maxWidth` so clients ask for sizes by name and the `full/max` alias can go.
