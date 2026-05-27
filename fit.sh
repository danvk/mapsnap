#!/usr/bin/env bash
# This georeferences images, makes an IIIF and compares it with OIM.

set -o errexit
set -x

dir=$1
tag=$2

uv run mapsnap/georef_from_labels.py $dir/*.scaled.jpg \
    --centerlines $dir/centerlines.geojson \
    --min-long-side 50 \
    --min-short-side 20 \
    --min-confidence 0.15

uv run python mapsnap/make_iiif_georef.py \
    $dir/main.iiif.json $dir'/*.georef.json' \
    --centerlines $dir/centerlines.geojson \
    --output $dir/$tag.iiif.json

uv run python mapsnap/compare_iiif_georef.py \
    $dir/main.iiif.json $dir/$tag.iiif.json \
    | tee $dir/$tag.txt
