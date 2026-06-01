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

# --edge-margin 0

if [ -e $dir/main.iiif.json ]; then
    ref_iiif=$dir/main.iiif.json
else
    ref_iiif=$dir/loc.iiif.json
fi

uv run python mapsnap/make_iiif_georef.py \
    $ref_iiif $dir'/*.georef.json' \
    --centerlines $dir/centerlines.geojson \
    --output $dir/$tag.iiif.json

if [ -e $dir/main.iiif.json ]; then
    uv run python mapsnap/compare_iiif_georef.py \
        $dir/main.iiif.json $dir/$tag.iiif.json \
        | tee $dir/$tag.txt
else
    echo "\nNo main.iiif.json, skipping comparison step.\n"
fi
