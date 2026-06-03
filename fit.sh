#!/usr/bin/env bash
# This georeferences images, makes an IIIF and compares it with OIM.

set -o errexit
set -x

dir=$1
tag=$2

centerlines=$dir/../centerlines.geojson

uv run mapsnap/georef_from_labels.py $dir/p*.raw.jpg \
    --centerlines $centerlines \
    --min-long-side 45 \
    --min-short-side 20 \
    --edge-margin 0 \
    --min-confidence 0.15 \
    --min-aspect-ratio 1.75

if [ -e $dir/main.iiif.json ]; then
    ref_iiif=$dir/main.iiif.json
elif [ -e $dir/loc.iiif.json ]; then
    ref_iiif=$dir/loc.iiif.json
else
    ref_iiif=$dir/*.manifest.json
fi

uv run python mapsnap/make_iiif_georef.py \
    $ref_iiif $dir'/*.georef.json' \
    --centerlines $centerlines \
    --output $dir/$tag.iiif.json

if [ -e $dir/main.iiif.json ]; then
    uv run python mapsnap/compare_iiif_georef.py \
        $dir/main.iiif.json $dir/$tag.iiif.json \
        | tee $dir/$tag.txt
else
    echo "\nNo main.iiif.json, skipping comparison step.\n"
fi
