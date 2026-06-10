#!/usr/bin/env bash
# This georeferences images, makes an IIIF and compares it with OIM.

set -o errexit
set -x

dir=$1
tag=$2

centerlines=$dir/centerlines.geojson
if [ ! -e $centerlines ]; then
    centerlines=$dir/../centerlines.geojson
fi

if compgen -G "$dir/p*.scaled.jpg" > /dev/null; then
    input_images=$dir/p*.scaled.jpg
elif compgen -G "$dir/p*.raw.jpg" > /dev/null; then
    input_images=$dir/p*.raw.jpg
else
    echo "Error: no p*.scaled.jpg or p*.raw.jpg found in $dir" >&2
    exit 1
fi

uv run mapsnap/georef_from_labels.py $input_images \
    --centerlines $centerlines \
    --min-long-side 45 \
    --min-short-side 20 \
    --edge-margin 0 \
    --min-confidence 0.15 \
    --min-aspect-ratio 1.75 \
    --one-gcp-fits

if [ -e $dir/main.iiif.json ]; then
    ref_iiif=$dir/main.iiif.json
elif [ -e $dir/loc.iiif.json ]; then
    ref_iiif=$dir/loc.iiif.json
else
    ref_iiif=$dir/*manifest.json
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
