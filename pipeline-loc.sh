#!/usr/bin/env bash
# This runs the full pipeline on a map from the Library of Congress (LoC).
# This script assumes you've already downloaded the manifest.json file to <dir>/manifest.json.
# - Downloads images from OIM and streets from OSM
# - Runs OCR on the images
# - Georeferences the images
# - Makes an IIIF file

set -o errexit
set -x

dir=$1
relation=$2

echo $dir
echo $relation

test -d $dir
test -e $dir/manifest.json

uv run python mapsnap/download_loc_iif.py --scale pct:25 $dir/manifest.json

uv run python mapsnap/download_osm.py $relation --output $dir/streets.osm.json

uv run python mapsnap/osm_to_centerlines.py \
    $dir/streets.osm.json \
    --output $dir/centerlines.geojson

uv run python mapsnap/detect_text.py \
    --centerlines $dir/centerlines.geojson \
    $dir/p*.raw.jpg

uv run mapsnap fit $dir init
