#!/usr/bin/env bash
# This runs the full pipeline on a map from the Library of Congress (LoC).
# Downloading the manifest and imagery is time-consuming and needs babysitting,
# so this script assumes you've already done that.
# The directory should contain a manifest.json file, and images named p*.raw.jpg.
# - downloading images from OIM and streets from OSM
# - running OCR on the images
# - georeferencing the images
# - making an IIIF file
# - comparing the generated IIIF against OIM's

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
