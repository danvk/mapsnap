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

sanborn_slug=$1
dirname=$2
relation=$3

echo $sanborn_slug
echo $dirname
echo $relation

dir=data/$dirname
test -d $dir

uv run python mapsnap/download_osm.py $relation --output $dir/streets.osm.json

uv run python mapsnap/osm_to_centerlines.py \
    $dir/streets.osm.json \
    --output $dir/centerlines.geojson

uv run python mapsnap/detect_text.py \
    --min-long-side 20 \
    --centerlines $dir/centerlines.geojson \
    --num-workers 2 \
    $dir/*.scaled.jpg

./fit.sh $dir mapsnap
