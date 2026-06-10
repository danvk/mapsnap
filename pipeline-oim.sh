#!/usr/bin/env bash
# This runs the full pipeline on an OIM map:
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
oim_prefix=$4

echo $sanborn_slug
echo $dirname
echo $relation
echo $oim_prefix

dir=data/$dirname
mkdir -p $dir

# Download IIIF files from OIM for the main content and the key map.
# The key map is only needed for getting a bounding box.
# If you want to georeference a skeleton map or other layer, you'll need to modify this.
curl -o $dir/main.iiif.json "https://oldinsurancemaps.net/iiif/mosaic/$sanborn_slug/main-content/?trim=true"
curl -o $dir/key.iiif.json "https://oldinsurancemaps.net/iiif/mosaic/$sanborn_slug/key-map/?trim=true"

uv run python mapsnap/download_oim_iiif.py \
    $dir/main.iiif.json \
    --oim-url-prefix "$oim_prefix"

uv run python mapsnap/scale_images.py $dir/*.raw.jpg
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
