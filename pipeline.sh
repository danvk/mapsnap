#!/usr/bin/env bash
set -o errexit
set -x

sanborn_slug=$1
dirname=$2
oim_prefix=$3

echo $sanborn_slug
echo $dirname
echo $oim_prefix

dir=data/$dirname
mkdir -p $dir

curl -o $dir/main.iiif.json "https://oldinsurancemaps.net/iiif/mosaic/$sanborn_slug/main-content/?trim=true"
curl -o $dir/key.iiif.json "https://oldinsurancemaps.net/iiif/mosaic/$sanborn_slug/key-map/?trim=true"

echo <<"END"

While the pipeline is running, visit:

    https://www.loc.gov/item/$sanborn_slug/manifest.json

in your browser and save the results to:

    $dir/loc.iiif.json

This will allow the pipeline to generate an IIIF file.

END

uv run python mapsnap/download_oim_iiif.py \
    $dir/main.iiif.json \
    --oim-url-prefix "$oim_prefix"

for x in $dir/*.jpg; do
    magick convert -colorspace gray -resize '2048>' $x ${x/.jpg/.2048px.jpg}
done

BBOX=$(uv run python mapsnap/iiif_bbox.py $dir/key.iiif.json)
uv run python mapsnap/download_osm.py \
    $BBOX \
    --output $dir/streets.osm.json

uv run python mapsnap/osm_to_centerlines.py \
    $dir/streets.osm.json \
    --output $dir/centerlines.geojson

jq -r '.elements[].tags.name' $dir/streets.osm.json | grep -v '^null$' | sort | uniq > $dir/streets.txt
uv run python mapsnap/generate_intersections.py $dir/streets.osm.json $dir/intersections.csv

uv run python mapsnap/detect_text.py $dir/*.2048px.jpg

uv run mapsnap/georef_from_labels.py $dir/*.2048px.jpg \
    --centerlines $dir/centerlines.geojson \
    --min-long-side 50 \
    --min-short-side 20 \
    --min-confidence 0.15

uv run python mapsnap/make_iiif_georef.py \
    $dir/loc.iiif.json $dir'/*.georef.json' \
    --output $dir/generated.iiif.json

uv run python compare_iiif_georef.py \
    $dir/main.iiif.json $dir/generated.iiif.json \
    | tee $dir/comparison.txt
