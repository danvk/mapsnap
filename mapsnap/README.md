# Sanborn Insurance Map Rectification

The goal of this project is to automatically georeference Sanborn Insurance Maps.

## Pipeline

- Load a map into OldInsuranceMaps.net by filling out their form or messaging Adam Cox on Slack.
- Manually georeference the Key Map to get a bounding box for the "volume" of individual maps.
- "Prepare" all the other maps on OIM.

(assuming you're working with a previously georeferenced map)

The commands below are for https://oldinsurancemaps.net/map/sanborn03376_029 aka https://www.loc.gov/item/sanborn03376_029.

Download imagery from OIM and its S3 bucket using the IIIF file:

```
mkdir ~/Documents/ohm/new_orleans_la_1951_vol_5
curl -o ~/Documents/ohm/new_orleans_la_1951_vol_5/main.iiif.json 'https://oldinsurancemaps.net/iiif/mosaic/sanborn03376_029/main-content/?trim=true'
uv run python download_oim_iiif.py ~/Documents/ohm/new_orleans_la_1951_vol_5/main.iiif.json --oim-url-prefix 'https://s3.us-central-1.wasabisys.com/oldinsurancemaps/uploaded/documents/new_orleans_la_1951_vol_5_'
```

This downloads 109 full-resolution JPEG files. It's convenient to pull these from OIM's S3 bucket since the Library of Congress tile server is pretty aggressive about rate-limiting.

The full-resolution images are more than you need for OCR and georeferencing. To speed up the later steps, it's convenient to downscale them and convert to grayscale using ImageMagick:

```
cd ~/Documents/ohm/new_orleans_la_1951_vol_5
for x in *.jpg; convert -colorspace gray -resize '2048>' $x ${x/.jpg/.2048px.jpg}
```

### Street and Intersections

Find the southwest and northeast corner of the key map to get a bounding box.

Go to Overpass Turbo and paste the following in:

```
[out:json][timeout:60];
(
  way["highway"]["name"](
    29.909795,-90.125975,  // southwest
    29.946841,-90.083828   // northeast
  );
);
out body;
>;
out skel qt;
```

Run this query and look at the results in the map to make sure they look OK. Then click "Export" and download the raw OSM data. Save this in `streets.osm.json`.

Extract street names and intersections by running:

```
jq -r '.elements[].tags.name' streets.osm.json | grep -v '^null$' | sort | uniq > streets.txt
uv run python sanborn/generate_intersections.py .../streets.osm.json .../intersections.csv
uv run python osm_to_centerlines.py .../streets.osm.json --output .../centerlines.geojson
```

### Street Label OCR

Run `detect_text.py` over all the scaled-down images to find street labels + angles:

```
for x in /Users/danvk/Documents/ohm/new_orleans_la_1951_vol_5/*.2048px.jpg ; do echo $x; uv run python detect_text.py $x --min-long-side 60 --min-short-side 12 --streets /Users/danvk/Documents/ohm/new_orleans_la_1951_vol_5/streets.txt --visualize ${x/.2048px.jpg/.detect.png} > ${x/.2048px.jpg/.streets.json}; done
```

This is the slowest step if you're running CPU-only, ~1 minute/image.

### Fit georeference model

Given detected street labels and street centerlines, find GCPs and fit a four-parameter model for each map:

```
for x in /Users/danvk/Documents/ohm/new_orleans_la_1951_vol_5/*.2048px.jpg ; do echo $x; uv run python georef_from_labels.py --labels ${x/.2048px.jpg/.streets.json} --centerlines /Users/danvk/Documents/ohm/new_orleans_la_1951_vol_5/centerlines.geojson --output ${x/.2048px.jpg/.georef.json} --image $x; done
```

For maps without enough control points, this will fail to produce an output.

### Make an IIIF file

Download the IIIF file for these Sanborn maps from the Library of Congress:

```
curl -o ~/Documents/ohm/new_orleans_la_1951_vol_5/loc.iiif.json https://www.loc.gov/item/sanborn03376_029/manifest.json
```

Probably, though, you'll need to load https://www.loc.gov/item/sanborn03376_029/manifest.json directly in your browser to avoid getting denied.

Generate an IIIF file using the georeferences:

```
uv run python /Users/danvk/github/ohm/make_iiif_georef.py /Users/danvk/Documents/ohm/new_orleans_la_1951_vol_5/loc.iiif.json '/Users/danvk/Documents/ohm/new_orleans_la_1951_vol_5/*.georef.json' --output /Users/danvk/Documents/ohm/new_orleans_la_1951_vol_5/generated.iiif.json
```

You can paste this into viewer.allmaps.org to look at the results.

### Measuring accuracy

Compare the generated IIIF file to the data from OIM:

```
python compare_iiif_georef.py \
  ~/Documents/ohm/new_orleans_la_1951_vol_5/main.iiif.json \
  ~/Documents/ohm/new_orleans_la_1951_vol_5/generated.iiif.json
```
