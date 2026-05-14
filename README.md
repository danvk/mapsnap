# Mapsnap

The goal of mapsnap is to automatically georeference Sanborn Insurance Maps.

## Performance

I used [New Orleans 1951 Volume 5][nola5] on OldInsuranceMaps.net for testing:

- Of the 109 images, mapsnap was able to locate 99 of them (91%).
- On these 99 maps:
  - The average RMSE vs OIM's hand geocoding was 15ft.
  - The median RMSE was 11ft.
  - 76% of maps had an RMSE <= 15ft.
  - 88% had an RMSE <= 25ft.
  - 95% had an RMSE <= 50ft.
  - The worse RMSE was 77ft.

RMSE was measured across 49 equally-spaced points on each image.

TODO: more tests in more places

[nola5]: https://oldinsurancemaps.net/map/sanborn03376_029

## How it works

Here's an [example][p19] of a Sanborn Insurance Map:

![Brooklyn 1939 Volume 2 Page 19](/images/brooklyn_ny_1939_vol_2_p19.jpg)

This image depicts a small area of Brooklyn, NY in 1939. Our goal is to overlay this on a contemporary map by determining its location, scale, and rotation.

We can run EasyOCR against it to get candidate street labels:

![Same image with boxes drawn around text](/images/brooklyn_ny_1939_vol_2_p19.detect.png)

There's a lot of text in this image! The red boxes show labels that are likely to be streets.

From each bounding rectangle we get three bits of information:

- A candidate street name
- A location in the image (center of the bounding rectangle)
- A direction (long edge of the bounding rectangle)

This is the information we'll use to determine the map projection.

We generally know _roughly_ where the map is: the Library of Congress has organized their Sanborn collection by country, state and county. We can get an even better estimate by roughly locating the [key map] for the volume. This gives a bounding box that's at most a few miles in each dimension.

Next, we download contemporary streets in that bounding box from [OpenStreetMap] (OSM). Our hope is that enough streets have stayed the same that we can line them up between the Sanborn map and OSM.

We can use the contemporary streets to filter the OCR results:

![Detected streets in the image](/images/detected-streets.jpg)

Many of these are real streets, but many of them are not:

- The cut-off street names on the right are CLARK (👍), LIBERTY (👍) and CONGRESS (👎). This last one is matching "Library of Congress" and there is a Congress Street in Brooklyn.
- Most of the streets in the middle are correct: HENRY, MONROE, PIERREPONT, CLINTON, FULTON, WASHINGTON, JOHNSON, ADAMS, MYRTLE.
- It misreads "N" and "E" as streets, and it sees BROOKLYN all over the place. These are all bad detections that could potentially throw off alignment.

Next, we extrapolate the streets in both directions, following the direction of the text. If two streets intersect in the image and in the OSM data, we record a candidate intersection:

...

If we have two or more candidate intersections, we have enough data to fit a model. (Sometimes we can get a fit with just one — more on this soon.) For each pair of intersections, we can fit a model and see where it would place the street labels from OCR. If the label gets mapped close to the expected street in OSM, and the street is at the expected angle there, then that's an indicator of a good fit and this street is an "inlier." If not, it's an outlier.

We try each pair of intersections and find the one that produces the best fit with the most inliers. This is our mapping!

...

[p19]: https://oldinsurancemaps.net/document/85714
[key map]: https://oldinsurancemaps.net/document/85676
[OpenStreetMap]: https://www.openstreetmap.org/#map=18/40.683787/-73.978527

## Development

Quickstart:

```
uv sync
uv run pytest
uv run pyright
uv run ruff check
uv run ruff format
```

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

## Notes on the model

### Six Parameters vs. Four

OldInsuranceMaps.net (OIM) uses a six parameter affine model. In addition to translation (two parameters), rotation and scale, this adds two new parameters:

- Skew: the x- and y-directions in the map need not be exactly 90° from each other.
- Scale anisotropy: A pixel in the x- and y-directions need not be the same distance.

In practice, the Sanborn maps are all very well-made and well-scanned and don't exhibit much of either of these. Most fits on OIM have scale anisitropy of less than 3% and skew of less than 3°. These are both close enough to zero that it's unclear if they're real or the result of inaccurate georeferencing. Removing these parameters makes it easier to fit a model since you only need two GCPs rather than three. And it eliminates a failure mode of detecting heavily skewed maps due to an inaccurate GCP.

### How I developed this model

The initial idea was to ask the OpenAI API to find intersections in the Sanborn map. It did an OK job at this, but it was slow (~3 minutes/task), expensive (~$0.10/image) and not always very accurate. It didn't do well on the non-skeleton maps, and its process was opaque, so it was hard to improve on.

I realized I was mostly evaluating its intersections by mentally tracing the street labels, so maybe I should just have OpenAI do that. But detecting street labels is just OCR. So that led me to the local process in this repo.

My initial hope was to _only_ use the street label positions and angles to fit a model, without any reference to intersections. But in practice, since a street only has an angle at a position, this required a rough location to get going. Extrapolated intersections were a good way to get that location. This led me to a two part model: first use extrapolated intersections to fit a rough model, then refine that using street angles. Eventually I realized that the refinement step wasn't helping, and wound up with the current model.

Claude Code and ChatGPT were both instrumental in getting this to work.

### Alternatives considered

- Each map has a compass rose. You could find it and detect its angle to know the direction of north. In practice, compass isn't necessarily _that_ accurate. It's better to use the streets.
- Each map has a scale bar. You could use this to detect the scale, leaving just location and rotation. I haven't explored this too much, though the one GCP approach relies on the scales all being about the same.

## Questions

- Will this work with other types of maps?
- How does this relate to OldInsuranceMaps.net (OIM)?

## Prior Art

- Shensky (2025): they detect intersections using a custom ML model, then try to read street labels in horizontal or vertical strips from them using Tesseract. That's the reverse of what this repo does. It inherently cannot find diagonal streets, and Tesseract is generally a worse model than EasyOCR.
