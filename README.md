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

RMSE was measured across 49 equally-spaced points on each image. You can [view the full fit][nolaiiif] on Allmaps.

TODO: more tests in more places

[nola5]: https://oldinsurancemaps.net/map/sanborn03376_029
[nolaiiif]: https://viewer.allmaps.org/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdanvk%2Fmapsnap%2Frefs%2Fheads%2Fmain%2Fgallery%2Fnew_orleans_la_1951_vol_5.iiif.json

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

We generally know _roughly_ where the map is: the Library of Congress has organized their Sanborn collection by country, state and county. We can get an even better estimate by locating the [key map] for the volume. This gives a bounding box that's at most a few miles in each dimension.

Next, we download contemporary streets in that bounding box from [OpenStreetMap] (OSM). Our hope is that enough streets have stayed the same that we can line them up between the Sanborn map and OSM.

We can use the contemporary streets to filter the OCR results:

![Detected streets in the image](/images/detected-streets.png)

Many of these are real streets, but many of them are not:

- The cut-off street names on the right are CLARK (👍), LIBERTY (👍) and CONGRESS (👎). This last one is matching the "Library of Congress" stamp, because there _is_ a Congress Street in Brooklyn.
- Most of the streets in the middle are correct: HENRY, MONROE, PIERREPONT, CLINTON, FULTON, WASHINGTON, JOHNSON, ADAMS, MYRTLE.
- It gets thrown off by "BROOKLYN BRIDGE APPROACH" and sees BROOKLYN in a few other places. These are all bad detections that could potentially throw off alignment.

The street matching here is quite flexible. We only match MONROE, which could be Monroe Place or Monroe Street: both exist in this area of Brooklyn. So we throw in both and hope to sort it out later.

Next, we extrapolate the streets in both directions, following the direction of the text. If two streets intersect in the image and in the OSM data, we record a candidate intersection:

![Intersections](/images/intersections.png)

These are known as Ground Control Points (GCPs). If we have two or more GCPs, we have enough data to fit a model. (Sometimes we can get a fit with just one — more on this in a bit.)

For each pair of GCPs, we can fit a model and see where it would place the street labels from OCR. If the label gets mapped close to the expected street in OSM, and the street is at the expected angle there, then that's an indicator of a good fit and this street is an "inlier." If not, it's an outlier.

We try each pair of GCPs and find the one that produces the best fit with the most inliers. (This is roughly the [RANSAC algorithm].) This is our mapping!

In the image above, the chosen GCPs are JOHNSON x ADAMS and MONROE x CLARK. The orange street labels are inliers this this mapping, and the gray ones are outliers. This rejects spurious streets like BROOKLYN and CONGRESS. Interestingly, it also rejects FULTON, which continued into this area in 1937 but stops short today.

Here's what the resulting mapping looks like:

![Map overlay](/images/map-overlay.jpg)

The fit is excellent. The streets and intersections line up well. If we zoom in, we can even see that some of the individual parcels match between 1937 and today. This is a good indication that our fit is accurate within a few feet.

You can view the [full mapping][bkiiif] for this section of Brooklyn on Allmaps.

[p19]: https://oldinsurancemaps.net/document/85714
[key map]: https://oldinsurancemaps.net/document/85676
[OpenStreetMap]: https://www.openstreetmap.org/#map=18/40.683787/-73.978527
[RANSAC algorithm]: https://www.thinkautonomous.ai/blog/ransac-algorithm/
[bkiiif]: https://viewer.allmaps.org/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdanvk%2Fmapsnap%2Frefs%2Fheads%2Fmain%2Fgallery%2Fbrooklyn_ny_1939_vol_2.iiif.json

### One GCP fits

It takes two GCPs to define a four parameter model (translate x, translate y, scale, rotation). But sometimes we can get away with just one.

Consider [page 432] from New Orleans 1951 Volume 5. Here are the detected streets and intersections:

![Image showing three streets and one intersection](/images/1gcp.png)

There are three streets and one intersection (S. Front and Napoleon don't intersect). The one intersection gives us a location, and hence the two translation parameters of the model. The angles of the roads can give us the rotation. But what about the scale?

Assuming this map is from a larger volume, we can assume that all the maps in the volume have roughly the same scale. Specifically, we plug in the median scale across all the maps with more GCPs. When you run the pipeline, these sorts of fits show up as "deferred." These fits aren't quite as robust as they'd be if we had more GCPs. But they're often pretty good, and they're better than nothing!

[page 432]: https://oldinsurancemaps.net/document/46011

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
mkdir data/new_orleans_la_1951_vol_5
curl -o data/new_orleans_la_1951_vol_5/main.iiif.json 'https://oldinsurancemaps.net/iiif/mosaic/sanborn03376_029/main-content/?trim=true'
uv run python mapsnap/download_oim_iiif.py data/new_orleans_la_1951_vol_5/main.iiif.json --oim-url-prefix 'https://s3.us-central-1.wasabisys.com/oldinsurancemaps/uploaded/documents/new_orleans_la_1951_vol_5_'
```

This downloads 109 full-resolution JPEG files. It's convenient to pull these from OIM's S3 bucket since the Library of Congress tile server is pretty aggressive about rate-limiting.

The full-resolution images are more than you need for OCR and georeferencing. To speed up the later steps, it's convenient to downscale them and convert to grayscale using [ImageMagick]:

```
cd data/new_orleans_la_1951_vol_5
for x in *.jpg; convert -colorspace gray -resize '2048>' $x ${x/.jpg/.2048px.jpg}
```

[ImageMagick]: https://imagemagick.org/command-line-processing/#gsc.tab=0

### Street and Intersections

Find the southwest and northeast corner of the key map to get a bounding box.

Mapsnap needs all the streets from OSM in this bounding box. To get them using the Overpass API, run:

```
uv run python mapsnap/download_osm.py 29.909795 -90.125975 29.946841 -90.083828 --output data/new_orleans_la_1951_vol_5/streets.osm.json
```

The order of the parameters is sw lat, sw lng, ne lat, ne lng.

Convert the OSM dump to GeoJSON by running:

```
uv run python mapsnap/osm_to_centerlines.py data/new_orleans_la_1951_vol_5/streets.osm.json --output data/new_orleans_la_1951_vol_5/centerlines.geojson
```

Though it's not needed for the pipeline, it can be helpful to extract street names and intersections to text files by running:

```
jq -r '.elements[].tags.name' streets.osm.json | grep -v '^null$' | sort | uniq > streets.txt
uv run python mapsnap/generate_intersections.py data/new_orleans_la_1951_vol_5/streets.osm.json data/new_orleans_la_1951_vol_5/intersections.csv
```

### Street Label OCR

Run `detect_text.py` over all the scaled-down images to find street labels + angles:

```
uv run python mapsnap/detect_text.py data/new_orleans_la_1951_vol_5/*.2048px.jpg
```

This is the slowest step. If you can use a GPU, it's ~15s/image. Iif you're running CPU-only, it's more like ~1 minute/image. This writes a `streets.json` file next to each image with candidate street label detections.

### Fit georeference model

This is it! Given detected street labels and street centerlines, find GCPs and fit a four-parameter model for each map:

```
rm data/new_orleans_la_1951_vol_5/*.georef.json
uv run mapsnap/georef_from_labels.py data/new_orleans_la_1951_vol_5/*.2048px.jpg --centerlines data/new_orleans_la_1951_vol_5/centerlines.geojson --min-long-side 60 --min-short-side 12 --fuzzy-match-threshold 0.20 --visualize-ocr
```

The main output is `pNNN.georef.json`, which contains the four-parameter model and debug information. Because of the `--visualize-ocr` flag, this also outputs `pNNN.detect.png` to help you debug the OCR.

For maps without enough control points, this will fail to produce an output. It won't delete an existing georef.json file, so make sure to run the `rm` command first to avoid cross-run contamination!

### Make an IIIF file

Download the IIIF file for these Sanborn maps from the Library of Congress:

```
curl -o data/new_orleans_la_1951_vol_5/loc.iiif.json https://www.loc.gov/item/sanborn03376_029/manifest.json
```

Probably, though, you'll need to load https://www.loc.gov/item/sanborn03376_029/manifest.json directly in your browser to avoid getting denied.

Then generate an IIIF file using the georeferences:

```
uv run python mapsnap/make_iiif_georef.py data/new_orleans_la_1951_vol_5/loc.iiif.json 'data/new_orleans_la_1951_vol_5/*.georef.json' --output data/new_orleans_la_1951_vol_5/generated.iiif.json
```

You can paste this into viewer.allmaps.org to look at the results.

### Measuring accuracy

Compare the generated IIIF file to the data from OIM:

```
uv run python compare_iiif_georef.py data/new_orleans_la_1951_vol_5/main.iiif.json data/new_orleans_la_1951_vol_5/generated.iiif.json
```

This will print out per-image stats and overall summary statistics.

## Debugging output

This repo comes with a small web app to help you debug individual maps:

```
cd app
npm i
npm run dev
```

Then visit localhost:5173 and drag the image file and its associated georef.json file into the browser window.

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
- Does this use AI?

## Prior Art

- [Shensky et. al. (2025)][shensky]
  - They detect intersections using a custom ML model, then try to read street labels in horizontal or vertical strips from them using Tesseract. This is essentially the reverse of what this repo does. It inherently cannot find diagonal streets, and Tesseract is generally a worse model than EasyOCR.
  - They fit six-paramter affine models using _all_ the GCPs they detect. Mapsnap uses a four-parameter model and the [RANSAC algorithm], which is able to throw out large numbers of outliers.
  - At least for New Orleans 1951, Mapsnap gets within 15ft on 76% of the images it maps, vs. 14% for the Shensky paper. (I'm not sure exactly which maps they're testing on, and they have other criteria, so this may not be a fair comparison.)

[shensky]: https://repositories.lib.utexas.edu/items/3f080054-8ff0-4e4c-8ef7-ea93b0fc36e0

## Debugger

🌎 [Mapsnap Debugger](https://www.danvk.org/mapsnap/)

The Mapsnap debugger lets you view georeferences and OCR results overlaid on the image. To use it, drag & drop the image and either its associated `georef.json` or `streets.json` file.

Debugging OCR:

![map in New Orleans with recognized street names](/images/mapsnap-debug-ocr.png)

Debugging georefs:

![Sanborn map of New Orleans overlaid on an OSM map](/images/mapsnap-debug-georef.jpg)

To run the debugger locally:

```
cd app
npm i
npm run dev
```

Then visit localhost:5173/mapsnap/.

To deploy the debugger:

```
npm run build
rm -rf ~/github/danvk.github.io/mapsnap && cp -r dist ~/github/danvk.github.io/mapsnap
```
