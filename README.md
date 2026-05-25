# Mapsnap

The goal of mapsnap is to automatically georeference Sanborn Insurance Maps.

## Performance

Test data comes from hand-geocoding by volunteers on OldInsuranceMaps.net:

Volume | Pages | Num Fit | Median RMSE | Within 15ft | Within 25ft | Allmaps
------ | ----- | ------- | ----------- | ----------- | ----------- | -------
[New Orleans 1951 Vol 5][nola5] | 109 | 101 (93%) | 12ft | 67% | 84% | [view][nola5-iiif]
[New Orleans 1896 Vol 2][nola2] | 91 | 83 (91%) | 25ft | 37% | 49% | [view][nola2-iiif]
[Detroit 1929 Vol 11][detroit] | 103 | 85 (83%) | 13ft | 58% | 73% | [view][detroit-iiif]
[Chicago 1950 Vol 1][chicago] | 111 | 100 (90%) | 10ft | 70% | 83% | [view][chicago-iiif]
[Champaign, Ill. 1915][champaign] | 33 | 28 (85%) | 11ft | 79% | 93% | [view][champaign-iiif]

RMSE was measured across 49 equally-spaced points on each image. You can view the fits on Allmaps or get the IIIF files from the `gallery` directory.

[nola5]: https://oldinsurancemaps.net/map/sanborn03376_029
[nola2]: https://oldinsurancemaps.net/map/sanborn03376_006
[detroit]: https://oldinsurancemaps.net/map/sanborn03985_041
[chicago]: https://oldinsurancemaps.net/map/sanborn01790_085
[champaign]: https://oldinsurancemaps.net/map/sanborn01778_006

[nola5-iiif]: https://viewer.allmaps.org/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdanvk%2Fmapsnap%2Frefs%2Fheads%2Fmain%2Fgallery%2Fnew_orleans_la_1951_vol_5.iiif.json
[nola2-iiif]: https://viewer.allmaps.org/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdanvk%2Fmapsnap%2Frefs%2Fheads%2Fmain%2Fgallery%2Fnew_orleans_la_1896_vol_2.iiif.json
[detroit-iiif]: https://viewer.allmaps.org/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdanvk%2Fmapsnap%2Frefs%2Fheads%2Fmain%2Fgallery%2Fdetroit_mich_1929_vol_11.iiif.json
[chicago-iiif]: https://viewer.allmaps.org/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdanvk%2Fmapsnap%2Frefs%2Fheads%2Fmain%2Fgallery%2Fchicago_il_1950_vol_1.iiif.json
[champaign-iiif]: https://viewer.allmaps.org/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdanvk%2Fmapsnap%2Frefs%2Fheads%2Fmain%2Fgallery%2Fchampaign_ill_1915.iiif.json

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
- "Prepare" all the other maps on OIM.
- Run the following command:

```bash
./pipeline.sh sanborn03376_029 new_orleans_la_1951_vol_5 r1836428 'https://s3.us-central-1.wasabisys.com/oldinsurancemaps/uploaded/documents/new_orleans_la_1951_vol_5_'
```

The four arguments here are:

- sanborn03376_029: Library of Congress (LoC) Sanborn Map ID number, also in the OIM URL.
- new_orleans_la_1951_vol_5: directory slug. Output will go in `data/new_orleans_la_1951_vol_5`.
- r1836428: Relation containing this map in OSM, usually a county. This is used to download all the streets in the area of the map.
- 'https://...': OIM S3 bucket prefix for images. You can get this by downloading a JPEG of a page from the OIM web site.

See `pipeline.sh` details about how to run the pipeline. The high-level steps are:

- `download_oim_iiif.py`: download all the Sanborn images from OIM. This is easier than downloading directly from the Library of Congress and gets you splits.
- `scale_images.py`: reduces the size of the images by a uniform scale factor so that OCR runs faster.
- `download_osm.py`: downloads all street data in the area from OpenStreetMap.
- `osm_to_centerlines.py`: converts raw OSM data to GeoJSON.
- `detect_text.py`: runs OCR over the downscaled images, saving candidate detections to `streets.json` files.
- `georef_from_labels.py`: georeferences images based on street detections, writing out `georef.json` files where it can find a good fit.
- `make_iiif_georef.py`: produces a IIIF JSON file from the georeferences. You can find examples of these in the `gallery` directory.
- `compare_iiif_georef.py`: compares the generated IIIF file with the human-generated one from OIM, producing a report on the accuracy of the fit.

The pipeline is set up to get imagery from OIM, but it's not dependent on OIM in a deep way. You can absolutely run it on images taken directly from the Library of Congress or another source, you'll just have to run some steps manually. In particular, `make_iiif_georef.py` can use either OIM's IIIF Manifest or the LoC's.

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
