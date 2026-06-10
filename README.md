# Mapsnap

The goal of Mapsnap is to automatically georeference Sanborn Insurance Maps.

If you'd like to georeference a map using Mapsnap, read about [How it Works](#how-it-works) then head down to the [Pipeline](#pipeline) section.

## Performance

Test data comes from hand-geocoding by volunteers on OldInsuranceMaps.net:

Volume | Pages | Num Fit | Median RMSE | Within 15ft | Within 25ft | Allmaps
------ | ----- | ------- | ----------- | ----------- | ----------- | -------
[New Orleans 1951 Vol 5][nola5] | 109 | 101 (93%) | 12ft | 67% | 84% | [view][nola5-iiif]
[New Orleans 1896 Vol 2][nola2] | 91 | 83 (91%) | 25ft | 37% | 49% | [view][nola2-iiif]
[Detroit 1929 Vol 11][detroit] | 103 | 85 (83%) | 13ft | 58% | 73% | [view][detroit-iiif]
[Chicago 1950 Vol 1][chicago] | 111 | 100 (90%) | 10ft | 70% | 83% | [view][chicago-iiif]
[Champaign, Ill. 1915][champaign] | 33 | 28 (85%) | 11ft | 79% | 93% | [view][champaign-iiif]

RMSE was measured across 49 equally-spaced points on each image. You can view the fits on Allmaps or get the IIIF files from the `gallery` directory. For notes on poor fits, see [test data notes][].

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
[test data notes]: https://github.com/danvk/mapsnap/wiki/Test-data-notes

## How it Works

Here's an [example][p19] of a Sanborn Insurance Map:

![Brooklyn 1939 Volume 2 Page 19](/images/brooklyn_ny_1939_vol_2_p19.jpg)

This image depicts a small area of Brooklyn, NY in 1939. Our goal is to overlay this on a contemporary map by determining its location, scale, and rotation.

We can run EasyOCR against it to get candidate street labels:

![Same image with boxes drawn around text](/images/brooklyn_ny_1939_vol_2_p19.detect.jpg)

There's a lot of text in this image! The green boxes show labels that Mapsnap believes are likely to be streets. These are all rectangles that:

1. EasyOCR's CRAFT text recognizer thinks contain text.
2. Match a street name somewhere in Brooklyn.
3. Are larger than a minimum size.

From each bounding rectangle we get three bits of information:

- A candidate street name
- A location in the image (center of the bounding rectangle)
- A direction (long edge of the bounding rectangle)

This is the information we'll use to determine the map projection.

We generally know _roughly_ where the map is: the Library of Congress has organized their Sanborn collection by country, state and county. So to get street names, we can download all of OSM's features for that county. Our hope is that enough streets have stayed the same that we can line them up between the Sanborn map and OSM.

(You can get much tighter by locating the [key map] for the volume. This gives a bounding box that's at most a few miles in each dimension, but it comes at the cost of an additional, manual step. A county is usually precise enough for Mapsnap.)

Most of the detected streets are real, but some of them are not.

- The streets in the middle are correct: HENRY, MONROE, PIERREPONT, CLINTON, FULTON, WASHINGTON, JOHNSON, ADAMS, MYRTLE.
- It gets thrown off by "BROOKLYN BRIDGE APPROACH" and sees BROOKLYN in a few other places. There's a "Brooklyn Avenue" and a "Bridge Street" in Brooklyn, but this doesn't refer to either of them. These are bad detections that could potentially throw off alignment.
- It matches the "POST" in "POST OFFICE" because "POST COURT" is a street in Brooklyn. Again, this could create trouble.
- There are a few more lower-confidence street detections in red boxes.

The street matching here is quite flexible. We only match MONROE, which could be Monroe Place or Monroe Street: both exist in Brooklyn. So we throw in both and hope to sort it out later.

Next, we extrapolate the streets in both directions, following the direction of the text. If two streets intersect near the image _and_ in the OSM data, we record a candidate intersection:

![Intersections](/images/intersections.png)

These are known as Ground Control Points (GCPs). If we have two or more GCPs, we have enough data to fit a model. (If you thought you needed three, see [Notes on the Model](#six-parameters-vs-four), below. Sometimes we can get a fit with just one — more on this [in a bit](#one-gcp-fits).)

For each pair of GCPs, we can fit a model and see where it would place the street labels from OCR. If the label gets mapped close to the expected street in OSM, and the street is at the expected angle there, then that's an indicator of a good fit and this street is an "inlier." If not, it's an outlier.

We try each pair of GCPs and find the one that produces the best fit with the most inliers. (This is roughly the [RANSAC algorithm].) This is our mapping!

In the image above, the chosen GCPs are PIERREPONT STREET x HENRY STREET and PIERREPONT STREET x CLINTON STREET. The orange street labels are inliers this this mapping, and the gray ones are outliers. This rejects spurious streets like BROOKLYN and POST. Interestingly, it also rejects FULTON, which continued into this area in 1937 but stops short today.

Here's what the resulting mapping looks like:

![Map overlay](/images/map-overlay.jpg)

The fit is excellent. The streets and intersections line up well. If we zoom in, we can even see that some of the individual parcels match between 1937 and today. This is a good indication that our fit is accurate within a few feet.

You can view the [full mapping][bkiiif] for this section of Brooklyn on Allmaps.

[p19]: https://oldinsurancemaps.net/document/85714
[key map]: https://oldinsurancemaps.net/document/85676
[RANSAC algorithm]: https://www.thinkautonomous.ai/blog/ransac-algorithm/
[bkiiif]: https://viewer.allmaps.org/?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdanvk%2Fmapsnap%2Frefs%2Fheads%2Fmain%2Fgallery%2Fbrooklyn_ny_1939_vol_2.iiif.json

### One GCP fits

It takes two GCPs to define a four parameter model (translate x, translate y, scale, rotation). But sometimes we can get away with just one.

Consider [page 431] from New Orleans 1951 Volume 5. Here are the detected streets and intersections:

![Image showing three streets and one intersection](/images/1gcp.jpg)

There are three streets and one intersection (CADIZ and FRONT don't intersect). The one intersection gives us a location, and hence the two translation parameters of the model. The angles of the roads can give us the rotation. But what about the scale?

Assuming this map is from a larger volume, we can assume that all the maps in the volume have roughly the same scale. Specifically, we plug in the median scale across all the maps with more GCPs. When you run the pipeline, these sorts of fits show up as "deferred."

These fits aren't quite as robust as they'd be if we had more GCPs, so we require a little bit more evidence before accepting them. Specifically, we look at all the adjacent pages in the volume. If two or more of them match the rotation angle of the 1-gcp fit, then we keep it. Otherwise we toss it.

Sometimes these fits aren't the best, but they're often pretty good and they significantly increase Mapsnap's coverage for some volumes, e.g. [Detroit 1929 Vol 11][detroit].

[page 431]: https://oldinsurancemaps.net/document/46009

### Automatic Masks

Sanborn pages typically include colorful, detailed information in the center and are more sparse towards the edges. The detail for those areas is contained on other pages. If you make a georeferenced map with the full pages, there will be significant overlap. The detailed parts of one map might be hidden behind the margins of another.

The solution to this is a **mask**. Each image in the IIIF file has a clipping polygon that removes margins and areas that are better covered by other pages. (OldInsuranceMaps calls this a [multimask].)

![Page 22 of Brooklyn, NY map with a clipping polygon](/images/p22-mask.jpg)

Mapsnap automatically generates clipping polygons for each image using the underlying street grid. The idea is that each "block" should only be represented by one page from the Sanborn volume. We choose the page that has the most color for that block. This sometimes generates more complex polygons than a human would, but it tends to work well in practice, at least in areas where the street grid hasn't changed much since the map was made. See [PR #31] for details.

[multimask]: https://docs.oldinsurancemaps.net/guides/trimming/
[PR #31]: https://github.com/danvk/mapsnap/pull/31

## Pipeline

Mapsnap can, in principle, run on any type of map. But it's only been tested on Sanborn maps. This repo contains tools for downloading and georeferencing Sanborn maps from two sources:

1. **OldInsuranceMaps** (OIM). OIM hosts a subset of volumes (~1,000) that have already been manually split and georeferenced. You can download images from it reliably. Mapsnap uses it for truth data.
2. **Library of Congress** (loc.gov). The LoC hosts most of the Sanborn volumes that are in the public domain (~30,000). These have not been georeferenced, and downloads are somewhat unreliable.

Depending on where you get your Sanborn maps, the next steps will be different.

### OldInsuranceMaps

```bash
./pipeline-oim.sh sanborn03376_029 new_orleans_la_1951_vol_5 r1836428 'https://s3.us-central-1.wasabisys.com/oldinsurancemaps/uploaded/documents/new_orleans_la_1951_vol_5_'
```

The four arguments here are:

- sanborn03376_029: Library of Congress (LoC) Sanborn Map ID number, also in the OIM URL.
- new_orleans_la_1951_vol_5: directory slug. Output will go in `data/new_orleans_la_1951_vol_5`.
- r1836428: Relation containing this map in OSM, usually a county. This is used to download all the streets in the area of the map.
- 'https://...': OIM S3 bucket prefix for images. You can get this by downloading a JPEG of a page from the OIM web site.

See `pipeline-oim.sh` details about how to run the pipeline. The high-level steps are:

- `download_oim_iiif.py`: download all the Sanborn images from OIM. This is easier than downloading directly from the Library of Congress and gets you splits.
- `scale_images.py`: reduces the size of the images by a uniform scale factor so that OCR runs faster.
- `download_osm.py`: downloads all street data in the area from OpenStreetMap.
- `osm_to_centerlines.py`: converts raw OSM data to GeoJSON.
- `detect_text.py`: runs OCR over the downscaled images, saving candidate detections to `streets.json` files.
- `georef_from_labels.py`: georeferences images based on street detections, writing out `georef.json` files where it can find a good fit.
- `make_iiif_georef.py`: produces a IIIF Georeference Extension. You can find examples of these in the `gallery` directory. View them on Allmaps.
- `compare_iiif_georef.py`: compares the generated IIIF file with the human-generated one from OIM, producing a report on the accuracy of the fit.

The last three steps run relatively quickly. You can experiment with options and iterate on them using `fit.sh`.

### Library of Congress

Again, using New Orleans 1951 Vol 5 as an example:

- Go to https://www.loc.gov/item/sanborn03376_029.
- Download the IIIF Presentation Manifest ("Manifest (JSON/LD)"). You have to do this by hand in your browser due to the LoC's Cloudflare DoS protections.
- Make a directory and put this file in `data/new_orleans_la_1951_vol_5/loc.manifest.json`.

Next, download all the images at 25% resolution by running:

```bash
uv run mapsnap/download_loc_iiif.py --scale pct:25 data/new_orleans_la_1951_vol_5/loc.manifest.json
```

This will go slowly. It might fail and you might have to wait a few hours to restart it and try again. But it will eventually get all the images.

From here, the process is similar to OldInsuranceMaps:

```bash
./pipeline-loc.sh sanborn03376_029 new_orleans_la_1951_vol_5 r1836428
```

See above for an explanation of these three parameters. This script downloads OSM data, runs OCR over the images, and then runs `fit.sh`.

The end result is a IIIF Georeference Extension that you can view with Allmaps. Since there's no truth data, you won't get a comparison at the end.

## Debugging output

This repo comes with a small web app to help you debug individual maps. You can access it at:

🌎 [Mapsnap Debugger](https://www.danvk.org/mapsnap/)

Drag & drop an image and either a `streets.json` or `georef.json` file to use it. All of the screenshots above were taken using the debugger.

## Notes on the model

### Six Parameters vs. Four

OldInsuranceMaps.net (OIM) uses a six parameter affine model. In addition to translation (two parameters), rotation and scale, this adds two new parameters:

- Skew: the x- and y-directions in the map need not be exactly 90° from each other.
- Scale anisotropy: A pixel in the x- and y-directions need not be the same distance.

In practice, the Sanborn maps are all very well-made and well-scanned and don't exhibit much of either of these. Most fits on OIM have scale anisitropy of less than 3% and skew of less than 3°. These are both close enough to zero that it's unclear if they're real or the result of inaccurate georeferencing. Removing these parameters makes it easier to fit a model since you only need two GCPs rather than three. And it eliminates a failure mode of detecting heavily skewed maps due to an inaccurate GCP.

In the GIS world, the four parameter model is known as a [Helmert transformation].

[Helmert transformation]: https://en.wikipedia.org/wiki/Helmert_transformation#Variations

### Why RANSAC?

When you find GCPs on OldInsuranceMaps or Allmaps, they use all of them to fit a model. If you provide more GCPs than needed, they use a technique like [least squares] to find a best fit. Every GCP you provide influences the model.

That is _not_ how mapsnap works. To georeference an image, it uses each pair of GCPs to fit a candidate model. It then looks at how well this model can explain the street labels. The best pair of GCPs are chosen, and the rest are discarded. They have no direct influence on the fit.

On some level this feels wasteful. Why throw out a GCP? The key difference is that we have much less confidence Mapsnap's GCPs than we would in a human's. It's quite likely that some large fraction of our GCPs are wrong. This might be because of an OCR mistake or misinterpretation (the "POST" in "POST OFFICE" is not Post Street), because a street turns before reaching an intersection (so that extrapolation is invalid), or because the street label was misinterpreted ("17TH" meant 17th Street, not 17th Ave).

Averaging a mix of correct and incorrect GCPs is unlikely to produce a good model. But by using [RANSAC], which is extremely robust to outliers, we can find even a little bit of signal through the noise.

[least squares]: https://en.wikipedia.org/wiki/Least_squares

### When does this fail?

Mapsnap makes an assumption that the streets today are at least somewhat like the streets when the map was produced. This is usually the case. The streets don't have to be _exactly_ the same. If a highway was put through a map, it can still be georeferenced so long as some of the streets adjacent to the highway remain the same. These sorts of maps are harder to georeference, though.

There are cases where this breaks down, though. If the streets in a large area have been reworked, then there's nothing for Mapsnap to "snap" to.

If the streets have all been renamed, this can also throw off georeferencing. For example, the borough of Queens in New York City systematically renamed all its streets in 1911. Mapsnap does well on 1945 Queens maps, but it does terribly on the [1898 maps][queens1898].

[queens1898]: https://www.loc.gov/item/sanborn06198_001

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

## Development

Quickstart:

```
uv sync
uv run pytest
uv run pyright
uv run ruff check
uv run ruff format
```

### Debugger

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
