/**
 * Work around an express-iiif (1.7.0) size bug that squeezes Allmaps tiles.
 *
 * The library's non-upscaling `w,h` handler only treats a request as a plain
 * downscale when BOTH dimensions are strictly smaller than the region's; a
 * size exactly equal to the region falls into its "clamp" branches, which
 * rescale one dimension by the request's aspect ratio (655,768 for a 655×768
 * region comes back 558×768). Allmaps requests exactly this identity form for
 * its scale-factor-1 tiles, so every non-square edge tile rendered squeezed —
 * visible as sheets shifting when the viewer zooms across tile scale levels.
 *
 * Rewriting an identity size to `max` (which the library serves correctly)
 * side-steps the broken branch without changing the response semantics.
 */
export function normalizeIiifImageUrl(url: string): string {
  return url.replace(
    /\/(\d+),(\d+),(\d+),(\d+)\/(\d+),(\d+)\//,
    (match, x, y, width, height, sizeWidth, sizeHeight) =>
      sizeWidth === width && sizeHeight === height
        ? `/${x},${y},${width},${height}/max/`
        : match,
  );
}
