/**
 * Splitting a repo-root-relative annotation path into volume + file name.
 *
 * Its own module because it is pure and worth testing directly: importing
 * VolumeViewer to reach it pulls in maplibre-gl, which will not load under jsdom.
 */

/** How many directory levels below `data/` a volume may sit; mirrors the server. */
export const MAX_VOLUME_DEPTH = 2;

/**
 * Split "data/<volume>/<file>.iiif.json" into its volume and file name.
 *
 * A volume synced from the mirror keeps each run's outputs under
 * `runs/<tag>/`, so "data/wernersville_pa_1914/runs/corpus-v1/mapsnap.iiif.json"
 * is volume "wernersville_pa_1914", run "runs/corpus-v1", file
 * "mapsnap.iiif.json". `run` is null for an annotation at the volume root.
 *
 * The volume may be a subdirectory of a multi-volume atlas, so
 * "data/brooklyn_1904-1908/vol13/2026-08-04.iiif.json" yields the volume
 * "brooklyn_1904-1908/vol13". Matching only single-segment volumes left every
 * volume-scoped fetch -- key maps, notes, adjacency, failed georefs -- unmade for
 * those files, which is why their page links came up broken (#228).
 *
 * The depth limit mirrors the server's, so a path the API would reject never
 * parses into a volume the UI then queries for.
 */
export function parseAnnotationPath(
  path: string | null,
): { volume: string; file: string; run: string | null } | null {
  const inRun = path?.match(
    /^data\/([^/]+(?:\/[^/]+)?)\/(runs\/[^/]+)\/([^/]+)$/,
  );
  if (inRun) {
    return {
      volume: inRun[1] ?? '',
      run: inRun[2] ?? '',
      file: inRun[3] ?? '',
    };
  }
  const match = path?.match(/^data\/([^/]+(?:\/[^/]+)?)\/([^/]+)$/);
  return match
    ? { volume: match[1] ?? '', file: match[2] ?? '', run: null }
    : null;
}

/**
 * The page image a debug view should open for a page: its own if it has one,
 * else its sheet's.
 *
 * A volume synced from the mirror has only whole-sheet scans, so a split
 * panel (`p2__2`) has no `p2__2.jpg`. The debugger handles `p2.jpg` with the
 * panel's sidecars by mapping through `p2.panels.json`, so the sheet is the
 * right link. With no image list (an older server) the page's own stem is
 * kept, as before.
 */
export function debugImageStem(
  stem: string,
  pageImages: ReadonlySet<string> | null,
): string {
  if (!pageImages || pageImages.has(stem)) return stem;
  const sheet = stem.replace(/__\d+$/, '');
  return pageImages.has(sheet) ? sheet : stem;
}
