/**
 * Locate the artifact directory that produced a given IIIF annotation page.
 *
 * `mapsnap fit --tag <tag>` writes `<volume>/<tag>.iiif.json` and, for some
 * runs, a matching `<volume>/artifacts/<tag>/` holding the per-page sidecars
 * that run produced. Those sidecars are the ones worth linking to from the
 * viewer: the top-level `<stem>.georef.json` is whatever ran most recently and
 * may have nothing to do with the annotation on screen.
 */

/**
 * The `artifacts/<tag>` directory for an annotation, volume-relative, or null.
 *
 * Handles both ways an annotation is addressed: `<volume>/<tag>.iiif.json` (the
 * copy at the volume root) and `<volume>/artifacts/<tag>/<tag>.iiif.json` (the
 * one inside the run's own directory, which is its own answer).
 *
 * Returns a path only; whether it exists, and whether it holds anything useful,
 * is the caller's to check.
 */
export function runArtifactDir(relativePath: string): string | null {
  const parts = relativePath.split('/');
  const file = parts[parts.length - 1];
  if (!file?.endsWith('.iiif.json')) return null;
  const tag = file.slice(0, -'.iiif.json'.length);

  // Already inside the run's directory: `<volume>/artifacts/<tag>/<tag>.iiif.json`.
  if (parts.length >= 3 && parts[parts.length - 2] === tag) {
    return parts.slice(0, -1).join('/');
  }
  // At the volume root: `<volume>/<tag>.iiif.json`.
  if (parts.length >= 2) {
    return [...parts.slice(0, -1), 'artifacts', tag].join('/');
  }
  return null;
}

/** Page sidecars a run archived: `p12.georef.json`, `p12.georef-osm.json`, `p12.streets.json`. */
const PAGE_SIDECAR = /^(p[^.]+)\.(?:georef(?:-[a-z0-9-]+)?|streets)\.json$/;

/**
 * Page stems a run saved any sidecar for, from that directory's file names.
 *
 * Every per-page artifact counts, not only the plain georef. A page the run
 * demoted or rescued has `georef-osm.json` or `georef-nofit.json` and no plain
 * `georef.json`, and a page that never fitted still has `streets.json`. Matching
 * only `p<stem>.georef.json` reported "this run saved no sidecar for this page"
 * for 54 of 112 pages on one Fargo run -- precisely the unfitted pages someone
 * opening the viewer is most likely to be looking at.
 */
export function runArtifactStems(files: string[]): string[] {
  const stems = new Set<string>();
  for (const file of files) {
    const match = file.match(PAGE_SIDECAR);
    if (match?.[1]) stems.add(match[1]);
  }
  return [...stems].sort();
}
