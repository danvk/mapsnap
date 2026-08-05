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
