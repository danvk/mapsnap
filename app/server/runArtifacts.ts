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

  // A corpus run synced from the mirror: `<volume>/runs/<tag>/mapsnap.iiif.json`,
  // with every sidecar beside it.
  const run = splitRunPath(relativePath);
  if (run.runDir) return `${run.volume}/${run.runDir}`;

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

/** A run directory of the mirror's layout, relative to its volume: `runs/<tag>`. */
const RUN_DIR = /^runs\/[A-Za-z0-9._-]+$/;

/** Whether `runDir` names one run directory (`runs/corpus-v1`), and nothing else. */
export function isRunDir(runDir: unknown): runDir is string {
  return typeof runDir === 'string' && RUN_DIR.test(runDir);
}

/**
 * Split a volume-relative file path around a mirror run directory.
 *
 * A volume synced from the mirror keeps each corpus run under `runs/<tag>/`:
 * `wernersville_pa_1914/runs/corpus-v1/mapsnap.iiif.json` is the file
 * `mapsnap.iiif.json` of run `runs/corpus-v1` in volume `wernersville_pa_1914`.
 * Page scans stay at the volume root, which is why the two must be told apart.
 * A path with no run directory is a file at its own directory, as before.
 */
export function splitRunPath(relativePath: string): {
  volume: string;
  runDir: string | null;
  file: string;
} {
  const parts = relativePath.split('/');
  const file = parts[parts.length - 1] ?? '';
  const runsIndex = parts.length - 3;
  if (runsIndex >= 1 && parts[runsIndex] === 'runs') {
    return {
      volume: parts.slice(0, runsIndex).join('/'),
      runDir: parts.slice(runsIndex, -1).join('/'),
      file,
    };
  }
  return { volume: parts.slice(0, -1).join('/'), runDir: null, file };
}

/** Page sidecars a run archived: `p12.georef.json`, `p12.georef-snap.json`, `p12.streets.json`. */
const PAGE_SIDECAR = /^(p[^.]+)\.(?:georef(?:-[a-z0-9-]+)?|streets)\.json$/;

/**
 * Page stems a run saved any sidecar for, from that directory's file names.
 *
 * Every per-page artifact counts, not only the plain georef. A page the run
 * rescued has `georef-snap.json` as well as its plain `georef.json`, and a page
 * that never fitted still has `streets.json`. Matching
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
