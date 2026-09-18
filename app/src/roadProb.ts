/**
 * Where to find a page's P(road) map, across both naming eras.
 *
 * `mapsnap roadprob` writes `<stem>.roadprob.jpg` beside the page (#354), and
 * `mapsnap split` cuts a parent's map into its panels'. Volumes fitted before
 * that have PNGs under the volume's `artifacts/edge_join/roadprob/`. Neither
 * is a legacy curiosity: as of 2026-09-18 the 24 truth volumes under data/ have
 * ONLY the old PNGs and everything mirrored since has ONLY the new sidecars, so
 * a client that knows one name shows no P(road) map for half the corpus.
 *
 * The file names come from the server's own helpers, which resolve the same
 * pair in `alternatePageImage`.
 */
import { legacyPageImageFile, pageImageFile } from '../server/iiifAnnotations';

/** The P(road) maps to try for `stem` in `volumeDir`, current naming first. */
export function roadProbCandidates(
  volumeDir: string,
  stem: string,
): [string, string] {
  return [
    `${volumeDir}/${pageImageFile('roadprob', stem)}`,
    `${volumeDir}/${legacyPageImageFile(stem)}`,
  ];
}

/** Whether `url` is actually an image, not a dev-server index page or a 404. */
export async function isImage(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return (
      response.ok &&
      (response.headers.get('content-type')?.startsWith('image/') ?? false)
    );
  } catch {
    return false;
  }
}

/** The first of `urls` that is on disk, or null when none is. */
export async function firstImage(urls: string[]): Promise<string | null> {
  for (const url of urls) {
    if (await isImage(url)) return url;
  }
  return null;
}
