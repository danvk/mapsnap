/**
 * Where a split parent's `panels.json` lives, for either kind of annotation.
 *
 * Its own module because it is pure and worth testing directly: the s3 form is
 * a sibling-path rewrite, and getting it wrong is invisible -- the viewer
 * silently falls back to holing the sheet rectangle with the panel's own ring,
 * which triangulates into visible wedges rather than failing outright (#496).
 */

/**
 * The URL to fetch `<parentStem>.panels.json` from, or null if there is none.
 *
 * An annotation in the mirror keeps its sidecars beside it under the run tag,
 * reachable through the server's S3 object route; a local volume's sit under
 * `data/<volume>/`.
 */
export function panelsUrlFor(
  annotationPath: string | null,
  volume: string | undefined,
  parentStem: string,
): string | null {
  if (!parentStem) return null;
  if (annotationPath?.startsWith('s3://')) {
    const sibling = annotationPath.replace(
      /[^/]+$/,
      `${parentStem}.panels.json`,
    );
    return `/s3-api/object?uri=${encodeURIComponent(sibling)}`;
  }
  return volume ? `/data/${volume}/${parentStem}.panels.json` : null;
}
