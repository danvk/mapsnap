/**
 * The atlas's place index: every town the Sanborn collection covers.
 *
 * Built by scripts/atlas/build_places.py. The index is deliberately thin --
 * enough to draw a dot and match a search -- because it loads before anything
 * can be shown. A town's volumes live in a per-state file fetched on demand.
 */

/** One town, as the opening map draws it. */
export interface Place {
  /** "illinois/chicago": the state and town slugs, and the volumes-file key. */
  id: string;
  name: string;
  state: string;
  lon: number;
  lat: number;
  /** Catalogued volumes, and the sheets across them. */
  volumes: number;
  sheets: number;
  /** How many of those volumes the mirror holds, and so the run could fit. */
  mirrored: number;
  firstYear: number | null;
  lastYear: number | null;
}

export interface PlaceIndex {
  /** The corpus run whose annotations the app reads. */
  runTag: string;
  bucket: string;
  places: Place[];
}

/** One catalogued volume of a town. */
export interface Volume {
  item: string;
  /** The catalogue's date string, which may carry a month ("1907-06"). */
  date: string;
  year: number | null;
  sheets: number;
  title: string;
  /**
   * The mirror's state slug and year, present only for a mirrored volume.
   *
   * Both come from the mirror's own mapping rather than from the catalogue:
   * 310 items are filed under a year their catalogue date does not give, and
   * guessing sends the fetch to a key that does not exist.
   */
  state?: string;
  mirrorYear?: string;
}

/** The state slug of a place id, which is also its volumes file's name. */
export function stateSlug(placeId: string): string {
  return placeId.split('/')[0] ?? '';
}

/**
 * Where a mirrored volume's annotation lives in the bucket.
 *
 * Returns null for a volume the mirror never took, which has no run output to
 * read however the corpus run went.
 */
export function annotationUri(
  volume: Volume,
  index: Pick<PlaceIndex, 'bucket' | 'runTag'>,
): string | null {
  if (!volume.state || !volume.mirrorYear) return null;
  return (
    `s3://${index.bucket}/by-state/${volume.state}/${volume.mirrorYear}/` +
    `${volume.item}/runs/${index.runTag}/mapsnap.iiif.json`
  );
}

/** Descending distinct years a town has volumes for; undated ones last. */
export function yearsOf(volumes: Volume[]): (number | null)[] {
  const years = [...new Set(volumes.map((v) => v.year))];
  years.sort((a, b) => {
    if (a === null) return 1;
    if (b === null) return -1;
    return b - a;
  });
  return years;
}

/** The volumes of one year, in catalogue order. */
export function volumesOfYear(
  volumes: Volume[],
  year: number | null,
): Volume[] {
  return volumes.filter((volume) => volume.year === year);
}

/**
 * The year to open a town on: its most recent with something to draw.
 *
 * A town's newest volume is often one the mirror never took, and opening on a
 * year that renders nothing looks like a broken app rather than a gap in the
 * data. Falls back to the newest year of any kind when nothing is mirrored, so
 * the panel still has something selected to describe.
 */
export function defaultYear(volumes: Volume[]): number | null {
  const mirrored = volumes.filter((volume) => volume.state);
  const years = yearsOf(mirrored.length > 0 ? mirrored : volumes);
  return years[0] ?? null;
}

/**
 * Towns matching a typed query, best first.
 *
 * Ranked by where the match falls and how big the town is: typing "chic"
 * should offer Chicago before Chicopee, and a prefix match should beat one in
 * the middle of a name. "chicago, il" and "chicago illinois" both work, since
 * the query is matched against "<name>, <state>".
 */
export function searchPlaces(
  places: Place[],
  query: string,
  limit = 8,
): Place[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];
  const scored: { place: Place; score: number }[] = [];
  for (const place of places) {
    const haystack = `${place.name}, ${place.state}`.toLowerCase();
    const at = haystack.indexOf(needle);
    if (at < 0) continue;
    // A prefix match outranks any interior one, whatever the sizes; within a
    // tier the bigger town wins, which is what a one-word query usually means.
    const tier = at === 0 ? 2 : haystack[at - 1] === ' ' ? 1 : 0;
    scored.push({ place, score: tier * 1e9 + place.sheets });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit).map((entry) => entry.place);
}
