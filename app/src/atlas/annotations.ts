/**
 * Reading a corpus run's annotations as the atlas needs them.
 *
 * Two sources, because the obvious one does not hold up. A published
 * annotation points its image services at loc.gov, and using it unmodified is
 * the cheapest thing the atlas could do -- nothing but the annotation JSON
 * would ever touch our own storage. But a town-year is hundreds of sheets and
 * thousands of tiles, and tile.loc.gov rate-limits well below that: opening
 * Chicago 1951 (7 volumes, 719 sheets) drew 4,096 failed requests and left the
 * host answering 429 to everything for minutes afterwards. The browser reports
 * those as CORS failures, since a 429 error page carries no CORS headers.
 *
 * So the default is the mirror, through the same rewrite the debugger uses:
 * each page's service is repointed at our own cache of the mirrored 25% scans.
 * `loc` remains selectable, and is the right source for a single volume.
 */

import { pagesFromAnnotation, type PageGeo } from '../iiif/pages';
import type { GeorefAnnotationPage } from '../../server/iiifAnnotations';
import type { Volume } from './places';

/** A volume's annotation, and the page geometry derived from it. */
export interface LoadedVolume {
  volume: Volume;
  annotation: GeorefAnnotationPage;
  pages: PageGeo[];
}

/** A volume that has no annotation to draw, and why. */
export interface MissingVolume {
  volume: Volume;
  reason: 'not mirrored' | 'not in this run' | 'unreadable';
}

/**
 * The page stem an item's label names, e.g. "p66" or "p66__1".
 *
 * A published annotation labels each item
 * "Chicago, Illinois | 1950 | sanborn01790_085 p100W", with a split panel
 * carrying its number in a trailing "[1]". The debugger's own annotations
 * carry the stem as a `page` metadata entry instead, which is what
 * `pagesFromAnnotation` reads -- hence the normalisation below rather than a
 * second geometry path.
 */
export function stemFromLabel(label: string | undefined): string | null {
  const tokens = (label ?? '').trim().split(/\s+/);
  const last = tokens[tokens.length - 1];
  if (!last) return null;
  const panel = /^\[(\d+)\]$/.exec(last);
  if (panel) {
    const stem = tokens[tokens.length - 2];
    return stem ? `${stem}__${panel[1]}` : null;
  }
  return /^p/.test(last) ? last : null;
}

/**
 * Add the `page` metadata entry a published annotation does not carry.
 *
 * Mutates a copy, never the argument: the annotation is also what gets handed
 * to Allmaps, and a page entry is meaningless to it either way.
 */
export function withPageMetadata(
  annotation: GeorefAnnotationPage,
): GeorefAnnotationPage {
  const items = (annotation.items ?? []).map((item) => {
    const metadata = item.metadata ?? [];
    if (metadata.some((entry) => entry.label === 'page')) return item;
    const stem = stemFromLabel(item.label as string | undefined);
    if (!stem) return item;
    return { ...item, metadata: [...metadata, { label: 'page', value: stem }] };
  });
  return { ...annotation, items } as GeorefAnnotationPage;
}

/** Where a volume's sheet images come from. */
export type ImageSource = 'mirror' | 'loc';

/**
 * Where to fetch a volume's annotation, for the chosen image source.
 *
 * `loc` takes the published annotation verbatim, so its pages resolve to
 * loc.gov. `mirror` asks the server to rewrite it against the cached scans,
 * which also fills in the `page` metadata the raw file lacks.
 */
export function annotationUrl(uri: string, source: ImageSource): string {
  return source === 'loc'
    ? `/s3-api/object?uri=${encodeURIComponent(uri)}`
    : `/iiif-api/annotation?path=${encodeURIComponent(uri)}`;
}

/**
 * Fetch one volume's annotation, or say why there is none.
 *
 * A 404 is the ordinary case while a run is still going, not an error worth
 * throwing: the town simply has fewer volumes to draw today than it will
 * tomorrow.
 */
export async function loadVolume(
  volume: Volume,
  uri: string | null,
  source: ImageSource,
): Promise<LoadedVolume | MissingVolume> {
  if (!uri) return { volume, reason: 'not mirrored' };
  try {
    const response = await fetch(annotationUrl(uri, source));
    if (!response.ok) return { volume, reason: 'not in this run' };
    const body = (await response.json()) as GeorefAnnotationPage & {
      annotation?: GeorefAnnotationPage;
    };
    // The rewrite route wraps its result; the raw object route does not.
    const raw = body.annotation ?? body;
    const annotation = withPageMetadata(raw);
    return { volume, annotation, pages: pagesFromAnnotation(annotation) };
  } catch {
    return { volume, reason: 'unreadable' };
  }
}

/** Whether a load result carries something to draw. */
export function isLoaded(
  result: LoadedVolume | MissingVolume,
): result is LoadedVolume {
  return 'annotation' in result;
}

/**
 * Run `work` over `items`, at most `width` at a time, keeping input order.
 *
 * A city-year can be a dozen volumes and each annotation is a megabyte; firing
 * them all at once buries the one the viewer is waiting on behind the rest.
 */
export async function inParallel<T, R>(
  items: T[],
  width: number,
  work: (item: T) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.min(width, items.length) }, () =>
    (async () => {
      for (let index = next++; index < items.length; index = next++) {
        results[index] = await work(items[index] as T);
      }
    })(),
  );
  await Promise.all(runners);
  return results;
}
