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
 * So the default is the chronoscope CDN, a static (level 0) IIIF service built
 * from the mirrored 25% scans: each page's service is repointed there, in the
 * browser, and the tiles never touch our own server. A volume the CDN does not
 * hold yet falls back to `mirror`, the same rewrite the debugger uses against
 * our own cache of those scans. `loc` remains selectable, and is the right
 * source for a single volume.
 */

import { pagesFromAnnotation, type PageGeo } from '../iiif/pages';
import {
  rescaleSvgSelector,
  type GeorefAnnotationPage,
} from '../../server/iiifAnnotations';
import type { Volume } from './places';

/** The static IIIF service for the mirrored scans, one directory per sheet. */
export const CDN_BASE = 'https://cdn.chronoscope.io/mapsnap';

/** A volume's annotation, and the page geometry derived from it. */
export interface LoadedVolume {
  volume: Volume;
  annotation: GeorefAnnotationPage;
  /**
   * The annotation's items, as geometry. One per PLACED image, which counts a
   * split panel separately from its siblings -- so this is neither the
   * volume's sheet count nor the number of sheets that got placed.
   */
  pages: PageGeo[];
  /**
   * Images the run decomposed the volume into, placed or not, from the
   * annotation's own report. Null for an annotation that carries no report.
   */
  totalImages: number | null;
  /**
   * Where this volume's sheets are actually drawn from, which is not always
   * the source asked for: a volume the CDN does not hold falls back to the
   * mirror.
   */
  source: ImageSource;
}

/**
 * An integer the annotation reports about itself, e.g. how many images the run
 * split the volume into.
 *
 * `fit` writes a report card into the annotation page's top-level metadata --
 * pages, placed, unplaced, fit sources. Reading `pages` from there beats
 * recomputing it: the app never sees the images a run declined to place, so
 * counting what it can see would always say everything was placed.
 */
export function reportedCount(
  annotation: GeorefAnnotationPage,
  label: string,
): number | null {
  const entry = (
    annotation as { metadata?: { label?: string; value?: string }[] }
  ).metadata?.find((m) => m.label === label);
  const value = Number(entry?.value);
  return Number.isFinite(value) ? value : null;
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
export type ImageSource = 'cdn' | 'mirror' | 'loc';

/**
 * Where to fetch a volume's annotation, for the chosen image source.
 *
 * `loc` takes the published annotation verbatim, so its pages resolve to
 * loc.gov; `cdn` takes it verbatim too, and rewrites it in the browser.
 * `mirror` asks the server to rewrite it against the cached scans, which also
 * fills in the `page` metadata the raw file lacks.
 */
export function annotationUrl(uri: string, source: ImageSource): string {
  return source === 'mirror'
    ? `/iiif-api/annotation?path=${encodeURIComponent(uri)}`
    : `/s3-api/object?uri=${encodeURIComponent(uri)}`;
}

/**
 * The CDN's service for a sheet, from its loc.gov service URL.
 *
 * The CDN keeps LoC's own service id as the directory name, e.g.
 * `.../iiif/service:gmd:gmd410m:...:01778_1915-0010` becomes
 * `${CDN_BASE}/service:gmd:gmd410m:...:01778_1915-0010`. Null for a source
 * that is not a loc.gov image service.
 */
export function cdnServiceUrl(
  locServiceUrl: string | undefined,
): string | null {
  const match = /\/iiif\/(service:[^/]+)/.exec(locServiceUrl ?? '');
  return match ? `${CDN_BASE}/${match[1]}` : null;
}

/**
 * The size of the CDN's image for a sheet LoC serves at width x height.
 *
 * The CDN's images are the mirror's 25% scans, which LoC renders at
 * ceil(dimension / 4) -- 6450 x 7650 is 1613 x 1913, where rounding would give
 * 1612 x 1912 and misplace every control point by a fraction of a pixel. Held
 * for all 86 sheets checked against the CDN's info.json, across 40 volumes.
 */
export function cdnImageSize(size: { width: number; height: number }): {
  width: number;
  height: number;
} {
  return {
    width: Math.ceil(size.width / 4),
    height: Math.ceil(size.height / 4),
  };
}

// Round to 1 decimal, as the server's rewrite does.
function round1(value: number): number {
  return Math.round(value * 10) / 10;
}

/**
 * Repoint every loc.gov page of an annotation at the CDN.
 *
 * The published annotation's control points and clipping polygon are in the
 * pixels of LoC's full-resolution image, so both are rescaled into the CDN
 * image's frame, per axis. An item whose source is not a loc.gov service is
 * left as it is. The argument is not mutated.
 */
export function rewriteForCdn(
  annotation: GeorefAnnotationPage,
): GeorefAnnotationPage {
  const result = structuredClone(annotation);
  for (const item of result.items ?? []) {
    const source = item.target?.source;
    const service = cdnServiceUrl(source?.id);
    if (
      !item.target ||
      !source ||
      !service ||
      !source.width ||
      !source.height
    ) {
      continue;
    }
    const size = cdnImageSize(source);
    const scale = {
      scaleX: size.width / source.width,
      scaleY: size.height / source.height,
    };
    item.target.source = { id: service, type: 'ImageService3', ...size };
    for (const feature of item.body?.features ?? []) {
      const coords = feature.properties?.resourceCoords;
      if (coords && coords.length >= 2) {
        feature.properties.resourceCoords = [
          round1((coords[0] ?? 0) * scale.scaleX),
          round1((coords[1] ?? 0) * scale.scaleY),
        ];
      }
    }
    const selector = item.target.selector;
    if (selector?.type === 'SvgSelector') {
      selector.value = rescaleSvgSelector(selector.value, scale, size);
    }
  }
  return result;
}

/**
 * Whether the CDN holds this volume's sheets, judged by its first page.
 *
 * The CDN is being filled a volume at a time, and so far a volume is either
 * all there or not there at all, so one info.json stands for the rest. It is
 * the request Allmaps would make first anyway, and the CDN lets the browser
 * cache it.
 */
async function cdnHoldsVolume(
  annotation: GeorefAnnotationPage,
): Promise<boolean> {
  const service = (annotation.items ?? [])
    .map((item) => cdnServiceUrl(item.target?.source?.id))
    .find((url) => url !== null);
  if (!service) return false;
  try {
    return (await fetch(`${service}/info.json`)).ok;
  } catch {
    return false;
  }
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
    if (source === 'cdn' && !(await cdnHoldsVolume(raw))) {
      return loadVolume(volume, uri, 'mirror');
    }
    const annotation = withPageMetadata(
      source === 'cdn' ? rewriteForCdn(raw) : raw,
    );
    return {
      volume,
      annotation,
      pages: pagesFromAnnotation(annotation),
      totalImages: reportedCount(annotation, 'pages'),
      source,
    };
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
