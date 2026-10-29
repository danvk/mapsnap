/**
 * Reading a published run's annotations as the atlas needs them.
 *
 * A published run (scripts/atlas/publish_run.py) holds every volume's
 * annotation twice: under iiif/loc/, pointing at loc.gov's image services, and
 * under iiif/chronoscope/, pointing at the Chronoscope CDN's quarter-size
 * copies, with control points and clip polygons already rescaled. The CDN is
 * the default. A town-year is hundreds of sheets and thousands of tiles, and
 * tile.loc.gov rate-limits well below that: opening Chicago 1951 (7 volumes,
 * 719 sheets) drew 4,096 failed requests and left the host answering 429 to
 * everything for minutes afterwards. The browser reports those as CORS
 * failures, since a 429 error page carries no CORS headers. loc.gov remains
 * selectable: it is where the scans come from, and fine for one volume.
 */

import { pagesFromAnnotation, type PageGeo } from '../iiif/pages';
import type { GeorefAnnotationPage } from '../../server/iiifAnnotations';
import type { RunInfo } from './footprints';
import type { Volume } from './places';

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

/**
 * A volume that has no annotation to draw, and why.
 *
 * `nothing placed`: the run published nothing for it, having placed none of
 * its images (or dropped all it placed).
 */
export interface MissingVolume {
  volume: Volume;
  reason: 'not digitized' | 'nothing placed' | 'unreadable';
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

/** Where a volume's sheet images come from: the run's two copies of each annotation. */
export type ImageSource = 'chronoscope' | 'loc';

/** A volume's main annotation in a published run. */
export function annotationUrl(
  run: RunInfo,
  item: string,
  source: ImageSource,
): string {
  return `${run.url}/iiif/${source}/${item}.main.iiif.json`;
}

/**
 * Fetch one volume's annotation, or say why there is none.
 *
 * A 404 is ordinary, not an error worth throwing: a run publishes no file for
 * a volume it placed nothing of.
 */
export async function loadVolume(
  volume: Volume,
  run: RunInfo,
  source: ImageSource,
): Promise<LoadedVolume | MissingVolume> {
  if (!volume.state) return { volume, reason: 'not digitized' };
  try {
    const response = await fetch(annotationUrl(run, volume.item, source));
    if (!response.ok) return { volume, reason: 'nothing placed' };
    const annotation = withPageMetadata(
      (await response.json()) as GeorefAnnotationPage,
    );
    return {
      volume,
      annotation,
      pages: pagesFromAnnotation(annotation),
      totalImages: reportedCount(annotation, 'pages'),
    };
  } catch {
    return { volume, reason: 'unreadable' };
  }
}

/** What a run says about one placed image, from its annotation's metadata. */
export interface PageReport {
  /** P(good): the confidence model's estimate that the pose is within 25 ft. */
  confidence: number | null;
  /** Set for panels, which the model ranks less well than whole pages. */
  provisional: boolean;
  /** The confidence model's version, e.g. "corpus-v2". */
  model: string | null;
  /** Streets and intersections the fit matched. */
  streets: number | null;
  intersections: number | null;
  /** The corpus run that placed it ("corpus-v2", "corpus-v2.1"). */
  run: string | null;
}

/** An annotation item's report, read from its metadata entries. */
export function pageReport(
  annotation: GeorefAnnotationPage,
  itemIndex: number,
): PageReport {
  const entries = new Map(
    (annotation.items?.[itemIndex]?.metadata ?? []).map((entry) => [
      entry.label,
      String(entry.value),
    ]),
  );
  const number = (label: string): number | null => {
    const value = entries.get(label);
    return value !== undefined && Number.isFinite(Number(value))
      ? Number(value)
      : null;
  };
  const pageRun = (
    annotation as { metadata?: { label?: string; value?: string }[] }
  ).metadata?.find((entry) => entry.label === 'run')?.value;
  return {
    confidence: number('confidence'),
    provisional: entries.get('confidence_provisional') === 'true',
    model: entries.get('confidence_model') ?? null,
    streets: number('streets'),
    intersections: number('intersections'),
    run: entries.get('run') ?? pageRun ?? null,
  };
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
