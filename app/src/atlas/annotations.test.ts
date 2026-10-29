import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  annotationUrl,
  inParallel,
  loadVolume,
  pageReport,
  stemFromLabel,
  withPageMetadata,
} from './annotations.ts';
import type { Volume } from './places.ts';
import type { GeorefAnnotationPage } from '../../server/iiifAnnotations.ts';

describe('stemFromLabel', () => {
  it('takes the page off a published label', () => {
    expect(
      stemFromLabel('Chicago, Illinois | 1950 | sanborn01790_085 p100W'),
    ).toBe('p100W');
  });

  it('folds a split panel back into its stem', () => {
    // A published split labels its panel in brackets; the on-disk stem, which
    // is what the geometry code matches on, joins them with __.
    expect(
      stemFromLabel('Saint Louis, Missouri | 1916 | sanborn04858_015 p66 [1]'),
    ).toBe('p66__1');
  });

  it('is null for a label that names no page', () => {
    expect(stemFromLabel('Chicago, Illinois | 1950 | key map')).toBeNull();
    expect(stemFromLabel(undefined)).toBeNull();
    expect(stemFromLabel('')).toBeNull();
  });
});

describe('withPageMetadata', () => {
  const page = (label: string, metadata?: { label: string; value: string }[]) =>
    ({
      items: [{ label, ...(metadata ? { metadata } : {}) }],
    }) as unknown as GeorefAnnotationPage;

  it('adds the page entry a published annotation does not carry', () => {
    const out = withPageMetadata(page('X | 1950 | sanborn01_001 p12'));
    expect(out.items?.[0]?.metadata).toContainEqual({
      label: 'page',
      value: 'p12',
    });
  });

  it('keeps entries the annotation already has', () => {
    const out = withPageMetadata(
      page('X | 1950 | sanborn01_001 p12', [{ label: 'streets', value: '4' }]),
    );
    expect(out.items?.[0]?.metadata).toHaveLength(2);
  });

  it('leaves an annotation that already names its pages alone', () => {
    const existing = [{ label: 'page', value: 'p9' }];
    const out = withPageMetadata(page('whatever p12', existing));
    expect(out.items?.[0]?.metadata).toEqual(existing);
  });

  it('does not mutate its argument', () => {
    // The same object is handed to Allmaps; editing it in place would make the
    // annotation depend on whether the panel had read it first.
    const input = page('X | 1950 | sanborn01_001 p12');
    withPageMetadata(input);
    expect(input.items?.[0]?.metadata).toBeUndefined();
  });
});

describe('inParallel', () => {
  it('keeps input order however the work finishes', async () => {
    const delays = [30, 1, 20, 2, 10];
    const out = await inParallel(delays, 2, async (ms) => {
      await new Promise((resolve) => setTimeout(resolve, ms));
      return ms;
    });
    expect(out).toEqual(delays);
  });

  it('runs no more than the given width at once', async () => {
    let live = 0;
    let peak = 0;
    await inParallel([...Array(12).keys()], 3, async () => {
      peak = Math.max(peak, ++live);
      await new Promise((resolve) => setTimeout(resolve, 1));
      live--;
    });
    expect(peak).toBeLessThanOrEqual(3);
  });

  it('has nothing to do for an empty list', async () => {
    expect(await inParallel([], 4, async () => 1)).toEqual([]);
  });
});

const RUN = { version: 'v2.0', url: 'https://data.mapsnap.org/runs/v2.0' };

// Abbeville 1907 p1, as the v2.0 run publishes it under iiif/chronoscope/.
const abbeville = (): GeorefAnnotationPage =>
  ({
    type: 'AnnotationPage',
    metadata: [
      { label: 'run', value: 'corpus-v2' },
      { label: 'pages', value: '6' },
    ],
    items: [
      {
        type: 'Annotation',
        label: 'Abbeville, Alabama | 1907 | sanborn00001_001 p1',
        metadata: [
          { label: 'streets', value: '3' },
          { label: 'intersections', value: '1' },
          { label: 'confidence', value: '0.94' },
          { label: 'confidence_model', value: 'corpus-v2' },
        ],
        target: {
          source: {
            id: 'https://cdn.chronoscope.io/mapsnap/service:gmd:gmd397m:g3974m:g3974am:g3974am_g000011907:00001_1907-0001',
            type: 'ImageService3',
            width: 1613,
            height: 1913,
          },
          selector: {
            type: 'SvgSelector',
            value:
              '<svg><polygon points="0,0 1613,0 1613,1913 0,1913" /></svg>',
          },
        },
        body: {
          type: 'FeatureCollection',
          transformation: { type: 'polynomial', options: { order: 1 } },
          features: [
            {
              type: 'Feature',
              properties: { resourceCoords: [0, 0] },
              geometry: { type: 'Point', coordinates: [-85.26, 31.58] },
            },
            {
              type: 'Feature',
              properties: { resourceCoords: [1613, 0] },
              geometry: { type: 'Point', coordinates: [-85.25, 31.58] },
            },
            {
              type: 'Feature',
              properties: { resourceCoords: [0, 1913] },
              geometry: { type: 'Point', coordinates: [-85.26, 31.57] },
            },
          ],
        },
      },
      {
        type: 'Annotation',
        label: 'Abbeville, Alabama | 1907 | sanborn00001_001 p2 [1]',
        metadata: [
          { label: 'confidence', value: '0.71' },
          { label: 'confidence_provisional', value: 'true' },
        ],
      },
    ],
  }) as unknown as GeorefAnnotationPage;

describe('annotationUrl', () => {
  it("is the volume's main file under the source's directory", () => {
    expect(annotationUrl(RUN, 'sanborn00001_001', 'chronoscope')).toBe(
      'https://data.mapsnap.org/runs/v2.0/iiif/chronoscope/sanborn00001_001.main.iiif.json',
    );
    expect(annotationUrl(RUN, 'sanborn00001_001', 'loc')).toBe(
      'https://data.mapsnap.org/runs/v2.0/iiif/loc/sanborn00001_001.main.iiif.json',
    );
  });
});

describe('pageReport', () => {
  it("reads a page's P(good) and match counts", () => {
    expect(pageReport(abbeville(), 0)).toEqual({
      confidence: 0.94,
      provisional: false,
      model: 'corpus-v2',
      streets: 3,
      intersections: 1,
      run: 'corpus-v2',
    });
  });

  it('marks a panel provisional and leaves missing fields null', () => {
    const report = pageReport(abbeville(), 1);
    expect(report.confidence).toBe(0.71);
    expect(report.provisional).toBe(true);
    expect(report.streets).toBeNull();
  });
});

describe('loadVolume', () => {
  const volume = {
    item: 'sanborn00001_001',
    state: 'alabama',
    mirrorYear: '1907',
    sheets: 2,
  } as Volume;

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // A fetch that serves the published annotation, recording what was asked.
  function stubFetch(status = 200) {
    const requested: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => {
      requested.push(url);
      return status === 200
        ? Response.json(abbeville())
        : new Response('', { status });
    });
    return requested;
  }

  it('reads the published file for the source, and names its pages', async () => {
    const requested = stubFetch();
    const result = await loadVolume(volume, RUN, 'chronoscope');
    expect(requested).toEqual([
      annotationUrl(RUN, 'sanborn00001_001', 'chronoscope'),
    ]);
    expect('pages' in result && result.pages.map((p) => p.stem)).toEqual([
      'p1',
    ]);
    expect('totalImages' in result && result.totalImages).toBe(6);
  });

  it('says nothing was placed when the run published no file', async () => {
    stubFetch(404);
    expect(await loadVolume(volume, RUN, 'loc')).toEqual({
      volume,
      reason: 'nothing placed',
    });
  });

  it('fetches nothing for a volume never digitized', async () => {
    const requested = stubFetch();
    const paper = { item: 'sanborn00001_009' } as Volume;
    expect(await loadVolume(paper, RUN, 'chronoscope')).toEqual({
      volume: paper,
      reason: 'not digitized',
    });
    expect(requested).toEqual([]);
  });
});
