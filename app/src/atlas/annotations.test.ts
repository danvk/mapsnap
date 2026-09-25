import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  CDN_BASE,
  cdnImageSize,
  cdnServiceUrl,
  inParallel,
  loadVolume,
  rewriteForCdn,
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

// Champaign 1915 p10, as a corpus run publishes it: LoC's full-resolution
// frame, 6450 x 7650, which the CDN serves at 1613 x 1913.
const CHAMPAIGN_SERVICE =
  'service:gmd:gmd410m:g4104m:g4104cm:g017781915:01778_1915-0010';
const champaign = (): GeorefAnnotationPage =>
  ({
    type: 'AnnotationPage',
    items: [
      {
        type: 'Annotation',
        label: 'Champaign, Illinois | 1915 | sanborn01778_006 p10',
        target: {
          source: {
            id: `https://tile.loc.gov/image-services/iiif/${CHAMPAIGN_SERVICE}/info.json`,
            type: 'ImageService2',
            width: 6450,
            height: 7650,
          },
          selector: {
            type: 'SvgSelector',
            value:
              '<svg><polygon points="0,0 6450,0 6450,7650 130,550.8" /></svg>',
          },
        },
        body: {
          features: [
            {
              type: 'Feature',
              properties: { resourceCoords: [6450, 7650] },
              geometry: null,
            },
            {
              type: 'Feature',
              properties: { resourceCoords: [1000, 2000] },
              geometry: null,
            },
          ],
        },
      },
    ],
  }) as unknown as GeorefAnnotationPage;

describe('cdnServiceUrl', () => {
  it("keeps LoC's service id as the CDN's directory", () => {
    expect(
      cdnServiceUrl(
        `https://tile.loc.gov/image-services/iiif/${CHAMPAIGN_SERVICE}/info.json`,
      ),
    ).toBe(`${CDN_BASE}/${CHAMPAIGN_SERVICE}`);
    expect(
      cdnServiceUrl(
        `https://tile.loc.gov/image-services/iiif/${CHAMPAIGN_SERVICE}`,
      ),
    ).toBe(`${CDN_BASE}/${CHAMPAIGN_SERVICE}`);
  });

  it('is null for anything but a loc.gov image service', () => {
    expect(cdnServiceUrl('http://localhost:8182/iiif/vol/p10.jpg')).toBeNull();
    expect(cdnServiceUrl(undefined)).toBeNull();
  });
});

describe('cdnImageSize', () => {
  it('rounds a quarter up, as the CDN does', () => {
    // Rounding to nearest would give 1612 x 1912 (half to even) or 1613 x 1913
    // (half up) depending on the language; the CDN's info.json says 1613.
    expect(cdnImageSize({ width: 6450, height: 7650 })).toEqual({
      width: 1613,
      height: 1913,
    });
    expect(cdnImageSize({ width: 6452, height: 7652 })).toEqual({
      width: 1613,
      height: 1913,
    });
  });
});

describe('rewriteForCdn', () => {
  it("moves the page onto the CDN's image and frame", () => {
    const input = champaign();
    const item = rewriteForCdn(input).items[0]!;
    expect(item.target?.source).toEqual({
      id: `${CDN_BASE}/${CHAMPAIGN_SERVICE}`,
      type: 'ImageService3',
      width: 1613,
      height: 1913,
    });
    const coords = item.body?.features?.map((f) => f.properties.resourceCoords);
    // The far corner lands on the far corner, not a fraction of a pixel short.
    expect(coords?.[0]).toEqual([1613, 1913]);
    expect(coords?.[1]).toEqual([250.1, 500.1]);
    expect(item.target?.selector?.value).toBe(
      '<svg><polygon points="0,0 1613,0 1613,1913 32.5,137.7" /></svg>',
    );
    // The published annotation is untouched.
    expect(input.items[0]?.target?.source.width).toBe(6450);
  });

  it('leaves a page that is not on loc.gov alone', () => {
    const input = champaign();
    input.items[0]!.target!.source.id =
      'http://localhost:8182/iiif/vol/p10.jpg';
    expect(rewriteForCdn(input)).toEqual(input);
  });
});

describe('loadVolume', () => {
  const volume = { item: 'sanborn01778_006' } as Volume;
  const uri =
    's3://mapsnap-sanborn/by-state/illinois/1915/sanborn01778_006/runs/corpus-v1/mapsnap.iiif.json';

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // A fetch that serves the published annotation, recording what was asked.
  function stubFetch() {
    const requested: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => {
      requested.push(url);
      return Response.json(champaign());
    });
    return requested;
  }

  it('draws the published annotation from the CDN', async () => {
    const requested = stubFetch();
    const result = await loadVolume(volume, uri, 'cdn');
    expect(
      'annotation' in result && result.annotation.items[0]?.target?.source,
    ).toEqual({
      id: `${CDN_BASE}/${CHAMPAIGN_SERVICE}`,
      type: 'ImageService3',
      width: 1613,
      height: 1913,
    });
    // The annotation is the only request; the tiles are Allmaps' to fetch.
    expect(requested).toEqual([
      `/s3-api/object?uri=${encodeURIComponent(uri)}`,
    ]);
  });

  it('draws it verbatim from loc.gov', async () => {
    stubFetch();
    const result = await loadVolume(volume, uri, 'loc');
    expect(
      'annotation' in result && result.annotation.items[0]?.target?.source.id,
    ).toBe(
      `https://tile.loc.gov/image-services/iiif/${CHAMPAIGN_SERVICE}/info.json`,
    );
  });
});
