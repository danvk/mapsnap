import { describe, expect, it } from 'vitest';
import {
  rescaleSvgSelector,
  rewriteAnnotationPage,
  serviceUrlToPageKey,
  type GeorefAnnotationPage,
} from './iiifAnnotations';

describe('serviceUrlToPageKey', () => {
  it('extracts numeric page keys, stripping leading zeros', () => {
    expect(
      serviceUrlToPageKey(
        'https://tile.loc.gov/image-services/iiif/service:gmd:g3994nm:05791_06_1906-0011/info.json',
      ),
    ).toBe('p11');
    expect(serviceUrlToPageKey('service:gmd:x:03376_01_1951-0425')).toBe(
      'p425',
    );
  });

  it('lowercases directional letter suffixes', () => {
    expect(
      serviceUrlToPageKey('service:gmd:x:01790_01N_1950-0006N/info.json'),
    ).toBe('p6n');
    expect(serviceUrlToPageKey('service:gmd:x:01790_01N_1950-0103W')).toBe(
      'p103w',
    );
    expect(serviceUrlToPageKey('service:gmd:x:05791_02_1939-0027s')).toBe(
      'p27s',
    );
  });

  it('decodes the Sanborn sb-format', () => {
    expect(serviceUrlToPageKey('service:gmd:x:sb001250')).toBe('p125');
    expect(serviceUrlToPageKey('service:gmd:x:sb00154s')).toBe('p154s');
  });

  it('returns null for covers, indexes, and missing URLs', () => {
    expect(serviceUrlToPageKey('service:gmd:x:05791_06_1906-covr')).toBeNull();
    expect(serviceUrlToPageKey('service:gmd:x:05791_06_1906-titl')).toBeNull();
    expect(serviceUrlToPageKey(null)).toBeNull();
    expect(serviceUrlToPageKey(undefined)).toBeNull();
    expect(serviceUrlToPageKey('')).toBeNull();
  });
});

describe('rescaleSvgSelector', () => {
  it('rescales a full-page rectangle exactly', () => {
    const svg =
      '<svg><polygon points="0,7987 0,0 5484,0 5484,7987 0,7987" /></svg>';
    const bounds = { width: 2048, height: 2983 };
    expect(
      rescaleSvgSelector(
        svg,
        { scaleX: 2048 / 5484, scaleY: 2983 / 7987 },
        bounds,
      ),
    ).toBe(
      '<svg><polygon points="0,2983 0,0 2048,0 2048,2983 0,2983" /></svg>',
    );
  });

  it('clamps negative and out-of-bounds points', () => {
    const svg = '<svg><polygon points="-0.0,5 10,20.7" /></svg>';
    expect(
      rescaleSvgSelector(
        svg,
        { scaleX: 1, scaleY: 1 },
        { width: 8, height: 15 },
      ),
    ).toBe('<svg><polygon points="0,5 8,15" /></svg>');
  });
});

// A two-GCP helmert annotation for p11 (source 4096×6000), plus a cover page
// and a page whose local image is missing.
function fixturePage(): GeorefAnnotationPage {
  return {
    id: 'https://example.com/generated',
    type: 'AnnotationPage',
    label: 'Test volume',
    items: [
      {
        id: 'https://example.com/p11/georef',
        type: 'Annotation',
        label: 'Page 11',
        metadata: [{ label: 'streets', value: '7' }],
        motivation: 'georeferencing',
        target: {
          type: 'SpecificResource',
          source: {
            id: 'https://tile.loc.gov/image-services/iiif/service:gmd:x:05791_06_1906-0011/info.json',
            type: 'ImageService2',
            width: 4096,
            height: 6000,
          },
          selector: {
            type: 'SvgSelector',
            value:
              '<svg><polygon points="0,6000 0,0 4096,0 4096,6000 0,6000" /></svg>',
          },
        },
        body: {
          type: 'FeatureCollection',
          transformation: { type: 'helmert' },
          features: [
            {
              type: 'Feature',
              properties: { resourceCoords: [400, 800], type: 'gcp' },
              geometry: { type: 'Point', coordinates: [-74.02, 40.64] },
            },
            {
              type: 'Feature',
              properties: { resourceCoords: [2048, 3000], type: 'gcp' },
              geometry: { type: 'Point', coordinates: [-74.01, 40.65] },
            },
          ],
        },
      },
      {
        id: 'https://example.com/covr/georef',
        type: 'Annotation',
        label: 'Cover',
        target: {
          type: 'SpecificResource',
          source: {
            id: 'service:gmd:x:05791_06_1906-covr',
            type: 'ImageService2',
            width: 4096,
            height: 6000,
          },
        },
      },
      {
        id: 'https://example.com/p99/georef',
        type: 'Annotation',
        label: 'Page 99',
        target: {
          type: 'SpecificResource',
          source: {
            id: 'service:gmd:x:05791_06_1906-0099',
            type: 'ImageService2',
            width: 4096,
            height: 6000,
          },
        },
      },
    ],
  };
}

describe('rewriteAnnotationPage', () => {
  const localPages = new Map([['p11', { width: 1024, height: 1500 }]]);
  const baseUrl = 'http://localhost:8182/iiif/test_volume';

  it('rewrites the source and rescales coordinates into the local frame', () => {
    const { annotation, skipped } = rewriteAnnotationPage(
      fixturePage(),
      localPages,
      baseUrl,
    );
    expect(annotation.items).toHaveLength(1);
    const item = annotation.items[0]!;
    expect(item.target?.source).toEqual({
      id: 'http://localhost:8182/iiif/test_volume/p11.jpg',
      type: 'ImageService3',
      width: 1024,
      height: 1500,
    });
    // 4096×6000 → 1024×1500 is a uniform 4× reduction.
    expect(item.body?.features?.[0]?.properties.resourceCoords).toEqual([
      100, 200,
    ]);
    expect(item.body?.features?.[1]?.properties.resourceCoords).toEqual([
      512, 750,
    ]);
    expect(item.target?.selector?.value).toBe(
      '<svg><polygon points="0,1500 0,0 1024,0 1024,1500 0,1500" /></svg>',
    );
    expect(item.body?.transformation).toEqual({ type: 'helmert' });
    expect(item.metadata).toContainEqual({ label: 'page', value: 'p11' });
    expect(skipped).toEqual([
      { label: 'Cover', pageKey: null, reason: 'not-a-page' },
      { label: 'Page 99', pageKey: 'p99', reason: 'missing-image' },
    ]);
  });

  it('does not mutate its input', () => {
    const input = fixturePage();
    rewriteAnnotationPage(input, localPages, baseUrl);
    expect(input).toEqual(fixturePage());
  });
});
