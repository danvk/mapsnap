import { describe, expect, it } from 'vitest';
import {
  imageStemsByLowercase,
  rescaleSvgSelector,
  rewriteAnnotationPage,
  labelToPageKey,
  splitIndexFor,
  withTiles,
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

  it('takes the generated split-panel variant from the id, not the label', () => {
    const label = 'Fargo, N.D. | 1958 p45 [2]';
    const base = 'https://tile.loc.gov/…:06536_1958-0045';
    expect(
      serviceUrlToPageKey(`${base}/info.json`, label, `${base}__1/georef`),
    ).toBe('p45__1');
    expect(
      serviceUrlToPageKey(`${base}/info.json`, label, `${base}__3/georef`),
    ).toBe('p45__3');
    // A generated whole page keeps no variant despite the stray label.
    expect(
      serviceUrlToPageKey(`${base}/info.json`, label, `${base}/georef`),
    ).toBe('p45');
  });

  it('takes the split-panel variant from the label, which no URL records', () => {
    // Without this every panel of a sheet collapses onto the parent key and
    // all but one annotation is lost (228 panels across the truth volumes).
    expect(
      serviceUrlToPageKey(
        'service:gmd:x:03376_01_1951-0428',
        'New Orleans, La. | 1951 | Vol. 5 p428 [2]',
      ),
    ).toBe('p428__2');
    expect(
      serviceUrlToPageKey(
        'service:gmd:x:sb001250',
        'New Orleans, La. | 1896 | Vol. 2 p125 [3]',
      ),
    ).toBe('p125__3');
    // A label with no bracket leaves the key unsuffixed.
    expect(
      serviceUrlToPageKey(
        'service:gmd:x:03376_01_1951-0428',
        'New Orleans, La. | 1951 | Vol. 5 p428',
      ),
    ).toBe('p428');
  });

  it('falls back to the label when the volume links no image service', () => {
    // Grand Rapids 1953 vol 7: 73 of 83 truth annotations carry source.id null.
    expect(
      serviceUrlToPageKey(null, 'Grand Rapids, Mich. | 1953 | Vol. 7 p844 [3]'),
    ).toBe('p844__3');
    expect(
      serviceUrlToPageKey(null, 'Grand Rapids, Mich. | 1953 | Vol. 7 p712'),
    ).toBe('p712');
    // Letter-only key-map sheets still resolve.
    expect(
      serviceUrlToPageKey(null, 'Los Angeles, Calif. | 1949 | Vol. 14 pa [2]'),
    ).toBe('pa__2');
    // No URL and no page identifier in the label is still null.
    expect(
      serviceUrlToPageKey(null, 'Grand Rapids, Mich. | 1953 | Vol. 7'),
    ).toBeNull();
  });
});

describe('splitIndexFor', () => {
  it('prefers the id, because generated labels are copied from truth', () => {
    // All three of fargo p45's generated panels are labelled "p45 [2]"; only
    // the id distinguishes them. Trusting the label pointed every panel at
    // p45__2.jpg and rendered the wrong sheet in the volume viewer.
    const label = 'Fargo, N.D. | 1958 p45 [2]';
    const base = 'https://tile.loc.gov/…:06536_1958-0045';
    expect(splitIndexFor(`${base}__1/georef`, label)).toBe(1);
    expect(splitIndexFor(`${base}__2/georef`, label)).toBe(2);
    expect(splitIndexFor(`${base}__3/georef`, label)).toBe(3);
  });

  it('ignores a stray label variant on a generated whole page', () => {
    expect(
      splitIndexFor(
        'https://tile.loc.gov/…:06536_1958-0045/georef',
        'Fargo, N.D. | 1958 p45 [2]',
      ),
    ).toBeNull();
  });

  it('falls back to the label for a truth item, whose id has no variant', () => {
    expect(
      splitIndexFor(
        'https://oldinsurancemaps.net/iiif/resource/54284/',
        'Grand Rapids, Mich. | 1953 | Vol. 7 p844 [3]',
      ),
    ).toBe(3);
    expect(splitIndexFor(undefined, 'Fargo, N.D. | 1958 p45')).toBeNull();
  });
});

describe('labelToPageKey', () => {
  it('reads the page id from the last pipe-separated segment', () => {
    expect(labelToPageKey('New Orleans, La. | 1896 | Vol. 2 p156')).toBe(
      'p156',
    );
    expect(labelToPageKey('New Orleans, La. | 1951 | Vol. 5 p428 [2]')).toBe(
      'p428__2',
    );
    expect(labelToPageKey('Chicago | 1950 | Vol. 1 p103W')).toBe('p103w');
    expect(labelToPageKey('Los Angeles, Calif. | 1949 | Vol. 14 pa [2]')).toBe(
      'pa__2',
    );
  });

  it('returns null when the label carries no page identifier', () => {
    expect(labelToPageKey('Grand Rapids, Mich. | 1953 | Vol. 7')).toBeNull();
    expect(labelToPageKey('')).toBeNull();
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

describe('rewriteAnnotationPage split panels', () => {
  // A split panel is georeferenced against its PARENT sheet: the source is the
  // parent's full-resolution size and the coords/selector are parent pixels.
  // Serving the panel image rescaled them by the panel's aspect and rendered
  // the sheet at ~2x scale, misaligned (fargo p45__1).
  const parentSource = {
    id: 'https://tile.loc.gov/…:06536_1958-0045/info.json',
    type: 'ImageService2',
    width: 6660,
    height: 8070,
  };
  function panel(index: number) {
    return {
      id: `https://tile.loc.gov/…:06536_1958-0045__${index}/georef`,
      label: 'Fargo, N.D. | 1958 p45 [2]',
      target: { source: { ...parentSource } },
      body: {
        features: [
          {
            type: 'Feature',
            properties: { resourceCoords: [3330, 4035], type: 'gcp' },
            geometry: { type: 'Point', coordinates: [-96.8, 46.88] },
          },
        ],
      },
    };
  }

  it('serves the parent image while keeping the panel identity', () => {
    const { annotation, skipped } = rewriteAnnotationPage(
      { items: [panel(1), panel(3)] } as never,
      new Map([['p45', { width: 1665, height: 2018 }]]),
      'http://localhost:8182/iiif/fargo_nd_1958',
      new Map([['p45', 'p45']]),
    );
    expect(skipped).toEqual([]);
    const keys = annotation.items.map(
      (item) =>
        item.metadata?.find((entry) => entry.label === 'page')?.value ?? '',
    );
    expect(keys).toEqual(['p45__1', 'p45__3']);
    for (const item of annotation.items) {
      // Parent image, and coords scaled by the parent ratio (0.25) on BOTH
      // axes -- the panel image would have given 1665/6660 vs 989/8070.
      expect(item.target?.source?.id).toBe(
        'http://localhost:8182/iiif/fargo_nd_1958/p45.jpg',
      );
      expect(item.target?.source?.width).toBe(1665);
      expect(item.body?.features?.[0]?.properties.resourceCoords).toEqual([
        832.5, 1009,
      ]);
    }
  });

  it('keeps a panel whose own image was never split out locally', () => {
    // grand_rapids p729__2 has truth but no p729__2.jpg; the parent exists, and
    // the parent is what the annotation is drawn against anyway.
    const { annotation, skipped } = rewriteAnnotationPage(
      {
        items: [
          {
            id: 'https://oldinsurancemaps.net/iiif/resource/54284/',
            label: 'Grand Rapids, Mich. | 1953 | Vol. 7 p729 [2]',
            target: { source: { ...parentSource, id: null } },
            body: { features: [] },
          },
        ],
      } as never,
      new Map([['p729', { width: 1665, height: 2018 }]]),
      'http://localhost:8182/iiif/grand_rapids_mi_1953_vol7',
      new Map([['p729', 'p729']]),
    );
    expect(skipped).toEqual([]);
    expect(
      annotation.items[0]?.metadata?.find((e) => e.label === 'page')?.value,
    ).toBe('p729__2');
  });
});

describe('imageStemsByLowercase', () => {
  it('maps a lowercased stem to the real filename case', async () => {
    const { mkdtemp, writeFile } = await import('fs/promises');
    const { tmpdir } = await import('os');
    const { join } = await import('path');
    const dir = await mkdtemp(join(tmpdir(), 'stems-'));
    // The two conventions that coexist in the corpus: Chicago writes p103w.jpg
    // for a URL ending 0103W, Asheville writes p33A.jpg for one ending 0033A.
    await writeFile(join(dir, 'p103w.jpg'), '');
    await writeFile(join(dir, 'p33A.jpg'), '');
    await writeFile(join(dir, 'p7.jpg'), '');
    const stems = await imageStemsByLowercase(dir);
    expect(stems.get('p103w')).toBe('p103w');
    expect(stems.get('p33a')).toBe('p33A');
    expect(stems.get('p7')).toBe('p7');
  });

  it('returns an empty map for a missing directory', async () => {
    expect((await imageStemsByLowercase('/nonexistent-volume')).size).toBe(0);
  });
});

describe('withTiles', () => {
  it('always yields at least one scale factor, however small the image', () => {
    // The bug this exists for: @allmaps/iiif-parser's fallback tileset uses
    // Array.from({length: maxExponent}), which is EMPTY at maxExponent 0, so
    // any image under one tile wide renders zero zoom levels and throws.
    // grand_rapids p844__3 (682x568) and fargo p62__4 (365x488) are real cases.
    const small = withTiles({ width: 682, height: 568 }) as {
      tiles: { width: number; scaleFactors: number[] }[];
    };
    expect(small.tiles).toEqual([{ width: 512, scaleFactors: [1, 2] }]);
    const tiny = withTiles({ width: 100, height: 80 }) as {
      tiles: { width: number; scaleFactors: number[] }[];
    };
    expect(tiny.tiles[0]!.scaleFactors).toEqual([1]);
  });

  it('covers a full-size sheet down to a single tile', () => {
    const sheet = withTiles({ width: 6660, height: 8070 }) as {
      tiles: { width: number; scaleFactors: number[] }[];
    };
    // 8070 / 512 = 15.8 -> the coarsest factor must reduce it under one tile.
    const coarsest = sheet.tiles[0]!.scaleFactors.at(-1)!;
    expect(8070 / coarsest).toBeLessThanOrEqual(512);
    expect(sheet.tiles[0]!.scaleFactors[0]).toBe(1);
  });

  it('leaves a body that already advertises tiles alone', () => {
    const existing = { width: 100, height: 100, tiles: [{ width: 256 }] };
    expect(withTiles(existing)).toBe(existing);
  });

  it('passes through a body with no usable dimensions', () => {
    const noDims = { type: 'ImageService3' };
    expect(withTiles(noDims)).toBe(noDims);
  });
});
