import { mkdtemp, mkdir, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const readCachedS3Text = vi.fn();
const readS3Head = vi.fn();

vi.mock('./s3Objects.ts', async () => {
  const actual =
    await vi.importActual<typeof import('./s3Objects.ts')>('./s3Objects.ts');
  return {
    ...actual,
    readCachedS3Text: (...args: unknown[]) => readCachedS3Text(...args),
    readS3Head: (...args: unknown[]) => readS3Head(...args),
    ensureCached: vi.fn(),
  };
});

const { s3Annotation } = await import('./s3Routes.ts');

const URI =
  's3://mapsnap-sanborn/by-state/florida/1950/sanborn01309_018/runs/corpus-v1/mapsnap.iiif.json';
const PREFIX = 'by-state/florida/1950/sanborn01309_018';

/** An annotation page naming one sheet and one of its split panels. */
function annotationJson(): string {
  const item = (suffix: string) => ({
    id: `https://example.com/p19${suffix}/georef`,
    type: 'Annotation',
    label: `Page 19${suffix}`,
    motivation: 'georeferencing',
    target: {
      type: 'SpecificResource',
      source: {
        id: 'https://tile.loc.gov/image-services/iiif/service:gmd:x:01309_01_1950-0019/info.json',
        type: 'ImageService2',
        width: 6508,
        height: 7680,
      },
      selector: {
        type: 'SvgSelector',
        value: '<svg><polygon points="0,7680 0,0 6508,0 6508,7680" /></svg>',
      },
    },
    body: {
      type: 'FeatureCollection',
      transformation: { type: 'helmert' },
      features: [
        {
          type: 'Feature',
          properties: { resourceCoords: [100, 200] },
          geometry: { type: 'Point', coordinates: [-80.2, 25.78] },
        },
        {
          type: 'Feature',
          properties: { resourceCoords: [900, 800] },
          geometry: { type: 'Point', coordinates: [-80.19, 25.79] },
        },
      ],
    },
  });
  return JSON.stringify({
    id: 'https://example.com/generated',
    type: 'AnnotationPage',
    items: [item(''), item('__1')],
  });
}

/** metadata.json as the mirror writes it: every sheet with its scaled size. */
function metadataJson(): string {
  return JSON.stringify({
    item: 'sanborn01309_018',
    scale_percent: 25,
    sheets: [
      { seq: 25, key: 'p19', width: 1627, height: 1920 },
      { seq: 26, key: 'p20', width: 1629, height: 1920 },
    ],
  });
}

describe('s3Annotation', () => {
  let cacheRoot: string;

  beforeEach(async () => {
    cacheRoot = await mkdtemp(join(tmpdir(), 'mapsnap-s3-test-'));
    readCachedS3Text.mockReset();
    readS3Head.mockReset();
    readS3Head.mockRejectedValue(new Error('no ranged read expected'));
  });

  it('takes page sizes from metadata.json without reading any scan', async () => {
    readCachedS3Text.mockImplementation(
      async (_root: string, { key }: { key: string }) =>
        key.endsWith('metadata.json') ? metadataJson() : annotationJson(),
    );

    const { annotation } = await s3Annotation(
      URI,
      'http://localhost:5173',
      cacheRoot,
    );

    expect(readS3Head).not.toHaveBeenCalled();
    // One object for the whole volume: the annotation, then the metadata.
    expect(readCachedS3Text).toHaveBeenCalledTimes(2);
    const service = annotation.items[0]?.target?.source;
    expect(service?.width).toBe(1627);
    expect(service?.height).toBe(1920);
    expect(service?.id).toContain(`/s3-iiif/mapsnap-sanborn/${PREFIX}/p19`);
  });

  it('measures a cached scan rather than fetching it again', async () => {
    // metadata.json without this sheet: the page has to be measured.
    readCachedS3Text.mockImplementation(
      async (_root: string, { key }: { key: string }) =>
        key.endsWith('metadata.json')
          ? JSON.stringify({ sheets: [{ key: 'p20', width: 1, height: 1 }] })
          : annotationJson(),
    );
    // A 20x10 JPEG, already in the cache.
    const jpeg = Buffer.from([
      0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x0a, 0x00, 0x14, 0x03,
    ]);
    await mkdir(join(cacheRoot, 'mapsnap-sanborn', PREFIX), {
      recursive: true,
    });
    await writeFile(
      join(cacheRoot, 'mapsnap-sanborn', PREFIX, 'p19.jpg'),
      jpeg,
    );

    const { annotation } = await s3Annotation(
      URI,
      'http://localhost:5173',
      cacheRoot,
    );

    expect(readS3Head).not.toHaveBeenCalled();
    expect(annotation.items[0]?.target?.source?.width).toBe(20);
    expect(annotation.items[0]?.target?.source?.height).toBe(10);
  });

  it('falls back to a ranged read when neither knows the size', async () => {
    readCachedS3Text.mockImplementation(
      async (_root: string, { key }: { key: string }) =>
        key.endsWith('metadata.json') ? '{}' : annotationJson(),
    );
    readS3Head.mockResolvedValue(
      Buffer.from([
        0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x0a, 0x00, 0x14, 0x03,
      ]),
    );

    const { annotation } = await s3Annotation(
      URI,
      'http://localhost:5173',
      cacheRoot,
    );

    expect(readS3Head).toHaveBeenCalledTimes(1);
    expect(annotation.items[0]?.target?.source?.width).toBe(20);
  });

  it('survives an item whose metadata.json is missing', async () => {
    readCachedS3Text.mockImplementation(
      async (_root: string, { key }: { key: string }) => {
        if (key.endsWith('metadata.json')) throw new Error('NoSuchKey');
        return annotationJson();
      },
    );
    readS3Head.mockRejectedValue(new Error('NoSuchKey'));

    const { annotation, skipped } = await s3Annotation(
      URI,
      'http://localhost:5173',
      cacheRoot,
    );

    // Nothing measurable: the rewrite reports the pages rather than throwing.
    expect(annotation.items).toEqual([]);
    expect(skipped.length).toBeGreaterThan(0);
  });
});
