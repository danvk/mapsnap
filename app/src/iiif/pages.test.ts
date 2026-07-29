import { describe, expect, it } from 'vitest';
import type { GeorefAnnotationPage } from '../../server/iiifAnnotations';
import {
  bearingDegrees,
  missingTruthPages,
  pagesFromAnnotation,
  svgPolygonPoints,
  type PageGeo,
} from './pages';

const METERS_PER_DEGREE_LAT = 110574;
const METERS_PER_DEGREE_LON_AT_EQUATOR = 111320;

const LON0 = -74.0;
const LAT0 = 40.65;

// Offset a base point by meters using the same local approximation as pages.ts.
function offset(eastMeters: number, northMeters: number): [number, number] {
  const metersPerLon =
    Math.cos((LAT0 * Math.PI) / 180) * METERS_PER_DEGREE_LON_AT_EQUATOR;
  return [
    LON0 + eastMeters / metersPerLon,
    LAT0 + northMeters / METERS_PER_DEGREE_LAT,
  ];
}

function gcpFeature(x: number, y: number, lonLat: [number, number]) {
  return {
    type: 'Feature',
    properties: { resourceCoords: [x, y], type: 'gcp' },
    geometry: { type: 'Point', coordinates: lonLat },
  };
}

// A one-item AnnotationPage for a width×height page with the given GCPs.
function annotationWith(
  features: unknown[],
  width = 1000,
  height = 2000,
  selector?: string,
): GeorefAnnotationPage {
  return {
    type: 'AnnotationPage',
    items: [
      {
        type: 'Annotation',
        metadata: [{ label: 'page', value: 'p7' }],
        target: {
          type: 'SpecificResource',
          source: { id: 'x/p7.jpg', type: 'ImageService3', width, height },
          ...(selector
            ? { selector: { type: 'SvgSelector', value: selector } }
            : {}),
        },
        body: {
          type: 'FeatureCollection',
          features: features as never[],
        },
      },
    ],
  };
}

describe('svgPolygonPoints', () => {
  it('parses the vertex list', () => {
    expect(
      svgPolygonPoints('<svg><polygon points="0,10 5.5,0 10,10" /></svg>'),
    ).toEqual([
      [0, 10],
      [5.5, 0],
      [10, 10],
    ]);
  });

  it('returns [] when there is no points attribute', () => {
    expect(svgPolygonPoints('<svg />')).toEqual([]);
  });
});

describe('bearingDegrees', () => {
  it('is 0 north, 90 east, -90 west', () => {
    expect(bearingDegrees([LON0, LAT0], offset(0, 100))).toBeCloseTo(0, 5);
    expect(bearingDegrees([LON0, LAT0], offset(100, 0))).toBeCloseTo(90, 5);
    expect(bearingDegrees([LON0, LAT0], offset(-100, 0))).toBeCloseTo(-90, 5);
  });
});

describe('pagesFromAnnotation', () => {
  it('fits an unrotated helmert page from two GCPs', () => {
    // 1000×2000 page, 2 meters per pixel, top edge pointing east.
    const annotation = annotationWith([
      gcpFeature(0, 0, offset(0, 0)),
      gcpFeature(1000, 0, offset(2000, 0)),
    ]);
    annotation.items[0]!.body!.transformation = { type: 'helmert' };
    const pages = pagesFromAnnotation(annotation);
    expect(pages).toHaveLength(1);
    const page = pages[0]!;
    expect(page.pageKey).toBe('p7');
    expect(page.itemIndex).toBe(0);
    expect(page.gcps).toHaveLength(2);
    expect(page.gcps[0]).toEqual({
      x: 0,
      y: 0,
      lon: LON0,
      lat: LAT0,
      type: 'gcp',
    });
    expect(page.transformationType).toBe('helmert');
    const [nw, ne, se, sw] = page.corners;
    expect(nw[0]).toBeCloseTo(LON0, 6);
    expect(nw[1]).toBeCloseTo(LAT0, 6);
    expect(ne[0]).toBeCloseTo(offset(2000, 0)[0], 5);
    expect(se[1]).toBeCloseTo(offset(0, -4000)[1], 5);
    expect(sw[0]).toBeCloseTo(LON0, 5);
    expect(page.rotationDegrees).toBeCloseTo(0, 3);
    // 2 m/px → 0.3048/2 pixels per foot.
    expect(page.scalePixelsPerFoot).toBeCloseTo(0.1524, 4);
  });

  it('reports clockwise rotation for a rotated helmert page', () => {
    // Top edge pointing south = page rotated 90° clockwise. The GCPs sit
    // symmetric about LAT0 so the fit's reference latitude matches offset()'s.
    const annotation = annotationWith([
      gcpFeature(0, 0, offset(0, 1000)),
      gcpFeature(1000, 0, offset(0, -1000)),
    ]);
    const page = pagesFromAnnotation(annotation)[0]!;
    expect(page.rotationDegrees).toBeCloseTo(90, 2);
    // Image y-down now points west.
    const sw = page.corners[3];
    expect(sw[0]).toBeCloseTo(offset(-4000, 1000)[0], 5);
    expect(sw[1]).toBeCloseTo(offset(-4000, 1000)[1], 5);
    expect(page.scalePixelsPerFoot).toBeCloseTo(0.1524, 4);
  });

  it('fits an affine page from corner GCPs and projects the clip polygon', () => {
    // 1000×2000 page, 1 meter per pixel, axis-aligned; clip = right half.
    const annotation = annotationWith(
      [
        gcpFeature(0, 0, offset(0, 0)),
        gcpFeature(1000, 0, offset(1000, 0)),
        gcpFeature(1000, 2000, offset(1000, -2000)),
        gcpFeature(0, 2000, offset(0, -2000)),
      ],
      1000,
      2000,
      '<svg><polygon points="500,0 1000,0 1000,2000 500,2000 500,0" /></svg>',
    );
    const page = pagesFromAnnotation(annotation)[0]!;
    expect(page.rotationDegrees).toBeCloseTo(0, 3);
    expect(page.scalePixelsPerFoot).toBeCloseTo(0.3048, 4);
    expect(page.gcps).toHaveLength(4);
    expect(page.transformationType).toBe('polynomial');
    expect(page.rectRing).toHaveLength(5);
    expect(page.rectRing[0]).toEqual(page.rectRing[4]);
    expect(page.clipRing).toHaveLength(5);
    // First clip vertex is the top-edge midpoint, 500 m east of the NW corner.
    expect(page.clipRing[0]![0]).toBeCloseTo(offset(500, 0)[0], 5);
    expect(page.clipRing[0]![1]).toBeCloseTo(LAT0, 5);
  });

  it('reads the split index and file stem from a generated annotation id', () => {
    const annotation = annotationWith([
      gcpFeature(0, 0, offset(0, 0)),
      gcpFeature(1000, 0, offset(2000, 0)),
    ]);
    annotation.items[0]!.id = 'https://oim/…656_14_1949-1499M__2/georef';
    const page = pagesFromAnnotation(annotation)[0]!;
    expect(page.splitIndex).toBe(2);
    expect(page.stem).toBe('p7__2');
  });

  it('leaves a whole page with no split index (stem === pageKey)', () => {
    const annotation = annotationWith([
      gcpFeature(0, 0, offset(0, 0)),
      gcpFeature(1000, 0, offset(2000, 0)),
    ]);
    annotation.items[0]!.id = 'https://oim/…656_14_1949-1499L/georef';
    // A generated whole page can carry a stray "[3]" label copied from the truth; ignore it.
    annotation.items[0]!.label = 'Los Angeles p1499L [3]';
    const page = pagesFromAnnotation(annotation)[0]!;
    expect(page.splitIndex).toBeNull();
    expect(page.stem).toBe('p7');
  });

  it('reads a truth split index from a trailing "[N]" label', () => {
    const annotation = annotationWith([
      gcpFeature(0, 0, offset(0, 0)),
      gcpFeature(1000, 0, offset(2000, 0)),
    ]);
    // Truth items have a plain resource id (no "/georef") and the split number in the label.
    annotation.items[0]!.id =
      'https://oldinsurancemaps.net/iiif/resource/57213/';
    annotation.items[0]!.label =
      'Los Angeles, Calif. | 1949 | Vol. 14 p1404 [2]';
    const page = pagesFromAnnotation(annotation)[0]!;
    expect(page.splitIndex).toBe(2);
    expect(page.stem).toBe('p7__2');
  });

  it('skips items without page metadata or enough GCPs', () => {
    const noMetadata = annotationWith([
      gcpFeature(0, 0, offset(0, 0)),
      gcpFeature(1000, 0, offset(2000, 0)),
    ]);
    noMetadata.items[0]!.metadata = [];
    expect(pagesFromAnnotation(noMetadata)).toHaveLength(0);

    const oneGcp = annotationWith([gcpFeature(0, 0, offset(0, 0))]);
    expect(pagesFromAnnotation(oneGcp)).toHaveLength(0);
  });
});

// A stand-in PageGeo carrying the fields missingTruthPages reads.
function pageGeo(stem: string, itemIndex: number): PageGeo {
  return {
    stem,
    pageKey: stem.split('__')[0]!,
    itemIndex,
  } as PageGeo;
}

describe('missingTruthPages', () => {
  it('selects the truth pages the comparison marked as unplaced', () => {
    const truth = [pageGeo('p1', 0), pageGeo('p2', 1), pageGeo('p3', 2)];
    const missing = missingTruthPages(truth, ['p3']);
    expect(missing.map((p) => p.stem)).toEqual(['p3']);
    // Negative id keeps it out of the fit pages' itemIndex space.
    expect(missing[0]!.itemIndex).toBe(-1);
  });

  it('lists each unplaced panel of a sheet under its own stem', () => {
    // Hudson: both p92 panels are unplaced. The old parent-key rule collapsed them
    // into one row called "p92", hiding p92__1 -- the largest single miss there is.
    const truth = [
      pageGeo('p92__1', 5),
      pageGeo('p92__2', 6),
      pageGeo('p91', 7),
    ];
    const missing = missingTruthPages(truth, ['p92__1', 'p92__2']);
    expect(missing.map((p) => p.stem)).toEqual(['p92__1', 'p92__2']);
    expect(missing.map((p) => p.itemIndex)).toEqual([-1, -2]);
  });

  it('keeps a panel whose sibling was placed', () => {
    // Champaign: p4__1 is placed and p4__2 is not. Keying on the parent page counted
    // the whole sheet as present, so the volume showed no misses against four reported.
    const truth = [pageGeo('p4__1', 0), pageGeo('p4__2', 1)];
    expect(missingTruthPages(truth, ['p4__2']).map((p) => p.stem)).toEqual([
      'p4__2',
    ]);
  });

  it('matches keys case-insensitively', () => {
    // Lettered pages are p1499J in the annotations and p1499j on disk.
    expect(
      missingTruthPages([pageGeo('p1499j', 0)], ['p1499J']).map((p) => p.stem),
    ).toEqual(['p1499j']);
  });

  it('returns nothing when the comparison reports no misses', () => {
    expect(missingTruthPages([pageGeo('p1', 0)], [])).toEqual([]);
  });
});

describe('missingTruthPages resilience', () => {
  it('reports nothing when the response carries no missing field', () => {
    // An older server, or a compare sidecar predating the field: no misses beats a
    // crashed page.
    expect(missingTruthPages([pageGeo('p1', 0)], undefined)).toEqual([]);
  });
});
