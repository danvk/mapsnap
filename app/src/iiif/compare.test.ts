import { describe, expect, it } from 'vitest';

import {
  comparePage,
  compareToTruth,
  haversineFeet,
  sampleGrid,
} from './compare';
import type { PageGeo } from './pages';

// A page whose corner quad places (0,0)..(w,h) on a small lon/lat rectangle.
function page(
  overrides: Partial<PageGeo> & { pageKey: string; itemIndex: number },
): PageGeo {
  const lon0 = overrides.corners?.[0]?.[0] ?? -74.0;
  const lat0 = overrides.corners?.[0]?.[1] ?? 40.7;
  return {
    splitIndex: null,
    stem: overrides.pageKey,
    width: 1000,
    height: 1000,
    corners: [
      [lon0, lat0],
      [lon0 + 0.01, lat0],
      [lon0 + 0.01, lat0 - 0.01],
      [lon0, lat0 - 0.01],
    ],
    rectRing: [],
    clipRing: [],
    clipPolygon: [
      [0, 0],
      [1000, 0],
      [1000, 1000],
      [0, 1000],
    ],
    scalePixelsPerFoot: 1,
    rotationDegrees: 0,
    gcps: [],
    transformationType: 'polynomial',
    ...overrides,
  };
}

// Corner quad placing (0,0)..(1000,1000) on a 0.01°×0.01° box at (lon0, lat0).
function cornersAt(lon0: number, lat0: number): PageGeo['corners'] {
  return [
    [lon0, lat0],
    [lon0 + 0.01, lat0],
    [lon0 + 0.01, lat0 - 0.01],
    [lon0, lat0 - 0.01],
  ];
}

// A left/right half [x0,x1] clip polygon over the 1000×1000 frame.
function halfClip(x0: number, x1: number): [number, number][] {
  return [
    [x0, 0],
    [x1, 0],
    [x1, 1000],
    [x0, 1000],
  ];
}

describe('haversineFeet', () => {
  it('measures a degree of latitude as ~364,000 ft', () => {
    const feet = haversineFeet([-74, 40], [-74, 41]);
    expect(feet).toBeGreaterThan(360_000);
    expect(feet).toBeLessThan(368_000);
  });
});

describe('sampleGrid', () => {
  it('is the 7x7 grid over the full image', () => {
    const grid = sampleGrid(600, 1200);
    expect(grid).toHaveLength(49);
    expect(grid[0]).toEqual([0, 0]);
    expect(grid.at(-1)).toEqual([600, 1200]);
  });
});

describe('comparePage', () => {
  it('reports zero error for identical transforms', () => {
    const a = page({ pageKey: 'p1', itemIndex: 0 });
    const b = page({ pageKey: 'p1', itemIndex: 0 });
    const stats = comparePage(a, b);
    expect(stats.rmseFt).toBeCloseTo(0, 5);
    expect(stats.maxFt).toBeCloseTo(0, 5);
    expect(stats.translationFt).toBeCloseTo(0, 5);
  });

  it('reports a pure shift as equal rmse/max/translation', () => {
    const a = page({ pageKey: 'p1', itemIndex: 0 });
    // Shift ~one ten-thousandth of a degree of latitude south: ~36.4 ft.
    const shifted = page({
      pageKey: 'p1',
      itemIndex: 0,
      corners: a.corners.map(([lon, lat]) => [lon, lat - 0.0001]) as never,
    });
    const stats = comparePage(a, shifted);
    expect(stats.rmseFt).toBeGreaterThan(30);
    expect(stats.rmseFt).toBeLessThan(40);
    expect(stats.maxFt).toBeCloseTo(stats.rmseFt, 1);
    expect(stats.translationFt).toBeCloseTo(stats.rmseFt, 1);
  });

  it('reports scale and rotation differences', () => {
    const a = page({ pageKey: 'p1', itemIndex: 0, scalePixelsPerFoot: 1.0 });
    const b = page({
      pageKey: 'p1',
      itemIndex: 0,
      scalePixelsPerFoot: 1.05,
      rotationDegrees: 2,
    });
    const stats = comparePage(a, b);
    expect(stats.scaleErrorPercent).toBeCloseTo(5, 5);
    expect(stats.rotationErrorDegrees).toBeCloseTo(-2, 5);
  });
});

describe('compareToTruth', () => {
  it('pairs whole pages by page key and skips pages with no truth', () => {
    const pages = [
      page({ pageKey: 'p1', itemIndex: 0 }),
      page({ pageKey: 'p9', itemIndex: 2 }), // no truth
    ];
    const truth = [page({ pageKey: 'p1', itemIndex: 0 })];
    const stats = compareToTruth(pages, truth);
    expect(stats.get(0)?.rmseFt).toBeCloseTo(0, 5);
    expect(stats.has(2)).toBe(false);
  });

  it('compares a page we kept whole against the truth largest split', () => {
    // We kept p5 whole; truth split it into a large left panel and a small right one.
    // The whole page should pair with the largest split — the left one, which here
    // carries our exact fit — not the small right one (a different, far-off fit).
    const pages = [page({ pageKey: 'p5', itemIndex: 0 })];
    const truth = [
      page({
        pageKey: 'p5',
        itemIndex: 0,
        clipPolygon: halfClip(0, 800), // large left panel, same fit as ours
      }),
      page({
        pageKey: 'p5',
        itemIndex: 1,
        clipPolygon: halfClip(800, 1000), // small right panel, far-off fit
        corners: cornersAt(-70.0, 40.7),
      }),
    ];
    const stats = compareToTruth(pages, truth);
    expect(stats.get(0)).not.toBeNull();
    expect(stats.get(0)!.rmseFt).toBeCloseTo(0, 3);
  });

  it('matches split panels by overlap even when the numbering differs', () => {
    // We split p6 into left (__1) and right (__2); truth numbered them the other way.
    // Overlap must pair each of our panels with the truth panel it covers, so both
    // read ~0 error (each panel carries the matching truth fit).
    const pages = [
      page({
        pageKey: 'p6',
        itemIndex: 0,
        splitIndex: 1,
        clipPolygon: halfClip(0, 500), // our left panel
        corners: cornersAt(-74.0, 40.7),
      }),
      page({
        pageKey: 'p6',
        itemIndex: 1,
        splitIndex: 2,
        clipPolygon: halfClip(500, 1000), // our right panel
        corners: cornersAt(-73.0, 40.7),
      }),
    ];
    const truth = [
      page({
        pageKey: 'p6',
        itemIndex: 0,
        clipPolygon: halfClip(500, 1000), // truth right panel
        corners: cornersAt(-73.0, 40.7),
      }),
      page({
        pageKey: 'p6',
        itemIndex: 1,
        clipPolygon: halfClip(0, 500), // truth left panel
        corners: cornersAt(-74.0, 40.7),
      }),
    ];
    const stats = compareToTruth(pages, truth);
    expect(stats.get(0)!.rmseFt).toBeCloseTo(0, 3);
    expect(stats.get(1)!.rmseFt).toBeCloseTo(0, 3);
  });
});
