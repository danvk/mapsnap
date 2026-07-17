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
    scalePixelsPerFoot: 1,
    rotationDegrees: 0,
    gcps: [],
    transformationType: 'polynomial',
    ...overrides,
  };
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
  it('pairs by page key and nulls split panels', () => {
    const pages = [
      page({ pageKey: 'p1', itemIndex: 0 }),
      page({ pageKey: 'p2__1', itemIndex: 1 }),
      page({ pageKey: 'p9', itemIndex: 2 }), // no truth
    ];
    const truth = [
      page({ pageKey: 'p1', itemIndex: 0 }),
      page({ pageKey: 'p2', itemIndex: 1 }),
    ];
    const stats = compareToTruth(pages, truth);
    expect(stats.get(0)?.rmseFt).toBeCloseTo(0, 5);
    expect(stats.get(1)).toBeNull(); // split: parent-frame truth
    expect(stats.has(2)).toBe(false);
  });

  it('nulls pages whose truth key is ambiguous', () => {
    const pages = [page({ pageKey: 'p3', itemIndex: 0 })];
    const truth = [
      page({ pageKey: 'p3', itemIndex: 0 }),
      page({ pageKey: 'p3', itemIndex: 1 }),
    ];
    expect(compareToTruth(pages, truth).get(0)).toBeNull();
  });
});
